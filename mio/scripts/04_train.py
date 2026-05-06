"""
Script 04 - Train the trading model (v2, anti-overfitting).

Cambios vs v1:
    1. Modelo más pequeño por defecto (d_model=32, n_patch_layers=2) → ~80K params
    2. Data augmentation temporal:
         - Gaussian noise en scalar features (sigma=0.03)
         - Dropout de noticias (30% de probabilidad de silenciar toda la rama)
         - Time shift: offset aleatorio ±2 pasos en la ventana de secuencia
    3. Stochastic Weight Averaging (SWA) a partir de epoch 30 → generalización
    4. Early stopping sobre val_f1 suavizado con EMA (α=0.4) en lugar del Sharpe
       ruidoso. Con 360 muestras / 12 activos el Sharpe tiene altísima varianza.
    5. LR warmup (5 epochs) + CosineDecay
    6. Label smoothing 0.10 (subió de 0.05) para reducir confianza excesiva
    7. weight_decay 1e-2 (subió de 1e-4) para más regularización L2
    8. batch_size 32 (bajó de 64) → más pasos de grad por epoch, gradientes menos
       correlacionados con series temporales.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.architecture import TradingModel, ModelConfig, MultiTaskLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train")

FINAL_DIR = PROJECT_ROOT / "data" / "final"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

def set_seed(seed=42):
    import random
    import os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TensorBatchDataset(Dataset):
    KEYS = [
        "symbol_id", "asset_type_crypto",
        "scalar_seq", "scalar_now",
        "news_embs", "news_weights", "news_mask", "news_sentiment",
        "filing_emb", "has_filings",
        "target_class", "target_return_norm",
    ]

    def __init__(self, path: Path):
        log.info(f"loading {path}")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.tensors = {k: blob[k] for k in self.KEYS if k in blob}
        self.dates = blob.get("dates", [])
        self.symbols = blob.get("symbols", [])
        self.n = len(self.tensors["target_class"])
        log.info(f"  {self.n} samples")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tensors.items()}


def collate(batch_list):
    out = {}
    for k in batch_list[0]:
        v = batch_list[0][k]
        if torch.is_tensor(v):
            out[k] = torch.stack([b[k] for b in batch_list])
        else:
            out[k] = [b[k] for b in batch_list]
    return out


# ---------------------------------------------------------------------------
# Data augmentation (solo en training)
# ---------------------------------------------------------------------------

def augment_batch(batch: dict, args, device) -> dict:
    """
    Aplica augmentation in-place sobre el batch ya en device.
    NUNCA se aplica en eval/test.
    """
    aug = {k: v for k, v in batch.items()}

    # 1. Ruido gaussiano en features escalares
    if args.aug_noise > 0:
        aug["scalar_now"] = aug["scalar_now"] + torch.randn_like(aug["scalar_now"]) * args.aug_noise
        aug["scalar_seq"] = aug["scalar_seq"] + torch.randn_like(aug["scalar_seq"]) * args.aug_noise

    # 2. News dropout: con prob p, silenciar toda la rama de noticias de esa muestra.
    #    Enseña al modelo a decidir sin noticias (muchos días no tienen news).
    if args.aug_news_drop > 0:
        B = aug["news_mask"].shape[0]
        drop = (torch.rand(B, device=device) < args.aug_news_drop).float().unsqueeze(-1)
        aug["news_mask"] = aug["news_mask"] * (1 - drop)

    # 3. Time shift: desplazar aleatoriamente la ventana temporal ±k pasos
    #    Reduce memorización de posiciones fijas dentro del window.
    if args.aug_time_shift > 0:
        shift = int(torch.randint(-args.aug_time_shift, args.aug_time_shift + 1, (1,)).item())
        if shift != 0:
            aug["scalar_seq"] = torch.roll(aug["scalar_seq"], shifts=shift, dims=1)

    return aug


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_class_weights(targets: torch.Tensor, n_classes: int = 3) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=n_classes).float().clamp(min=1.0)
    return counts.sum() / (n_classes * counts)



@torch.no_grad()
def evaluate(model, loader, device, threshold=0.48):
    model.eval()
    all_logits, all_targets, all_ret_true, all_ret_pred = [], [], [], []
    for batch in loader:
        batch_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch_dev)
        all_logits.append(out["logits"].cpu())
        all_targets.append(batch["target_class"])
        all_ret_true.append(batch["target_return_norm"])
        all_ret_pred.append(out["return_pred"].cpu()) # Guardamos la predicción de regresión

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    ret_true = torch.cat(all_ret_true)
    ret_preds = torch.cat(all_ret_pred)

    # --- INFERENCIA HÍBRIDA ---
    probs = torch.softmax(logits, dim=-1)
    
    threshold = 0.48
    
    preds = torch.ones(len(logits), dtype=torch.long)
    preds[probs[:, 0] > threshold] = 0
    preds[probs[:, 2] > threshold] = 2

    acc = (preds == targets).float().mean().item()

    f1s = []
    for c in range(3):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    f1_active = float(np.mean([f1s[0], f1s[2]]))  # SELL + BUY (excluye HOLD)

    action_signed = torch.where(preds == 2, 1.0, torch.where(preds == 0, -1.0, 0.0))
    pnl = (action_signed * ret_true).numpy()
    sharpe = pnl.mean() / (pnl.std() + 1e-8) * math.sqrt(252)
    cum_ret = float(pnl.sum())
    active = action_signed != 0
    
    if active.sum() > 0:
        win_rate = ((action_signed * ret_true) > 0)[active].float().mean().item()
        gross_profit = pnl[pnl > 0].sum().item()
        gross_loss = abs(pnl[pnl < 0].sum().item())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    else:
        win_rate = 0.0
        profit_factor = 0.0

    out_metrics = {
        "accuracy": acc, 
        "macro_f1": macro_f1, 
        "f1_active": f1_active,
        "sharpe_sim": float(sharpe), 
        "cum_return_sim": cum_ret, 
        "win_rate": win_rate,           
        "profit_factor": profit_factor, 
        "n_buy": int((preds == 2).sum().item()),
        "n_hold": int((preds == 1).sum().item()),
        "n_sell": int((preds == 0).sum().item()),
    }
    
    return out_metrics


# ---------------------------------------------------------------------------
# LR schedule: warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr_lambda(warmup_epochs: int, total_epochs: int, min_factor: float = 0.05):
    def fn(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
    return fn


# ---------------------------------------------------------------------------
# EMA suavizado para early stopping estable
# ---------------------------------------------------------------------------

class EMAMetric:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    ##set_seed(42)
    device = torch.device(args.device)
    log.info(f"device: {device}")

    train_ds = TensorBatchDataset(FINAL_DIR / "train.pt")
    val_ds   = TensorBatchDataset(FINAL_DIR / "val.pt")
    test_ds  = TensorBatchDataset(FINAL_DIR / "test.pt")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    with open(FINAL_DIR / "manifest.json") as f:
        manifest = json.load(f)

    cfg = ModelConfig(
        n_scalar_features=manifest["n_scalar_features"],
        n_symbols=manifest["n_symbols"],
        seq_len=manifest["seq_len"],
        max_news=manifest["max_news_per_day"],
        finbert_dim=manifest["finbert_dim"],
        d_model=args.d_model,
        d_id=args.d_id,
        n_patch_layers=args.n_patch_layers,
        dropout=args.dropout,
    )
    log.info(f"config: F={cfg.n_scalar_features}, S={cfg.n_symbols}, D={cfg.d_model}, layers={cfg.n_patch_layers}")

    model = TradingModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"params: {n_params:,}")

    class_weights = compute_class_weights(train_ds.tensors["target_class"]).to(device)
    log.info(f"class weights: {class_weights.tolist()}")

    loss_fn = MultiTaskLoss(
        class_weights=class_weights,
        alpha_cls=args.alpha_cls,
        alpha_reg=args.alpha_reg,
        label_smoothing=args.label_smoothing,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_lambda = get_lr_lambda(args.warmup_epochs, args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    swa_model = AveragedModel(model) if args.swa_start > 0 else None
    swa_scheduler = SWALR(optim, swa_lr=args.lr * 0.05) if args.swa_start > 0 else None

    best_val_score = -float("inf")
    best_path = CHECKPOINT_DIR / "best.pt"
    epochs_no_improve = 0
    ema_f1 = EMAMetric(alpha=0.4)

    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_cls = total_reg = 0
        n_batches = 0

        for batch in train_loader:
            batch_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch_dev = augment_batch(batch_dev, args, device)
            out = model(batch_dev)
            losses = loss_fn(out, batch_dev["target_class"], batch_dev["target_return_norm"])

            optim.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optim.step()

            total_loss += losses["total"].item()
            total_cls  += losses["cls"].item()
            total_reg  += losses["reg"].item()
            n_batches  += 1

        in_swa = args.swa_start > 0 and epoch >= args.swa_start
        if in_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        train_loss = total_loss / max(n_batches, 1)
        current_lr = optim.param_groups[0]["lr"]

        val_metrics = evaluate(model, val_loader, device)
        ema_f1_val = ema_f1.update(val_metrics["macro_f1"])

        # score: 80% EMA-F1 (estable) + 20% Sharpe suavizado
        val_score = 0.8 * ema_f1_val + 0.2 * np.tanh(val_metrics["sharpe_sim"] / 2)

        log.info(
            f"epoch {epoch:3d} | lr {current_lr:.2e} | tr {train_loss:.4f} "
            f"(cls {total_cls/n_batches:.4f} reg {total_reg/n_batches:.4f}) | "
            f"val f1 {val_metrics['macro_f1']:.3f} ema_f1 {ema_f1_val:.3f} "
            f"sharpe {val_metrics['sharpe_sim']:.2f} win {val_metrics['win_rate']:.3f} "
            f"PF {val_metrics['profit_factor']:.2f} | "
            f"B/H/S {val_metrics['n_buy']}/{val_metrics['n_hold']}/{val_metrics['n_sell']} | "
            f"score {val_score:.4f}" + (" [SWA]" if in_swa else "")
        )

        history.append({
            "epoch": epoch, "lr": current_lr, "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "val_ema_f1": ema_f1_val, "val_score": val_score,
        })

        if val_score > best_val_score + 1e-4:
            best_val_score = val_score
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "epoch": epoch,
                "val_score": val_score,
                "val_metrics": val_metrics,
            }, best_path)
            epochs_no_improve = 0
            log.info(f"  -> saved best.pt (val_score={val_score:.4f}, ema_f1={ema_f1_val:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                log.info(f"early stop at epoch {epoch}")
                break

    # SWA final
    if swa_model is not None and args.swa_start > 0:
        log.info("updating SWA batch norm...")
        update_bn(train_loader, swa_model, device=device)
        swa_metrics = evaluate(swa_model, val_loader, device)
        log.info(f"SWA val: f1={swa_metrics['macro_f1']:.3f} sharpe={swa_metrics['sharpe_sim']:.2f}")
        torch.save({
            "model_state": swa_model.module.state_dict(),
            "config": cfg.__dict__,
            "epoch": -1,
            "val_score": swa_metrics["macro_f1"],
            "val_metrics": swa_metrics,
            "is_swa": True,
        }, CHECKPOINT_DIR / "best_swa.pt")
        log.info("guardado best_swa.pt")

    log.info(f"training done in {time.time()-t_start:.1f}s")

    # test
    log.info("--- TEST (best checkpoint) ---")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device, threshold=0.48)
    log.info(
        f"TEST: acc {test_metrics['accuracy']:.3f} f1 {test_metrics['macro_f1']:.3f} "
        f"f1_active {test_metrics['f1_active']:.3f} "
        f"sharpe {test_metrics['sharpe_sim']:.2f} cum_ret {test_metrics['cum_return_sim']:.3f} "
        f"win {test_metrics['win_rate']:.3f} PF {test_metrics['profit_factor']:.2f}"
    )

    with open(CHECKPOINT_DIR / "training_history.json", "w") as f:
        json.dump({
            "history": history,
            "test_metrics": {k: v for k, v in test_metrics.items() if k != "pnl_array"},
            "best_epoch": ckpt["epoch"],
            "best_val_score": ckpt["val_score"],
        }, f, indent=2)

    log.info(f"DONE — best_epoch={ckpt['epoch']} best_val_score={ckpt['val_score']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs",          type=int,   default=80)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight_decay",    type=float, default=1e-2)
    p.add_argument("--dropout",         type=float, default=0.35)
    p.add_argument("--d_model",         type=int,   default=32)
    p.add_argument("--d_id",            type=int,   default=8)
    p.add_argument("--n_patch_layers",  type=int,   default=2)
    p.add_argument("--patience",        type=int,   default=20)
    p.add_argument("--warmup_epochs",   type=int,   default=5)
    p.add_argument("--grad_clip",       type=float, default=0.5)
    p.add_argument("--alpha_cls",       type=float, default=0.8)
    p.add_argument("--alpha_reg",       type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.10)
    p.add_argument("--swa_start",       type=int,   default=30,  help="epoch donde empieza SWA (0=desactivar)")
    p.add_argument("--aug_noise",       type=float, default=0.03)
    p.add_argument("--aug_news_drop",   type=float, default=0.30)
    p.add_argument("--aug_time_shift",  type=int,   default=2)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
