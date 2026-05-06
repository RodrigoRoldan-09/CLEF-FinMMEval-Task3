"""
Multi-branch trading model for FinMMEval Task 3.

Branches:
    1) scalar_now     (F,)             -> MLP -> (D,)
    2) scalar_seq     (T, F)           -> PatchTST-tiny -> (D,)
    3) news           (N, 768) + mask  -> attention pool (query=symbol+type) -> proj (D,)
    4) filing         (768,)           -> proj -> (D,)
    5) identity       (symbol_id, asset_type) -> emb (D_id,)

Fusion: Gated Multimodal Fusion (one gate per modality, per sample) ->
        cross-attention between branches -> head (3-class + scalar regression).

The forward also returns:
    - logits, return_pred              (for losses)
    - gate_weights (B, 4)              (interpretability)
    - news_attn_weights (B, N)         (which news mattered most)
    - feature_grad_handle (callable)   (gradient x input post-hoc, lazy)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    n_scalar_features: int        # se completa al instanciar (depende de selección)
    n_symbols: int = 12
    seq_len: int = 30
    max_news: int = 8
    finbert_dim: int = 768
    d_model: int = 64             # dim común para fusión
    d_id: int = 8                 # dim del embedding de símbolo
    patch_len: int = 5
    n_patch_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.20
    n_classes: int = 3            # SELL, HOLD, BUY


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------

class ScalarMLP(nn.Module):
    """Rama escalar: features del día actual."""

    def __init__(self, n_in: int, d_out: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchTSTBlock(nn.Module):
    """Bloque transformer simple para serie temporal."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        h = self.norm2(x)
        x = x + self.drop(self.ffn(h))
        return x


class PatchTSTEncoder(nn.Module):
    """
    PatchTST-tiny: parte la serie en patches no solapados, los embeda y los pasa por un
    pequeño transformer. Hace pooling final con mean para obtener un vector por muestra.
    """

    def __init__(self, n_features: int, seq_len: int, patch_len: int, d_model: int,
                 n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = math.ceil(seq_len / patch_len)
        self.pad_len = self.n_patches * patch_len - seq_len

        self.patch_proj = nn.Linear(patch_len * n_features, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            PatchTSTBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        B, T, F_ = x.shape
        if self.pad_len > 0:
            # left-pad replicando primer step (no info futura)
            pad = x[:, :1, :].expand(B, self.pad_len, F_)
            x = torch.cat([pad, x], dim=1)
        # patches
        x = x.view(B, self.n_patches, self.patch_len * F_)
        h = self.patch_proj(x) + self.pos_emb
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return h.mean(dim=1)  # (B, d_model)


class NewsAttentionPool(nn.Module):
    """
    Attention pooling sobre las N noticias del día. La query es el embedding del símbolo
    concatenado con el flag de tipo de activo. Devuelve (B, d_model) y los pesos de atención.
    """

    def __init__(self, finbert_dim: int, query_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, d_model)
        self.k_proj = nn.Linear(finbert_dim, d_model)
        self.v_proj = nn.Linear(finbert_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, news_embs, news_weights, news_mask, query):
        """
        news_embs:    (B, N, 768)
        news_weights: (B, N)        peso temporal (1.0 día actual, decay para weekend carry)
        news_mask:    (B, N)        1 si hay noticia, 0 si padding
        query:        (B, query_dim)
        """
        q = self.q_proj(query).unsqueeze(1)            # (B, 1, D)
        k = self.k_proj(news_embs)                     # (B, N, D)
        v = self.v_proj(news_embs)                     # (B, N, D)

        scores = (q * k).sum(dim=-1) * self.scale      # (B, N)
        # multiplicamos por weight temporal antes del softmax (logit space sería mejor pero esto funciona bien)
        # mejor: sumar log(weight) en logit space para que weight=0 -> -inf
        log_w = torch.log(news_weights.clamp(min=1e-6))
        scores = scores + log_w

        # mask
        scores = scores.masked_fill(news_mask < 0.5, -1e9)
        # si TODA la fila es mask, evitar NaN: forzar al menos 1 slot
        all_masked = (news_mask.sum(dim=-1, keepdim=True) < 0.5)
        # cuando todo está enmascarado dejamos un attention uniforme dummy y luego output=0
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(all_masked, torch.zeros_like(attn), attn)

        pooled = (attn.unsqueeze(-1) * v).sum(dim=1)   # (B, D)
        pooled = self.norm(self.drop(self.out_proj(pooled)))
        return pooled, attn


class FilingProj(nn.Module):
    """Proyección del embedding del filing-state (con decay ya aplicado)."""

    def __init__(self, finbert_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(finbert_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
# Gated Multimodal Fusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """
    Cada rama produce un vector de d_model. Un gate aprendido decide cuánto pesa cada
    modalidad por muestra. Los gates se calculan sobre la concatenación de todas las
    ramas (so the model knows what's available per sample) y son sigmoid (0-1) — no
    softmax, porque no son mutuamente excluyentes.

    Después de aplicar gates, hacemos cross-attention ligero entre las 4 ramas tratadas
    como tokens, y hacemos pooling final (mean).
    """

    def __init__(self, d_model: int, n_branches: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_branches = n_branches
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * n_branches, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_branches),
        )
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, branches: list[torch.Tensor], availability: torch.Tensor):
        """
        branches: lista de (B, D)
        availability: (B, n_branches)  - 1 si la rama tiene info, 0 si está enmascarada
                                         (e.g. has_filings=0 para cripto, has_news=0 si no hay noticias)
        """
        stacked = torch.stack(branches, dim=1)          # (B, K, D)
        flat = stacked.view(stacked.size(0), -1)        # (B, K*D)
        gate_logits = self.gate_net(flat)
        # multiplicamos por availability: las ramas no disponibles tienen gate=0
        gates = torch.sigmoid(gate_logits) * availability  # (B, K)

        gated = stacked * gates.unsqueeze(-1)           # (B, K, D)

        # cross-attention entre ramas (cada rama atiende a las otras)
        h, _ = self.cross_attn(gated, gated, gated, need_weights=False)
        h = self.norm(gated + self.drop(h))

        # pooling: mean sobre las ramas, ponderando por gate (las ramas pesan más cuando el gate es alto)
        gates_norm = gates / gates.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        pooled = (h * gates_norm.unsqueeze(-1)).sum(dim=1)  # (B, D)

        return pooled, gates


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TradingModel(nn.Module):
    BRANCH_NAMES = ["scalar_now", "scalar_seq", "news", "filing"]

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        # identidad
        self.symbol_emb = nn.Embedding(cfg.n_symbols, cfg.d_id)
        # query para news attention = symbol_emb + asset_type_flag (1-dim)
        query_dim = cfg.d_id + 1

        # ramas
        self.scalar_now_branch = ScalarMLP(cfg.n_scalar_features, D, cfg.dropout)
        self.scalar_seq_branch = PatchTSTEncoder(
            n_features=cfg.n_scalar_features,
            seq_len=cfg.seq_len,
            patch_len=cfg.patch_len,
            d_model=D,
            n_layers=cfg.n_patch_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )
        self.news_branch = NewsAttentionPool(cfg.finbert_dim, query_dim, D, cfg.dropout)
        self.filing_branch = FilingProj(cfg.finbert_dim, D, cfg.dropout)

        # fusión
        n_branches = 4
        self.fusion = GatedFusion(D, n_branches, cfg.n_heads, cfg.dropout)

        # añadimos identidad al vector pooled antes de la cabeza
        self.head_in_dim = D + cfg.d_id + 1

        # cabezas
        self.cls_head = nn.Sequential(
            nn.Linear(self.head_in_dim, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(D, cfg.n_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(self.head_in_dim, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(D, 1),
        )

    def forward(self, batch: dict) -> dict:
        """
        batch keys (todos en device del módulo):
            scalar_now:        (B, F)
            scalar_seq:        (B, T, F)
            news_embs:         (B, N, 768)
            news_weights:      (B, N)
            news_mask:         (B, N)
            filing_emb:        (B, 768)
            symbol_id:         (B,) long
            asset_type_crypto: (B,) float
            has_filings:       (B,) float
        """
        sym_emb = self.symbol_emb(batch["symbol_id"])                    # (B, d_id)
        asset_flag = batch["asset_type_crypto"].unsqueeze(-1)            # (B, 1)
        identity = torch.cat([sym_emb, asset_flag], dim=-1)              # (B, d_id+1)

        b1 = self.scalar_now_branch(batch["scalar_now"])
        b2 = self.scalar_seq_branch(batch["scalar_seq"])
        b3, news_attn = self.news_branch(
            batch["news_embs"], batch["news_weights"], batch["news_mask"], identity
        )
        b4 = self.filing_branch(batch["filing_emb"])

        # availability:
        # scalar_now y scalar_seq siempre 1
        # news: 1 si hay al menos una noticia
        # filing: has_filings (1 stocks, 0 crypto)
        has_news = (batch["news_mask"].sum(dim=-1) > 0).float()
        availability = torch.stack([
            torch.ones_like(has_news),
            torch.ones_like(has_news),
            has_news,
            batch["has_filings"],
        ], dim=-1)  # (B, 4)

        fused, gates = self.fusion([b1, b2, b3, b4], availability)
        h = torch.cat([fused, identity], dim=-1)

        logits = self.cls_head(h)
        ret_pred = self.reg_head(h).squeeze(-1)

        return {
            "logits": logits,
            "return_pred": ret_pred,
            "gates": gates,                # (B, 4)
            "news_attn": news_attn,        # (B, N)
            "branch_repr": {
                "scalar_now": b1, "scalar_seq": b2, "news": b3, "filing": b4
            },
        }


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    def __init__(self, class_weights=None,
                 alpha_cls: float = 0.7, alpha_reg: float = 0.3,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.alpha_cls = alpha_cls
        self.alpha_reg = alpha_reg
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.reg_loss = nn.HuberLoss(delta=1.0)

    def forward(self, out: dict, target_class: torch.Tensor, target_return_norm: torch.Tensor):
        l_cls = self.cls_loss(out["logits"], target_class)
        l_reg = self.reg_loss(out["return_pred"], target_return_norm)
        total = self.alpha_cls * l_cls + self.alpha_reg * l_reg
        return {"total": total, "cls": l_cls.detach(), "reg": l_reg.detach()}
