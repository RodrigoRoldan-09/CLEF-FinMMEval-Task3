"""
Script 03 - Build final tensorized dataset.

Toma el raw + las features seleccionadas, calcula embeddings de FinBERT (con cache
en disco) para noticias y filings (chunking + mean pooling para textos largos),
arma tensores PyTorch con todas las ramas y splits temporales por activo.

Output:
    data/final/train.pt           - dict de tensores
    data/final/val.pt
    data/final/test.pt
    data/final/scaler.json        - mean/std de features escalares (fit en train)
    data/final/symbol_vocab.json  - mapeo symbol -> id para embedding aprendido
    data/final/manifest.json      - metadata, shapes, version
    data/cache/finbert_news.h5    - cache de embeddings (compartido entre runs)
    data/cache/finbert_filings.h5

Splits temporales por activo: 65% train / 15% val / 20% test.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_final")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw_features"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "feature_analysis"
OUT_DIR = PROJECT_ROOT / "data" / "final"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Config
FINBERT_MODEL = "yiyanghkust/finbert-tone"
FINBERT_DIM = 768
SEQ_LEN = 30  # ventana temporal para la rama secuencial
MAX_NEWS_PER_DAY = 8  # capamos noticias para tensor de tamaño fijo
NEWS_MAX_TOKENS = 256
FILING_CHUNK_TOKENS = 384
FILING_MAX_CHUNKS = 12
SPLIT_TRAIN = 0.65
SPLIT_VAL = 0.15
# test = 1 - train - val

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Cache simple basado en HDF5 (un archivo) o npz por hash. Usamos npz porque
    HDF5 a veces da problemas de concurrencia y el dataset es chico."""

    def __init__(self, name: str):
        self.path = CACHE_DIR / f"{name}.npz"
        self.data = {}
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as f:
                self.data = {k: f[k] for k in f.files}
            log.info(f"  loaded cache {name}: {len(self.data)} entries")

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]

    def get(self, text: str) -> Optional[np.ndarray]:
        k = self.key(text)
        return self.data.get(k)

    def put(self, text: str, emb: np.ndarray):
        self.data[self.key(text)] = emb.astype(np.float32)

    def save(self):
        if not self.data:
            return
        np.savez_compressed(self.path, **self.data)
        log.info(f"  saved cache {self.path.name}: {len(self.data)} entries")


# ---------------------------------------------------------------------------
# FinBERT
# ---------------------------------------------------------------------------

class FinBertEncoder:
    """Frozen FinBERT. Devuelve embedding [CLS] (768-dim) y sentiment logits (3-dim)."""

    def __init__(self):
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

        log.info(f"loading FinBERT ({FINBERT_MODEL}) on {DEVICE}")
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        # modelo base para embeddings
        self.encoder = AutoModel.from_pretrained(FINBERT_MODEL).to(DEVICE).eval()
        # mismo modelo con head de clasificación para sentiment scores
        try:
            self.classifier = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(DEVICE).eval()
            self.has_classifier = True
        except Exception as e:
            log.warning(f"no se pudo cargar classifier head: {e}")
            self.has_classifier = False

        if DEVICE == "cuda":
            self.encoder = self.encoder.half()
            if self.has_classifier:
                self.classifier = self.classifier.half()

    @torch.no_grad()
    def embed(self, text: str, max_tokens: int = NEWS_MAX_TOKENS) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            cls_emb: (768,) embedding [CLS]
            sentiment: (3,) softmax probs [neg, neu, pos] o zeros si no hay classifier
        """
        if not text or not text.strip():
            return np.zeros(FINBERT_DIM, dtype=np.float32), np.array([0, 1, 0], dtype=np.float32)

        toks = self.tokenizer(text, truncation=True, max_length=max_tokens, return_tensors="pt").to(DEVICE)
        out = self.encoder(**toks)
        cls = out.last_hidden_state[:, 0, :].squeeze(0).float().cpu().numpy()

        if self.has_classifier:
            logits = self.classifier(**toks).logits.squeeze(0).float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # FinBERT-tone orden: [neutral, positive, negative] - normalizamos a [neg, neu, pos]
            # según docs de yiyanghkust: 0=neutral, 1=positive, 2=negative
            sentiment = np.array([probs[2], probs[0], probs[1]], dtype=np.float32)
        else:
            sentiment = np.array([0, 1, 0], dtype=np.float32)
        return cls, sentiment

    @torch.no_grad()
    def embed_long(self, text: str, chunk_tokens: int = FILING_CHUNK_TOKENS, max_chunks: int = FILING_MAX_CHUNKS) -> np.ndarray:
        """Para filings largos: chunking + mean pooling sobre los [CLS] de cada chunk."""
        if not text or not text.strip():
            return np.zeros(FINBERT_DIM, dtype=np.float32)

        # tokenizamos completo y partimos en chunks
        all_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not all_ids:
            return np.zeros(FINBERT_DIM, dtype=np.float32)

        cls_token = self.tokenizer.cls_token_id
        sep_token = self.tokenizer.sep_token_id
        chunk_ids_list = []
        # menos 2 por [CLS] y [SEP]
        step = chunk_tokens - 2
        for i in range(0, len(all_ids), step):
            chunk = all_ids[i:i + step]
            chunk = [cls_token] + chunk + [sep_token]
            chunk_ids_list.append(chunk)
            if len(chunk_ids_list) >= max_chunks:
                break

        embeddings = []
        for chunk in chunk_ids_list:
            ids = torch.tensor([chunk], device=DEVICE)
            attn = torch.ones_like(ids)
            out = self.encoder(input_ids=ids, attention_mask=attn)
            embeddings.append(out.last_hidden_state[:, 0, :].squeeze(0).float().cpu().numpy())

        return np.mean(embeddings, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Embeddings por día
# ---------------------------------------------------------------------------

def embed_news_day(weighted_news: list[dict], encoder: FinBertEncoder, cache: EmbeddingCache):
    """
    Devuelve:
        embs: (MAX_NEWS_PER_DAY, FINBERT_DIM)
        weights: (MAX_NEWS_PER_DAY,)  - peso temporal (1.0 para news del día, decay para weekend carry)
        mask: (MAX_NEWS_PER_DAY,)     - 1 si la slot tiene news, 0 si padding
        sentiment_agg: (3,)           - sentiment promedio ponderado del día
        n_news: int
    """
    embs = np.zeros((MAX_NEWS_PER_DAY, FINBERT_DIM), dtype=np.float32)
    weights = np.zeros(MAX_NEWS_PER_DAY, dtype=np.float32)
    mask = np.zeros(MAX_NEWS_PER_DAY, dtype=np.float32)
    sentiments = []
    sent_weights = []

    if not weighted_news:
        return embs, weights, mask, np.array([0, 1, 0], dtype=np.float32), 0

    # ordenar por peso descendente (las más relevantes primero)
    sorted_news = sorted(weighted_news, key=lambda x: -float(x.get("weight", 1.0)))

    for i, wn in enumerate(sorted_news[:MAX_NEWS_PER_DAY]):
        text = wn["text"]
        w = float(wn.get("weight", 1.0))
        cached = cache.get(text)
        if cached is not None and cached.shape[0] == FINBERT_DIM + 3:
            cls = cached[:FINBERT_DIM]
            sent = cached[FINBERT_DIM:]
        else:
            cls, sent = encoder.embed(text, max_tokens=NEWS_MAX_TOKENS)
            cache.put(text, np.concatenate([cls, sent]))

        embs[i] = cls
        weights[i] = w
        mask[i] = 1.0
        sentiments.append(sent)
        sent_weights.append(w)

    if sentiments:
        sw = np.array(sent_weights)
        sw = sw / sw.sum()
        sent_agg = np.average(np.stack(sentiments), axis=0, weights=sw).astype(np.float32)
    else:
        sent_agg = np.array([0, 1, 0], dtype=np.float32)

    return embs, weights, mask, sent_agg, len(sorted_news)


def embed_filing_day(filing_texts: list[str], encoder: FinBertEncoder, cache: EmbeddingCache) -> np.ndarray:
    """Combina (mean) los embeddings de los filings del día. Para no-stocks o sin filings, zeros."""
    if not filing_texts:
        return np.zeros(FINBERT_DIM, dtype=np.float32)

    embs = []
    for text in filing_texts:
        cached = cache.get(text)
        if cached is not None and cached.shape[0] == FINBERT_DIM:
            embs.append(cached)
        else:
            emb = encoder.embed_long(text)
            cache.put(text, emb)
            embs.append(emb)
    return np.mean(embs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Filing carry-forward
# ---------------------------------------------------------------------------

def compute_filing_state_per_row(df_symbol: pd.DataFrame, encoder: FinBertEncoder, cache: EmbeddingCache, asset_type: str) -> np.ndarray:
    """
    Por cada día, mantiene el embedding del último filing visto (con decay temporal).
    Si no hay filings vistos aún, zeros + flag.
    Salida: (n_days, FINBERT_DIM)
    """
    n = len(df_symbol)
    out = np.zeros((n, FINBERT_DIM), dtype=np.float32)
    if asset_type == "crypto":
        return out

    last_filing_emb = None
    last_filing_day = -1
    half_life = 45.0
    lam = math.log(2) / half_life

    for i, row in enumerate(df_symbol.itertuples(index=False)):
        # combinar 10k + 10q de hoy
        today_filings = []
        if hasattr(row, "ten_k") and len(row.ten_k) > 0:
            today_filings.extend(list(row.ten_k))
        if hasattr(row, "ten_q") and len(row.ten_q) > 0:
            today_filings.extend(list(row.ten_q))

        if today_filings:
            emb = embed_filing_day(today_filings, encoder, cache)
            last_filing_emb = emb
            last_filing_day = i

        if last_filing_emb is not None:
            age = i - last_filing_day
            decay = math.exp(-lam * age)
            out[i] = last_filing_emb * decay

    return out


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def build_sequence_features(df_symbol: pd.DataFrame, scalar_cols: list[str], seq_len: int) -> np.ndarray:
    """
    Para cada día t, construye una ventana de [t-seq_len+1 ... t] de las features escalares.
    Si t < seq_len-1, pad con el primer valor disponible (forward fill desde el inicio).
    Salida: (n_days, seq_len, n_features)
    """
    n = len(df_symbol)
    f = len(scalar_cols)
    arr = df_symbol[scalar_cols].values.astype(np.float32)
    # forward-fill NaNs en cada columna
    for c in range(f):
        col = arr[:, c]
        last = 0.0
        first_valid = False
        for i in range(n):
            if np.isnan(col[i]):
                col[i] = last
            else:
                last = col[i]
                first_valid = True
        if not first_valid:
            col[:] = 0.0
        arr[:, c] = col

    out = np.zeros((n, seq_len, f), dtype=np.float32)
    for t in range(n):
        start = max(0, t - seq_len + 1)
        window = arr[start:t + 1]
        if len(window) < seq_len:
            pad = np.repeat(window[:1], seq_len - len(window), axis=0)
            window = np.vstack([pad, window])
        out[t] = window
    return out


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def make_splits_per_symbol(df: pd.DataFrame) -> dict[str, dict]:
    """Por cada símbolo: índices de train/val/test (respetando orden temporal)."""
    splits = {}
    for sym in df["symbol"].unique():
        mask = df["symbol"] == sym
        sub = df.loc[mask].sort_values("date")
        n = len(sub)
        n_tr = int(n * SPLIT_TRAIN)
        n_val = int(n * SPLIT_VAL)
        train_idx = sub.index[:n_tr].tolist()
        val_idx = sub.index[n_tr:n_tr + n_val].tolist()
        test_idx = sub.index[n_tr + n_val:].tolist()
        splits[sym] = {"train": train_idx, "val": val_idx, "test": test_idx}
    return splits


# ---------------------------------------------------------------------------
# Targets (replicamos lógica del script 02 — fuente única de verdad sería ideal,
# pero los thresholds vienen de target_distribution.json)
# ---------------------------------------------------------------------------

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_return"] = df["future_price_diff"] / df["price"]
    df["target_class"] = np.nan
    df["target_return_norm"] = np.nan

    dist_path = ANALYSIS_DIR / "target_distribution.json"
    if dist_path.exists():
        with open(dist_path) as f:
            dist = json.load(f)
        thresholds = dist["thresholds_per_symbol"]
        log.info(f"using thresholds from feature_analysis/target_distribution.json")
    else:
        thresholds = {}

    for sym in df["symbol"].unique():
        mask = df["symbol"] == sym
        sub = df.loc[mask].sort_values("date")
        ret = sub["target_return"]
        vol = ret.rolling(20, min_periods=5).std().shift(1)
        norm = ret / vol.replace(0, np.nan)
        df.loc[sub.index, "target_return_norm"] = norm.values

        thr = thresholds.get(sym)
        if thr is None:
            n = len(sub)
            train_cut = int(n * 0.65)
            train_ret = sub["target_return"].iloc[:train_cut].dropna()
            thr = float(np.quantile(np.abs(train_ret), 0.33)) if len(train_ret) >= 20 else 0.005

        cls = pd.Series(1, index=sub.index, dtype=float)
        cls.loc[ret > thr] = 2
        cls.loc[ret < -thr] = 0
        cls.loc[ret.isna()] = np.nan
        df.loc[sub.index, "target_class"] = cls.values

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # cargar selección
    sel_path = ANALYSIS_DIR / "selected_features.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"corre primero 02_feature_selection.py")
    with open(sel_path) as f:
        selection = json.load(f)
    scalar_cols = selection["features"]
    log.info(f"using {len(scalar_cols)} selected scalar features")

    # cargar dataset raw
    df = pd.read_parquet(RAW_DIR / "_all.parquet")
    log.info(f"loaded raw: {len(df)} rows")

    df = build_targets(df)

    # vocab de símbolos
    symbols = sorted(df["symbol"].unique())
    symbol_vocab = {sym: i for i, sym in enumerate(symbols)}
    with open(OUT_DIR / "symbol_vocab.json", "w") as f:
        json.dump(symbol_vocab, f, indent=2)

    # scaler fit en train: lo computamos al final cuando ya tenemos los splits
    splits_idx = make_splits_per_symbol(df)
    train_idx_global = sum([splits_idx[s]["train"] for s in symbols], [])

    train_subset = df.loc[train_idx_global, scalar_cols].astype(np.float32)
    means = train_subset.mean().fillna(0).values
    stds = train_subset.std().fillna(1).replace(0, 1).values
    log.info(f"scaler fit on {len(train_subset)} train rows")

    with open(OUT_DIR / "scaler.json", "w") as f:
        json.dump({
            "feature_columns": scalar_cols,
            "mean": means.tolist(),
            "std": stds.tolist(),
        }, f, indent=2)

    # standardize
    df_z = df.copy()
    for i, c in enumerate(scalar_cols):
        df_z[c] = (df[c].astype(float) - means[i]) / stds[i]

    # encoder
    log.info("loading FinBERT encoder")
    encoder = FinBertEncoder()
    news_cache = EmbeddingCache("finbert_news")
    filing_cache = EmbeddingCache("finbert_filings")

    # vamos a llenar listas por split
    splits_data = {"train": [], "val": [], "test": []}

    for sym_idx, sym in enumerate(symbols):
        mask = df_z["symbol"] == sym
        sub = df_z.loc[mask].sort_values("date").reset_index(drop=True)
        sub_raw = df.loc[mask].sort_values("date").reset_index(drop=True)
        asset_type = "crypto" if sub_raw["asset_type_crypto"].iloc[0] == 1 else "stock"
        log.info(f"[{sym}] type={asset_type}, n_days={len(sub)}")

        # secuencia temporal de features escalares
        seq = build_sequence_features(sub, scalar_cols, SEQ_LEN)  # (n, T, F)

        # filing state por día (con decay)
        log.info(f"  computing filing embeddings")
        filing_state = compute_filing_state_per_row(sub_raw, encoder, filing_cache, asset_type)

        # noticias por día
        log.info(f"  computing news embeddings")
        news_embs = np.zeros((len(sub), MAX_NEWS_PER_DAY, FINBERT_DIM), dtype=np.float32)
        news_weights = np.zeros((len(sub), MAX_NEWS_PER_DAY), dtype=np.float32)
        news_mask = np.zeros((len(sub), MAX_NEWS_PER_DAY), dtype=np.float32)
        news_sentiment = np.zeros((len(sub), 3), dtype=np.float32)
        news_count = np.zeros(len(sub), dtype=np.int32)

        for i, row in enumerate(sub_raw.itertuples(index=False)):
            wn = list(row.news_weighted) if hasattr(row, "news_weighted") else []
            embs, weights, mask_arr, sent, n_news = embed_news_day(wn, encoder, news_cache)
            news_embs[i] = embs
            news_weights[i] = weights
            news_mask[i] = mask_arr
            news_sentiment[i] = sent
            news_count[i] = n_news
            if (i + 1) % 50 == 0:
                log.info(f"    {i+1}/{len(sub)} days processed")

        # save caches periódicamente
        news_cache.save()
        filing_cache.save()

        # split
        sp = splits_idx[sym]
        # mapear índices globales a locales en sub
        global_to_local = {gi: li for li, gi in enumerate(df.loc[mask].sort_values("date").index)}

        for split_name in ["train", "val", "test"]:
            for gi in sp[split_name]:
                if gi not in global_to_local:
                    continue
                li = global_to_local[gi]
                # skip si target es NaN (último día del activo)
                if pd.isna(sub_raw["target_class"].iloc[li]):
                    continue
                sample = {
                    "symbol_id": symbol_vocab[sym],
                    "asset_type_crypto": float(sub_raw["asset_type_crypto"].iloc[li]),
                    "scalar_seq": seq[li],                      # (T, F)
                    "scalar_now": seq[li, -1],                  # (F,) último día
                    "news_embs": news_embs[li],                 # (MAX_NEWS, 768)
                    "news_weights": news_weights[li],           # (MAX_NEWS,)
                    "news_mask": news_mask[li],                 # (MAX_NEWS,)
                    "news_sentiment": news_sentiment[li],       # (3,)
                    "news_count": int(news_count[li]),
                    "filing_emb": filing_state[li],             # (768,)
                    "has_filings": float(asset_type == "stock"),
                    "target_class": int(sub_raw["target_class"].iloc[li]),
                    "target_return_norm": float(sub_raw["target_return_norm"].iloc[li]) if not pd.isna(sub_raw["target_return_norm"].iloc[li]) else 0.0,
                    "date": str(sub_raw["date"].iloc[li].date()),
                }
                splits_data[split_name].append(sample)

    # save caches finales
    news_cache.save()
    filing_cache.save()

    # convertir a tensores y guardar
    for split_name, samples in splits_data.items():
        if not samples:
            log.warning(f"{split_name} vacío!")
            continue
        log.info(f"packing {split_name}: {len(samples)} samples")
        tensors = {
            "symbol_id": torch.tensor([s["symbol_id"] for s in samples], dtype=torch.long),
            "asset_type_crypto": torch.tensor([s["asset_type_crypto"] for s in samples], dtype=torch.float32),
            "scalar_seq": torch.tensor(np.stack([s["scalar_seq"] for s in samples]), dtype=torch.float32),
            "scalar_now": torch.tensor(np.stack([s["scalar_now"] for s in samples]), dtype=torch.float32),
            "news_embs": torch.tensor(np.stack([s["news_embs"] for s in samples]), dtype=torch.float32),
            "news_weights": torch.tensor(np.stack([s["news_weights"] for s in samples]), dtype=torch.float32),
            "news_mask": torch.tensor(np.stack([s["news_mask"] for s in samples]), dtype=torch.float32),
            "news_sentiment": torch.tensor(np.stack([s["news_sentiment"] for s in samples]), dtype=torch.float32),
            "news_count": torch.tensor([s["news_count"] for s in samples], dtype=torch.long),
            "filing_emb": torch.tensor(np.stack([s["filing_emb"] for s in samples]), dtype=torch.float32),
            "has_filings": torch.tensor([s["has_filings"] for s in samples], dtype=torch.float32),
            "target_class": torch.tensor([s["target_class"] for s in samples], dtype=torch.long),
            "target_return_norm": torch.tensor([s["target_return_norm"] for s in samples], dtype=torch.float32),
            "dates": [s["date"] for s in samples],
            "symbols": [symbols[s["symbol_id"]] for s in samples],
        }
        torch.save(tensors, OUT_DIR / f"{split_name}.pt")
        log.info(f"  saved {split_name}.pt — shapes: scalar_now={tensors['scalar_now'].shape}, news_embs={tensors['news_embs'].shape}")

    # manifest
    manifest = {
        "version": 1,
        "n_scalar_features": len(scalar_cols),
        "scalar_features": scalar_cols,
        "seq_len": SEQ_LEN,
        "max_news_per_day": MAX_NEWS_PER_DAY,
        "finbert_dim": FINBERT_DIM,
        "finbert_model": FINBERT_MODEL,
        "n_symbols": len(symbols),
        "symbols": symbols,
        "splits": {k: len(v) for k, v in splits_data.items()},
        "class_distribution": {
            split: dict(zip(*np.unique([s["target_class"] for s in v], return_counts=True))) if v else {}
            for split, v in splits_data.items()
        },
    }
    # convertir numpy ints a int python para json
    manifest["class_distribution"] = {k: {int(c): int(n) for c, n in v.items()} for k, v in manifest["class_distribution"].items()}
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"DONE. manifest: {manifest['splits']}")


if __name__ == "__main__":
    main()
