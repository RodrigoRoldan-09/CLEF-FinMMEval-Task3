"""
Feature runtime for serving.

Para cada request del API:
    1) Persistir el día (price + news + filings + momentum) en el state store.
    2) Leer historial de >=200 días para ese símbolo.
    3) Leer cross-asset (mega-caps + BTC) para los últimos ~60 días.
    4) Llamar a model.feature_engineering para reproducir EXACTAMENTE las features
       que vio el training.
    5) Tomar la última fila (el día del request) y empacarla como tensor escalar +
       construir la ventana temporal (30 días) hacia atrás.
    6) Para news: pasar a embedding_runtime.
    7) Para filings: mantener "filing-state" con decay (igual que en script 03).

Returns: dict listo para model.forward().
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.feature_engineering import (
    SYMBOLS_CRYPTO, MEGACAP,
    build_features_for_symbol,
    add_cross_asset_features,
)

log = logging.getLogger("feature_runtime")


def _get(obj, key, default=None):
    """Acceso uniforme a atributo (dataclass) o key (dict)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# Filing state with decay (running through history)
# ---------------------------------------------------------------------------

def compute_filing_state_today(history_df: pd.DataFrame, embed_filing_long_fn) -> np.ndarray:
    """
    Recorre el histórico de un símbolo, va embeddando cada filing visto y mantiene
    el último embedding con decay exponencial half-life=45d.
    Devuelve el embedding del último día (FINBERT_DIM,).
    """
    last_filing_emb = None
    last_filing_idx = -1
    half_life = 45.0
    lam = math.log(2) / half_life

    n = len(history_df)
    if n == 0:
        return np.zeros(768, dtype=np.float32)

    for i in range(n):
        row = history_df.iloc[i]
        today_filings = []
        if isinstance(row.get("ten_k"), list) and len(row["ten_k"]) > 0:
            today_filings.extend(row["ten_k"])
        if isinstance(row.get("ten_q"), list) and len(row["ten_q"]) > 0:
            today_filings.extend(row["ten_q"])
        if today_filings:
            # combinar (mean) los embeddings de los filings del día
            embs = [embed_filing_long_fn(t) for t in today_filings]
            last_filing_emb = np.mean(embs, axis=0).astype(np.float32)
            last_filing_idx = i

    if last_filing_emb is None:
        return np.zeros(768, dtype=np.float32)

    age = (n - 1) - last_filing_idx
    decay = math.exp(-lam * age)
    return (last_filing_emb * decay).astype(np.float32)


# ---------------------------------------------------------------------------
# News embedding for the day's request
# ---------------------------------------------------------------------------

def embed_news_for_day(
    weighted_news: list,
    embed_news_fn,
    max_news: int = 8,
    finbert_dim: int = 768,
):
    """
    weighted_news: lista de objetos con .text y .weight (tipo WeightedNews).
    embed_news_fn: callable(text) -> (cls_emb (768,), sentiment (3,))

    Returns: dict con embs (max_news, 768), weights (max_news,), mask (max_news,),
             sentiment_agg (3,), top_text (str), n_news (int)
    """
    embs = np.zeros((max_news, finbert_dim), dtype=np.float32)
    weights = np.zeros(max_news, dtype=np.float32)
    mask = np.zeros(max_news, dtype=np.float32)
    sentiments, sent_weights = [], []
    top_text = ""

    if not weighted_news:
        return {
            "embs": embs, "weights": weights, "mask": mask,
            "sentiment_agg": np.array([0, 1, 0], dtype=np.float32),
            "top_text": "", "n_news": 0,
            "all_texts_in_slots": [],
        }

    sorted_news = sorted(weighted_news, key=lambda x: -float(_get(x, "weight", 1.0)))
    slot_texts = []
    for i, wn in enumerate(sorted_news[:max_news]):
        text = _get(wn, "text", str(wn))
        w = float(_get(wn, "weight", 1.0))
        cls, sent = embed_news_fn(text)
        embs[i] = cls
        weights[i] = w
        mask[i] = 1.0
        sentiments.append(sent)
        sent_weights.append(w)
        slot_texts.append(text)

    if sentiments:
        sw = np.array(sent_weights)
        sw = sw / sw.sum()
        sent_agg = np.average(np.stack(sentiments), axis=0, weights=sw).astype(np.float32)
    else:
        sent_agg = np.array([0, 1, 0], dtype=np.float32)

    top_text = slot_texts[0] if slot_texts else ""
    return {
        "embs": embs, "weights": weights, "mask": mask,
        "sentiment_agg": sent_agg,
        "top_text": top_text,
        "n_news": len(sorted_news),
        "all_texts_in_slots": slot_texts,
    }


# ---------------------------------------------------------------------------
# Build the sequence window from history
# ---------------------------------------------------------------------------

def _ffill_array(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs en cada columna; si toda la col es NaN -> zeros."""
    out = arr.copy()
    n, f = out.shape
    for c in range(f):
        col = out[:, c]
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
        out[:, c] = col
    return out


def build_sequence_window(features_df: pd.DataFrame, scalar_cols: list, seq_len: int) -> np.ndarray:
    """
    Toma las últimas seq_len filas y las apila en (seq_len, F).
    Si hay menos filas, left-pad con replicación de la primera fila disponible.
    """
    n_rows = len(features_df)
    f = len(scalar_cols)
    if n_rows == 0:
        return np.zeros((seq_len, f), dtype=np.float32)

    arr = features_df[scalar_cols].astype(float).values
    arr = _ffill_array(arr)

    if n_rows >= seq_len:
        return arr[-seq_len:].astype(np.float32)
    pad = np.repeat(arr[:1], seq_len - n_rows, axis=0)
    return np.vstack([pad, arr]).astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

class FeatureRuntime:
    """
    Mantiene la config (cols seleccionadas, scaler, vocab) cargada en memoria.
    Llama a state_store, módulo de features compartido, y embedding_runtime.
    """

    def __init__(
        self, state_store, embedding_runtime,
        scaler_path: Path, manifest_path: Path, symbol_vocab_path: Path,
        seq_len: int = 30, max_news: int = 8, finbert_dim: int = 768,
        lookback_days: int = 200,
    ):
        import json
        with open(scaler_path) as f:
            scaler = json.load(f)
        self.scalar_cols = scaler["feature_columns"]
        self.scalar_mean = np.array(scaler["mean"], dtype=np.float32)
        self.scalar_std = np.array(scaler["std"], dtype=np.float32)

        with open(manifest_path) as f:
            self.manifest = json.load(f)
        with open(symbol_vocab_path) as f:
            self.symbol_vocab = json.load(f)

        self.state = state_store
        self.embed = embedding_runtime
        self.seq_len = seq_len
        self.max_news = max_news
        self.finbert_dim = finbert_dim
        self.lookback_days = lookback_days

    def ingest_request(self, symbol: str, date: str, price: float,
                       news: list, ten_k: list, ten_q: list, momentum: str | None,
                       history_price: list[dict] | None):
        """Persiste todo lo que viene en el request al state store."""
        # primero: history_price (no pisa news/filings de esos días si ya existían)
        if history_price:
            self.state.upsert_history_prices(symbol, history_price)
        # después: el día actual (sí trae news/filings)
        self.state.upsert_day(
            symbol=symbol, date=date, price=price,
            news=list(news or []),
            ten_k=list(ten_k or []),
            ten_q=list(ten_q or []),
            momentum_raw=(momentum or "neutral").lower(),
        )

    def build_tensors(self, symbol: str, date: str) -> dict:
        """
        Lee del state store, calcula features, embeddings, y devuelve un dict listo
        para pasar a model.forward() (con batch dim = 1).
        """
        asset_type = "crypto" if symbol in SYMBOLS_CRYPTO else "stock"

        # 1. histórico del símbolo
        history = self.state.get_history(symbol, until_date=date, lookback_days=self.lookback_days)
        if not history:
            raise RuntimeError(f"no hay histórico en el state store para {symbol} (¿corriste el seed?)")
        df_sym = pd.DataFrame(history)
        df_sym["date"] = pd.to_datetime(df_sym["date"])
        df_sym["symbol"] = symbol

        # 2. features single-symbol
        feats = build_features_for_symbol(df_sym, symbol, asset_type)
        if feats.empty:
            raise RuntimeError(f"feature engineering vacío para {symbol}")

        # 3. cross-asset: necesitamos return_1d de mega-caps + BTC
        #    cargamos historial de todos los símbolos relevantes (price-only) y calculamos returns
        all_recent = self.state.get_all_recent(until_date=date, lookback_days=60)
        if all_recent:
            df_all = pd.DataFrame(all_recent)
            df_all["date"] = pd.to_datetime(df_all["date"])
            df_all = df_all.sort_values(["symbol", "date"])
            df_all["return_1d"] = df_all.groupby("symbol")["price"].pct_change(1)

            # añadimos return_1d del propio símbolo desde feats para que merge no pierda data
            feats_for_merge = feats[["date", "symbol", "return_1d"]].copy()
            df_all = pd.concat(
                [df_all[df_all["symbol"] != symbol], feats_for_merge],
                ignore_index=True,
            )
            df_all = add_cross_asset_features(df_all)
            cross = df_all[df_all["symbol"] == symbol][
                ["date", "mkt_proxy_return_1d", "btc_return_1d", "corr_to_market_30d"]
            ]
            feats = feats.merge(cross, on="date", how="left")
        else:
            feats["mkt_proxy_return_1d"] = 0.0
            feats["btc_return_1d"] = 0.0
            feats["corr_to_market_30d"] = 0.0

        # 4. ventana temporal de scalar features
        # filtrar columnas seleccionadas que existan en feats; las que no, las creamos a 0
        for c in self.scalar_cols:
            if c not in feats.columns:
                log.warning(f"feature {c} no presente en runtime, rellenando con 0")
                feats[c] = 0.0

        # estandarizar
        scalar_arr = feats[self.scalar_cols].astype(float).values
        scalar_arr_z = (scalar_arr - self.scalar_mean) / np.where(self.scalar_std == 0, 1, self.scalar_std)
        feats_z = feats.copy()
        for i, c in enumerate(self.scalar_cols):
            feats_z[c] = scalar_arr_z[:, i]

        scalar_seq = build_sequence_window(feats_z, self.scalar_cols, self.seq_len)
        scalar_now = scalar_seq[-1]

        # 5. noticias del día (con weighted carry-over si aplica)
        last_row = feats.iloc[-1]
        weighted_news = last_row.get("news_weighted", []) or []
        news_pack = embed_news_for_day(
            weighted_news,
            embed_news_fn=self.embed.embed_news,
            max_news=self.max_news,
            finbert_dim=self.finbert_dim,
        )

        # 6. filing state (running through full history)
        filing_emb = compute_filing_state_today(df_sym, self.embed.embed_filing_long) \
            if asset_type == "stock" else np.zeros(self.finbert_dim, dtype=np.float32)

        # 7. identidad
        symbol_id = self.symbol_vocab.get(symbol)
        if symbol_id is None:
            raise RuntimeError(f"symbol {symbol} no está en el vocab del modelo")

        # 8. tensores con batch=1
        batch = {
            "scalar_now": torch.from_numpy(scalar_now).unsqueeze(0).float(),
            "scalar_seq": torch.from_numpy(scalar_seq).unsqueeze(0).float(),
            "news_embs": torch.from_numpy(news_pack["embs"]).unsqueeze(0).float(),
            "news_weights": torch.from_numpy(news_pack["weights"]).unsqueeze(0).float(),
            "news_mask": torch.from_numpy(news_pack["mask"]).unsqueeze(0).float(),
            "filing_emb": torch.from_numpy(filing_emb).unsqueeze(0).float(),
            "symbol_id": torch.tensor([symbol_id], dtype=torch.long),
            "asset_type_crypto": torch.tensor([1.0 if asset_type == "crypto" else 0.0], dtype=torch.float32),
            "has_filings": torch.tensor([1.0 if asset_type == "stock" else 0.0], dtype=torch.float32),
        }

        # extras para explicación
        meta = {
            "asset_type": asset_type,
            "n_history_rows": len(df_sym),
            "n_news_today": news_pack["n_news"],
            "top_news_text": news_pack["top_text"],
            "news_slot_texts": news_pack["all_texts_in_slots"],
            "news_sentiment": news_pack["sentiment_agg"].tolist(),
            "scalar_features": dict(zip(self.scalar_cols, scalar_now.tolist())),
            "scalar_features_unscaled": dict(zip(self.scalar_cols, feats[self.scalar_cols].iloc[-1].tolist())),
        }
        return batch, meta
