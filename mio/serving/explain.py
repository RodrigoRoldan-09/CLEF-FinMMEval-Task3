"""
Deterministic rationale generator (<= 50 words).

NOT a language model. Llena una plantilla con:
    - acción predicha
    - probabilidad de la acción (para hedge en lenguaje)
    - top modalidad por gate
    - top-3 features más importantes (gradient × input post-hoc)
    - noticia top-1 por attention (truncada)

Si quieres deshabilitar la justificación (en caso de que la API ya no la pida),
pasa enabled=False y devuelve string vacío. El endpoint sigue respondiendo
{"recommended_action": "..."} igual.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import torch


BRANCH_NAMES = ["scalar_now", "scalar_seq", "news", "filing"]
BRANCH_HUMAN = {
    "scalar_now": "today's metrics",
    "scalar_seq": "30-day trend",
    "news": "news flow",
    "filing": "recent filings",
}

# Etiquetas humanas para features comunes (extiende según necesites)
FEATURE_LABELS = {
    "rsi_14": "RSI(14)",
    "rsi_7": "RSI(7)",
    "macd_hist": "MACD histogram",
    "return_1d": "1d return",
    "return_5d": "5d return",
    "return_10d": "10d return",
    "return_20d": "20d return",
    "realized_vol_20d": "20d volatility",
    "realized_vol_5d": "5d volatility",
    "zscore_20d": "20d z-score",
    "bollinger_position": "Bollinger position",
    "momentum_encoded": "momentum signal",
    "log_news_count": "news volume",
    "filing_decay_10q": "fresh 10-Q",
    "filing_decay_10k": "fresh 10-K",
    "mkt_proxy_return_1d": "market 1d",
    "btc_return_1d": "BTC 1d",
    "corr_to_market_30d": "market corr",
    "sma_short_dist": "SMA short dist",
    "sma_mid_dist": "SMA mid dist",
    "sma_long_dist": "SMA long dist",
    "ema_cross": "EMA cross",
    "williams_r_14": "Williams %R",
    "vol_regime_high": "high vol regime",
}


def _truncate_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def feature_attribution(model, batch: dict, target_class: int) -> np.ndarray:
    """
    Gradient × input sobre scalar_now para el target_class.
    Devuelve np.ndarray de shape (F,) con las contribuciones firmadas por feature.
    NO modifica el modelo (todo en eval mode + grad re-enabled localmente).
    """
    model.eval()
    inp = batch["scalar_now"].clone().detach().requires_grad_(True)
    new_batch = {**batch, "scalar_now": inp}
    out = model(new_batch)
    logits = out["logits"]
    # gradiente del logit de la clase objetivo respecto al input
    score = logits[0, target_class]
    grads = torch.autograd.grad(score, inp, retain_graph=False, create_graph=False)[0]
    contrib = (grads * inp).detach().squeeze(0).cpu().numpy()
    return contrib  # vector len = F_scalar


def render_rationale(
    action: str,
    probabilities: np.ndarray,
    gates: np.ndarray,
    news_attn: np.ndarray,
    feature_attribution_scores: np.ndarray,
    scalar_columns: list[str],
    scalar_features_unscaled: dict,
    news_slot_texts: list[str],
    sentiment: np.ndarray,
    enabled: bool = True,
    max_words: int = 50,
) -> str:
    """
    Construye la justificación.

    action: 'BUY' / 'HOLD' / 'SELL'
    probabilities: (3,) softmax over [SELL, HOLD, BUY]
    gates: (4,) gates de fusión por modalidad
    news_attn: (N,) attention sobre las noticias (en orden de slots)
    feature_attribution_scores: (F,) gradient × input, signed
    scalar_columns: nombres en el mismo orden
    scalar_features_unscaled: dict raw value para reportar magnitudes legibles
    news_slot_texts: textos en el mismo orden de slots
    sentiment: (3,) [neg, neu, pos] del agregado del día
    """
    if not enabled:
        return ""

    # 1. modalidad principal
    branch_idx = int(np.argmax(gates))
    top_branch = BRANCH_HUMAN[BRANCH_NAMES[branch_idx]]

    # 2. top-3 features (por |attribution|), excluyendo dummies de identidad
    abs_scores = np.abs(feature_attribution_scores)
    order = np.argsort(-abs_scores)
    top_features = []
    for idx in order:
        col = scalar_columns[idx]
        if col in {"asset_type_crypto", "asset_type_stock", "is_megacap", "is_tech"}:
            continue
        score = float(feature_attribution_scores[idx])
        sign = "+" if score >= 0 else "-"
        label = FEATURE_LABELS.get(col, col)
        # valor crudo formateado
        raw_val = scalar_features_unscaled.get(col, None)
        if raw_val is None or not np.isfinite(raw_val):
            top_features.append(f"{sign}{label}")
        else:
            if abs(raw_val) >= 100:
                fval = f"{raw_val:.0f}"
            elif abs(raw_val) >= 1:
                fval = f"{raw_val:.2f}"
            else:
                fval = f"{raw_val:.3f}"
            top_features.append(f"{label}={fval}")
        if len(top_features) >= 3:
            break

    # 3. sentiment del día
    sent_label = ["neg", "neu", "pos"][int(np.argmax(sentiment))]

    # 4. confianza
    p = float(probabilities[{"SELL": 0, "HOLD": 1, "BUY": 2}[action]])
    if p >= 0.55:
        conf = ""
    elif p >= 0.40:
        conf = "tilted "
    else:
        conf = "weakly "

    # 5. noticia top-1 (si la modalidad news pesó relevantemente)
    news_clip = ""
    if len(news_slot_texts) > 0 and news_attn is not None and len(news_attn) > 0 and gates[2] > 0.15:
        top_idx = int(np.argmax(news_attn[:len(news_slot_texts)]))
        if 0 <= top_idx < len(news_slot_texts):
            top_text = news_slot_texts[top_idx]
            news_clip = ' news: "' + _truncate_words(top_text, 8) + '"'

    feat_str = ", ".join(top_features) if top_features else ""
    text = f"{action} ({conf}driven by {top_branch}; {feat_str}; sentiment {sent_label}){news_clip}"

    # 6. enforzar límite de 50 palabras
    return _truncate_words(text, max_words)
