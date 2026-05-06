"""
Pure feature-engineering functions, shared between training (script 01) and
serving (feature_runtime).

CRITICAL: si modificas una función aquí, NO la dupliques en script 01. La idea es
que ambos pipelines llamen al mismo código para garantizar reproducibilidad.

Las funciones operan sobre DataFrames con columnas: date, price, symbol, news,
ten_k, ten_q, momentum_raw, future_price_diff (opcional), y devuelven DataFrames
con features añadidas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants (mantener en sync con script 01)
# ---------------------------------------------------------------------------

WEEKEND_NEWS_LAMBDA = 0.18

MEGACAP = {"AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA"}
TECH = {"AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA", "ADBE"}

SYMBOLS_CRYPTO = {"BTC", "ETH"}


@dataclass
class WeightedNews:
    text: str
    weight: float
    age_days: int


# ---------------------------------------------------------------------------
# Weekend news consolidation
# ---------------------------------------------------------------------------

def consolidate_weekend_news(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """
    Para stocks: si hay filas de S/D (price NaN), acumular news al siguiente trading day.
    Para cripto: passthrough.
    Devuelve solo las filas que son trading days.
    """
    df = df.copy()
    if asset_type == "crypto":
        df["news_weighted"] = df["news"].apply(lambda lst: [WeightedNews(t, 1.0, 0) for t in lst])
        df["weekend_news_count"] = 0
        df["news_raw_strings"] = df["news"]
        return df

    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["is_trading_day"] = (~df["price"].isna()) & (df["dow"] < 5)

    n_weekend = int((~df["is_trading_day"]).sum())
    if n_weekend == 0:
        df["news_weighted"] = df["news"].apply(lambda lst: [WeightedNews(t, 1.0, 0) for t in lst])
        df["weekend_news_count"] = 0
        df["news_raw_strings"] = df["news"]
        return df.drop(columns=["dow", "is_trading_day"]).reset_index(drop=True)

    pending: list[WeightedNews] = []
    weighted_col, weekend_count_col, raw_strings_col = [], [], []
    keep = []

    for _, row in df.iterrows():
        if not row["is_trading_day"]:
            for n in row["news"]:
                pending.append(WeightedNews(n, 1.0, 0))
            keep.append(False)
            weighted_col.append(None)
            weekend_count_col.append(0)
            raw_strings_col.append([])
            continue

        today = [WeightedNews(t, 1.0, 0) for t in row["news"]]
        carry = []
        if pending:
            n_pending = len(pending)
            for k, wn in enumerate(pending):
                age = n_pending - k
                w = math.exp(-WEEKEND_NEWS_LAMBDA * age)
                carry.append(WeightedNews(wn.text, w, age))

        all_news = today + carry
        weighted_col.append(all_news)
        weekend_count_col.append(len(carry))
        raw_strings_col.append([wn.text for wn in all_news])
        keep.append(True)
        pending = []

    df["news_weighted"] = weighted_col
    df["weekend_news_count"] = weekend_count_col
    df["news_raw_strings"] = raw_strings_col
    df = df[keep].drop(columns=["dow", "is_trading_day"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def _safe_pct_change(s: pd.Series, periods: int) -> pd.Series:
    return s.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def _williams_r(series: pd.Series, period: int = 14) -> pd.Series:
    high = series.rolling(period).max()
    low = series.rolling(period).min()
    return -100.0 * (high - series) / (high - low).replace(0, np.nan)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / s.replace(0, np.nan)


def _slope(series: pd.Series, window: int) -> pd.Series:
    def _fit(arr):
        if np.isnan(arr).any():
            return np.nan
        x = np.arange(len(arr))
        m = np.polyfit(x, arr, 1)[0]
        mean_level = arr.mean()
        return m / mean_level if mean_level != 0 else np.nan
    return series.rolling(window).apply(_fit, raw=True)


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------

def add_price_features(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    df = df.copy()
    p = df["price"]

    if asset_type == "crypto":
        short, mid, long_ = 7, 14, 30
    else:
        short, mid, long_ = 5, 20, 50

    for k in [1, 3, 5, 10, 20]:
        df[f"return_{k}d"] = _safe_pct_change(p, k)
    df["log_return_1d"] = np.log(p / p.shift(1))

    lr = df["log_return_1d"]
    df["realized_vol_5d"] = lr.rolling(short).std()
    df["realized_vol_20d"] = lr.rolling(mid).std()
    df["realized_vol_60d"] = lr.rolling(long_).std()
    df["vol_of_vol"] = df["realized_vol_5d"].rolling(mid).std()
    df["vol_regime_high"] = (df["realized_vol_20d"] > df["realized_vol_60d"]).astype(float)

    df["rsi_7"] = _rsi(p, 7)
    df["rsi_14"] = _rsi(p, 14)
    macd, sig, hist = _macd(p)
    df["macd"] = macd
    df["macd_signal"] = sig
    df["macd_hist"] = hist
    df["roc_10"] = _safe_pct_change(p, 10)
    df["roc_20"] = _safe_pct_change(p, 20)
    df["williams_r_14"] = _williams_r(p, 14)

    sma_short = p.rolling(short).mean()
    sma_mid = p.rolling(mid).mean()
    sma_long = p.rolling(long_).mean()
    df["sma_short_dist"] = (p - sma_short) / sma_short
    df["sma_mid_dist"] = (p - sma_mid) / sma_mid
    df["sma_long_dist"] = (p - sma_long) / sma_long
    df["sma_short_above_long"] = (sma_short > sma_long).astype(float)
    df["ema_12"] = p.ewm(span=12, adjust=False).mean()
    df["ema_26"] = p.ewm(span=26, adjust=False).mean()
    df["ema_cross"] = (df["ema_12"] > df["ema_26"]).astype(float)
    df["price_slope_10d"] = _slope(p, 10)
    df["price_slope_20d"] = _slope(p, mid)

    df["zscore_20d"] = _zscore(p, mid)
    bb_mid = p.rolling(mid).mean()
    bb_std = p.rolling(mid).std()
    df["bollinger_position"] = (p - bb_mid) / (2 * bb_std).replace(0, np.nan)

    win_year = min(252, len(df))
    df["dist_from_max"] = (p - p.rolling(win_year, min_periods=20).max()) / p.rolling(win_year, min_periods=20).max()
    df["dist_from_min"] = (p - p.rolling(win_year, min_periods=20).min()) / p.rolling(win_year, min_periods=20).min()

    df["gap_overnight_proxy"] = lr if asset_type == "stock" else 0.0
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df["date"])
    df["dow"] = d.dt.dayofweek.astype(float)
    for i in range(7):
        df[f"dow_{i}"] = (d.dt.dayofweek == i).astype(float)
    df["day_of_month"] = d.dt.day.astype(float)
    df["month"] = d.dt.month.astype(float)
    df["is_month_end"] = d.dt.is_month_end.astype(float)
    df["is_quarter_end"] = d.dt.is_quarter_end.astype(float)
    return df


def add_news_scalar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["news_count"] = df["news_raw_strings"].apply(len).astype(float)
    df["log_news_count"] = np.log1p(df["news_count"])
    df["has_news"] = (df["news_count"] > 0).astype(float)
    df["weekend_news_carry_flag"] = (df.get("weekend_news_count", 0) > 0).astype(float) if "weekend_news_count" in df else 0.0
    df["news_total_chars"] = df["news_raw_strings"].apply(lambda lst: sum(len(s) for s in lst)).astype(float)
    df["log_news_total_chars"] = np.log1p(df["news_total_chars"])
    return df


def add_filing_features(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    df = df.copy()
    if asset_type == "crypto":
        df["days_since_10k"] = -1.0
        df["days_since_10q"] = -1.0
        df["filing_decay_10k"] = 0.0
        df["filing_decay_10q"] = 0.0
        df["has_recent_filing_30d"] = 0.0
        df["has_filings"] = 0.0
        return df

    df["has_filings"] = 1.0
    n = len(df)
    days_10k = np.full(n, np.nan)
    days_10q = np.full(n, np.nan)
    last_10k_idx = -1
    last_10q_idx = -1
    for i in range(n):
        if len(df.iloc[i]["ten_k"]) > 0:
            last_10k_idx = i
        if len(df.iloc[i]["ten_q"]) > 0:
            last_10q_idx = i
        if last_10k_idx >= 0:
            days_10k[i] = i - last_10k_idx
        if last_10q_idx >= 0:
            days_10q[i] = i - last_10q_idx
    df["days_since_10k"] = days_10k
    df["days_since_10q"] = days_10q

    half_life = 45.0
    lam = math.log(2) / half_life
    df["filing_decay_10k"] = np.where(np.isnan(days_10k), 0.0, np.exp(-lam * np.where(np.isnan(days_10k), 0, days_10k)))
    df["filing_decay_10q"] = np.where(np.isnan(days_10q), 0.0, np.exp(-lam * np.where(np.isnan(days_10q), 0, days_10q)))
    recent_10k = (~np.isnan(days_10k)) & (days_10k <= 30)
    recent_10q = (~np.isnan(days_10q)) & (days_10q <= 30)
    df["has_recent_filing_30d"] = (recent_10k | recent_10q).astype(float)
    df["days_since_10k"] = df["days_since_10k"].fillna(999.0)
    df["days_since_10q"] = df["days_since_10q"].fillna(999.0)
    return df


def add_momentum_encoded(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
    df["momentum_encoded"] = df["momentum_raw"].map(mapping).fillna(0.0)
    df["momentum_bullish"] = (df["momentum_encoded"] > 0).astype(float)
    df["momentum_bearish"] = (df["momentum_encoded"] < 0).astype(float)
    return df


def add_identity_features(df: pd.DataFrame, symbol: str, asset_type: str) -> pd.DataFrame:
    df = df.copy()
    df["asset_type_crypto"] = float(asset_type == "crypto")
    df["asset_type_stock"] = float(asset_type == "stock")
    df["is_megacap"] = float(symbol in MEGACAP)
    df["is_tech"] = float(symbol in TECH)
    return df


# ---------------------------------------------------------------------------
# Pipeline single-symbol
# ---------------------------------------------------------------------------

def build_features_for_symbol(
    df: pd.DataFrame, symbol: str, asset_type: str | None = None,
) -> pd.DataFrame:
    """
    Pipeline completo single-symbol. df debe tener: date, price, symbol, news, ten_k,
    ten_q, momentum_raw. Devuelve DataFrame con TODAS las features candidatas.
    """
    if asset_type is None:
        asset_type = "crypto" if symbol in SYMBOLS_CRYPTO else "stock"

    df = df.sort_values("date").reset_index(drop=True)
    df = consolidate_weekend_news(df, asset_type)
    df = add_price_features(df, asset_type)
    df = add_calendar_features(df)
    df = add_news_scalar_features(df)
    df = add_filing_features(df, asset_type)
    df = add_momentum_encoded(df)
    df = add_identity_features(df, symbol, asset_type)
    return df


def add_cross_asset_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega:
        - mkt_proxy_return_1d (media de return_1d de mega-caps tech)
        - btc_return_1d
        - corr_to_market_30d (rolling por activo)

    df_all debe ser concat de varios símbolos con sus features ya calculadas.
    """
    df_all = df_all.copy()
    pivot_ret = df_all.pivot_table(index="date", columns="symbol", values="return_1d")

    mega_cols = [c for c in pivot_ret.columns if c in MEGACAP]
    if mega_cols:
        mkt_proxy = pivot_ret[mega_cols].mean(axis=1)
    else:
        mkt_proxy = pivot_ret.mean(axis=1)
    btc_proxy = pivot_ret["BTC"] if "BTC" in pivot_ret.columns else pd.Series(index=pivot_ret.index, dtype=float)

    mkt_df = pd.DataFrame({
        "date": mkt_proxy.index,
        "mkt_proxy_return_1d": mkt_proxy.values,
        "btc_return_1d": btc_proxy.reindex(mkt_proxy.index).values,
    })
    df_all = df_all.merge(mkt_df, on="date", how="left")

    df_all["corr_to_market_30d"] = np.nan
    for sym in df_all["symbol"].unique():
        mask = df_all["symbol"] == sym
        sub = df_all.loc[mask].sort_values("date")
        corr = sub["return_1d"].rolling(30).corr(sub["mkt_proxy_return_1d"])
        df_all.loc[sub.index, "corr_to_market_30d"] = corr.values
    return df_all
