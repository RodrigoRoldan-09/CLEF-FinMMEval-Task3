"""
Script 01 - Build raw feature dataset for FinMMEval Task 3.

Descarga los 12 parquets desde HuggingFace, normaliza schema, maneja fines de semana
en stocks con decay temporal de noticias, y calcula ~60 features candidatas.

Output:
    data/raw_features/{symbol}.parquet         - features por activo
    data/raw_features/_all.parquet             - dataset consolidado
    data/raw_features/_metadata.json           - info de cada activo (rangos, conteos)
"""
from __future__ import annotations

import json
import logging
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_raw")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HF_BASE = "hf://datasets/TheFinAI/daily_news/data"

SYMBOLS_STOCK = ["TSLA", "MSFT", "BMRN", "MRNA", "ADBE", "AAPL", "AMZN", "GOOGL", "META", "NVDA"]
SYMBOLS_CRYPTO = ["BTC", "ETH"]
ALL_SYMBOLS = SYMBOLS_CRYPTO + SYMBOLS_STOCK

MEGACAP = {"AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA"}
TECH = {"AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA", "ADBE"}

# Decay para noticias acumuladas en stocks (viernes -> lunes)
# weight = exp(-lambda * days_since)
WEEKEND_NEWS_LAMBDA = 0.18  # half-life ~3.85 días, día 0=1.0, día 1=0.83, día 3=0.58


# ---------------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------------

# Posibles nombres alternativos vistos en datasets de FinAI / similares.
# Si en tu copia local tiene otros nombres, agrégalos aquí.
COLUMN_ALIASES = {
    "date": ["date", "Date", "timestamp", "ds"],
    "price": ["price", "prices", "close", "Close", "adj_close", "Adj Close"],
    "news": ["news", "news_list", "headlines"],
    "asset": ["asset", "symbol", "ticker"],
    "momentum": ["momentum", "momentum_signal"],
    "ten_k": ["10k", "ten_k", "10-K", "10K"],
    "ten_q": ["10q", "ten_q", "10-Q", "10Q"],
    "future_price_diff": ["future_price_diff", "future_diff", "next_day_diff"],
}


def _resolve_columns(df: pd.DataFrame) -> dict:
    """Mapea nombres canónicos -> nombres reales en df, o None si no existe."""
    found = {}
    for canonical, candidates in COLUMN_ALIASES.items():
        match = next((c for c in candidates if c in df.columns), None)
        found[canonical] = match
    return found


def _coerce_list(x) -> list:
    """news/10k/10q a veces vienen como np.ndarray, str-json, list, o NaN."""
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    if isinstance(x, np.ndarray):
        return [str(item) for item in x.tolist() if item is not None and str(item) != "nan"]
    if isinstance(x, list):
        return [str(item) for item in x if item is not None and str(item) != "nan"]
    if isinstance(x, str):
        s = x.strip()
        if not s or s == "[]":
            return []
        # intentar parsear si es lista en formato str
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                return [s]
        return [s]
    return [str(x)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_symbol(symbol: str) -> pd.DataFrame:
    """Descarga y normaliza el parquet de un símbolo."""
    url = f"{HF_BASE}/{symbol}-00000-of-00001.parquet"
    log.info(f"  loading {symbol} from HF")
    df = pd.read_parquet(url)

    cols = _resolve_columns(df)
    missing_required = [k for k in ("date", "price") if cols[k] is None]
    if missing_required:
        raise RuntimeError(f"{symbol}: faltan columnas requeridas: {missing_required}. Columnas reales: {list(df.columns)}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[cols["date"]])
    out["price"] = pd.to_numeric(df[cols["price"]], errors="coerce")
    out["symbol"] = symbol
    out["news"] = df[cols["news"]].apply(_coerce_list) if cols["news"] else [[] for _ in range(len(df))]

    if cols["momentum"]:
        out["momentum_raw"] = df[cols["momentum"]].astype(str).str.lower()
    else:
        out["momentum_raw"] = "neutral"

    out["ten_k"] = df[cols["ten_k"]].apply(_coerce_list) if cols["ten_k"] else [[] for _ in range(len(df))]
    out["ten_q"] = df[cols["ten_q"]].apply(_coerce_list) if cols["ten_q"] else [[] for _ in range(len(df))]

    if cols["future_price_diff"]:
        out["future_price_diff"] = pd.to_numeric(df[cols["future_price_diff"]], errors="coerce")
    else:
        # lo calculamos nosotros si no viene
        out["future_price_diff"] = out["price"].shift(-1) - out["price"]

    out = out.sort_values("date").reset_index(drop=True)
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Weekend news handling for stocks
# ---------------------------------------------------------------------------

@dataclass
class WeightedNews:
    """Una noticia con su peso (decay temporal)."""
    text: str
    weight: float
    age_days: int


def consolidate_weekend_news(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """
    Para stocks: las filas de S/D no tienen precio (no trading day) pero pueden tener
    news. Las acumulamos en el siguiente día hábil con decay temporal.

    Estrategia: emitimos una sola fila por trading day. La columna `news` ahora
    contiene lista de WeightedNews (text, weight, age_days). Mantenemos también
    `news_raw_strings` para compatibilidad/embeddings.
    """
    if asset_type == "crypto":
        # cripto opera 7 días, no hay nada que acumular
        df["news_weighted"] = df["news"].apply(lambda lst: [WeightedNews(t, 1.0, 0) for t in lst])
        df["weekend_news_count"] = 0
        df["news_raw_strings"] = df["news"]
        return df

    # stock: detectar trading days vs no-trading days (los no-trading tienen price NaN o repetido)
    # criterio robusto: si price es NaN o si el día es sábado/domingo
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek  # 0=Mon ... 6=Sun
    df["is_trading_day"] = (~df["price"].isna()) & (df["dow"] < 5)

    # Si en el dataset las filas de fin de semana simplemente no existen (gap),
    # entonces todas las filas son trading days y no hay nada que acumular.
    # Detectamos eso:
    n_weekend_rows = int((~df["is_trading_day"]).sum())
    if n_weekend_rows == 0:
        log.info(f"    no hay filas de fin de semana, news ya está limpio")
        df["news_weighted"] = df["news"].apply(lambda lst: [WeightedNews(t, 1.0, 0) for t in lst])
        df["weekend_news_count"] = 0
        df["news_raw_strings"] = df["news"]
        df = df.drop(columns=["dow"])
        return df

    log.info(f"    consolidando {n_weekend_rows} filas no-trading hacia el siguiente trading day")

    # Acumulador de noticias pendientes (de días no-trading anteriores)
    pending: list[WeightedNews] = []
    weighted_col = []
    weekend_count_col = []
    raw_strings_col = []

    current_date = None
    for idx, row in df.iterrows():
        if not row["is_trading_day"]:
            # acumulamos las noticias para el próximo trading day
            for n in row["news"]:
                # age_days se calcula al momento del flush
                pending.append(WeightedNews(n, 1.0, 0))
            weighted_col.append(None)
            weekend_count_col.append(0)
            raw_strings_col.append([])
            continue

        # es trading day: noticias del día (peso 1.0) + pendientes con decay
        today_news = [WeightedNews(t, 1.0, 0) for t in row["news"]]
        weekend_carry = []
        if pending:
            ref_date = row["date"]
            for wn in pending:
                # buscamos la fecha original de esa noticia para calcular age
                # truco: contamos desde el último trading day. Como pending se llena en orden,
                # asumimos age = días desde la fila que la generó hasta ref_date.
                # Para simplicidad y dado que pending se vacía cada trading day, usamos
                # decay basado en posición inversa (la primera pending es la más vieja).
                pass
            # asignamos age por orden cronológico de inserción
            n_pending = len(pending)
            for k, wn in enumerate(pending):
                # el más antiguo tiene mayor age
                age = n_pending - k
                weight = math.exp(-WEEKEND_NEWS_LAMBDA * age)
                weekend_carry.append(WeightedNews(wn.text, weight, age))

        all_news = today_news + weekend_carry
        weighted_col.append(all_news)
        weekend_count_col.append(len(weekend_carry))
        raw_strings_col.append([wn.text for wn in all_news])

        pending = []  # vaciar

    df["news_weighted"] = weighted_col
    df["weekend_news_count"] = weekend_count_col
    df["news_raw_strings"] = raw_strings_col

    # filtrar solo trading days
    df = df[df["is_trading_day"]].drop(columns=["dow", "is_trading_day"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
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
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def _williams_r(series: pd.Series, period: int = 14) -> pd.Series:
    high = series.rolling(period).max()
    low = series.rolling(period).min()
    return -100.0 * (high - series) / (high - low).replace(0, np.nan)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / s.replace(0, np.nan)


def _slope(series: pd.Series, window: int) -> pd.Series:
    """Pendiente normalizada de regresión lineal sobre rolling window."""
    def _fit(arr):
        if np.isnan(arr).any():
            return np.nan
        x = np.arange(len(arr))
        # slope normalizado por nivel medio para que sea comparable entre activos
        m = np.polyfit(x, arr, 1)[0]
        mean_level = arr.mean()
        return m / mean_level if mean_level != 0 else np.nan
    return series.rolling(window).apply(_fit, raw=True)


def add_price_features(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """Calcula features de precio/momentum/volatilidad/trend."""
    df = df.copy()
    p = df["price"]

    # cripto opera 7 días, ajustamos ventanas
    if asset_type == "crypto":
        short, mid, long_ = 7, 14, 30
    else:
        short, mid, long_ = 5, 20, 50

    # returns
    for k in [1, 3, 5, 10, 20]:
        df[f"return_{k}d"] = _safe_pct_change(p, k)
    df["log_return_1d"] = np.log(p / p.shift(1))

    # volatilidad (std de log returns)
    lr = df["log_return_1d"]
    df["realized_vol_5d"] = lr.rolling(short).std()
    df["realized_vol_20d"] = lr.rolling(mid).std()
    df["realized_vol_60d"] = lr.rolling(long_).std()
    df["vol_of_vol"] = df["realized_vol_5d"].rolling(mid).std()
    df["vol_regime_high"] = (df["realized_vol_20d"] > df["realized_vol_60d"]).astype(float)

    # momentum técnico
    df["rsi_7"] = _rsi(p, 7)
    df["rsi_14"] = _rsi(p, 14)
    macd, sig, hist = _macd(p)
    df["macd"] = macd
    df["macd_signal"] = sig
    df["macd_hist"] = hist
    df["roc_10"] = _safe_pct_change(p, 10)
    df["roc_20"] = _safe_pct_change(p, 20)
    df["williams_r_14"] = _williams_r(p, 14)

    # trend
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

    # mean reversion
    df["zscore_20d"] = _zscore(p, mid)
    bb_mid = p.rolling(mid).mean()
    bb_std = p.rolling(mid).std()
    df["bollinger_position"] = (p - bb_mid) / (2 * bb_std).replace(0, np.nan)

    # 52-week high/low (con la data que haya)
    win_year = min(252, len(df))
    df["dist_from_max"] = (p - p.rolling(win_year, min_periods=20).max()) / p.rolling(win_year, min_periods=20).max()
    df["dist_from_min"] = (p - p.rolling(win_year, min_periods=20).min()) / p.rolling(win_year, min_periods=20).min()

    # gap overnight (solo stocks; aprox como log_return_1d ya que no tenemos open/close separados)
    if asset_type == "stock":
        df["gap_overnight_proxy"] = lr  # placeholder; si en el dataset hay open lo cambiamos
    else:
        df["gap_overnight_proxy"] = 0.0

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["date"]
    df["dow"] = d.dt.dayofweek.astype(float)
    for i in range(5):  # solo lun-vie como dummies; sat/sun es para cripto
        df[f"dow_{i}"] = (d.dt.dayofweek == i).astype(float)
    df["dow_5"] = (d.dt.dayofweek == 5).astype(float)
    df["dow_6"] = (d.dt.dayofweek == 6).astype(float)
    df["day_of_month"] = d.dt.day.astype(float)
    df["month"] = d.dt.month.astype(float)
    df["is_month_end"] = d.dt.is_month_end.astype(float)
    df["is_quarter_end"] = d.dt.is_quarter_end.astype(float)
    return df


def add_news_scalar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Solo conteos y flags. El sentiment de FinBERT se calcula en script 03 (lento)."""
    df = df.copy()
    df["news_count"] = df["news_raw_strings"].apply(len).astype(float)
    df["log_news_count"] = np.log1p(df["news_count"])
    df["has_news"] = (df["news_count"] > 0).astype(float)
    df["weekend_news_carry_flag"] = (df.get("weekend_news_count", 0) > 0).astype(float) if "weekend_news_count" in df else 0.0
    # longitud total como proxy de "intensidad informativa"
    df["news_total_chars"] = df["news_raw_strings"].apply(lambda lst: sum(len(s) for s in lst)).astype(float)
    df["log_news_total_chars"] = np.log1p(df["news_total_chars"])
    return df


def add_filing_features(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """
    days_since_last_10K, days_since_last_10Q, decay weight.
    Para cripto todo es 0/inf.
    """
    df = df.copy()
    if asset_type == "crypto":
        df["days_since_10k"] = -1.0  # marcador
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

    # decay con half-life de 45 días
    half_life = 45.0
    lam = math.log(2) / half_life
    df["filing_decay_10k"] = np.where(np.isnan(days_10k), 0.0, np.exp(-lam * np.where(np.isnan(days_10k), 0, days_10k)))
    df["filing_decay_10q"] = np.where(np.isnan(days_10q), 0.0, np.exp(-lam * np.where(np.isnan(days_10q), 0, days_10q)))

    # flag de filing reciente
    recent_10k = (~np.isnan(days_10k)) & (days_10k <= 30)
    recent_10q = (~np.isnan(days_10q)) & (days_10q <= 30)
    df["has_recent_filing_30d"] = (recent_10k | recent_10q).astype(float)

    # fillna de days_since con un valor grande (nunca visto = mucho tiempo)
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


def add_cross_asset_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Features que requieren info de otros activos:
      - mkt_proxy_return_1d: media de return_1d de mega-caps tech (proxy de SPY/QQQ)
      - btc_return_1d: return_1d de BTC como risk-on proxy
      - corr_to_market_30d: correlación rolling del activo con el proxy
    """
    df_all = df_all.copy()
    pivot_ret = df_all.pivot_table(index="date", columns="symbol", values="return_1d")

    mega_cols = [c for c in pivot_ret.columns if c in MEGACAP]
    if mega_cols:
        mkt_proxy = pivot_ret[mega_cols].mean(axis=1)
    else:
        mkt_proxy = pivot_ret.mean(axis=1)
    btc_proxy = pivot_ret["BTC"] if "BTC" in pivot_ret.columns else pd.Series(index=pivot_ret.index, dtype=float)

    mkt_df = pd.DataFrame({"date": mkt_proxy.index, "mkt_proxy_return_1d": mkt_proxy.values, "btc_return_1d": btc_proxy.reindex(mkt_proxy.index).values})

    df_all = df_all.merge(mkt_df, on="date", how="left")

    # correlación rolling con el proxy, por activo
    df_all["corr_to_market_30d"] = np.nan
    for sym in df_all["symbol"].unique():
        mask = df_all["symbol"] == sym
        sub = df_all.loc[mask].sort_values("date")
        corr = sub["return_1d"].rolling(30).corr(sub["mkt_proxy_return_1d"])
        df_all.loc[sub.index, "corr_to_market_30d"] = corr.values

    return df_all


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_symbol(symbol: str) -> pd.DataFrame:
    asset_type = "crypto" if symbol in SYMBOLS_CRYPTO else "stock"
    log.info(f"[{symbol}] type={asset_type}")
    df = load_symbol(symbol)
    log.info(f"    raw rows: {len(df)}, date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    df = consolidate_weekend_news(df, asset_type)
    log.info(f"    after weekend consolidation: {len(df)} rows")

    df = add_price_features(df, asset_type)
    df = add_calendar_features(df)
    df = add_news_scalar_features(df)
    df = add_filing_features(df, asset_type)
    df = add_momentum_encoded(df)
    df = add_identity_features(df, symbol, asset_type)

    log.info(f"    final cols: {len(df.columns)}, non-null sample: price={df['price'].notna().sum()}, return_1d={df['return_1d'].notna().sum()}")
    return df


def main():
    log.info(f"output dir: {OUT_DIR}")
    log.info(f"processing {len(ALL_SYMBOLS)} symbols")

    per_symbol = {}
    for sym in ALL_SYMBOLS:
        try:
            df = process_symbol(sym)
            per_symbol[sym] = df
        except Exception as e:
            log.error(f"[{sym}] FAILED: {type(e).__name__}: {e}")

    if not per_symbol:
        raise RuntimeError("ningún símbolo procesado correctamente")

    log.info("computing cross-asset features")
    consolidated = pd.concat(per_symbol.values(), ignore_index=True)
    consolidated = add_cross_asset_features(consolidated)

    # guardar por símbolo (re-split después de cross-asset)
    for sym in per_symbol:
        sub = consolidated[consolidated["symbol"] == sym].sort_values("date").reset_index(drop=True)
        # convertir news_weighted a algo serializable (lo necesitamos en script 03)
        if "news_weighted" in sub.columns:
            sub["news_weighted"] = sub["news_weighted"].apply(
                lambda lst: [{"text": w.text, "weight": w.weight, "age_days": w.age_days} for w in lst] if lst else []
            )
        sub.to_parquet(OUT_DIR / f"{sym}.parquet", index=False)
        log.info(f"  saved {sym}.parquet ({len(sub)} rows, {len(sub.columns)} cols)")

    if "news_weighted" in consolidated.columns:
        consolidated["news_weighted"] = consolidated["news_weighted"].apply(
            lambda lst: [{"text": w.text, "weight": w.weight, "age_days": w.age_days} for w in lst] if lst else []
        )
    consolidated.to_parquet(OUT_DIR / "_all.parquet", index=False)
    log.info(f"saved _all.parquet ({len(consolidated)} rows)")

    # metadata
    meta = {
        "symbols": list(per_symbol.keys()),
        "n_rows_total": int(len(consolidated)),
        "date_min": str(consolidated["date"].min().date()),
        "date_max": str(consolidated["date"].max().date()),
        "n_features_raw": int(len(consolidated.columns)),
        "feature_columns": [c for c in consolidated.columns if c not in {"date", "symbol", "news", "news_raw_strings", "news_weighted", "ten_k", "ten_q", "future_price_diff", "momentum_raw"}],
        "per_symbol": {
            sym: {
                "n_rows": int(len(df)),
                "date_min": str(df["date"].min().date()),
                "date_max": str(df["date"].max().date()),
                "n_with_10k": int(df["ten_k"].apply(len).gt(0).sum()),
                "n_with_10q": int(df["ten_q"].apply(len).gt(0).sum()),
                "n_with_news": int(df["news_raw_strings"].apply(len).gt(0).sum()) if "news_raw_strings" in df.columns else int(df["news"].apply(len).gt(0).sum()),
            }
            for sym, df in per_symbol.items()
        },
    }
    with open(OUT_DIR / "_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"saved _metadata.json")
    log.info(f"DONE. {len(meta['feature_columns'])} feature columns generated.")


if __name__ == "__main__":
    main()
