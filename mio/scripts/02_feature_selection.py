"""
Script 02 - Feature selection.

Carga el dataset raw, define el target con dead-zone basado en percentiles del retorno
(por activo), y combina cuatro métodos de importancia:
    1) Spearman correlation con target continuo (return normalizado)
    2) Mutual Information con target categórico (BUY/HOLD/SELL)
    3) LightGBM gain importance (con purged walk-forward CV)
    4) VIF para descartar multicolineales

Output:
    data/feature_analysis/spearman.csv
    data/feature_analysis/mi.csv
    data/feature_analysis/lgbm_importance.csv
    data/feature_analysis/vif.csv
    data/feature_analysis/combined_ranking.csv
    data/feature_analysis/selected_features.json
    data/feature_analysis/plots/*.png
    data/feature_analysis/target_distribution.json
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("feature_selection")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw_features"
OUT_DIR = PROJECT_ROOT / "data" / "feature_analysis"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Columnas que NUNCA son features (identificadores, raw text, target leak)
NON_FEATURE_COLS = {
    "date", "symbol", "news", "news_raw_strings", "news_weighted",
    "ten_k", "ten_q", "future_price_diff", "momentum_raw", "price",
    "target_class", "target_return", "target_return_norm",
}

# Configuración del target
TARGET_DEADZONE_PERCENTILE = 0.33  # ±33% del retorno absoluto -> HOLD


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Por cada activo, definimos:
      target_return       = future_price_diff / price
      target_return_norm  = z-score del retorno (volatilidad rolling 20d)
      target_class        = BUY (2) / HOLD (1) / SELL (0) usando dead-zone por percentil

    El dead-zone se calcula con percentiles del |target_return| en train (aproximado
    aquí como primer 70% temporal). Usando 33% inferior -> HOLD, 33-66 -> según signo,
    66+ -> claramente BUY/SELL.

    En realidad el split es: si |ret| < umbral -> HOLD; si ret > umbral -> BUY; else SELL.
    El umbral se estima por activo para respetar volatilidades distintas (TSLA != BTC).
    """
    df = df.copy()
    df["target_return"] = df["future_price_diff"] / df["price"]

    # vol rolling 20d para normalizar
    df["target_return_norm"] = np.nan
    df["target_class"] = np.nan

    thresholds = {}
    for sym in df["symbol"].unique():
        mask = df["symbol"] == sym
        sub = df.loc[mask].sort_values("date")
        ret = sub["target_return"]
        # vol rolling con shift para no usar info futura
        vol = ret.rolling(20, min_periods=5).std().shift(1)
        norm = ret / vol.replace(0, np.nan)
        df.loc[sub.index, "target_return_norm"] = norm.values

        # umbral usando solo primer 70% del activo (sin look-ahead)
        n = len(sub)
        train_cut = int(n * 0.70)
        train_ret = sub["target_return"].iloc[:train_cut].dropna()
        if len(train_ret) < 20:
            thr = 0.005  # fallback
        else:
            thr = np.quantile(np.abs(train_ret), TARGET_DEADZONE_PERCENTILE)
        thresholds[sym] = float(thr)

        cls = pd.Series(1, index=sub.index, dtype=int)  # default HOLD
        cls.loc[ret > thr] = 2  # BUY
        cls.loc[ret < -thr] = 0  # SELL
        # las filas con target_return NaN (último día) las dejamos NaN
        cls.loc[ret.isna()] = np.nan
        df.loc[sub.index, "target_class"] = cls.values

    # log distribución
    dist = df["target_class"].value_counts(dropna=True).to_dict()
    log.info(f"target class distribution: {dist}  (0=SELL, 1=HOLD, 2=BUY)")
    log.info(f"thresholds per symbol: {thresholds}")

    # guardar
    with open(OUT_DIR / "target_distribution.json", "w") as f:
        json.dump({
            "global_distribution": {str(k): int(v) for k, v in dist.items()},
            "thresholds_per_symbol": thresholds,
            "deadzone_percentile": TARGET_DEADZONE_PERCENTILE,
        }, f, indent=2)

    return df


# ---------------------------------------------------------------------------
# Feature column detection
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if df[c].dtype == "object":
            continue
        if df[c].dtype.name.startswith("datetime"):
            continue
        cols.append(c)
    return cols


# ---------------------------------------------------------------------------
# 1) Spearman
# ---------------------------------------------------------------------------

def spearman_analysis(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    from scipy.stats import spearmanr

    target = df["target_return_norm"]
    rows = []
    for f in features:
        x = df[f]
        valid = x.notna() & target.notna()
        if valid.sum() < 30:
            rows.append({"feature": f, "spearman_rho": 0.0, "spearman_pvalue": 1.0, "n": int(valid.sum())})
            continue
        try:
            rho, pval = spearmanr(x[valid], target[valid])
            if np.isnan(rho):
                rho, pval = 0.0, 1.0
        except Exception:
            rho, pval = 0.0, 1.0
        rows.append({"feature": f, "spearman_rho": float(rho), "spearman_pvalue": float(pval), "n": int(valid.sum())})
    out = pd.DataFrame(rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    out.to_csv(OUT_DIR / "spearman.csv", index=False)
    log.info(f"spearman done. top5: {out.head(5)['feature'].tolist()}")
    return out


# ---------------------------------------------------------------------------
# 2) Mutual Information
# ---------------------------------------------------------------------------

def mi_analysis(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    from sklearn.feature_selection import mutual_info_classif

    valid = df["target_class"].notna()
    sub = df.loc[valid, features + ["target_class"]].copy()
    # imputamos con mediana (MI no tolera NaN)
    for f in features:
        sub[f] = sub[f].fillna(sub[f].median())
    X = sub[features].values
    y = sub["target_class"].astype(int).values
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    out = pd.DataFrame({"feature": features, "mutual_info": mi}).sort_values("mutual_info", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_DIR / "mi.csv", index=False)
    log.info(f"MI done. top5: {out.head(5)['feature'].tolist()}")
    return out


# ---------------------------------------------------------------------------
# 3) LightGBM importance with purged walk-forward CV
# ---------------------------------------------------------------------------

def purged_walkforward_splits(df: pd.DataFrame, n_folds: int = 4, embargo_days: int = 5):
    """
    Walk-forward por fechas globales (no por activo) con embargo.
    Yield (train_idx, val_idx) por fold.
    """
    df_sorted = df.sort_values("date").reset_index()
    dates = df_sorted["date"].unique()
    n_dates = len(dates)
    fold_size = n_dates // (n_folds + 1)

    for k in range(n_folds):
        train_end = fold_size * (k + 1)
        val_start = train_end + embargo_days
        val_end = val_start + fold_size

        if val_end >= n_dates:
            break

        train_dates = set(dates[:train_end])
        val_dates = set(dates[val_start:val_end])

        train_idx = df_sorted.loc[df_sorted["date"].isin(train_dates), "index"].values
        val_idx = df_sorted.loc[df_sorted["date"].isin(val_dates), "index"].values
        yield train_idx, val_idx


def lgbm_importance(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("lightgbm no instalado, intentando sklearn GradientBoosting como fallback")
        return _sklearn_gb_importance_fallback(df, features)

    valid = df["target_class"].notna()
    sub = df.loc[valid].copy()
    for f in features:
        sub[f] = sub[f].fillna(sub[f].median())

    X = sub[features].values
    y = sub["target_class"].astype(int).values

    importance_acc = np.zeros(len(features))
    fold_count = 0

    for train_idx, val_idx in purged_walkforward_splits(sub.reset_index(drop=True), n_folds=4, embargo_days=5):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        if len(np.unique(y_tr)) < 2:
            continue

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            objective="multiclass",
            num_class=3,
            verbosity=-1,
            random_state=42,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
        importance_acc += model.booster_.feature_importance(importance_type="gain")
        fold_count += 1

    if fold_count == 0:
        log.warning("ningún fold válido en LGBM")
        return pd.DataFrame({"feature": features, "lgbm_gain": np.zeros(len(features))})

    importance_acc /= fold_count
    out = pd.DataFrame({"feature": features, "lgbm_gain": importance_acc}).sort_values("lgbm_gain", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_DIR / "lgbm_importance.csv", index=False)
    log.info(f"LGBM done ({fold_count} folds). top5: {out.head(5)['feature'].tolist()}")
    return out


def _sklearn_gb_importance_fallback(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    from sklearn.ensemble import GradientBoostingClassifier
    valid = df["target_class"].notna()
    sub = df.loc[valid].copy()
    for f in features:
        sub[f] = sub[f].fillna(sub[f].median())
    X = sub[features].values
    y = sub["target_class"].astype(int).values
    importance_acc = np.zeros(len(features))
    fold_count = 0
    for train_idx, val_idx in purged_walkforward_splits(sub.reset_index(drop=True)):
        if len(np.unique(y[train_idx])) < 2:
            continue
        m = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
        m.fit(X[train_idx], y[train_idx])
        importance_acc += m.feature_importances_
        fold_count += 1
    if fold_count == 0:
        return pd.DataFrame({"feature": features, "lgbm_gain": np.zeros(len(features))})
    importance_acc /= fold_count
    out = pd.DataFrame({"feature": features, "lgbm_gain": importance_acc}).sort_values("lgbm_gain", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_DIR / "lgbm_importance.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 4) VIF (multicolinealidad)
# ---------------------------------------------------------------------------

def vif_analysis(df: pd.DataFrame, features: list[str], vif_threshold: float = 10.0) -> tuple[pd.DataFrame, list[str]]:
    """
    Calcula VIF y elimina iterativamente features con VIF > threshold (greedy: el peor primero).
    Retorna (vif_df_inicial, features_sobrevivientes).
    """
    sub = df[features].copy()
    for f in features:
        sub[f] = sub[f].fillna(sub[f].median())

    # estandarizar
    sub = (sub - sub.mean()) / sub.std().replace(0, 1)
    sub = sub.dropna(axis=1, how="all")
    surviving = list(sub.columns)

    initial_vifs = _compute_vif(sub[surviving])
    initial_df = pd.DataFrame({"feature": surviving, "vif": initial_vifs}).sort_values("vif", ascending=False)
    initial_df.to_csv(OUT_DIR / "vif.csv", index=False)

    # eliminación greedy
    max_iter = 50
    for _ in range(max_iter):
        if len(surviving) < 2:
            break
        vifs = _compute_vif(sub[surviving])
        worst_idx = int(np.argmax(vifs))
        if vifs[worst_idx] <= vif_threshold:
            break
        dropped = surviving.pop(worst_idx)
        log.info(f"  VIF drop: {dropped} (vif={vifs[worst_idx]:.1f})")

    log.info(f"VIF: {len(features)} -> {len(surviving)} features sobreviven (threshold={vif_threshold})")
    return initial_df, surviving


def _compute_vif(X: pd.DataFrame) -> np.ndarray:
    """VIF vía R^2 de cada columna vs el resto. Implementación simple sin statsmodels."""
    cols = X.columns.tolist()
    vifs = []
    for i, c in enumerate(cols):
        y = X[c].values
        Xrest = X.drop(columns=c).values
        # regresión lineal por mínimos cuadrados
        try:
            beta, *_ = np.linalg.lstsq(Xrest, y, rcond=None)
            yhat = Xrest @ beta
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            r2 = min(max(r2, 0), 0.9999)
            vif = 1.0 / (1.0 - r2)
        except Exception:
            vif = 1.0
        vifs.append(vif)
    return np.array(vifs)


# ---------------------------------------------------------------------------
# Combined ranking + selection
# ---------------------------------------------------------------------------

def combine_rankings(spearman: pd.DataFrame, mi: pd.DataFrame, lgbm: pd.DataFrame, vif_survivors: list[str]) -> pd.DataFrame:
    """
    Combina los 3 rankings normalizados (rank/n) en un score promedio. Solo considera
    features que sobrevivieron a VIF.
    """
    sp = spearman.copy()
    sp["spearman_score"] = sp["spearman_rho"].abs()
    sp["spearman_rank"] = sp["spearman_score"].rank(ascending=False, pct=True)

    mi = mi.copy()
    mi["mi_rank"] = mi["mutual_info"].rank(ascending=False, pct=True)

    lg = lgbm.copy()
    lg["lgbm_rank"] = lg["lgbm_gain"].rank(ascending=False, pct=True)

    merged = sp[["feature", "spearman_score", "spearman_rank"]].merge(
        mi[["feature", "mutual_info", "mi_rank"]], on="feature", how="outer"
    ).merge(
        lg[["feature", "lgbm_gain", "lgbm_rank"]], on="feature", how="outer"
    )

    # menor pct rank = mejor; combinamos como media (1 - rank) -> mayor = mejor
    merged["combined_score"] = (
        (1 - merged["spearman_rank"].fillna(1)) * 0.25 +
        (1 - merged["mi_rank"].fillna(1)) * 0.25 +
        (1 - merged["lgbm_rank"].fillna(1)) * 0.50  # más peso a LGBM porque captura interacciones
    )
    merged["passes_vif"] = merged["feature"].isin(vif_survivors)

    merged = merged.sort_values("combined_score", ascending=False).reset_index(drop=True)
    merged.to_csv(OUT_DIR / "combined_ranking.csv", index=False)
    return merged


def find_elbow(scores: np.ndarray) -> int:
    """Encuentra el codo en una curva de scores (decreciente). Método: max distancia a la línea entre primer y último punto."""
    n = len(scores)
    if n < 3:
        return n
    x = np.arange(n)
    y = scores
    # línea entre (0, y[0]) y (n-1, y[-1])
    p1 = np.array([0, y[0]])
    p2 = np.array([n - 1, y[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return n
    line_unit = line_vec / line_len
    distances = []
    for i in range(n):
        pt = np.array([x[i], y[i]])
        vec = pt - p1
        proj = np.dot(vec, line_unit) * line_unit
        perp = vec - proj
        distances.append(np.linalg.norm(perp))
    elbow = int(np.argmax(distances)) + 1
    return max(elbow, 5)  # mínimo 5 features


def select_features(combined: pd.DataFrame) -> dict:
    """Aplica filtros: passes_vif, elbow del combined_score, garantiza mínimos por categoría."""
    passing = combined[combined["passes_vif"]].copy().reset_index(drop=True)
    if len(passing) == 0:
        log.warning("ninguna feature pasó VIF, usando todas")
        passing = combined.copy()

    elbow_k = find_elbow(passing["combined_score"].values)
    elbow_k = min(max(elbow_k, 15), 40)  # entre 15 y 40 features
    log.info(f"elbow detectado en k={elbow_k}")

    selected = passing.head(elbow_k)["feature"].tolist()

    # garantizar que features clave estén (si pasan VIF)
    must_have_if_present = [
        "momentum_encoded", "rsi_14", "macd_hist", "return_5d", "return_1d",
        "realized_vol_20d", "zscore_20d", "asset_type_crypto", "has_news",
        "log_news_count", "filing_decay_10q",
    ]
    vif_set = set(passing["feature"])
    for f in must_have_if_present:
        if f in vif_set and f not in selected:
            selected.append(f)
            log.info(f"  forced add: {f}")

    return {
        "n_features": len(selected),
        "elbow_k": elbow_k,
        "features": selected,
        "ranking_top20": passing.head(20).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(combined: pd.DataFrame, vif_df: pd.DataFrame, selected: list[str]):
    # combined ranking top 30
    top = combined.head(30)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2a9d8f" if f in selected else "#adb5bd" for f in top["feature"]]
    ax.barh(top["feature"][::-1], top["combined_score"][::-1], color=colors[::-1])
    ax.set_title("Combined feature ranking (top 30)")
    ax.set_xlabel("combined score")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "combined_ranking_top30.png", dpi=120)
    plt.close()

    # elbow plot
    scores = combined[combined["passes_vif"]]["combined_score"].values
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(scores) + 1), scores, marker="o", markersize=3)
    ax.axvline(len(selected), color="red", linestyle="--", label=f"selected k={len(selected)}")
    ax.set_xlabel("feature rank")
    ax.set_ylabel("combined score")
    ax.set_title("Elbow analysis (VIF-survivors only)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "elbow.png", dpi=120)
    plt.close()

    # VIF histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    vif_clipped = vif_df["vif"].clip(upper=50)
    ax.hist(vif_clipped, bins=30, color="#e76f51")
    ax.axvline(10, color="black", linestyle="--", label="threshold=10")
    ax.set_xlabel("VIF (clipped at 50)")
    ax.set_title("VIF distribution (initial)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "vif_distribution.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw_path = RAW_DIR / "_all.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"no se encontró {raw_path}. Corre primero 01_build_raw_dataset.py")

    log.info(f"loading {raw_path}")
    df = pd.read_parquet(raw_path)
    log.info(f"  {len(df)} rows, {len(df.columns)} cols")

    df = build_targets(df)
    features = get_feature_columns(df)
    log.info(f"candidate features: {len(features)}")

    log.info("=" * 60)
    log.info("1/4 Spearman analysis")
    sp = spearman_analysis(df, features)

    log.info("=" * 60)
    log.info("2/4 Mutual Information")
    mi = mi_analysis(df, features)

    log.info("=" * 60)
    log.info("3/4 LightGBM importance with purged walk-forward CV")
    lg = lgbm_importance(df, features)

    log.info("=" * 60)
    log.info("4/4 VIF analysis")
    vif_df, vif_survivors = vif_analysis(df, features, vif_threshold=10.0)

    log.info("=" * 60)
    log.info("Combining rankings")
    combined = combine_rankings(sp, mi, lg, vif_survivors)
    selection = select_features(combined)

    with open(OUT_DIR / "selected_features.json", "w") as f:
        json.dump(selection, f, indent=2, default=str)
    log.info(f"selected {selection['n_features']} features")
    log.info(f"  top 10: {selection['features'][:10]}")

    log.info("Generating plots")
    make_plots(combined, vif_df, selection["features"])

    log.info(f"DONE. results in {OUT_DIR}")


if __name__ == "__main__":
    main()
