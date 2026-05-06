"""
Script 05 - Seed the SQLite state store with historical data.

Lee data/raw_features/_all.parquet (o los .parquet por símbolo) y carga al SQLite
toda la historia disponible. Esto se corre UNA VEZ antes del primer request en
producción para que el día 1 ya tenga >200 días de histórico para calcular RSI_14,
SMA_50, vol_60d, etc.

Después, cada request del API añade el día nuevo vía StateStore.upsert_day().
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from serving.state_store import StateStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("seed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw_features")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "state.db")
    parser.add_argument("--reset", action="store_true", help="borrar la DB antes de sembrar")
    args = parser.parse_args()

    if args.reset and args.db.exists():
        args.db.unlink()
        log.info(f"removed existing {args.db}")

    raw_path = args.raw_dir / "_all.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"no se encontró {raw_path}")

    df = pd.read_parquet(raw_path)
    log.info(f"loaded {len(df)} rows from {raw_path}")

    store = StateStore(args.db)

    # tomamos las columnas crudas: en el raw, las noticias originales del día están en
    # `news` (lista de strings), y "news_raw_strings" es la versión post-consolidación
    # de fines de semana (incluye carry-over). Para sembrar usamos la cruda original
    # porque queremos que el store represente el estado tal-cual del día. La consolidación
    # de fines de semana se aplica en runtime cuando construimos features.
    news_col = "news" if "news" in df.columns else "news_raw_strings"

    n_inserted = 0
    for sym in df["symbol"].unique():
        sub = df[df["symbol"] == sym].sort_values("date")
        log.info(f"[{sym}] inserting {len(sub)} days")
        for _, row in sub.iterrows():
            date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
            news = list(row[news_col]) if news_col in sub.columns else []
            ten_k = list(row["ten_k"]) if "ten_k" in sub.columns else []
            ten_q = list(row["ten_q"]) if "ten_q" in sub.columns else []
            momentum_raw = str(row["momentum_raw"]) if "momentum_raw" in sub.columns else None
            store.upsert_day(
                symbol=sym,
                date=date_str,
                price=float(row["price"]) if pd.notna(row["price"]) else None,
                news=[str(n) for n in news],
                ten_k=[str(t) for t in ten_k],
                ten_q=[str(t) for t in ten_q],
                momentum_raw=momentum_raw,
            )
            n_inserted += 1

    store.set_meta("seeded_from", str(raw_path))
    store.set_meta("seeded_rows", str(n_inserted))

    stats = store.stats()
    log.info(f"DONE. {n_inserted} rows inserted.")
    log.info(f"per-symbol stats: {json.dumps(stats['per_symbol'], indent=2)}")


if __name__ == "__main__":
    main()
