"""
Persistent state store for API runtime.

Mantiene una tabla `daily_state` con todo lo que necesitamos para reconstruir features
para cualquier (symbol, date). Idempotente por (symbol, date) — si llega el mismo día
dos veces (re-envío), se actualiza en su lugar.

Schema:
    daily_state(
        symbol TEXT, date TEXT,
        price REAL,
        news_json TEXT,                  -- JSON list[str]
        ten_k_json TEXT,                 -- JSON list[str] (puede ser '[]')
        ten_q_json TEXT,                 -- JSON list[str]
        momentum_raw TEXT,
        future_price_diff REAL NULL,     -- llenado al día siguiente cuando llega next price
        ingested_at TEXT,
        PRIMARY KEY (symbol, date)
    )

    daily_action(
        symbol TEXT, date TEXT,
        action TEXT,                     -- BUY/HOLD/SELL
        rationale TEXT,
        gates_json TEXT,                 -- JSON dict
        top_news TEXT,
        latency_ms INTEGER,
        model_version TEXT,
        served_at TEXT,
        PRIMARY KEY (symbol, date)
    )
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "state.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_state (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    price REAL,
    news_json TEXT NOT NULL DEFAULT '[]',
    ten_k_json TEXT NOT NULL DEFAULT '[]',
    ten_q_json TEXT NOT NULL DEFAULT '[]',
    momentum_raw TEXT,
    future_price_diff REAL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_state_symbol_date ON daily_state(symbol, date);

CREATE TABLE IF NOT EXISTS daily_action (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    action TEXT NOT NULL,
    rationale TEXT,
    gates_json TEXT,
    top_news TEXT,
    latency_ms INTEGER,
    model_version TEXT,
    served_at TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_action_symbol_date ON daily_action(symbol, date);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class StateStore:
    """Thread-safe wrapper. Permite acceso concurrente desde FastAPI."""

    def __init__(self, db_path: Path = DEFAULT_DB):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self._lock = threading.RLock()
        # check_same_thread=False + lock manual para que FastAPI workers compartan conexión.
        # Para producción de verdad usaríamos un pool, pero esto es suficiente.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=10.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()

    # --- meta ---
    def set_meta(self, key: str, value: str):
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            cur = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    # --- daily_state ---
    def upsert_day(
        self, symbol: str, date: str, price: Optional[float],
        news: list[str], ten_k: list[str], ten_q: list[str],
        momentum_raw: Optional[str],
    ):
        """Inserta o actualiza un día. Idempotente."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO daily_state(symbol,date,price,news_json,ten_k_json,ten_q_json,momentum_raw,ingested_at)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol,date) DO UPDATE SET
                    price=COALESCE(excluded.price, daily_state.price),
                    news_json=excluded.news_json,
                    ten_k_json=excluded.ten_k_json,
                    ten_q_json=excluded.ten_q_json,
                    momentum_raw=COALESCE(excluded.momentum_raw, daily_state.momentum_raw),
                    ingested_at=excluded.ingested_at
                """,
                (
                    symbol, date, price,
                    json.dumps(news, ensure_ascii=False),
                    json.dumps(ten_k, ensure_ascii=False),
                    json.dumps(ten_q, ensure_ascii=False),
                    momentum_raw, now,
                ),
            )
            # actualizar future_price_diff del día anterior si el price está disponible
            if price is not None:
                cur = self._conn.execute(
                    "SELECT date, price FROM daily_state WHERE symbol=? AND date<? ORDER BY date DESC LIMIT 1",
                    (symbol, date),
                )
                prev = cur.fetchone()
                if prev and prev[1] is not None:
                    diff = price - prev[1]
                    self._conn.execute(
                        "UPDATE daily_state SET future_price_diff=? WHERE symbol=? AND date=?",
                        (diff, symbol, prev[0]),
                    )
            self._conn.commit()

    def upsert_history_prices(self, symbol: str, history: list[dict]):
        """Inserta los precios históricos del request si los días no existen aún.
        history: [{'date': 'YYYY-MM-DD', 'price': float}, ...]
        No pisa news/filings de días ya existentes."""
        with self._lock:
            for h in history:
                d = h["date"]
                p = float(h["price"])
                # solo inserta si no existe; si existe, actualiza el precio
                self._conn.execute(
                    """
                    INSERT INTO daily_state(symbol,date,price,news_json,ten_k_json,ten_q_json,ingested_at)
                    VALUES (?,?,?,'[]','[]','[]', ?)
                    ON CONFLICT(symbol,date) DO UPDATE SET
                        price=COALESCE(excluded.price, daily_state.price)
                    """,
                    (symbol, d, p, datetime.now(timezone.utc).isoformat()),
                )
            self._conn.commit()

    def get_history(self, symbol: str, until_date: str, lookback_days: int = 200) -> list[dict]:
        """
        Devuelve hasta `lookback_days` filas <= until_date para ese símbolo, ordenadas asc.
        El día until_date va incluido si existe.
        """
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT date, price, news_json, ten_k_json, ten_q_json, momentum_raw
                FROM daily_state
                WHERE symbol=? AND date<=?
                ORDER BY date DESC
                LIMIT ?
                """,
                (symbol, until_date, lookback_days),
            )
            rows = cur.fetchall()
        rows = list(reversed(rows))
        out = []
        for r in rows:
            out.append({
                "date": r[0],
                "price": r[1],
                "news": json.loads(r[2]) if r[2] else [],
                "ten_k": json.loads(r[3]) if r[3] else [],
                "ten_q": json.loads(r[4]) if r[4] else [],
                "momentum_raw": r[5] or "neutral",
            })
        return out

    def get_all_recent(self, until_date: str, lookback_days: int = 60) -> list[dict]:
        """Para cross-asset features (mkt_proxy_return, btc_return, etc.)."""
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT symbol, date, price
                FROM daily_state
                WHERE date<=? AND price IS NOT NULL
                ORDER BY symbol, date
                """,
                (until_date,),
            )
            rows = cur.fetchall()
        return [{"symbol": r[0], "date": r[1], "price": r[2]} for r in rows]

    # --- actions log ---
    def log_action(
        self, symbol: str, date: str, action: str, rationale: str,
        gates: dict, top_news: str, latency_ms: int, model_version: str,
    ):
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO daily_action(symbol,date,action,rationale,gates_json,top_news,latency_ms,model_version,served_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol,date) DO UPDATE SET
                    action=excluded.action,
                    rationale=excluded.rationale,
                    gates_json=excluded.gates_json,
                    top_news=excluded.top_news,
                    latency_ms=excluded.latency_ms,
                    model_version=excluded.model_version,
                    served_at=excluded.served_at
                """,
                (symbol, date, action, rationale, json.dumps(gates), top_news, latency_ms, model_version, now),
            )
            self._conn.commit()

    def get_last_action(self, symbol: str) -> Optional[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT date, action, rationale FROM daily_action WHERE symbol=? ORDER BY date DESC LIMIT 1",
                (symbol,),
            )
            row = cur.fetchone()
        if row:
            return {"date": row[0], "action": row[1], "rationale": row[2]}
        return None

    def stats(self) -> dict:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM daily_state")
            n_state = cur.fetchone()[0]
            cur = self._conn.execute("SELECT COUNT(*) FROM daily_action")
            n_action = cur.fetchone()[0]
            cur = self._conn.execute("SELECT symbol, COUNT(*), MIN(date), MAX(date) FROM daily_state GROUP BY symbol")
            per_sym = {r[0]: {"rows": r[1], "min_date": r[2], "max_date": r[3]} for r in cur.fetchall()}
        return {"total_state_rows": n_state, "total_actions": n_action, "per_symbol": per_sym}
