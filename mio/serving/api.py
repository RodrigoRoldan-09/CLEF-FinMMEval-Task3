"""
FastAPI server for FinMMEval Task 3.

Endpoint POST /trading_action/  -- mismo schema que el reference.

Lifecycle:
    startup: carga modelo, FinBERT, state store, scaler/manifest.
    request: ingest -> features -> embeddings (cached) -> model -> action -> rationale -> log.
    shutdown: flush caches, close DB.

Defaults to GPU. Si CUDA no está disponible, log error y arranca en CPU (lento).

Render deployment: si STATE_DB apunta a un disco persistente vacío, el startup
copia automáticamente el state.db del repositorio como fallback.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import TradingModel, ModelConfig
from serving.state_store import StateStore
from serving.embedding_runtime import FinBertRuntime
from serving.feature_runtime import FeatureRuntime
from serving.explain import render_rationale, feature_attribution

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("api")


# ---------------------------------------------------------------------------
# Schema (idéntico al reference)
# ---------------------------------------------------------------------------

class HistoricalPrice(BaseModel):
    date: str
    price: float


class TradingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    date: str
    price: Dict[str, float]
    news: Dict[str, List[str]]
    symbol: List[str]
    momentum: Optional[Dict[str, str]] = None
    history_price: Dict[str, List[HistoricalPrice]] = Field(default_factory=dict)
    ten_k: Optional[Dict[str, List[str]]] = Field(default=None, alias="10k")
    ten_q: Optional[Dict[str, List[str]]] = Field(default=None, alias="10q")


class TradingResponse(BaseModel):
    recommended_action: str
    rationale: Optional[str] = None  # se llena solo si RATIONALE_ENABLED


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="FinMMEval Task 3 — Multi-Branch Trading API", version="1.0.0")
RATIONALE_ENABLED = os.getenv("RATIONALE_ENABLED", "1") == "1"


class ServerState:
    def __init__(self):
        self.model: TradingModel | None = None
        self.feature_runtime: FeatureRuntime | None = None
        self.embed_runtime: FinBertRuntime | None = None
        self.state_store: StateStore | None = None
        self.device: torch.device | None = None
        self.model_version: str = "v1"
        self.idx_to_label = {0: "SELL", 1: "HOLD", 2: "BUY"}


SERVER = ServerState()


@app.on_event("startup")
def startup():
    final_dir = PROJECT_ROOT / "data" / "final"
    cache_dir = PROJECT_ROOT / "data" / "cache"
    ckpt_path = Path(os.getenv("CHECKPOINT_PATH", PROJECT_ROOT / "checkpoints" / "best.pt"))
    db_path = Path(os.getenv("STATE_DB", PROJECT_ROOT / "data" / "state.db"))

    # Render fallback: si STATE_DB apunta a /data/state.db (disco persistente)
    # y no existe todavía, copiamos el state.db del repo como punto de partida.
    if not db_path.exists():
        local_db = PROJECT_ROOT / "data" / "state.db"
        if local_db.exists() and db_path != local_db:
            import shutil
            db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_db, db_path)
            log.info(f"Render fallback: copiado {local_db} -> {db_path}")

    if not ckpt_path.exists():
        raise RuntimeError(f"checkpoint no existe: {ckpt_path}. Corre train primero.")
    if not db_path.exists():
        raise RuntimeError(f"state DB no existe: {db_path}. Corre 05_seed_state_db.py primero.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log.warning("CUDA no disponible. Sirviendo en CPU (latencia alta, riesgo de timeout).")
    SERVER.device = device

    # 1. State store
    SERVER.state_store = StateStore(db_path)
    stats = SERVER.state_store.stats()
    log.info(f"state store: {stats['total_state_rows']} rows, "
             f"symbols: {sorted(stats['per_symbol'].keys())}")

    # 2. FinBERT runtime
    SERVER.embed_runtime = FinBertRuntime(
        cache_news_path=cache_dir / "finbert_news.npz",
        cache_filings_path=cache_dir / "finbert_filings.npz",
        device=str(device),
        use_fp16=(device.type == "cuda"),
    )

    # 3. Feature runtime
    SERVER.feature_runtime = FeatureRuntime(
        state_store=SERVER.state_store,
        embedding_runtime=SERVER.embed_runtime,
        scaler_path=final_dir / "scaler.json",
        manifest_path=final_dir / "manifest.json",
        symbol_vocab_path=final_dir / "symbol_vocab.json",
    )

    # 4. Model
    log.info(f"loading checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = TradingModel(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    SERVER.model = model
    SERVER.model_version = f"v1-epoch{ckpt.get('epoch', '?')}"
    log.info(f"model loaded ({sum(p.numel() for p in model.parameters()):,} params, version {SERVER.model_version})")
    log.info(f"rationale enabled: {RATIONALE_ENABLED}")


@app.on_event("shutdown")
def shutdown():
    if SERVER.embed_runtime:
        SERVER.embed_runtime.flush_caches()
    if SERVER.state_store:
        SERVER.state_store.close()
    log.info("shutdown complete")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {"service": "FinMMEval Task 3", "version": SERVER.model_version, "rationale_enabled": RATIONALE_ENABLED}


@app.get("/health")
def health():
    if SERVER.model is None:
        raise HTTPException(503, "model not loaded")
    cuda_available = torch.cuda.is_available()
    return {
        "status": "healthy",
        "device": str(SERVER.device),
        "cuda_available": cuda_available,
        "model_version": SERVER.model_version,
        "state_rows": SERVER.state_store.stats()["total_state_rows"] if SERVER.state_store else 0,
    }


@app.get("/stats")
def stats():
    if SERVER.state_store is None:
        raise HTTPException(503, "store not loaded")
    return SERVER.state_store.stats()


@app.post("/trading_action/", response_model=TradingResponse)
def trading_action(request: TradingRequest):
    t0 = time.time()
    if SERVER.model is None:
        raise HTTPException(503, "model not loaded")

    if not request.symbol:
        raise HTTPException(400, "no symbol provided")
    symbol = request.symbol[0]

    if symbol not in request.price:
        raise HTTPException(400, f"no price for symbol {symbol}")
    price = float(request.price[symbol])

    # extraer
    news_for_sym = list((request.news or {}).get(symbol, []))
    momentum = (request.momentum or {}).get(symbol)
    ten_k_for_sym = list((request.ten_k or {}).get(symbol, [])) if request.ten_k else []
    ten_q_for_sym = list((request.ten_q or {}).get(symbol, [])) if request.ten_q else []
    history_for_sym = [{"date": h.date, "price": float(h.price)} for h in request.history_price.get(symbol, [])]

    # 1. ingest
    try:
        SERVER.feature_runtime.ingest_request(
            symbol=symbol, date=request.date, price=price,
            news=news_for_sym, ten_k=ten_k_for_sym, ten_q=ten_q_for_sym,
            momentum=momentum, history_price=history_for_sym,
        )
    except Exception as e:
        log.exception("ingest failed")
        raise HTTPException(500, f"ingest failed: {type(e).__name__}: {e}")

    # 2. features + embeddings
    try:
        batch, meta = SERVER.feature_runtime.build_tensors(symbol=symbol, date=request.date)
    except Exception as e:
        log.exception("feature build failed")
        # fallback HOLD para no romper el concurso
        SERVER.state_store.log_action(
            symbol, request.date, "HOLD",
            rationale="fallback: feature build error",
            gates={}, top_news="", latency_ms=int((time.time()-t0)*1000),
            model_version=SERVER.model_version,
        )
        return TradingResponse(recommended_action="HOLD", rationale="HOLD (system fallback)")

    # 3. mover a device
    batch = {k: v.to(SERVER.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    # 4. inferencia
    with torch.no_grad():
        out = SERVER.model(batch)
        probs = torch.softmax(out["logits"], dim=-1)[0].cpu().numpy()
        gates = out["gates"][0].cpu().numpy()
        news_attn = out["news_attn"][0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    action = SERVER.idx_to_label[pred_idx]

    # 5. rationale (si habilitado)
    rationale = ""
    if RATIONALE_ENABLED:
        try:
            attribution = feature_attribution(SERVER.model, batch, pred_idx)
            rationale = render_rationale(
                action=action,
                probabilities=probs,
                gates=gates,
                news_attn=news_attn,
                feature_attribution_scores=attribution,
                scalar_columns=SERVER.feature_runtime.scalar_cols,
                scalar_features_unscaled=meta["scalar_features_unscaled"],
                news_slot_texts=meta["news_slot_texts"],
                sentiment=np.array(meta["news_sentiment"]),
                enabled=True,
                max_words=50,
            )
        except Exception as e:
            log.exception("rationale failed (non-fatal)")
            rationale = f"{action} (rationale unavailable)"

    # 6. log
    latency_ms = int((time.time() - t0) * 1000)
    SERVER.state_store.log_action(
        symbol=symbol, date=request.date, action=action,
        rationale=rationale,
        gates={"scalar_now": float(gates[0]), "scalar_seq": float(gates[1]),
               "news": float(gates[2]), "filing": float(gates[3])},
        top_news=meta.get("top_news_text", ""),
        latency_ms=latency_ms,
        model_version=SERVER.model_version,
    )
    log.info(
        f"[{request.date} {symbol}] -> {action} "
        f"(p={probs[pred_idx]:.3f}, gates={gates.round(2).tolist()}, "
        f"news_count={meta['n_news_today']}, latency={latency_ms}ms)"
    )

    if RATIONALE_ENABLED:
        return TradingResponse(recommended_action=action, rationale=rationale)
    return TradingResponse(recommended_action=action)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=62237)
    p.add_argument("--no-rationale", action="store_true", help="desactivar rationale")
    args = p.parse_args()

    if args.no_rationale:
        os.environ["RATIONALE_ENABLED"] = "0"

    log.info(f"starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
