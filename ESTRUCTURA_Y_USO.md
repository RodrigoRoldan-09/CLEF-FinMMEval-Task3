# FinMMEval Task 3 — Guía de Estructura y Uso

## Estructura de carpetas

```
finmmeval_task3/              ← raíz del proyecto (ejecutar todos los comandos desde aquí)
│
├── model/                    ← paquete del modelo (NO editar sin leer abajo)
│   ├── __init__.py           ← exports lazy (no arrastra torch innecesariamente)
│   ├── architecture.py       ← TradingModel, ModelConfig, GatedFusion, MultiTaskLoss
│   └── feature_engineering.py ← FUENTE ÚNICA de features (training Y serving)
│
├── serving/                  ← paquete del API en producción
│   ├── __init__.py           ← exports lazy
│   ├── state_store.py        ← SQLite persistente con upsert idempotente
│   ├── embedding_runtime.py  ← FinBERT en GPU fp16 + caché .npz compartida
│   ├── feature_runtime.py    ← convierte request → tensores listos para el modelo
│   ├── explain.py            ← genera justificación ≤50 palabras (plantilla, no LLM)
│   └── api.py                ← FastAPI endpoint compatible con schema del concurso
│
├── scripts/                  ← pipeline de entrenamiento (correr en orden 01→05)
│   ├── 01_build_raw_dataset.py  ← descarga HF, consolida fines de semana, ~60 features
│   ├── 02_feature_selection.py  ← Spearman + MI + LGBM + VIF → 19 features
│   ├── 03_build_final_dataset.py ← embeddings FinBERT + tensores PyTorch por split
│   ├── 04_train.py              ← entrenamiento con augmentation + SWA
│   └── 05_seed_state_db.py      ← siembra el SQLite con todo el histórico (1 vez)
│
├── data/                     ← generada automáticamente por los scripts
│   ├── raw_features/
│   │   ├── AAPL.parquet      ← un archivo por activo + columnas de features candidatas
│   │   ├── BTC.parquet
│   │   ├── ... (12 archivos)
│   │   ├── _all.parquet      ← todos los activos concatenados
│   │   └── _metadata.json    ← estadísticas del dataset
│   │
│   ├── feature_analysis/
│   │   ├── selected_features.json  ← las 19 features elegidas con sus scores
│   │   ├── feature_ranking.png
│   │   └── class_distribution.png
│   │
│   ├── final/
│   │   ├── train.pt          ← tensores de entrenamiento (1 588 muestras)
│   │   ├── val.pt            ← tensores de validación (360 muestras)
│   │   ├── test.pt           ← tensores de prueba (496 muestras)
│   │   ├── scaler.json       ← media/std de cada feature (fit en train)
│   │   ├── symbol_vocab.json ← {"AAPL": 0, "BTC": 1, ...}
│   │   └── manifest.json     ← config del dataset (n_features, seq_len, etc.)
│   │
│   ├── cache/
│   │   ├── finbert_news.npz     ← embeddings de noticias cacheados (SHA1 → vector)
│   │   └── finbert_filings.npz  ← embeddings de filings cacheados
│   │
│   └── state.db              ← SQLite de producción (generado por script 05)
│
└── checkpoints/              ← generada por script 04
    ├── best.pt               ← mejor checkpoint por EMA-F1 en validación
    ├── best_swa.pt           ← checkpoint SWA (suele generalizar mejor en test)
    └── training_history.json ← métricas de todas las épocas + test final


TOTAL archivos que debes tener antes de iniciar el serving:
  ✓  checkpoints/best.pt          (del script 04)
  ✓  data/state.db                (del script 05)
  ✓  data/final/scaler.json       (del script 03)
  ✓  data/final/manifest.json     (del script 03)
  ✓  data/final/symbol_vocab.json (del script 03)
  ✓  data/cache/finbert_news.npz  (del script 03)
  ✓  data/cache/finbert_filings.npz (del script 03)
```

---

## Orden de ejecución

### Fase 1 — Entrenamiento (solo una vez)

```bash
# Desde la raíz del proyecto
cd finmmeval_task3

# 1. Descargar datos y calcular features candidatas (~5-10 min)
python scripts/01_build_raw_dataset.py

# 2. Seleccionar las 19 mejores features (~5 min)
python scripts/02_feature_selection.py

# 3. Generar embeddings FinBERT y tensores PyTorch (~20-40 min primera vez)
python scripts/03_build_final_dataset.py

# 4. Entrenar el modelo (~5-10 min en GPU)
python scripts/04_train.py --epochs 80 --batch_size 32
```

### Fase 2 — Preparar producción (solo una vez)

```bash
# 5. Sembrar la base de datos SQLite con todo el histórico
python scripts/05_seed_state_db.py
```

### Fase 3 — Servidor (mantener activo durante el concurso)

```bash
# Iniciar el API (GPU obligatorio para FinBERT)
python -m serving.api --host 0.0.0.0 --port 62237

# En otra terminal: exponer al internet para que el organizador lo alcance
cloudflared tunnel --url http://localhost:62237
# o: ngrok http 62237
```

---

## Qué hace cada archivo del serving

### `serving/state_store.py`
Base de datos SQLite que acumula toda la historia de precios, noticias y filings día a día.
- Siembra: script 05 la llena con los ~252 días históricos por activo
- Producción: cada request añade el día nuevo con `upsert_day()`
- Tablas: `daily_state` (datos de mercado) y `daily_action` (auditoría de predicciones)
- Idempotente: re-enviar el mismo día actualiza la fila existente, no duplica

### `serving/embedding_runtime.py`
FinBERT en GPU fp16 con caché en disco compartida con el entrenamiento.
- Noticias: [CLS] embedding (768 dims) + sentiment 3-dims por texto
- Filings largos: chunking (384 tokens) + mean pooling de hasta 12 chunks
- Caché: SHA1[:20] del texto → vector .npz. Hits de caché = ~0ms

### `serving/feature_runtime.py`
Orquesta el pipeline completo: state store → features → embeddings → tensores.
- Llama a `model/feature_engineering.py` (mismo código que el script 01)
- Devuelve el batch dict listo para `model.forward()`

### `serving/explain.py`
Genera la justificación ≤50 palabras con una plantilla (no un LLM).
- `feature_attribution()`: calcula gradiente × entrada para encontrar las features más influyentes
- `render_rationale()`: llena la plantilla con la rama dominante, top-3 features, sentimiento y clip de noticia

### `serving/api.py`
El endpoint FastAPI. En el startup carga modelo + FinBERT + SQLite + scaler.
- `POST /trading_action/` → procesa request → devuelve `{"recommended_action": "BUY", "rationale": "..."}`
- `GET /health` → estado del servidor y disponibilidad de CUDA
- `GET /stats` → resumen de la base de datos (filas por activo, total de acciones)
- Fallback: cualquier error en el pipeline → responde `HOLD` automáticamente

---

## Variables de entorno (opcionales)

```bash
RATIONALE_ENABLED=0        # deshabilitar la justificación (más rápido)
CHECKPOINT_PATH=checkpoints/best_swa.pt  # usar checkpoint SWA en vez del normal
STATE_DB=data/state.db     # ruta a la base de datos
```

---

## Dependencias

```bash
pip install pandas pyarrow huggingface_hub
pip install numpy scipy scikit-learn lightgbm matplotlib
pip install torch transformers sentencepiece
pip install fastapi uvicorn pydantic python-dotenv
```

Requisito obligatorio en producción: **GPU con CUDA** (FinBERT en CPU tarda 200-400ms por noticia y puede superar el timeout de 3 min del concurso).

---

## Prueba rápida del endpoint

```bash
curl -X POST http://localhost:62237/trading_action/ \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2026-04-26",
    "price": {"TSLA": 250.50},
    "news": {"TSLA": ["Tesla beats Q1 earnings expectations"]},
    "symbol": ["TSLA"],
    "momentum": {"TSLA": "bullish"},
    "10k": null,
    "10q": null,
    "history_price": {"TSLA": [
      {"date": "2026-04-24", "price": 248.0},
      {"date": "2026-04-25", "price": 249.5}
    ]}
  }'
```

Respuesta esperada:
```json
{
  "recommended_action": "BUY",
  "rationale": "BUY (driven by today's metrics; RSI(14)=62, 5d return=+0.02, MACD hist=0.45; sentiment pos)"
}
```

---

## Flujo interno de un request (qué pasa por dentro)

```
POST /trading_action/
        │
        ▼
1. upsert_history_prices()     ← guarda precios históricos del request en SQLite
2. upsert_day()                ← guarda precio + noticias + filings de hoy
        │
        ▼
3. get_history(lookback=200d)  ← lee 200 días de contexto del SQLite
4. build_features_for_symbol() ← calcula las 19 features (mismo código que training)
5. add_cross_asset_features()  ← mkt_proxy y BTC return
6. estandarizar con scaler.json
        │
        ▼
7. embed_news()                ← FinBERT GPU (o caché .npz si ya fue vista)
8. compute_filing_state()      ← embedding de filings con decay 45d
        │
        ▼
9. model.forward()             ← logits + gates + news_attn
        │
        ▼
10. render_rationale()         ← plantilla determinística <50 palabras
11. log_action()               ← auditoría en SQLite
        │
        ▼
{"recommended_action": "BUY", "rationale": "..."}
```

---

## Consejos operativos

- **Checkpoint a usar**: compara `best.pt` vs `best_swa.pt` en test. `best_swa.pt` habitualmente generaliza mejor. Para cambiarlo: `CHECKPOINT_PATH=checkpoints/best_swa.pt python -m serving.api`
- **Ver el log de predicciones**: `sqlite3 data/state.db "SELECT date, symbol, action, rationale, latency_ms FROM daily_action ORDER BY served_at DESC LIMIT 20;"`
- **Ver historia por activo**: `sqlite3 data/state.db "SELECT * FROM daily_state WHERE symbol='TSLA' ORDER BY date DESC LIMIT 5;"`
- **Re-sembrar la DB** (si algo se corrompió): `python scripts/05_seed_state_db.py --reset`
- **Sin rationale** (más rápido, menos riesgo de error): `python -m serving.api --no-rationale`
