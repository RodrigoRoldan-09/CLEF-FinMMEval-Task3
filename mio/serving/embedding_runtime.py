"""
Runtime FinBERT encoder for serving.

Usa el mismo cache .npz que el script 03 (training) para que requests con noticias
ya vistas en el training set sean instantáneas. Si llega una noticia nueva, se
embeddá una sola vez y se persiste el cache (write-through).
"""
from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("embedding_runtime")

FINBERT_MODEL = "yiyanghkust/finbert-tone"
FINBERT_DIM = 768
NEWS_MAX_TOKENS = 256
FILING_CHUNK_TOKENS = 384
FILING_MAX_CHUNKS = 12


class DiskCacheNPZ:
    """Cache write-through compartido con training. Persiste cada N escrituras."""

    def __init__(self, path: Path, persist_every: int = 25):
        self.path = Path(path)
        self.persist_every = persist_every
        self._dirty = 0
        self._lock = threading.RLock()
        self.data = {}
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as f:
                self.data = {k: f[k] for k in f.files}
            log.info(f"loaded {len(self.data)} cached embeddings from {self.path.name}")

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]

    def get(self, text: str) -> Optional[np.ndarray]:
        with self._lock:
            return self.data.get(self.key(text))

    def put(self, text: str, emb: np.ndarray):
        with self._lock:
            self.data[self.key(text)] = emb.astype(np.float32)
            self._dirty += 1
            if self._dirty >= self.persist_every:
                self._save_locked()

    def flush(self):
        with self._lock:
            if self._dirty > 0:
                self._save_locked()

    def _save_locked(self):
        if not self.data:
            return
        # escritura atómica: a tmp y luego rename
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        np.savez_compressed(tmp, **self.data)
        tmp.replace(self.path)
        self._dirty = 0


class FinBertRuntime:
    """FinBERT en GPU para serving. Embedding [CLS] + sentiment, con cache."""

    def __init__(
        self,
        cache_news_path: Path,
        cache_filings_path: Path,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

        self.device = torch.device(device)
        log.info(f"loading FinBERT on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        self.encoder = AutoModel.from_pretrained(FINBERT_MODEL).to(self.device).eval()
        try:
            self.classifier = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(self.device).eval()
            self.has_classifier = True
        except Exception as e:
            log.warning(f"sin classifier head: {e}")
            self.has_classifier = False

        if use_fp16 and self.device.type == "cuda":
            self.encoder = self.encoder.half()
            if self.has_classifier:
                self.classifier = self.classifier.half()
            self.fp16 = True
        else:
            self.fp16 = False

        self.news_cache = DiskCacheNPZ(cache_news_path)
        self.filing_cache = DiskCacheNPZ(cache_filings_path)

    @torch.no_grad()
    def embed_news(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cls_emb (768,), sentiment (3,) ordered [neg, neu, pos])."""
        if not text or not text.strip():
            return np.zeros(FINBERT_DIM, dtype=np.float32), np.array([0, 1, 0], dtype=np.float32)

        cached = self.news_cache.get(text)
        if cached is not None and cached.shape[0] == FINBERT_DIM + 3:
            return cached[:FINBERT_DIM].copy(), cached[FINBERT_DIM:].copy()

        toks = self.tokenizer(text, truncation=True, max_length=NEWS_MAX_TOKENS, return_tensors="pt").to(self.device)
        out = self.encoder(**toks)
        cls = out.last_hidden_state[:, 0, :].squeeze(0).float().cpu().numpy()

        if self.has_classifier:
            logits = self.classifier(**toks).logits.squeeze(0).float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # yiyanghkust/finbert-tone: 0=neutral, 1=positive, 2=negative
            sentiment = np.array([probs[2], probs[0], probs[1]], dtype=np.float32)
        else:
            sentiment = np.array([0, 1, 0], dtype=np.float32)

        self.news_cache.put(text, np.concatenate([cls, sentiment]))
        return cls, sentiment

    @torch.no_grad()
    def embed_filing_long(self, text: str) -> np.ndarray:
        """Chunking + mean pooling para filings largos (10K/10Q)."""
        if not text or not text.strip():
            return np.zeros(FINBERT_DIM, dtype=np.float32)

        cached = self.filing_cache.get(text)
        if cached is not None and cached.shape[0] == FINBERT_DIM:
            return cached.copy()

        all_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not all_ids:
            return np.zeros(FINBERT_DIM, dtype=np.float32)

        cls_token = self.tokenizer.cls_token_id
        sep_token = self.tokenizer.sep_token_id
        step = FILING_CHUNK_TOKENS - 2
        chunks = []
        for i in range(0, len(all_ids), step):
            ch = all_ids[i:i + step]
            ch = [cls_token] + ch + [sep_token]
            chunks.append(ch)
            if len(chunks) >= FILING_MAX_CHUNKS:
                break

        embeddings = []
        for ch in chunks:
            ids = torch.tensor([ch], device=self.device)
            attn = torch.ones_like(ids)
            out = self.encoder(input_ids=ids, attention_mask=attn)
            embeddings.append(out.last_hidden_state[:, 0, :].squeeze(0).float().cpu().numpy())

        emb = np.mean(embeddings, axis=0).astype(np.float32)
        self.filing_cache.put(text, emb)
        return emb

    def flush_caches(self):
        self.news_cache.flush()
        self.filing_cache.flush()
