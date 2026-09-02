"""Local embedding generation via SentenceTransformers.

Model loading and ``.encode()`` calls are synchronous/CPU-bound, so both
are run in a worker thread through ``asyncio.to_thread`` to keep the
FastAPI event loop free for other requests. The model itself is loaded
once (lazily, on first use) and cached as a module-level singleton.
"""

import asyncio
import threading

from core.config import EMBEDDING_MODEL_NAME
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _encode(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of texts (e.g. document chunks)."""
    if not texts:
        return []
    return await asyncio.to_thread(_encode, texts)


async def embed_query(text: str) -> list[float]:
    """Embeds a single query string."""
    vectors = await embed_texts([text])
    return vectors[0]
