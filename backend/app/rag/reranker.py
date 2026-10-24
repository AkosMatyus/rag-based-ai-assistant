"""Cross-encoder-based reranking (optional, off by default).

The search in vector_store.py (a "bi-encoder") is fast, but somewhat
inaccurate: the query and each piece of code are converted into vectors
INDEPENDENTLY, and the two vectors are then compared afterwards. In
contrast, the cross-encoder receives the query and a candidate TOGETHER
and directly produces a relevance score for them - much more accurate,
but also much slower, which is why we only run it on the candidate list
already narrowed down by the bi-encoder, rather than on all stored chunks.

Unlike the embedding model and the LLM, this one does NOT run in Ollama:
Ollama has no reranking API. It runs inside the backend process via
sentence-transformers, which (together with PyTorch) is an optional
dependency - install it with ``pip install -r requirements-rerank.txt``
before setting RERANK_ENABLED=true. The import is deferred to first use, so
the backend starts and works without it while reranking is disabled.
"""


import asyncio
import math
import threading

from core.config import RERANKER_MODEL_NAME
from rag.vector_store import RetrievedChunk

_model = None
_model_lock = threading.Lock()


class RerankerUnavailableError(RuntimeError):
    """RERANK_ENABLED=true, but sentence-transformers isn't installed."""


def is_reranker_installed() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:
                    raise RerankerUnavailableError(
                        "Reranking is enabled (RERANK_ENABLED=true), but "
                        "sentence-transformers isn't installed. Run "
                        "`pip install -r requirements-rerank.txt` in backend/, "
                        "or set RERANK_ENABLED=false."
                    ) from exc
                _model = CrossEncoder(RERANKER_MODEL_NAME)
    return _model


def _rerank(
    question: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    model = _get_model()
    pairs = [(question, chunk.content) for chunk in chunks]
    raw_scores = model.predict(pairs)

    for chunk, raw_score in zip(chunks, raw_scores):
        chunk.rerank_score = 1 / (1 + math.exp(-float(raw_score)))

    chunks.sort(key=lambda chunk: chunk.rerank_score, reverse=True)
    return chunks[:top_k]


async def rerank(
    question: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    if not chunks:
        return []
    # Model loading and scoring are synchronous and CPU/GPU-bound, so they
    # run in a worker thread to keep the event loop free.
    return await asyncio.to_thread(_rerank, question, chunks, top_k)
