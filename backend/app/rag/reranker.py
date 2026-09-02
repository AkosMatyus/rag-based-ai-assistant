"""Cross-encoder-based reranking.
The search in vector_store.py (a "bi-encoder") is fast, but somewhat
inaccurate: the query and each piece of code are converted into vectors
INDEPENDENTLY, and the two vectors are then compared afterwards. In
contrast, the cross-encoder receives the query and a candidate TOGETHER
and directly produces a relevance score for them - much more accurate,
but also much slower, which is why we only run it on the candidate list
already narrowed down by the bi-encoder, rather than on all stored chunks.
"""


import asyncio
import math
import threading

from core.config import RERANKER_MODEL_NAME
from rag.vector_store import RetrievedChunk
from sentence_transformers import CrossEncoder

_model: CrossEncoder | None = None
_model_lock = threading.Lock()


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
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
    return await asyncio.to_thread(_rerank, question, chunks, top_k)