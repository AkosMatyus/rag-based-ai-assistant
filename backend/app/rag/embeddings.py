"""Embedding generation through Ollama's ``/api/embed`` endpoint.

The embedding model (EMBEDDING_MODEL_NAME, e.g. ``qwen3-embedding:0.6b``)
runs inside Ollama, which returns L2-normalized vectors - so cosine
similarity in pgvector works on them directly. Chunks are sent in batches
of EMBEDDING_BATCH_SIZE per request while indexing.

Questions and indexed chunks get different prefixes
(EMBEDDING_QUERY_PREFIX / EMBEDDING_DOCUMENT_PREFIX): instruction-tuned
embedding models such as Qwen3-Embedding expect a task instruction in front
of the query only.
"""

from core.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_DOCUMENT_PREFIX,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_QUERY_PREFIX,
    OLLAMA_KEEP_ALIVE,
)
from rag.ollama_client import OllamaError, ollama_request


async def _embed(inputs: list[str]) -> list[list[float]]:
    data = await ollama_request(
        "POST",
        "/api/embed",
        {
            "model": EMBEDDING_MODEL_NAME,
            "input": inputs,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        },
    )

    vectors = data.get("embeddings") or []
    if len(vectors) != len(inputs):
        raise OllamaError(
            f"Ollama returned {len(vectors)} embeddings for {len(inputs)} texts."
        )

    # Checked here so a wrong setting produces a clear message instead of a
    # pgvector "expected N dimensions" error at insert time.
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise OllamaError(
                f'The embedding model "{EMBEDDING_MODEL_NAME}" returns '
                f"{len(vector)}-dimensional vectors, but EMBEDDING_DIMENSIONS is "
                f"{EMBEDDING_DIMENSIONS}. Set EMBEDDING_DIMENSIONS={len(vector)} in "
                "backend/.env and restart the backend."
            )

    return vectors


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds document chunks (used while indexing)."""

    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        vectors.extend(await _embed([EMBEDDING_DOCUMENT_PREFIX + text for text in batch]))
    return vectors


async def embed_query(text: str) -> list[float]:
    """Embeds a single question (used when answering)."""

    vectors = await _embed([EMBEDDING_QUERY_PREFIX + text])
    return vectors[0]
