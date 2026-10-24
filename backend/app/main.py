import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from core.config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    OLLAMA_BASE_URL,
    RERANK_ENABLED,
)
from database.database import AsyncSessionLocal, Base, engine
from rag.embeddings import embed_query
from rag.ingestion import fail_interrupted_indexing
from rag.ollama_client import OllamaError, missing_models
from rag.reranker import is_reranker_installed
from rag.vector_store import reset_chunks_if_dimensions_changed
from routers import chat, github, rag, user

# Import so the ORM models register on Base.metadata before create_all() runs.
import models.repository  # noqa: F401,E402
import models.user  # noqa: F401,E402

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(user.router, prefix="/users", tags=["Users"])
app.include_router(github.router, prefix="/github", tags=["GitHub"])
app.include_router(rag.router, prefix="/rag", tags=["RAG"])

# Keeps a reference to the background model check so it isn't garbage-collected.
_background_tasks: set[asyncio.Task] = set()


async def _check_models() -> None:
    """Checks the model side of the setup and logs what's missing.

    Only ever logs: the API (sign-in, repository lists) keeps working while
    Ollama is down, and indexing or asking then fails with the same
    explanation in the UI.
    """

    if RERANK_ENABLED and not is_reranker_installed():
        logger.warning(
            "RERANK_ENABLED=true, but sentence-transformers isn't installed. "
            "Run `pip install -r requirements-rerank.txt` in backend/."
        )

    try:
        missing = await missing_models([EMBEDDING_MODEL_NAME, LLM_MODEL_NAME])
    except OllamaError as exc:
        logger.warning("%s Indexing and answering questions won't work until it is.", exc)
        return

    for name in missing:
        logger.warning('Ollama model "%s" is not downloaded yet. Run: ollama pull %s', name, name)

    if EMBEDDING_MODEL_NAME in missing:
        return

    try:
        # Also loads the embedding model into memory ahead of the first request.
        await embed_query("startup check")
    except OllamaError as exc:  # e.g. EMBEDDING_DIMENSIONS doesn't match the model
        logger.warning("%s", exc)
        return

    if not missing:
        logger.info(
            "Ollama is ready at %s (embeddings: %s, LLM: %s).",
            OLLAMA_BASE_URL,
            EMBEDDING_MODEL_NAME,
            LLM_MODEL_NAME,
        )


@app.on_event("startup")
async def startup() -> None:
    async with engine.begin() as connection:
        # Required once per database before any `vector` column can be created.
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # If the embedding model was switched to one with a different output
        # size, the stored chunks can't be used anymore: drop them here, so
        # create_all() below recreates the table with the new vector size.
        old_dimensions = await reset_chunks_if_dimensions_changed(connection, EMBEDDING_DIMENSIONS)
        if old_dimensions is not None:
            logger.warning(
                "EMBEDDING_DIMENSIONS changed from %d to %d: dropped all indexed chunks. "
                "Every repository is back to 'Not indexed' and has to be indexed again.",
                old_dimensions,
                EMBEDDING_DIMENSIONS,
            )

        await connection.run_sync(Base.metadata.create_all)

    # Best-effort ANN index for cosine similarity search, in its own
    # transaction: not fatal if the installed pgvector version doesn't
    # support HNSW yet (older than 0.5) - similarity search still works via
    # a plain (slower) sequential scan, which is perfectly fine at the repo
    # sizes this test setup targets. Kept out of the transaction above so a
    # failure here can never roll back the extension/table creation.
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
                    "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("HNSW index nem hozható létre, sequential scan lesz használva: %s", exc)

    # Indexing runs as an in-process background task, so any repository
    # still marked "indexing" at this point belongs to a job that died with
    # the previous process (a crash, Ctrl+C, or a --reload restart). Nothing
    # would ever finish it, so mark it as failed - the UI then shows a Retry
    # button instead of polling forever.
    # Assumes a single backend process (one uvicorn worker), which is how
    # this project runs: with several workers, one worker starting up would
    # also fail jobs that are still running in the others.
    async with AsyncSessionLocal() as db:
        interrupted = await fail_interrupted_indexing(db)
    if interrupted:
        logger.warning("Marked %d interrupted indexing job(s) as failed.", interrupted)

    # In the background, so a slow or missing Ollama never delays startup.
    task = asyncio.create_task(_check_models())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


@app.get("/")
async def root():
    return {"message": "RAG Code Assistant backend is running"}
