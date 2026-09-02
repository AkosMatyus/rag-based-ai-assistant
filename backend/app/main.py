import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from database.database import Base, engine
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


@app.on_event("startup")
async def startup() -> None:
    async with engine.begin() as connection:
        # Required once per database before any `vector` column can be created.
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

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


@app.get("/")
async def root():
    return {"message": "RAG Code Assistant backend is running"}

