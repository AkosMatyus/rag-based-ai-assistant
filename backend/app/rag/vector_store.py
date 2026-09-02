"""Storage and retrieval of chunk embeddings in Postgres via pgvector.

All queries go through SQLAlchemy's async ORM (asyncpg driver), so this
module never blocks the event loop - the actual similarity search is
executed by Postgres itself using the pgvector ``<=>`` cosine-distance
operator.
"""

from dataclasses import dataclass
from typing import TypedDict

from models.repository import DocumentChunk
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession


class ChunkRecord(TypedDict):
    file_path: str
    chunk_index: int
    content: str
    start_line: int | None
    end_line: int | None
    embedding: list[float]


@dataclass
class RetrievedChunk:
    file_path: str
    content: str
    start_line: int | None
    end_line: int | None
    chunk_index: int
    distance: float  # cosine distance
    rerank_score: float | None = None


async def replace_repository_chunks(
    db: AsyncSession,
    repository_id: int,
    chunks: list[ChunkRecord],
) -> None:
    """Atomically swaps out all stored chunks for a repository (re-indexing)."""

    await db.execute(delete(DocumentChunk).where(DocumentChunk.repository_id == repository_id))

    db.add_all(
        [
            DocumentChunk(
                repository_id=repository_id,
                file_path=chunk["file_path"],
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                start_line=chunk["start_line"],
                end_line=chunk["end_line"],
                embedding=chunk["embedding"],
            )
            for chunk in chunks
        ]
    )

    await db.commit()


async def similarity_search(
    db: AsyncSession,
    repository_id: int,
    query_embedding: list[float],
    top_k: int,
) -> list[RetrievedChunk]:
    """Returns the ``top_k`` chunks of ``repository_id`` closest to the query."""

    distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding)

    stmt = (
        select(DocumentChunk, distance_expr.label("distance"))
        .where(DocumentChunk.repository_id == repository_id)
        .order_by(distance_expr)
        .limit(top_k)
    )

    result = await db.execute(stmt)

    return [
        RetrievedChunk(
            file_path=chunk.file_path,
            content=chunk.content,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_index=chunk.chunk_index,
            distance=float(distance),
        )
        for chunk, distance in result.all()
    ]
