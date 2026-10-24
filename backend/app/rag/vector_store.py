"""Storage and retrieval of chunk embeddings in Postgres via pgvector.

All queries go through SQLAlchemy's async ORM (asyncpg driver), so this
module never blocks the event loop - the actual similarity search is
executed by Postgres itself using the pgvector ``<=>`` cosine-distance
operator.
"""

from dataclasses import dataclass
from typing import TypedDict

from models.repository import DocumentChunk, Repository, RepositoryStatus
from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession


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


async def stored_embedding_dimensions(connection: AsyncConnection) -> int | None:
    """Dimension of the existing ``document_chunks.embedding`` column, or None
    if the table doesn't exist yet. (For pgvector's ``vector(n)`` type the
    column's type modifier, ``atttypmod``, is ``n`` itself.)"""

    result = await connection.execute(
        text(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = to_regclass('document_chunks') "
            "AND attname = 'embedding' AND NOT attisdropped"
        )
    )
    dimensions = result.scalar_one_or_none()
    return dimensions if dimensions and dimensions > 0 else None


async def reset_chunks_if_dimensions_changed(
    connection: AsyncConnection,
    expected_dimensions: int,
) -> int | None:
    """Drops ``document_chunks`` when its vector size no longer matches
    EMBEDDING_DIMENSIONS, i.e. the embedding model was switched to one with
    a different output size.

    The stored vectors can't be compared with the new model's vectors at all,
    and every insert or search would fail, so they are useless either way -
    and they are derived data that re-indexing recreates. Every repository is
    set back to "not indexed" so the UI offers to index it again. The caller
    recreates the table (create_all) afterwards in the same transaction.

    Returns the old dimension when a reset happened, otherwise None.
    """

    stored = await stored_embedding_dimensions(connection)
    if stored is None or stored == expected_dimensions:
        return None

    await connection.execute(text("DROP TABLE document_chunks"))
    await connection.execute(
        update(Repository)
        .where(Repository.status != RepositoryStatus.NOT_INDEXED)
        .values(
            status=RepositoryStatus.NOT_INDEXED,
            file_count=0,
            chunk_count=0,
            indexed_at=None,
            error_message=None,
        )
    )
    return stored
