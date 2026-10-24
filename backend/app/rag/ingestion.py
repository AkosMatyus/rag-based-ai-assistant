"""Orchestrates the end-to-end ingestion pipeline for one repository:

    GitHub tarball -> filter files -> split into chunks -> embed (Ollama) -> pgvector

Runs as a FastAPI background task, so it owns its own DB session (the
request-scoped session used by the endpoint that triggered it is already
closed by the time this executes).
"""

import logging
from datetime import datetime, timedelta, timezone

from database.database import AsyncSessionLocal
from models.repository import Repository, RepositoryStatus, RepositoryAccess
from rag.embeddings import embed_texts
from rag.github_loader import download_repo_files, fetch_default_branch
from rag.text_splitter import split_file
from rag.vector_store import ChunkRecord, replace_repository_chunks
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def get_or_create_repository(db: AsyncSession, owner: str, name: str) -> Repository:
    full_name = f"{owner}/{name}"

    result = await db.execute(select(Repository).where(Repository.full_name == full_name))
    repository = result.scalar_one_or_none()

    if repository is None:
        repository = Repository(full_name=full_name, owner=owner, name=name)
        db.add(repository)
        await db.commit()
        await db.refresh(repository)

    return repository


async def grant_repository_access(db: AsyncSession, user_id: int, repository_id: int) -> None:
    """Feljegyzi (vagy frissíti), hogy a felhasználó hozzáfér a repóhoz - ezt
    mindig egy sikeres, élő GitHub jogosultság-ellenőrzés (check_repo_access)
    után hívjuk, sosem magától."""

    result = await db.execute(
        select(RepositoryAccess).where(
            RepositoryAccess.user_id == user_id,
            RepositoryAccess.repository_id == repository_id,
        )
    )
    access = result.scalar_one_or_none()

    if access is None:
        db.add(RepositoryAccess(user_id=user_id, repository_id=repository_id))
    else:
        access.last_verified_at = datetime.now(timezone.utc)

    await db.commit()


async def fail_interrupted_indexing(db: AsyncSession, older_than_minutes: int | None = None) -> int:
    """Marks repositories stuck in "indexing" as failed, so they can be started again.

    Ingestion runs as an in-process background task, so if the backend stops
    mid-run (crash, Ctrl+C, a --reload restart) nothing is left to move the
    repository out of "indexing" - and /rag/ingest refuses to start a repo
    that is already indexing.

    - Without ``older_than_minutes`` (called once at startup): every
      "indexing" repository is failed, since no job can be running yet in a
      process that has just started.
    - With ``older_than_minutes`` (called from the status endpoints): only
      jobs that started longer ago than that are failed - a safety net for a
      job that hangs while the server keeps running.

    Returns the number of repositories that were marked as failed.
    """

    stmt = update(Repository).where(Repository.status == RepositoryStatus.INDEXING)

    if older_than_minutes is None:
        message = "Indexing was interrupted because the server restarted. Start it again."
    else:
        # updated_at is last written when the job sets the status to
        # "indexing", so it is effectively the job's start time.
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=older_than_minutes)
        stmt = stmt.where(Repository.updated_at < cutoff)
        message = (
            f"Indexing didn't finish within {older_than_minutes} minutes, "
            "so it was marked as failed. Start it again."
        )

    result = await db.execute(
        stmt.values(status=RepositoryStatus.ERROR, error_message=message)
        # Callers run this before loading any Repository into the session,
        # so there are no in-memory objects that would need syncing.
        .execution_options(synchronize_session=False)
    )
    await db.commit()

    return result.rowcount or 0


async def run_ingestion(repository_id: int, github_token: str | None) -> None:
    async with AsyncSessionLocal() as db:
        repository = await db.get(Repository, repository_id)
        if repository is None:
            logger.warning("Ingestion requested for unknown repository_id=%s", repository_id)
            return

        repository.status = RepositoryStatus.INDEXING
        repository.error_message = None
        await db.commit()

        try:
            branch = await fetch_default_branch(repository.owner, repository.name, github_token)

            files = await download_repo_files(repository.owner, repository.name, branch, github_token)

            if not files:
                raise ValueError("No indexable text or source files were found in the repository.")

            chunk_records: list[ChunkRecord] = []
            all_texts: list[str] = []

            for file in files:
                for chunk in split_file(file.path, file.content):
                    chunk_records.append(
                        ChunkRecord(
                            file_path=file.path,
                            chunk_index=chunk.chunk_index,
                            content=chunk.content,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            embedding=[],  # filled in below
                        )
                    )
                    all_texts.append(chunk.content)

            embeddings = await embed_texts(all_texts)
            for record, embedding in zip(chunk_records, embeddings):
                record["embedding"] = embedding

            await replace_repository_chunks(db, repository.id, chunk_records)

            repository.status = RepositoryStatus.READY
            repository.default_branch = branch
            repository.file_count = len(files)
            repository.chunk_count = len(chunk_records)
            repository.indexed_at = datetime.now(timezone.utc)
            await db.commit()

            logger.info(
                "Indexed %s: %d files, %d chunks",
                repository.full_name,
                repository.file_count,
                repository.chunk_count,
            )

        except Exception as exc:  # noqa: BLE001 - surface any failure as repo status
            logger.exception("Ingestion failed for repository_id=%s", repository_id)
            repository.status = RepositoryStatus.ERROR
            repository.error_message = str(exc)[:2000]
            await db.commit()
