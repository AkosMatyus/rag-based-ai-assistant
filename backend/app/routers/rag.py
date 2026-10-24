from core.config import GITHUB_API_TOKEN, INDEXING_TIMEOUT_MINUTES
from database.database import get_db
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from models.repository import (
    IngestRequest,
    Repository,
    RepositoryAccess,
    RepositoryStatus,
    RepositoryStatusResponse,
)
from models.user import User
from rag.github_loader import check_repo_access
from rag.ingestion import (
    fail_interrupted_indexing,
    get_or_create_repository,
    grant_repository_access,
    run_ingestion,
)
from routers.user import get_current_user
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


def _split_full_name(repo_full_name: str) -> tuple[str, str]:
    parts = repo_full_name.strip().split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise HTTPException(
            status_code=400,
            detail="Use the owner/repository format, for example psf/requests.",
        )
    return parts[0], parts[1]


def _to_status_response(repository: Repository) -> RepositoryStatusResponse:
    return RepositoryStatusResponse(
        repo_full_name=repository.full_name,
        status=repository.status,
        file_count=repository.file_count,
        chunk_count=repository.chunk_count,
        indexed_at=repository.indexed_at.isoformat() if repository.indexed_at else None,
        error_message=repository.error_message,
    )


def _resolve_github_token(current_user: User) -> str | None:
    return current_user.github_access_token or GITHUB_API_TOKEN or None


async def _fail_timed_out_indexing(db: AsyncSession) -> None:
    """Fails indexing jobs that have been running for too long. Runs before
    every status read (the frontend polls these while a repo is indexing),
    so a hung job turns into "error" with a Retry button on its own."""

    await fail_interrupted_indexing(db, older_than_minutes=INDEXING_TIMEOUT_MINUTES)


@router.post("/ingest", response_model=RepositoryStatusResponse)
async def ingest_repository(
    payload: IngestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Ellenőrzi a hozzáférést, majd elindítja a (újra-)indexelést a háttérben."""

    owner, name = _split_full_name(payload.repo_full_name)
    github_token = _resolve_github_token(current_user)

    has_access = await check_repo_access(owner, name, github_token)
    if not has_access:
        raise HTTPException(
            status_code=404,
            detail="Repository not found, or you don't have access to it.",
        )

    # Must run before the repository is loaded below, so a timed-out job no
    # longer counts as "already indexing" and can be started again.
    await _fail_timed_out_indexing(db)

    repository = await get_or_create_repository(db, owner, name)
    await grant_repository_access(db, current_user.id, repository.id)

    # A job for this repo is genuinely still running - don't start a second one.
    if repository.status == RepositoryStatus.INDEXING:
        return _to_status_response(repository)

    repository.status = RepositoryStatus.INDEXING
    repository.error_message = None
    repository.requested_by_user_id = current_user.id
    await db.commit()
    await db.refresh(repository)

    background_tasks.add_task(run_ingestion, repository.id, github_token)

    return _to_status_response(repository)


@router.get("/status", response_model=RepositoryStatusResponse)
async def get_repository_status(
    repo_full_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Used by the frontend to poll indexing progress for the selected repo."""

    await _fail_timed_out_indexing(db)

    result = await db.execute(select(Repository).where(Repository.full_name == repo_full_name))
    repository = result.scalar_one_or_none()

    if repository is None:
        return RepositoryStatusResponse(
            repo_full_name=repo_full_name,
            status=RepositoryStatus.NOT_INDEXED,
            file_count=0,
            chunk_count=0,
        )

    return _to_status_response(repository)


@router.post("/repos", response_model=RepositoryStatusResponse)
async def add_repository(
    payload: IngestRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Hozzáadja a repót a felhasználó saját listájához - ellenőrzi élőben a
    GitHub felé, hogy van-e hozzáférése, de NEM indítja el az indexelést."""

    owner, name = _split_full_name(payload.repo_full_name)
    github_token = _resolve_github_token(current_user)

    has_access = await check_repo_access(owner, name, github_token)
    if not has_access:
        raise HTTPException(
            status_code=404,
            detail="Repository not found, or you don't have access to it.",
        )

    repository = await get_or_create_repository(db, owner, name)
    await grant_repository_access(db, current_user.id, repository.id)

    return _to_status_response(repository)


@router.get("/repos/mine", response_model=list[RepositoryStatusResponse])
async def list_my_repositories(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """A felhasználó által hozzáadott / lekérdezett repók listája."""

    await _fail_timed_out_indexing(db)

    result = await db.execute(
        select(Repository)
        .join(RepositoryAccess, RepositoryAccess.repository_id == Repository.id)
        .where(RepositoryAccess.user_id == current_user.id)
        .order_by(Repository.full_name)
    )
    repositories = result.scalars().all()

    return [_to_status_response(repository) for repository in repositories]