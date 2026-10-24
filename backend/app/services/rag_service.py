"""High-level orchestration for answering a question about an indexed repo.

Ties together the individual RAG building blocks that live under
``app/rag/``: embeddings (question -> vector, via Ollama), vector_store
(retrieval), the optional reranker, and llm (answer generation from the
retrieved context, via Ollama).
"""

from core.config import (
    DOC_FILE_PENALTY,
    GITHUB_API_TOKEN,
    RAG_TOP_K,
    RERANK_CANDIDATES, 
    RERANK_ENABLED,
    REPOSITORY_ACCESS_TTL_HOURS,
    CODE_FILE_BOOST,
)
from fastapi import HTTPException
from models.chat import ChatResponse, SourceChunk, SourceGroup
from models.repository import Repository, RepositoryStatus, RepositoryAccess
from models.user import User
from rag.embeddings import embed_query
from rag.github_loader import check_repo_access
from rag.llm import generate_answer
from rag.ollama_client import OllamaError
from rag.vector_store import RetrievedChunk, similarity_search
from rag.reranker import RerankerUnavailableError, rerank
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from datetime import datetime, timedelta, timezone


def _group_by_file(
    chunks: list[RetrievedChunk],
) -> list[tuple[str, list[RetrievedChunk]]]:
    """Csoportosítja a chunkokat fájlonként, MEGTARTVA az összeset (nem csak
    a legjobbat fájlonként) - a csoportok és az egyes csoportokon belüli
    chunkok sorrendje is a bejövő (relevancia szerinti) sorrendet követi."""
    groups: dict[str, list[RetrievedChunk]] = {}
    order: list[str] = []
    for chunk in chunks:
        if chunk.file_path not in groups:
            groups[chunk.file_path] = []
            order.append(chunk.file_path)
        groups[chunk.file_path].append(chunk)
    return [(file_path, groups[file_path]) for file_path in order]

def _relevance_score(chunk: RetrievedChunk) -> float:
    if chunk.rerank_score is not None:
        return round(chunk.rerank_score, 4)
    return round(max(0.0, 1 - chunk.distance / 2), 4)


DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}

def _is_documentation_file(file_path: str) -> bool:
    filename = file_path.rsplit("/", 1)[-1].lower()
    if "." not in filename:
        return False
    extension = "." + filename.rsplit(".", 1)[-1]
    return extension in DOC_EXTENSIONS


def _apply_file_type_bias(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Slightly favor source code over documentation in the final ranking -
    the instruction provided in the prompt alone was not reliable enough
    with a smaller local model, so we enforce this here, at the retrieval
    level."""

    def biased_score(chunk: RetrievedChunk) -> float:
        base = _relevance_score(chunk)
        if _is_documentation_file(chunk.file_path):
            return base - DOC_FILE_PENALTY
        return base + CODE_FILE_BOOST

    return sorted(chunks, key=biased_score, reverse=True)


def _build_source_url(repo_full_name: str, branch: str | None, chunk: RetrievedChunk) -> str:
    ref = branch or "main"
    url = f"https://github.com/{repo_full_name}/blob/{ref}/{chunk.file_path}"
    if chunk.start_line:
        url += f"#L{chunk.start_line}"
        if chunk.end_line and chunk.end_line != chunk.start_line:
            url += f"-L{chunk.end_line}"
    return url


def _build_context(chunks: list[RetrievedChunk]) -> str:
    sections = []
    for chunk in chunks:
        location = chunk.file_path
        if chunk.start_line:
            location += f" (lines {chunk.start_line}-{chunk.end_line})"
        sections.append(f"### {location}\n{chunk.content}")
    return "\n\n".join(sections)


async def answer_question(
    db: AsyncSession,
    question: str,
    repo_full_name: str | None,
    current_user: User,
) -> ChatResponse:
    if not repo_full_name:
        raise HTTPException(status_code=400, detail="Select a repository to ask about.")

    result = await db.execute(select(Repository).where(Repository.full_name == repo_full_name))
    repository = result.scalar_one_or_none()

    if repository is None:
        raise HTTPException(
            status_code=404,
            detail="This repository doesn't exist or you don't have access to it.",
        )

    access_result = await db.execute(
        select(RepositoryAccess).where(
            RepositoryAccess.user_id == current_user.id,
            RepositoryAccess.repository_id == repository.id,
        )
    )
    access = access_result.scalar_one_or_none()

    if access is None:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this repository in the app yet. Add it to your list first.",
        )

    access_age = datetime.now(timezone.utc) - access.last_verified_at
    if access_age > timedelta(hours=REPOSITORY_ACCESS_TTL_HOURS):
        github_token = current_user.github_access_token or GITHUB_API_TOKEN or None
        still_has_access = await check_repo_access(repository.owner, repository.name, github_token)

        if not still_has_access:
            await db.delete(access)
            await db.commit()
            raise HTTPException(
                status_code=403,
                detail="Your GitHub access to this repository has been revoked.",
            )

        access.last_verified_at = datetime.now(timezone.utc)
        await db.commit()

    if repository.status != RepositoryStatus.READY:
        raise HTTPException(
            status_code=409,
            detail=(
                "This repository isn't fully indexed yet. Start indexing it "
                "and wait until it finishes."
            ),
        )
    
    try:
        query_embedding = await embed_query(question)

        candidate_count = RERANK_CANDIDATES if RERANK_ENABLED else RAG_TOP_K
        retrieved = await similarity_search(db, repository.id, query_embedding, top_k=candidate_count)

        if RERANK_ENABLED and retrieved:
            retrieved = await rerank(question, retrieved, top_k=RAG_TOP_K)
    except (OllamaError, RerankerUnavailableError) as exc:
        # A model problem (Ollama not running, model not downloaded, ...):
        # 503 with the explanation, which the frontend shows in the chat.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    retrieved = _apply_file_type_bias(retrieved)[:RAG_TOP_K]

    if not retrieved:
        return ChatResponse(
            answer="No relevant code was found in this repository for that question.",
            sources=[],
        )

    print("---------------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------------")
    for index, chunk in enumerate(retrieved):
        print(f"{index}: {chunk.file_path}")
    print("---------------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------------")

    context = _build_context(retrieved)
    try:
        answer_text = await generate_answer(question, context)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    sources = [
        SourceGroup(
            file_path=file_path,
            chunks=[
                SourceChunk(
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=_relevance_score(chunk),
                    url=_build_source_url(repo_full_name, repository.default_branch, chunk),
                    content=chunk.content,
                )
                for chunk in file_chunks
            ],
        )
        for file_path, file_chunks in _group_by_file(retrieved)
    ]

    return ChatResponse(answer=answer_text, sources=sources)
