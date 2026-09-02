"""Downloads a GitHub repository and extracts its text/code files.

Uses the repo tarball endpoint (one HTTP request) instead of walking the
Git Trees/Contents API file by file - much friendlier to GitHub's rate
limits for repos with many files. The download itself is async (httpx);
the actual tar extraction is sync/CPU-bound so it is offloaded to a
worker thread via ``asyncio.to_thread``.
"""

import asyncio
import io
import tarfile
from dataclasses import dataclass

import httpx
from core.config import RAG_MAX_FILE_SIZE_BYTES, RAG_MAX_FILES_PER_REPO

# Extensions treated as indexable text/code.
TEXT_EXTENSIONS = {
    ".py", ".pyi", ".ipynb",
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".java", ".kt", ".kts", ".scala",
    ".go", ".rs", ".rb", ".php", ".pl",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cs",
    ".swift", ".m",
    ".sh", ".bash", ".zsh", ".ps1",
    ".sql",
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
    ".md", ".mdx", ".rst", ".txt",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env.example",
    ".graphql", ".proto", ".xml",
}

# Bare filenames (no / lowercase extension) that should still be indexed.
TEXT_FILENAMES = {"dockerfile", "makefile", "readme", "license", "procfile"}

# Any path segment matching one of these means "skip this file".
SKIP_DIR_PARTS = {
    "node_modules", ".git", "dist", "build", "vendor", "venv", ".venv",
    "env", "__pycache__", ".next", ".nuxt", "target", "bin", "obj",
    ".idea", ".vscode", "coverage", "out", ".pytest_cache", ".mypy_cache",
    "egg-info", ".tox",
}

SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "cargo.lock", "composer.lock", "gemfile.lock",
}


@dataclass
class RepoFile:
    path: str
    content: str


def _headers(token: str | None) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def check_repo_access(owner: str, repo: str, token: str | None) -> bool:
    """Megmondja, hogy a megadott GitHub-tokennel (vagy token nélkül, publikus
    repóknál) van-e legalább olvasási jog owner/repo-hoz. A GitHub maga dönti
    el ezt (200 = van hozzáférés, 404 = nincs, vagy nem is létezik a repó) -
    nekünk nem kell tudnunk, hogy a felhasználó tulajdonos, collaborator vagy
    szervezeti tag-e, csak a végeredményt kérdezzük meg."""
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers=_headers(token),
        )
    return response.status_code == 200


async def fetch_default_branch(owner: str, repo: str, token: str | None) -> str:
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers=_headers(token),
        )
        response.raise_for_status()
        data = response.json()
        return data.get("default_branch") or "main"


async def download_repo_files(
    owner: str,
    repo: str,
    ref: str,
    token: str | None,
) -> list[RepoFile]:
    """Downloads the repo tarball for ``ref`` and returns its indexable files."""

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        response = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/tarball/{ref}",
            headers=_headers(token),
        )
        response.raise_for_status()
        raw_bytes = response.content

    return await asyncio.to_thread(_extract_text_files, raw_bytes)


def _is_indexable(relative_path: str) -> bool:
    path_parts = set(relative_path.split("/"))
    if path_parts & SKIP_DIR_PARTS:
        return False

    filename = relative_path.rsplit("/", 1)[-1]
    if filename.lower() in SKIP_FILENAMES:
        return False

    if "." in filename:
        extension = "." + filename.rsplit(".", 1)[-1].lower()
    else:
        extension = ""

    if extension in TEXT_EXTENSIONS:
        return True

    return filename.lower() in TEXT_FILENAMES


def _extract_text_files(raw_bytes: bytes) -> list[RepoFile]:
    files: list[RepoFile] = []

    with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            if len(files) >= RAG_MAX_FILES_PER_REPO:
                break

            if not member.isfile():
                continue

            if member.size <= 0 or member.size > RAG_MAX_FILE_SIZE_BYTES:
                continue

            # Tarball members look like "<owner>-<repo>-<sha>/path/to/file".
            relative_path = member.name.split("/", 1)[1] if "/" in member.name else member.name

            if not relative_path or not _is_indexable(relative_path):
                continue

            extracted = archive.extractfile(member)
            if extracted is None:
                continue

            raw = extracted.read()

            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue  # binary file, skip

            if not text.strip():
                continue

            files.append(RepoFile(path=relative_path, content=text))

    return files
