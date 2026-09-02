"""Splits file contents into overlapping chunks for embedding.

Uses ``langchain-text-splitters``. When the file extension maps to a known
programming language, ``RecursiveCharacterTextSplitter.from_language`` is
used so chunks are split on language-aware boundaries (function/class
definitions etc.) instead of arbitrary character counts. Everything else
(plain text, markdown, config files, unknown extensions) falls back to a
generic recursive splitter.
"""

from dataclasses import dataclass

from core.config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

EXTENSION_LANGUAGE_MAP: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".pyi": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".mjs": Language.JS,
    ".cjs": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".c": Language.CPP,
    ".h": Language.CPP,
    ".cpp": Language.CPP,
    ".hpp": Language.CPP,
    ".cc": Language.CPP,
    ".cs": Language.CSHARP,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".md": Language.MARKDOWN,
    ".mdx": Language.MARKDOWN,
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".proto": Language.PROTO,
}


@dataclass
class TextChunk:
    content: str
    chunk_index: int
    start_line: int
    end_line: int


def _extension_of(file_path: str) -> str:
    filename = file_path.rsplit("/", 1)[-1]
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def _splitter_for(file_path: str) -> RecursiveCharacterTextSplitter:
    language = EXTENSION_LANGUAGE_MAP.get(_extension_of(file_path))

    if language is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


def split_file(file_path: str, content: str) -> list[TextChunk]:
    """Splits one file's content into TextChunks with best-effort line numbers."""

    splitter = _splitter_for(file_path)
    raw_chunks = splitter.split_text(content)

    chunks: list[TextChunk] = []
    search_cursor = 0

    for index, chunk_text in enumerate(raw_chunks):
        start_pos = content.find(chunk_text, search_cursor)
        if start_pos == -1:
            # Overlap can make chunks appear "before" the last search cursor;
            # fall back to a full-content search rather than losing the chunk.
            start_pos = content.find(chunk_text)

        if start_pos == -1:
            start_line = 1
            end_line = 1 + chunk_text.count("\n")
        else:
            start_line = content.count("\n", 0, start_pos) + 1
            end_line = start_line + chunk_text.count("\n")
            # Advance past most of this chunk so the next `find` stays fast
            # and roughly in order, while still tolerating overlap.
            search_cursor = start_pos + max(len(chunk_text) - RAG_CHUNK_OVERLAP, 1)

        chunks.append(
            TextChunk(
                content=chunk_text,
                chunk_index=index,
                start_line=start_line,
                end_line=end_line,
            )
        )

    return chunks
