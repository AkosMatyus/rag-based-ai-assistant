import os

from dotenv import load_dotenv

load_dotenv()

# --- Database ---------------------------------------------------------------
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# --- Auth ---------------------------------------------------------------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_secret_key")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30)
ACCESS_TOKEN_EXPIRE_MINUTES = int(ACCESS_TOKEN_EXPIRE_MINUTES)

# --- GitHub OAuth / API ---------------------------------------------------------------
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://127.0.0.1:8000/github/callback")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Optional fallback personal access token used for repo ingestion when the
# current user has no github_access_token (e.g. locally registered users).
# Without any token, only public repos can be ingested (subject to GitHub's
# lower anonymous rate limit).
GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN", "")

# --- Repository access control ---
# Ennyi óráig számít "frissnek" egy hozzáférés-ellenőrzés, mielőtt a
# rendszer újra megkérdezné élőben a GitHubot, hogy a felhasználónak még
# mindig van-e hozzáférése az adott repóhoz.
REPOSITORY_ACCESS_TTL_HOURS = int(os.getenv("REPOSITORY_ACCESS_TTL_HOURS", "24"))

# --- RAG: text splitting ---------------------------------------------------------------
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# --- RAG: GitHub ingestion limits ---------------------------------------------------------------
# Skip any single file bigger than this (bytes) - avoids indexing generated/binary-ish files.
RAG_MAX_FILE_SIZE_BYTES = int(os.getenv("RAG_MAX_FILE_SIZE_BYTES", str(300 * 1024)))
# Hard cap on number of files read from a single repo per ingestion run.
RAG_MAX_FILES_PER_REPO = int(os.getenv("RAG_MAX_FILES_PER_REPO", "800"))
# Safety net for a hung indexing job: a repository that has been "indexing"
# for longer than this is marked as failed, so the user can start it again.
# (Jobs killed by a server restart are failed immediately at startup instead.)
INDEXING_TIMEOUT_MINUTES = int(os.getenv("INDEXING_TIMEOUT_MINUTES", "60"))

# --- Ollama (runs the embedding model and the LLM) ---------------------------------------------------------------
# Both models run in Ollama, a separate local program the backend talks to
# over HTTP - see README "Setup - 2. Ollama". 127.0.0.1 rather than
# "localhost": on Windows, "localhost" may be tried as IPv6 (::1) first,
# which Ollama doesn't listen on, and every request would wait for that
# attempt to fail.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
# Upper limit for one Ollama call. Generous on purpose: the first call after
# a model was unloaded also includes loading it into (GPU) memory.
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
# How long Ollama keeps a model loaded after its last use (Ollama's own
# default is 5 minutes, after which the next question waits for a reload).
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# --- RAG: embeddings ---------------------------------------------------------------
# Ollama model name. qwen3-embedding:0.6b: multilingual (100+ languages)
# and trained for code retrieval, 32k-token context, 1024-dim output.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
# Must match the embedding model's output size; it is also the size of the
# document_chunks.embedding column (see main.py for what happens when it changes).
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
# How many chunks are sent to Ollama in one /api/embed request while indexing.
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
# Text put in front of every question / every indexed chunk before embedding.
# Qwen3-Embedding expects a one-sentence task instruction on queries (in
# English, in exactly this "Instruct: ...\nQuery:" shape) and nothing on
# documents. Other models need other prefixes (e.g. nomic-embed-text uses
# "search_query: " / "search_document: "), or none at all.
EMBEDDING_QUERY_PREFIX = os.getenv(
    "EMBEDDING_QUERY_PREFIX",
    "Instruct: Given a question about a software repository, retrieve the "
    "source code or documentation that answers it\nQuery:",
)
EMBEDDING_DOCUMENT_PREFIX = os.getenv("EMBEDDING_DOCUMENT_PREFIX", "")

# --- RAG: retrieval ---------------------------------------------------------------
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

# --- RAG: file type bias (code > documentation) ---
# The value added to/subtracted from the relevance score when determining
# the final ranking - source code should receive a slight advantage over documentation.
CODE_FILE_BOOST = float(os.getenv("CODE_FILE_BOOST", "0.05"))
DOC_FILE_PENALTY = float(os.getenv("DOC_FILE_PENALTY", "0.08"))


# --- RAG: LLM used for answer generation (runs in Ollama) ---------------------------------------------------------------
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3.5:4b")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
# Context window requested from Ollama for each answer (prompt + answer, in
# tokens). Set explicitly because Ollama's own default is only 4096 tokens
# on GPUs with less than 24 GB of VRAM, and a longer prompt would be cut.
LLM_CONTEXT_LENGTH = int(os.getenv("LLM_CONTEXT_LENGTH", "8192"))
