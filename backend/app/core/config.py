import os

from dotenv import load_dotenv

load_dotenv()

# --- Database ---------------------------------------------------------------
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")

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

# --- RAG: embeddings ---------------------------------------------------------------
# Local, CPU-friendly sentence-transformers model. 384-dim output.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))

# --- RAG: retrieval ---------------------------------------------------------------
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

# --- RAG: file type bias (code > documentation) ---
# The value added to/subtracted from the relevance score when determining
# the final ranking - source code should receive a slight advantage over documentation.
CODE_FILE_BOOST = float(os.getenv("CODE_FILE_BOOST", "0.05"))
DOC_FILE_PENALTY = float(os.getenv("DOC_FILE_PENALTY", "0.08"))


# --- RAG: local Hugging Face LLM for answer generation ---------------------------------------------------------------

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
# "auto" | "cpu" | "cuda" | a specific CUDA device index (e.g. "0")
LLM_DEVICE = os.getenv("LLM_DEVICE", "auto")

