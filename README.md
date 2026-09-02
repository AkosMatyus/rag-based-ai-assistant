# RAG-based GitHub Codebase Assistant

A full-stack application that indexes a GitHub repository and answers natural-language
questions about it using a **Retrieval-Augmented Generation (RAG)** pipeline. Every
model in the pipeline (embeddings, optional reranker, answer-generation LLM) runs
**locally** — no external AI API is called.

This README covers both the backend (FastAPI) and the frontend (React) in one place,
and is written to let someone with no prior context clone the repo and get it running.

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [Tech stack](#tech-stack)
5. [Prerequisites](#prerequisites)
6. [Setup — 1. PostgreSQL + pgvector](#setup--1-postgresql--pgvector)
7. [Setup — 2. Backend](#setup--2-backend)
8. [Setup — 3. GPU / CUDA (optional but recommended)](#setup--3-gpu--cuda-optional-but-recommended)
9. [Setup — 4. Frontend](#setup--4-frontend)
10. [Quick start / smoke test (no GitHub App needed)](#quick-start--smoke-test-no-github-app-needed)
11. [Optional: GitHub OAuth App (for GitHub login and private repos)](#optional-github-oauth-app-for-github-login-and-private-repos)
12. [Configuration reference](#configuration-reference)
13. [How a question is answered, end to end](#how-a-question-is-answered-end-to-end)
14. [Known limitations](#known-limitations)
15. [Troubleshooting](#troubleshooting)

---

## What this project does

A large language model's knowledge is frozen at training time — it cannot know the
contents of a private (or simply post-training) GitHub repository. This project solves
that with RAG: instead of feeding an entire codebase to a model, it

1. downloads and **indexes** a chosen GitHub repository (splits it into chunks,
   computes a vector embedding for each chunk, stores them in PostgreSQL via the
   `pgvector` extension),
2. on each question, **retrieves** the most semantically relevant chunks for that
   specific repository (vector similarity search, with an optional reranking step),
3. and **generates** an answer with a locally-running LLM, instructed to rely only on
   the retrieved chunks — with every answer showing exactly which files and line
   ranges it came from, so it can be verified against the real source.

## Architecture

```
┌──────────────┐   HTTP (JSON)     ┌───────────────┐
│   Frontend   │ ───────────────▶ │    Backend     │
│ React + Vite │ ◀─────────────── │    FastAPI     │
└──────────────┘                   └───────┬───────┘
                                           │
                    ┌──────────────────────┼──────────────────────────────┐
                    ▼                      ▼                              ▼
          ┌─────────────────────┐  ┌─────────────────────────┐  ┌────────────────────┐
          │   PostgreSQL        │  │  Local models           │  │   GitHub API       │
          │   + pgvector        │  │  (Hugging Face,         │  │  (repo contents,   │
          │   (chunks, vectors, │  │  sentence-transformers) │  │  OAuth, access     │
          │   users, access)    │  │                         │  │  checks)           │
          └─────────────────────┘  └─────────────────────────┘  └────────────────────┘
```

Two independent flows exist:

- **Indexing** (`POST /rag/ingest`): download repo → filter to text/code files → split
  into chunks → embed each chunk → store in `pgvector`. Runs as a background task; the
  frontend polls `GET /rag/status` until it's done.
- **Asking a question** (`POST /chat/ask`): embed the question → vector search →
  (optionally) rerank → build a prompt from the retrieved chunks → generate the answer
  locally → return the answer with its sources.

Everything I/O-bound (HTTP calls, DB queries) is `async`. Everything CPU/GPU-bound
(embedding, generation) is synchronous by nature and is pushed onto a worker thread via
`asyncio.to_thread`, so a slow model call doesn't block the whole server.

## Project structure

```
backend/
  app/
    core/config.py           All settings, loaded from .env
    database/database.py     Async SQLAlchemy engine/session (asyncpg)

    models/
      user.py                  User ORM model + auth schemas
      chat.py                  ChatRequest/ChatResponse/SourceGroup/SourceChunk schemas
      repository.py            Repository / DocumentChunk / RepositoryAccess ORM models

    routers/
      user.py                  Register/login, get_current_user dependency
      github.py                GitHub OAuth login + repo listing (own vs collaborator)
      rag.py                   Ingestion, indexing status, add-repo, "my repos"
      chat.py                  POST /chat/ask

    rag/                      The RAG building blocks, one concern per file
      github_loader.py           Downloads a repo tarball, filters to text/code files,
                                  and checks whether a user has access to a repo
      text_splitter.py           langchain-text-splitters, language-aware chunking
      embeddings.py               sentence-transformers embeddings
      vector_store.py             pgvector storage + cosine-similarity retrieval
      reranker.py                 Optional cross-encoder reranking (see Known limitations)
      llm.py                       Local Hugging Face model, generates the answer
      ingestion.py                 Orchestrates the pipeline above, runs as a background task

    services/
      rag_service.py            Orchestrates the "ask" flow: embed -> retrieve ->
                                 (rerank) -> bias -> generate -> format sources

    main.py                    FastAPI app, CORS, startup (pgvector extension + tables)

  requirements.txt
  .env.example

frontend/
  src/
    App.tsx                   All pages/components (landing, auth, chat) in one file
    App.css
  package.json
```

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Backend framework | Python, FastAPI | async-native, automatic request validation, auto-generated API docs |
| DB access | SQLAlchemy (async), asyncpg | ORM + native async PostgreSQL driver |
| Storage | PostgreSQL + `pgvector` | relational data and vector similarity search in one database |
| Text splitting | `langchain-text-splitters` | language-aware chunking (splits on function/class boundaries per language) |
| Embeddings | `sentence-transformers` | small, locally-runnable embedding models |
| Reranking (optional) | `sentence-transformers` `CrossEncoder` | second-pass, more precise relevance scoring |
| Answer generation | `transformers` (Hugging Face) | runs a local instruction-tuned LLM, no external API |
| Auth | JWT, GitHub OAuth 2.0 | local accounts, plus linking a GitHub account for repo access |
| Frontend | React, TypeScript, Vite | component UI, static typing, fast dev server |
| Code display | `react-syntax-highlighter` | syntax-highlighted source previews |

## Prerequisites

- **Python 3.11+** (developed and tested on 3.13)
- **Node.js 18+** and npm
- **PostgreSQL** (any recent version, 14–17 all work) — see the pgvector section below
- **Git**
- **(Optional) An NVIDIA GPU with recent drivers** — the app runs on CPU too, just
  slower; see the CUDA section for GPU setup
- **(Optional) A GitHub OAuth App** — only needed for GitHub login and for browsing
  your own/collaborator repos; see [Quick start](#quick-start--smoke-test-no-github-app-needed),
  which works without one

## Setup — 1. PostgreSQL + pgvector

The `vector` Postgres extension has to be installed on the **database server itself**,
not just `pip install`-ed. You need a PostgreSQL server already installed.

**Windows**

1. Install "Desktop development with C++" via the
   [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   (the free Community edition works too — you only need the C++ workload).
2. Open **"x64 Native Tools Command Prompt for VS"** (search for it in the Start menu)
   **as Administrator** — a plain `cmd`/PowerShell window won't have the right compiler
   on `PATH`.
3. Run (adjust the PostgreSQL version to match what you have installed, and the
   pgvector tag to the latest release if a newer one exists):

   ```bat
   set "PGROOT=C:\Program Files\PostgreSQL\17"
   cd %TEMP%
   git clone --branch v0.8.6 https://github.com/pgvector/pgvector.git
   cd pgvector
   nmake /F Makefile.win
   nmake /F Makefile.win install
   ```

   If `nmake /F Makefile.win install` fails with "Access is denied", you're not running
   the command prompt as Administrator — close it and reopen it with "Run as
   administrator".
4. Restart the PostgreSQL service (Services app → `postgresql-x64-17` → Restart).

**macOS**

```bash
brew install pgvector
```

(If PostgreSQL was also installed via Homebrew, this just works.)

**Linux (Debian/Ubuntu)**

```bash
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
sudo apt-get install postgresql-17-pgvector   # match your Postgres major version
```

Or build from source:

```bash
sudo apt-get install build-essential postgresql-server-dev-17
git clone --branch v0.8.6 https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
```

**Then, on any OS**, create the database once:

```bash
psql -U postgres -c "CREATE DATABASE rag_assistant;"
```

Nothing else needs to be configured manually — the backend creates the `vector`
extension inside the database itself (`CREATE EXTENSION IF NOT EXISTS vector`) and
creates all tables automatically the first time it starts. The extension just needs to
be *installable* on the server, which is what the steps above set up.

## Setup — 2. Backend

From the `backend/` directory:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- If PowerShell refuses to run the activation script ("running scripts is disabled on
  this system"), run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then
  retry.
- **Never move or rename the project folder after creating the venv on Windows.** The
  `.exe` launchers inside `.venv\Scripts` (`pip.exe`, `uvicorn.exe`, ...) have the
  venv's absolute path baked in at creation time. If the folder moves, every launcher
  breaks with `Fatal error in launcher: Unable to create process using "..."`. If this
  happens, delete `.venv` and recreate it at the folder's current location — or avoid
  the problem entirely by always calling tools through `python -m` instead of the
  launcher `.exe` directly:
  ```bash
  python -m pip install -r requirements.txt
  python -m uvicorn main:app --reload
  ```
  (run from `backend/app` for the `uvicorn` command).

Copy the example environment file and fill it in:

```bash
cp .env.example .env
```

At minimum, set `DB_PASS` to your PostgreSQL password and `JWT_SECRET_KEY` to any
random string. Everything else has a working default — see the
[Configuration reference](#configuration-reference) for what each variable does.
GitHub-related variables can be left blank; see [Quick start](#quick-start--smoke-test-no-github-app-needed).

Run the backend:

```bash
cd app
uvicorn main:app --reload --port 8000
```

On first startup it connects to Postgres, creates the `vector` extension and all
tables. The first `/rag/ingest` and the first `/chat/ask` call will each download their
respective models from the Hugging Face Hub on first use (the embedding model, then
the LLM) and cache them under `~/.cache/huggingface`
(Windows: `%USERPROFILE%\.cache\huggingface`) — that first call is slow (the LLM alone
is a multi-gigabyte download), subsequent calls reuse the cached, already-loaded model.

Verify it's up: open `http://127.0.0.1:8000/docs` — you should see the interactive API
documentation (Swagger UI).

## Setup — 3. GPU / CUDA (optional but recommended)

**The app works fine on CPU** — this section is only for making it faster. Skip it if
you don't have an NVIDIA GPU.

### The problem you will hit

`pip install -r requirements.txt` installs the **CPU-only** build of PyTorch, even on a
machine with a GPU — because the plain `torch` package on PyPI defaults to CPU-only.
Nothing in `requirements.txt` can reliably force a CUDA build (the correct package
depends on your CUDA version and isn't expressible as a normal PyPI dependency), so
this is always a **second, manual install step**, every time the virtual environment is
(re)created:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Run this **after** `pip install -r requirements.txt` — if you run it before, or run
`pip install -r requirements.txt` again afterwards, it gets silently overwritten by the
CPU-only build again.

### Why specifically `cu128`

Recent NVIDIA GPUs (the RTX 50 series / "Blackwell" architecture, `sm_120`) require
**CUDA 12.8 or newer** to be recognized by PyTorch. If you have an older GPU, an older
`cuXXX` index (e.g. `cu121`, `cu124`) may also work, but `cu128` is safe for both older
and newer cards as of this writing. If in doubt, get the exact, currently-correct
command for your OS/CUDA version from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) (select
Stable → your OS → Pip → the newest CUDA option).

### Verifying it worked

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The version string must contain `+cu...` (e.g. `2.7.0+cu128`) — if it has no `+cu`
suffix, or shows `+cpu`, the CPU-only build is installed and `torch.cuda.is_available()`
will print `False` regardless of how good your GPU is. Also double-check with:

```bash
pip show torch
```

and look at the `Version:` line for the same `+cu...` suffix.

If `torch.cuda.is_available()` is still `False` after confirming a `+cu...` build is
installed, check that your GPU driver is up to date and is actually visible to the
system:

```bash
nvidia-smi
```

If this command fails or doesn't list your GPU, the driver — not PyTorch — is the
problem; install/update it from NVIDIA's website and reboot.

### How the app picks CPU vs. GPU

Once a CUDA-enabled `torch` is installed, nothing else needs to change — `LLM_DEVICE`
in `.env` defaults to `auto`, which uses the GPU automatically if
`torch.cuda.is_available()` is `True`, and falls back to CPU otherwise. The embedding
model (via `sentence-transformers`) does the same detection independently and needs no
configuration either way.

## Setup — 4. Frontend

From the `frontend/` directory:

```bash
npm install
npm run dev
```

Vite will print the local URL, typically `http://localhost:5173`. Open it in a
browser. The frontend expects the backend at `http://127.0.0.1:8000` (hardcoded as
`API_URL` in `src/App.tsx`) — if you run the backend on a different host/port, update
that constant.

## Quick start / smoke test (no GitHub App needed)

The fastest way to verify everything works end to end, **without** setting up a GitHub
OAuth App:

1. Open the frontend, go to **Register**, create a local account (username/password —
   this does not touch GitHub at all).
2. Log in. You land on the chat page with three tabs: **Own**, **Added**,
   **Collaborator**. The "Own" and "Collaborator" tabs will show an error, because they
   require a linked GitHub account — that's expected and fine for this test.
3. Click the **"Added"** tab. Type a small, well-known **public** repository, e.g.
   `octocat/Hello-World` or any small public repo of your choice, and click
   **"Add"**. The backend checks (via the GitHub API, unauthenticated) that the repo
   exists and is accessible, and adds it to your list — no GitHub login is required for
   *public* repositories.
4. Select it, click **"Start indexing"**. The status badge changes to "Indexing..." and
   polls automatically every 3 seconds; wait for **"Ready"**.
5. Ask a question about the repo in plain language. The answer appears with a
   **Sources** section below it; click a source pill to open the exact code
   snippet (with syntax highlighting) in the right-hand panel, with Previous/Next
   navigation if a file contributed more than one chunk.

If this all works, the full pipeline (ingestion, embeddings, vector search, local LLM
generation) is functioning correctly.

Note: without setting `GITHUB_API_TOKEN` in `.env`, GitHub API calls (used for both the
access check and the repo download) are subject to GitHub's unauthenticated rate limit
of 60 requests/hour, which is generally enough for a single smoke test but can be hit
quickly with repeated testing. See [Configuration reference](#configuration-reference)
for how to set a personal access token if needed.

## Optional: GitHub OAuth App (for GitHub login and private repos)

To test "Log in with GitHub" and the **Own**/**Collaborator** tabs (which list repos
from *your* actual GitHub account), you need your own GitHub OAuth App — this is free
and takes about two minutes:

1. Go to <https://github.com/settings/developers> → **OAuth Apps** → **New OAuth App**.
2. **Homepage URL**: `http://localhost:5173`
3. **Authorization callback URL**: `http://127.0.0.1:8000/github/callback` (must match
   `GITHUB_REDIRECT_URI` in `.env` exactly).
4. After creating it, copy the **Client ID**, and generate + copy a **Client Secret**.
5. Put both into `backend/.env`:
   ```
   GITHUB_CLIENT_ID=...
   GITHUB_CLIENT_SECRET=...
   ```
6. Restart the backend. The "Log in with GitHub" button on the login page will now
   work, and after linking your account, private repos you own or collaborate on will
   appear in the **Own**/**Collaborator** tabs (subject to the access checks described
   in [Known limitations](#known-limitations)).

## Configuration reference

All settings live in `backend/.env` (copy `backend/.env.example` to start). Every
variable has a working default in code except where noted.

| Variable | Default | Purpose |
|---|---|---|
| `DB_USER` / `DB_PASS` / `DB_NAME` / `DB_HOST` | `postgres` / — / `rag_assistant` / `localhost` | PostgreSQL connection |
| `JWT_SECRET_KEY` | — | Signing key for login tokens — set this to a random string |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Login session length |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | — | GitHub OAuth App credentials (optional, see above) |
| `GITHUB_REDIRECT_URI` | `http://127.0.0.1:8000/github/callback` | Must match the OAuth App's callback URL |
| `FRONTEND_URL` | `http://localhost:5173` | Where GitHub login redirects back to |
| `GITHUB_API_TOKEN` | — | Optional fallback personal access token for repo ingestion when the user has no linked GitHub account — raises the anonymous GitHub API rate limit and is required to index private repos without GitHub login |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1200` / `200` | Text-splitting chunk size/overlap, in **characters** (not tokens) |
| `RAG_MAX_FILE_SIZE_BYTES` | `307200` (300 KB) | Files larger than this are skipped during ingestion |
| `RAG_MAX_FILES_PER_REPO` | `800` | Hard cap on files read per repository per ingestion run |
| `EMBEDDING_MODEL_NAME` | `ibm-granite/granite-embedding-97m-multilingual-r2` | Local `sentence-transformers` embedding model. Multilingual and code-aware; see rationale below |
| `EMBEDDING_DIMENSIONS` | `384` | Must match the embedding model's output size — changing the model to one with a different dimension requires dropping and recreating the `document_chunks` table |
| `RAG_TOP_K` | `6` | Number of chunks included in the final answer prompt |
| `RERANK_ENABLED` | `false` | Enables a second-pass cross-encoder reranking step — see [Known limitations](#known-limitations) for why it's off by default |
| `RERANK_CANDIDATES` | `25` | How many chunks the initial vector search retrieves before reranking (only used when reranking is enabled) |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder model used for reranking (only used when reranking is enabled) |
| `CODE_FILE_BOOST` | `0.05` | Small score bonus for source-code files over documentation files in the final ranking |
| `DOC_FILE_PENALTY` | `0.08` | Small score penalty for documentation files (`.md`, `.mdx`, `.rst`, `.txt`) |
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | Local Hugging Face model used to generate answers |
| `LLM_MAX_NEW_TOKENS` | `512` | Maximum length of a generated answer |
| `LLM_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| a specific CUDA device index |
| `REPOSITORY_ACCESS_TTL_HOURS` | `24` | How long a recorded repo-access grant is trusted before it's re-verified live against the GitHub API |

**Why `granite-embedding-97m-multilingual-r2` instead of the more commonly-referenced
`all-MiniLM-L6-v2`:** the project's questions are asked in Hungarian while the
indexed content (source code, some documentation) is predominantly English. IBM's
Granite embedding model is explicitly trained for both multilingual retrieval and code
retrieval, with an 8k–32k token context window (vs. 256 tokens for `all-MiniLM-L6-v2`,
which silently truncates anything longer). It also happens to share the same 384-dim
output size as `all-MiniLM-L6-v2`, so switching between them never requires a database
migration.

## How a question is answered, end to end

1. Browser sends `POST /chat/ask` with the question, the selected repo, and the auth
   token.
2. FastAPI resolves the current user from the token (`get_current_user` dependency).
3. `rag_service.answer_question`:
   - confirms the repository exists and the user has a recorded, still-valid access
     grant to it (re-verifying live against GitHub if the grant is older than
     `REPOSITORY_ACCESS_TTL_HOURS`),
   - confirms the repository has finished indexing,
   - embeds the question with the same model used for indexing,
   - runs a `pgvector` cosine-similarity search scoped to that repository,
   - if `RERANK_ENABLED=true`, reorders the candidates with a cross-encoder,
   - applies the source-code-vs-documentation bias and keeps the top `RAG_TOP_K`,
   - builds a prompt from the retrieved chunks and generates the answer locally,
   - groups the chunks by file for the response (`sources: [{file_path, chunks: [...]}]`).
4. The frontend renders the answer, one pill per source file, and a right-hand code
   preview panel with Previous/Next navigation across a file's chunks, sorted by
   relevance.

## Known limitations

These are deliberate scope decisions or open issues, not oversights — documented here
for transparency:

- **No 4-bit quantization.** The LLM currently always loads at its native precision
  (`torch_dtype="auto"`, i.e. bf16/fp16 on GPU). This limits which model sizes
  comfortably fit on a given GPU's VRAM — the tested configuration
  (`Qwen2.5-Coder-3B-Instruct`) fits comfortably unquantized on a 12 GB GPU.
- **No relevance threshold.** The system does not currently refuse to answer when the
  best-retrieved chunk is only weakly relevant — it always attempts an answer from
  whatever was retrieved, which can occasionally lead to a confidently-worded but
  unhelpful or partially hallucinated answer when the codebase genuinely doesn't
  contain what was asked about.
- **Reranking is implemented but disabled by default.** Both cross-encoders tested
  during development (`ms-marco-MiniLM-L-6-v2`, English-only, trained on natural-language
  web passages; `bge-reranker-v2-m3`, multilingual but still natural-language-trained)
  turned out to be a poor fit for this project's specific combination of
  Hungarian-language queries over source-code passages — one produced confidently
  wrong rankings, the other produced uniformly unconfident (~50%) scores for
  everything. `RERANK_ENABLED=true` can still be turned on and experimented with, but
  it is not a validated improvement for this codebase-and-language combination.
- **No conversation history.** Each question is answered independently; nothing is
  persisted to the database, and there is no follow-up/multi-turn context.
- **Full re-indexing only.** Re-indexing a repository replaces all of its chunks; there
  is no incremental/delta indexing of only changed files.
- **In-process model inference.** All three local models (embedding, optional
  reranker, LLM) run inside the same process as the web server, loaded once as
  singletons. This is simple to run and deploy for a single-user/demo setting, but
  means the model layer cannot be scaled independently of the API layer, and a crash or
  out-of-memory error in model inference can affect the whole backend process.
- **`.env` changes require a full restart.** `uvicorn --reload` only watches `.py`
  files; editing `.env` while the server is running has no effect until the process is
  stopped and started again.
- **Embedding/chunking changes require re-indexing.** Changing `EMBEDDING_MODEL_NAME`,
  `RAG_CHUNK_SIZE`, or similar settings only affects *future* indexing runs — existing
  repositories keep their old chunks/vectors until "Re-index" is clicked again for
  each of them.

## Troubleshooting

**`asyncpg.exceptions.FeatureNotSupportedError: extension "vector" is not available`**
on startup — the `pgvector` extension isn't installed on your PostgreSQL *server* (a
`pip install` alone never fixes this). Follow [Setup — 1](#setup--1-postgresql--pgvector)
for your OS.

**`torch.cuda.is_available()` prints `False`** — see the full
[GPU / CUDA section](#setup--3-gpu--cuda-optional-but-recommended); by far the most
common cause is the CPU-only `torch` build silently installed by
`pip install -r requirements.txt`.

**`Fatal error in launcher: Unable to create process using "..."`** (Windows) — the
project folder was moved or renamed after the virtual environment was created. Delete
`.venv` and recreate it at the folder's current path, or use `python -m pip ...` /
`python -m uvicorn ...` instead of the `pip`/`uvicorn` launcher executables going
forward.

**Changed a setting in `.env` but nothing happened** — you need a full backend
restart (stop with Ctrl+C, start again), not just a save; `--reload` does not watch
`.env`.

**Re-indexed a repo but answers still seem to reflect the old content, or a UI setting
change didn't do anything** — check whether the change was to something indexing-time
(chunking, embedding model) vs. query-time (LLM, retrieval, reranking); the former
needs the affected repositories to be explicitly re-indexed, the latter takes effect on
the very next question after a restart.

**A repo add/ingest request returns 404 "no such repository, or you don't have
access"** — this is often correct behavior (a typo in the repo name, or a genuinely
private repo without a linked GitHub account/token), not a bug. Verify the exact repo
name and, for private repos, that a GitHub account is linked or `GITHUB_API_TOKEN` is
set.

**CORS errors in the browser console** — the backend's CORS policy
(`backend/app/main.py`) only allows `http://localhost:5173` by default. If Vite starts
on a different port (it will if 5173 is already in use — check the terminal output),
either free up port 5173 or add the actual port to `allow_origins` in `main.py`.
