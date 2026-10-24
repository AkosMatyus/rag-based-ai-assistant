# RAG-based GitHub Codebase Assistant

A full-stack application that indexes a GitHub repository and answers natural-language
questions about it using a **Retrieval-Augmented Generation (RAG)** pipeline. Every
model in the pipeline runs **locally** — the embedding model and the answer-generation
LLM in [Ollama](https://ollama.com), the optional reranker inside the backend — so no
external AI API is called.

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
7. [Setup — 2. Ollama (models)](#setup--2-ollama-models)
8. [Setup — 3. Backend](#setup--3-backend)
9. [Setup — 4. Frontend](#setup--4-frontend)
10. [Quick start / smoke test (no GitHub App needed)](#quick-start--smoke-test-no-github-app-needed)
11. [Optional: GitHub OAuth App (for GitHub login and private repos)](#optional-github-oauth-app-for-github-login-and-private-repos)
12. [Optional: reranking](#optional-reranking)
13. [Configuration reference](#configuration-reference)
14. [How a question is answered, end to end](#how-a-question-is-answered-end-to-end)
15. [Known limitations](#known-limitations)
16. [Troubleshooting](#troubleshooting)

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
          │   PostgreSQL        │  │  Ollama (local)         │  │   GitHub API       │
          │   + pgvector        │  │  embedding model + LLM, │  │  (repo contents,   │
          │   (chunks, vectors, │  │  HTTP API on :11434     │  │  OAuth, access     │
          │   users, access)    │  │                         │  │  checks)           │
          └─────────────────────┘  └─────────────────────────┘  └────────────────────┘
```

Two independent flows exist:

- **Indexing** (`POST /rag/ingest`): download repo → filter to text/code files → split
  into chunks → embed the chunks (Ollama) → store in `pgvector`. Runs as a background task;
  while any repository is indexing, the frontend refreshes `GET /rag/repos/mine` every
  3 seconds (`GET /rag/status` returns the same information for a single repository).
- **Asking a question** (`POST /chat/ask`): embed the question (Ollama) → vector
  search → (optionally) rerank → build a prompt from the retrieved chunks → generate
  the answer (Ollama) → return the answer with its sources.

Everything is `async`: database queries (asyncpg) and the HTTP calls to Ollama and
GitHub (httpx). The heavy model work happens in the separate Ollama process, so the
backend only awaits its responses. The few CPU-bound steps left in the backend
(unpacking the repository tarball, the optional reranker) run in a worker thread via
`asyncio.to_thread`, so they don't block the server either.

## Project structure

```
backend/
  app/
    core/config.py           All settings, loaded from backend/.env
    database/database.py     Async SQLAlchemy engine/session (asyncpg), built from the DB_* settings
    utils.py                 Password hashing (bcrypt) and JWT creation/validation

    models/
      user.py                  User ORM model + auth schemas
      chat.py                  ChatRequest/ChatResponse/SourceGroup/SourceChunk schemas
      repository.py            Repository / DocumentChunk / RepositoryAccess ORM models

    routers/
      user.py                  Register/login, get_current_user dependency
      github.py                GitHub OAuth login + repo listing (own vs collaborator)
      rag.py                   Ingestion, indexing status, add-repo, "my repos"
      chat.py                  POST /chat/ask

    rag/                     The RAG building blocks, one concern per file
      github_loader.py         Downloads a repo tarball, filters to text/code files,
                               and checks whether a user has access to a repo
      text_splitter.py         langchain-text-splitters, language-aware chunking
      ollama_client.py         Async HTTP client for the local Ollama server, with
                               user-readable errors (not running, model not pulled)
      embeddings.py            Embeddings via Ollama (/api/embed)
      vector_store.py          pgvector storage + cosine-similarity retrieval
      reranker.py              Optional cross-encoder reranking (sentence-transformers,
                               in-process; see Optional: reranking)
      llm.py                   Answer generation via Ollama (/api/chat)
      ingestion.py             Orchestrates the pipeline above, runs as a background task

    services/
      rag_service.py           Orchestrates the "ask" flow: embed -> retrieve ->
                               (rerank) -> bias -> generate -> format sources

    main.py                  FastAPI app, CORS, startup (pgvector extension, tables,
                             HNSW index, failing interrupted indexing jobs, Ollama check)

  requirements.txt
  requirements-rerank.txt    Optional extra for the reranker (installs PyTorch)
  .env.example

frontend/
  index.html                 Page shell; applies the saved light/dark theme before first paint
  src/
    main.tsx                 React entry point
    App.tsx                  All pages/components (landing, sign-in/register, workspace) in one file
    App.css                  Component styles and responsive layout
    index.css                Design tokens (colors, fonts) for the light and dark themes
  package.json
```

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Backend framework | Python, FastAPI | async-native, automatic request validation, auto-generated API docs |
| DB access | SQLAlchemy (async), asyncpg | ORM + native async PostgreSQL driver |
| Storage | PostgreSQL + `pgvector` | relational data and vector similarity search in one database |
| Text splitting | `langchain-text-splitters` | language-aware chunking (splits on function/class boundaries per language) |
| Model runtime | Ollama | runs the embedding model and the LLM locally (GPU or CPU) behind an HTTP API, and handles GPU support itself |
| Embeddings | `qwen3-embedding:0.6b` via Ollama `/api/embed` | multilingual, code-aware embeddings |
| Answer generation | `qwen3.5:4b` via Ollama `/api/chat` | local instruction-tuned LLM, no external API |
| HTTP client | `httpx` | async calls to Ollama and the GitHub API |
| Reranking (optional) | `sentence-transformers` `CrossEncoder` | second-pass, more precise relevance scoring (in-process: Ollama has no reranking API) |
| Auth | JWT (`python-jose`), bcrypt (`passlib`), GitHub OAuth 2.0 | local username/password accounts, plus GitHub sign-in for repo access |
| Frontend | React 19, TypeScript, Vite 8, `react-router-dom` | component UI, static typing, fast dev server, client-side routing |
| Styling | Plain CSS with design tokens | light and dark theme without a UI framework dependency |
| Code display | `react-syntax-highlighter` | syntax-highlighted source previews |

## Prerequisites

- **Python 3.11+** (developed and tested on 3.13)
- **Node.js 20.19+ or 22.12+** and npm (the minimum Vite 8 supports)
- **PostgreSQL 13 or newer** (developed on 17) with the **pgvector** extension — see
  [Setup — 1](#setup--1-postgresql--pgvector), or use the Docker image described there
- **Git**
- **(Windows only) Visual Studio Build Tools with the C++ workload** — needed to compile
  pgvector, unless you use Docker
- **Ollama**, a recent version — runs the models; see [Setup — 2](#setup--2-ollama-models)
- **(Optional) A GPU that Ollama supports** (NVIDIA with a recent driver, Apple
  Silicon, or a supported AMD card) — without one the models run on the CPU, which
  works but makes answers much slower
- **(Optional) A GitHub OAuth App** — only needed for GitHub login and for browsing
  your own/collaborator repos; see [Quick start](#quick-start--smoke-test-no-github-app-needed),
  which works without one

## Setup — 1. PostgreSQL + pgvector

The backend needs three things from the database side: a running **PostgreSQL server**,
the **pgvector extension installed on that server**, and **one empty database**. It
creates everything *inside* that database by itself (extension, tables, index) on its
first start, so there's no SQL script to run.

### 1.1 Install PostgreSQL

**Windows** — use the installer from
[postgresql.org/download/windows](https://www.postgresql.org/download/windows/). During
setup, write down the password you choose for the `postgres` superuser (it becomes
`DB_PASS`) and keep the default port `5432`. The installer also adds **pgAdmin 4** and
**SQL Shell (psql)** to the Start menu; the Stack Builder step at the end can be
skipped.

**macOS** — either `brew install postgresql@17` followed by
`brew services start postgresql@17`, or [Postgres.app](https://postgresapp.com/), which
already includes pgvector. Both create a superuser named after your **macOS username**
(not `postgres`) that connects locally without a password — use that name as
`DB_USER` (`DB_PASS` is then ignored).

**Linux (Debian/Ubuntu)**

```bash
sudo apt-get install postgresql
# The postgres superuser has no password by default, but the backend connects over
# TCP with a password, so set one:
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'change_me';"
```

**Alternative on any OS: Docker.** This one command gives you PostgreSQL with pgvector
already installed *and* the database already created, so you can skip 1.2 and 1.3:

```bash
docker run -d --name rag-postgres -p 5432:5432 \
  -e POSTGRES_PASSWORD=change_me -e POSTGRES_DB=rag_assistant \
  -v rag_pgdata:/var/lib/postgresql/data pgvector/pgvector:pg17
```

In PowerShell, write the command on one line without the `\` line breaks. If a local
PostgreSQL already uses port 5432, map another one (`-p 5433:5432`) and set
`DB_HOST=localhost:5433` (see 1.4).

### 1.2 Install pgvector on the server

The `vector` extension has to be installed on the **database server itself**; the
`pgvector` package in `requirements.txt` is only the Python client and doesn't install
anything into PostgreSQL.

**Windows**

1. Install "Desktop development with C++" via the
   [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   (the free Community edition works too — you only need the C++ workload).
2. Open **"x64 Native Tools Command Prompt for VS"** (search for it in the Start menu)
   **as Administrator** — a plain `cmd`/PowerShell window won't have the right compiler
   on `PATH`.
3. Run (adjust `17` to your PostgreSQL version, and the pgvector tag to the latest
   release if a newer one exists):

   ```bat
   set "PGROOT=C:\Program Files\PostgreSQL\17"
   cd %TEMP%
   git clone --branch v0.8.7 https://github.com/pgvector/pgvector.git
   cd pgvector
   nmake /F Makefile.win
   nmake /F Makefile.win install
   ```

   If `nmake /F Makefile.win install` fails with "Access is denied", you're not running
   the command prompt as Administrator — close it and reopen it with "Run as
   administrator". If linking fails with `unresolved external symbol
   float_to_shortest_decimal_bufn`, you're on PostgreSQL 17.0–17.2; update to 17.3 or
   newer.
4. Restart the PostgreSQL service (Services app → `postgresql-x64-17` → Restart).

**macOS**

```bash
brew install pgvector
```

(This works with Homebrew's `postgresql@17` and `postgresql@18`; Postgres.app already
has it.)

**Linux (Debian/Ubuntu)**

```bash
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
sudo apt-get install postgresql-17-pgvector   # match your Postgres major version
```

Or build from source:

```bash
sudo apt-get install build-essential postgresql-server-dev-17
git clone --branch v0.8.7 https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
```

### 1.3 Create the database

Create one empty database (the examples use `rag_assistant`, which is also what
`.env.example` expects):

```bat
:: Windows (cmd) - psql isn't on PATH by default, so call it by its full path:
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "CREATE DATABASE rag_assistant;"
```

```powershell
# Windows (PowerShell) - same, but a quoted path needs the & call operator:
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "CREATE DATABASE rag_assistant;"
```

```bash
# macOS (Homebrew / Postgres.app)
createdb rag_assistant

# Linux
sudo -u postgres createdb rag_assistant
```

Without the command line: in **pgAdmin**, open *Servers → PostgreSQL 17*, right-click
*Databases → Create → Database…*, and name it `rag_assistant`. Or open **SQL Shell
(psql)**, press Enter at each prompt to accept the defaults, type the `postgres`
password, then run `CREATE DATABASE rag_assistant;`.

To check that pgvector (1.2) is installed correctly, run this in the same psql/pgAdmin
session:

```sql
SELECT name, default_version FROM pg_available_extensions WHERE name = 'vector';
```

One row means it's ready. No rows means the extension isn't installed on this server
yet; go back to 1.2.

You don't need to create any tables or run `CREATE EXTENSION` yourself.

### 1.4 Connect the backend to the database

The connection is configured with four variables in `backend/.env` (you create this file
in [Setup — 3](#setup--3-backend)):

```ini
DB_USER=postgres
DB_PASS=the_password_you_set_for_postgres
DB_NAME=rag_assistant
DB_HOST=localhost
```

`backend/app/database/database.py` builds the connection URL
`postgresql+asyncpg://DB_USER:DB_PASS@DB_HOST/DB_NAME` from these (`asyncpg` is the
async PostgreSQL driver). Two things to watch:

- **Port.** The backend connects on PostgreSQL's default port, `5432`. If your server
  listens on another one, put it into `DB_HOST`, e.g. `DB_HOST=localhost:5433`.
  (`config.py` also defines `DB_PORT`, but the connection URL doesn't use it yet.)
- **Special characters in the password.** The password is inserted into a URL, so an
  `@` or `%` in it must be URL-encoded: `@` → `%40`, `%` → `%25`.

On every start the backend then:

1. runs `CREATE EXTENSION IF NOT EXISTS vector` in the database,
2. if `EMBEDDING_DIMENSIONS` no longer matches the size of the stored vectors (the
   embedding model was switched), drops `document_chunks` and sets every repository
   back to "Not indexed" (see [2.5](#25-switching-models)),
3. creates any missing tables: `users`, `repositories`, `repository_access`,
   `document_chunks`,
4. creates the HNSW index on the embedding column (if missing) for fast
   cosine-similarity search,
5. marks repositories left in "indexing" by a previous run as failed (see
   [Troubleshooting](#troubleshooting)),
6. checks in the background that Ollama is reachable and has both models
   ([2.4](#24-how-the-backend-uses-ollama)).

After the first start you can check the result:

```bash
psql -U postgres -h localhost -d rag_assistant -c "\dt"
```

(On Windows, call `psql.exe` by its full path as in 1.3, with `&` in front in
PowerShell.) You should see the four
tables listed above.

Good to know:

- **Superuser rights for the extension.** pgvector isn't a "trusted" extension, so the
  first `CREATE EXTENSION` needs a superuser. Connecting as `postgres` (or as your
  Homebrew/Postgres.app user) is the simplest for a local setup. To use a dedicated
  role instead, create the database with `OWNER your_role` (so the role can create
  tables) and run `CREATE EXTENSION vector;` in it once as `postgres`.
- **Tables are never altered, only created.** If a model in `backend/app/models/`
  changes (for example after pulling an update that adds a column), drop the affected
  table, or drop and recreate the whole database, and restart the backend.
- **Starting over.** `DROP DATABASE rag_assistant;` plus `CREATE DATABASE
  rag_assistant;` wipes all users, repositories and chunks; the backend recreates the
  tables on its next start. `DROP TABLE document_chunks;` wipes only the indexed
  chunks; re-index every repository afterwards, since their status still says
  "Indexed" but they have no chunks left to search. (After a change of
  `EMBEDDING_DIMENSIONS` the backend does this by itself, including the status reset.)

## Setup — 2. Ollama (models)

Both models of the pipeline run in [Ollama](https://ollama.com), a free program that
downloads open models, runs them on your GPU (or on the CPU when there is none), and
serves them through a local HTTP API at `http://127.0.0.1:11434`. The backend only sends
requests to that API (`backend/app/rag/ollama_client.py`), so the Python environment
needs no PyTorch, no CUDA build and no Hugging Face downloads.

| Role | Default model | Download | Notes |
|---|---|---|---|
| Embeddings (indexing and questions) | `qwen3-embedding:0.6b` | ~640 MB | 1024-dim vectors, multilingual, trained for code retrieval |
| Answer generation | `qwen3.5:4b` | ~4 GB | 4-bit (Q4_K_M) build; the backend switches its "thinking" off |

Together they fit comfortably in 12 GB of VRAM, so both stay loaded at the same time.

### 2.1 Install Ollama

- **Windows / macOS** — download the app from [ollama.com/download](https://ollama.com/download)
  and install it. It runs in the background (tray / menu bar icon) and starts the
  Ollama server automatically, also after a reboot.
- **Linux** — `curl -fsSL https://ollama.com/install.sh | sh`, which installs Ollama and
  starts it as a systemd service.

Check it in a new terminal:

```bash
ollama --version
```

**GPU:** Ollama detects a supported GPU by itself (on NVIDIA cards it only needs a
recent driver, not the CUDA toolkit). Nothing has to be configured in the project.

### 2.2 Download the models

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:4b
ollama list
```

`ollama list` should show both. The names must match `EMBEDDING_MODEL_NAME` and
`LLM_MODEL_NAME` in `backend/.env` exactly, including the part after the colon (the
defaults already do).

### 2.3 Check that it works

```bash
ollama run qwen3.5:4b --think=false "Say hello in Hungarian."
ollama ps
```

`ollama ps` lists the loaded models. The `PROCESSOR` column should say `100% GPU`; if it
shows `CPU` or a CPU/GPU split, the model runs (partly) on the CPU and answers will be
much slower (see [Troubleshooting](#troubleshooting)).

You can also open `http://127.0.0.1:11434` in a browser: it answers "Ollama is
running".

### 2.4 How the backend uses Ollama

- **Startup check.** A few seconds after the backend starts, it logs whether Ollama is
  reachable, whether both models are downloaded (with the exact `ollama pull` command
  if not), and whether the embedding model's vector size matches
  `EMBEDDING_DIMENSIONS`. The backend still starts without Ollama; indexing and
  questions then fail with the same explanation in the UI.
- **Keeping models loaded.** Every request asks Ollama to keep the model in memory for
  `OLLAMA_KEEP_ALIVE` (30 minutes) after its last use. The first question after that
  waits a few seconds while the model loads again.
- **Context window.** Each answer request sets the context size explicitly
  (`LLM_CONTEXT_LENGTH`, 8192 tokens). Ollama's own default is only 4096 tokens on GPUs
  with less than 24 GB of VRAM, which a prompt with six code chunks can exceed — Ollama
  would then silently cut part of it.
- **Ollama on another machine.** Point `OLLAMA_BASE_URL` at it (the other machine has
  to run Ollama with `OLLAMA_HOST=0.0.0.0` to accept remote connections).

### 2.5 Switching models

1. `ollama pull <model>` the new model.
2. Set `LLM_MODEL_NAME` or `EMBEDDING_MODEL_NAME` in `backend/.env`. For an embedding
   model, also set `EMBEDDING_DIMENSIONS` to its vector size, and
   `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_DOCUMENT_PREFIX` to what its model page asks
   for (see [Configuration reference](#configuration-reference)).
3. Restart the backend.

A new LLM is used from the very next question. A new embedding model needs every
repository to be indexed again, because vectors from different models can't be
compared. If its vector size differs from the old one, the backend notices that at
startup, drops the stored chunks itself, and sets every repository back to "Not
indexed"; if the size is the same, click **Re-index** on each repository yourself.

## Setup — 3. Backend

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

At minimum, set the `DB_*` values to match your database (see
[1.4](#14-connect-the-backend-to-the-database)) and `JWT_SECRET_KEY` to any random
string. Everything else has a working default — see the
[Configuration reference](#configuration-reference) for what each variable does.
GitHub-related variables can be left blank; see [Quick start](#quick-start--smoke-test-no-github-app-needed).

Run the backend:

```bash
cd app
uvicorn main:app --reload --port 8000
```

On startup it connects to the database and sets it up as described in
[1.4](#14-connect-the-backend-to-the-database), then checks Ollama: a line like
`Ollama is ready at http://127.0.0.1:11434 (embeddings: qwen3-embedding:0.6b, LLM:
qwen3.5:4b)` in the log means both models are available. Otherwise the log says what's
missing (see [2.4](#24-how-the-backend-uses-ollama)).

Verify it's up: open `http://127.0.0.1:8000/docs` — you should see the interactive API
documentation (Swagger UI).

## Setup — 4. Frontend

From the `frontend/` directory:

```bash
npm install
npm run dev
```

Vite will print the local URL, typically `http://localhost:5173`. Open it in a
browser, using `localhost` rather than `127.0.0.1`: the backend's CORS setting only
allows `http://localhost:5173`. The frontend expects the backend at `http://127.0.0.1:8000` (hardcoded as
`API_URL` in `src/App.tsx`) — if you run the backend on a different host/port, update
that constant.

## Quick start / smoke test (no GitHub App needed)

The fastest way to verify everything works end to end, **without** setting up a GitHub
OAuth App. PostgreSQL, Ollama (with both models pulled), the backend and the frontend
all need to be running.

1. Open the frontend, go to **Register**, create a local account (username/password —
   this does not touch GitHub at all).
2. Sign in. You land on the workspace, whose sidebar has three tabs: **Own**,
   **Added**, **Collaborator**. A local account starts on **Added**; **Own** and
   **Collaborator** only show a "Connect GitHub" prompt, because they list repositories
   from a GitHub account. That's expected and fine for this test.
3. In the **"Added"** tab, type a small, well-known **public** repository, e.g.
   `octocat/Hello-World` or any small public repo of your choice, and click
   **"Add"**. The backend checks (via the GitHub API, unauthenticated) that the repo
   exists and is accessible, and adds it to your list — no GitHub login is required for
   *public* repositories.
4. Click the **"Index"** button on its row in the sidebar. The status changes to
   "Indexing…" and polls automatically every 3 seconds; wait for **"Indexed"**. If it
   ends in "Indexing failed", the main panel shows why, and **Retry** starts it again.
5. Ask a question about the repo in plain language (Enter sends, Shift+Enter adds a new
   line). The answer appears with a **Sources** list below it; click a source to open
   the exact code snippet (with syntax highlighting) in the right-hand panel, with
   Previous/Next navigation if that file contributed more than one chunk and a link
   that opens the same lines on GitHub.

If this all works, the full pipeline (ingestion, embeddings, vector search, local LLM
generation) is functioning correctly.

Note: without setting `GITHUB_API_TOKEN` in `.env`, GitHub API calls (used for both the
access check and the repo download) are subject to GitHub's unauthenticated rate limit
of 60 requests/hour, which is generally enough for a single smoke test but can be hit
quickly with repeated testing. See [Configuration reference](#configuration-reference)
for how to set a personal access token if needed.

## Optional: GitHub OAuth App (for GitHub login and private repos)

To test GitHub sign-in and the **Own**/**Collaborator** tabs (which list repos
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
6. Restart the backend. **Continue with GitHub** (sign-in page), **Sign up with
   GitHub** (register page) and **Connect GitHub** (Own/Collaborator tabs) now work.

How GitHub sign-in behaves:

- All three buttons sign you in with a **GitHub-based account**: the first GitHub
  sign-in creates a new app user tied to your GitHub ID, later ones reuse it. This
  account is **separate from any local username/password account** — repositories you
  added under a local account don't appear in it.
- **Own** lists repositories owned by your GitHub user; **Collaborator** lists the
  ones you can access as a collaborator or organization member. Both together show the
  50 most recently updated repositories.
- The app requests the `read:user` and `repo` scopes. `repo` is the scope a GitHub
  OAuth App needs to read private repositories; it also allows writing, which the app
  never does.

## Optional: reranking

The cross-encoder reranker (off by default; see [Known limitations](#known-limitations)
for why) is the one model that does **not** run in Ollama, because Ollama has no
reranking API. It runs inside the backend through `sentence-transformers`, which is
therefore an optional extra:

```bash
# from backend/, with the virtual environment active
pip install -r requirements-rerank.txt
```

Then set `RERANK_ENABLED=true` in `.env` and restart the backend. The reranker model
(`RERANKER_MODEL_NAME`) is downloaded from the Hugging Face Hub on first use and cached
under `~/.cache/huggingface` (Windows: `%USERPROFILE%\.cache\huggingface`).

This installs the CPU build of PyTorch, which is enough to score `RERANK_CANDIDATES`
chunks per question. To run it on an NVIDIA GPU, replace it with a CUDA build
afterwards — RTX 50 series ("Blackwell") cards need CUDA 12.8 or newer, i.e. `cu128`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The version must end in `+cu128` and the second value must be `True`. Running
`pip install -r requirements-rerank.txt` again later can switch it back to the CPU build.

## Configuration reference

All settings live in `backend/.env` (copy `backend/.env.example` to start). Every
variable has a working default in code except where noted.

| Variable | Default | Purpose |
|---|---|---|
| `DB_USER` / `DB_PASS` / `DB_NAME` / `DB_HOST` | `postgres` / `postgres` / `postgres` / `localhost` | PostgreSQL connection, see [1.4](#14-connect-the-backend-to-the-database). `.env.example` sets `DB_NAME=rag_assistant`, matching step 1.3. A non-default port goes into `DB_HOST` (`localhost:5433`) |
| `JWT_SECRET_KEY` | `your_secret_key` | Signing key for login tokens. The default is only a placeholder, so always set your own random string |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Login session length |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | — | GitHub OAuth App credentials (optional, see above) |
| `GITHUB_REDIRECT_URI` | `http://127.0.0.1:8000/github/callback` | Must match the OAuth App's callback URL |
| `FRONTEND_URL` | `http://localhost:5173` | Where GitHub login redirects back to |
| `GITHUB_API_TOKEN` | — | Optional fallback personal access token, used for GitHub API calls when the signed-in user is a local (non-GitHub) account — raises the anonymous GitHub API rate limit and is required to index private repos without GitHub sign-in |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1200` / `200` | Text-splitting chunk size/overlap, in **characters** (not tokens) |
| `RAG_MAX_FILE_SIZE_BYTES` | `307200` (300 KB) | Files larger than this are skipped during ingestion |
| `RAG_MAX_FILES_PER_REPO` | `800` | Hard cap on files read per repository per ingestion run |
| `INDEXING_TIMEOUT_MINUTES` | `60` | A repository stuck in "indexing" for longer than this is marked as failed, so it can be started again (jobs interrupted by a server restart are failed immediately at startup) |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Where the Ollama server listens |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | Upper limit for one Ollama call (generous, because it can include loading a model) |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps a model loaded after its last use (Ollama's own default is 5 minutes) |
| `EMBEDDING_MODEL_NAME` | `qwen3-embedding:0.6b` | Ollama embedding model (must be pulled); see the rationale below |
| `EMBEDDING_DIMENSIONS` | `1024` | Must match the embedding model's vector size. When it changes, the backend drops the stored chunks at the next start and every repository has to be indexed again |
| `EMBEDDING_BATCH_SIZE` | `32` | Chunks sent to Ollama per embedding request while indexing |
| `EMBEDDING_QUERY_PREFIX` | Qwen3-Embedding's task instruction (`Instruct: … \nQuery:`) | Text put in front of every question before embedding. Model-specific: change it together with the embedding model |
| `EMBEDDING_DOCUMENT_PREFIX` | *(empty)* | Text put in front of every indexed chunk before embedding (Qwen3-Embedding needs none) |
| `RAG_TOP_K` | `6` | Number of chunks included in the final answer prompt |
| `RERANK_ENABLED` | `false` | Enables a second-pass cross-encoder reranking step; needs `requirements-rerank.txt` (see [Optional: reranking](#optional-reranking)) and is off by default for the reasons in [Known limitations](#known-limitations) |
| `RERANK_CANDIDATES` | `15` (`.env.example`: `25`) | How many chunks the initial vector search retrieves before reranking (only used when reranking is enabled) |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder model used for reranking (only used when reranking is enabled) |
| `CODE_FILE_BOOST` | `0.05` | Small score bonus for source-code files over documentation files in the final ranking |
| `DOC_FILE_PENALTY` | `0.08` | Small score penalty for documentation files (`.md`, `.mdx`, `.rst`, `.txt`) |
| `LLM_MODEL_NAME` | `qwen3.5:4b` | Ollama chat model used to generate answers (must be pulled); see the note on the LLM below |
| `LLM_MAX_NEW_TOKENS` | `512` | Maximum length of a generated answer, in tokens |
| `LLM_CONTEXT_LENGTH` | `8192` | Context window (prompt + answer, in tokens) requested from Ollama for each answer |
| `REPOSITORY_ACCESS_TTL_HOURS` | `24` | How long a recorded repo-access grant is trusted before it's re-verified live against the GitHub API |

**Why `qwen3-embedding:0.6b`:** the project's questions are asked in Hungarian while
the indexed content (source code, some documentation) is mostly English, so the
embedding model has to be both multilingual and good at code. Qwen3-Embedding covers
100+ languages including programming languages, is trained for code retrieval, and
has a 32k-token context window, so chunks are never truncated. It's in Ollama's model
library (the earlier `granite-embedding-97m-multilingual-r2` isn't). Its vectors have
1024 dimensions, well within the 2,000 that pgvector's HNSW index supports. It expects
a one-sentence task instruction in front of each *query* (not in front of documents),
which `EMBEDDING_QUERY_PREFIX` adds.

**About the LLM:** the tested configuration is `qwen3.5:4b` on an RTX 5070 (12 GB).
Ollama's build of it is 4-bit quantized (Q4_K_M), about 4 GB including a vision
component this project doesn't use. Any chat model from Ollama's library can be used
instead; three things are handled in `backend/app/rag/llm.py`:

- Qwen3 and Qwen3.5 models "think" (write a long reasoning block before the answer) by
  default, which used up the `LLM_MAX_NEW_TOKENS` budget and cut answers off. Every
  request sends `"think": false`. (Ollama only rejects `think: true` for models that
  can't think, so this is safe for any model.)
- Sampling settings are sent with every request (temperature 0.3, top-p 0.8, top-k 20)
  and override the model's own Ollama defaults — `qwen3.5` ships with temperature 1
  and `presence_penalty` 1.5. Repetition and presence penalties are set to off on
  purpose: they also penalize tokens copied from the context, which garbled file names
  and identifiers.
- The context window is set explicitly (`LLM_CONTEXT_LENGTH`), see
  [2.4](#24-how-the-backend-uses-ollama).

## How a question is answered, end to end

1. Browser sends `POST /chat/ask` with the question, the selected repo, and the auth
   token.
2. FastAPI resolves the current user from the token (`get_current_user` dependency).
3. `rag_service.answer_question`:
   - confirms the repository exists and the user has a recorded, still-valid access
     grant to it (re-verifying live against GitHub if the grant is older than
     `REPOSITORY_ACCESS_TTL_HOURS`),
   - confirms the repository has finished indexing,
   - embeds the question through Ollama with the same model used for indexing (plus
     `EMBEDDING_QUERY_PREFIX`),
   - runs a `pgvector` cosine-similarity search scoped to that repository,
   - if `RERANK_ENABLED=true`, reorders the candidates with a cross-encoder,
   - re-sorts the chunks with the source-code-vs-documentation bias
     (`CODE_FILE_BOOST` / `DOC_FILE_PENALTY`) and keeps the top `RAG_TOP_K`. With
     reranking off, the vector search already returns exactly `RAG_TOP_K` chunks, so
     the bias changes their order (in the prompt and in the source list), not which
     chunks are included,
   - builds a prompt from the retrieved chunks and generates the answer with the LLM
     in Ollama,
   - groups the chunks by file for the response (`sources: [{file_path, chunks: [...]}]`).
4. The frontend renders the answer with one source entry per file (listing its line
   ranges), and a right-hand code preview panel with Previous/Next navigation across a
   file's chunks, sorted by relevance.

## Known limitations

These are deliberate scope decisions or open issues, not oversights — documented here
for transparency:

- **The models are quantized.** Ollama's default model builds are quantized GGUF
  files; `qwen3.5:4b` is 4-bit (Q4_K_M). That's what lets the LLM and the embedding
  model fit side by side in 12 GB of VRAM, at a small quality cost compared to the
  full-precision weights. A less-quantized build can be used by pulling another tag of
  the model, if its Ollama page offers one.
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
  it is not a validated improvement for this codebase-and-language combination. It
  also can't run in Ollama (no reranking API), so it needs the extra
  `requirements-rerank.txt` (PyTorch) inside the backend.
- **No conversation history.** Each question is answered independently; earlier
  questions are never sent to the model as context. The conversation shown in the UI
  lives only in the open browser tab and is lost on reload; nothing is saved to the
  database.
- **Full re-indexing only.** Re-indexing a repository replaces all of its chunks; there
  is no incremental/delta indexing of only changed files.
- **One local Ollama instance.** The models run in Ollama, outside the backend
  process, so a model crash or out-of-memory error no longer takes the API down, and
  Ollama can be restarted, upgraded or moved to another machine (`OLLAMA_BASE_URL`)
  independently. But it is a single instance that by default handles one request per
  model at a time (`OLLAMA_NUM_PARALLEL`), so simultaneous questions and indexing jobs
  queue up. The optional reranker still runs inside the backend process. Indexing jobs
  also run as background tasks inside the backend, so a restart stops them (they are
  then marked as failed and can be retried).
- **No database migrations.** On startup the backend only *creates* missing tables; it
  never changes existing ones. After a model change, the affected tables have to be
  dropped by hand (see [1.4](#14-connect-the-backend-to-the-database)).
- **GitHub sign-in creates a separate account.** A local account can't be linked to a
  GitHub account; signing in with GitHub always uses a GitHub-based app user.
- **Only 50 GitHub repositories are listed.** The Own/Collaborator tabs show the 50
  most recently updated repositories; older ones can still be added by name in the
  Added tab.
- **Sessions expire after `ACCESS_TOKEN_EXPIRE_MINUTES` (30 by default).** There is no
  refresh token; when the token expires, the UI sends you back to the sign-in page.
- **GitHub access tokens are stored unencrypted** in the `users` table. That's
  acceptable for this local, single-user setup, not for a shared deployment.
- **`.env` changes require a full restart.** `uvicorn --reload` only watches `.py`
  files; editing `.env` while the server is running has no effect until the process is
  stopped and started again.
- **Embedding/chunking changes require re-indexing.** Changing `EMBEDDING_MODEL_NAME`,
  `RAG_CHUNK_SIZE`, or similar settings only affects *future* indexing runs — existing
  repositories keep their old chunks/vectors until "Re-index" is clicked again for
  each of them. Only a change of the vector size is detected (and the chunks dropped)
  automatically; the model name used for indexing isn't recorded.

## Troubleshooting

**`asyncpg.exceptions.FeatureNotSupportedError: extension "vector" is not available`**
on startup — the `pgvector` extension isn't installed on your PostgreSQL *server* (a
`pip install` alone never fixes this). Follow [Setup — 1](#setup--1-postgresql--pgvector)
for your OS.

**`'psql' is not recognized as an internal or external command`** (Windows) — the
PostgreSQL installer doesn't add `psql` to `PATH`. Call it by its full path
(`"C:\Program Files\PostgreSQL\17\bin\psql.exe"`, with `&` in front in PowerShell), or use
SQL Shell (psql) or pgAdmin from the Start menu.

**`password authentication failed for user "postgres"`** on startup — `DB_USER` /
`DB_PASS` in `backend/.env` don't match the database. On Linux, also check that a
password was set for `postgres` (see [1.1](#11-install-postgresql)). If the password
contains `@` or `%`, URL-encode it (see [1.4](#14-connect-the-backend-to-the-database)).

**`database "rag_assistant" does not exist`** — the database from
[1.3](#13-create-the-database) hasn't been created, or `DB_NAME` points somewhere else.
If `DB_NAME` is missing from `.env` entirely, the backend uses the `postgres` database.

**`ConnectionRefusedError`** on startup (`[WinError 1225]` on Windows, `[Errno 111]` on
Linux) — the PostgreSQL server isn't running, or it listens on a different port. On
Windows, start it in the Services app (`postgresql-x64-17` → Start). For a non-default
port, use `DB_HOST=localhost:<port>`.

**`permission denied to create extension "vector"`** — the backend connects as a role
without superuser rights. Run `CREATE EXTENSION vector;` once in the database as
`postgres`, then restart the backend.

**"Can't reach Ollama at http://127.0.0.1:11434"** (in the startup log, as a
repository's indexing error, or in the chat) — Ollama isn't running. Start the Ollama
app (Windows/macOS) or run `ollama serve`, then check `http://127.0.0.1:11434` in a
browser ("Ollama is running"). If Ollama runs on another machine or port, set
`OLLAMA_BASE_URL`.

**"The model "…" isn't available in Ollama. Download it with: ollama pull …"** — run
the command shown. The name in `.env` must match `ollama list` exactly, including the
tag after the colon.

**Answers are very slow** — run `ollama ps` while a question is being answered. If
`PROCESSOR` shows `CPU` or a CPU/GPU split, the model doesn't run (fully) on the GPU:
update the GPU driver, close other programs that use a lot of VRAM, or switch to a
smaller model. The first question after startup, or after `OLLAMA_KEEP_ALIVE` expired,
also includes loading the model.

**"The embedding model "…" returns N-dimensional vectors, but EMBEDDING_DIMENSIONS is
M"** — set `EMBEDDING_DIMENSIONS=N` in `.env` and restart. The backend then drops the
old chunks itself; index the repositories again.

**Startup log: "EMBEDDING_DIMENSIONS changed from … to …: dropped all indexed
chunks"** — expected after switching to an embedding model with a different vector
size. Every repository shows "Not indexed"; click **Index** on each one.

**Log warning about `LLM_CONTEXT_LENGTH`** — the prompt plus the answer didn't fit
into the context window, so part of the retrieved code may have been cut. Raise
`LLM_CONTEXT_LENGTH` (uses more VRAM), or lower `RAG_TOP_K` or `RAG_CHUNK_SIZE`.

**"Reranking is enabled (RERANK_ENABLED=true), but sentence-transformers isn't
installed"** — install the optional extra (see [Optional: reranking](#optional-reranking))
or set `RERANK_ENABLED=false`.

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

**A repository shows "Indexing failed" with "Indexing was interrupted because the server
restarted"** — indexing runs as a background task inside the backend process, so
stopping or restarting the backend mid-run (Ctrl+C, a crash, or `--reload` picking up
a `.py` change) kills the job. On the next startup, any repository still marked
"indexing" is marked as failed; click **Retry** to index it again. A job that hangs
while the server keeps running is failed the same way after
`INDEXING_TIMEOUT_MINUTES`. (This startup check assumes a single backend process —
the default `uvicorn` setup.)

**Adding or indexing a repo fails with "Repository not found, or you don't have access
to it."** — this is often correct behavior (a typo in the repo name, or a private repo
while signed in with a local account and no `GITHUB_API_TOKEN`), not a bug. Verify the
exact `owner/name` and, for private repos, sign in with GitHub or set
`GITHUB_API_TOKEN`.

**`Invalid GitHub OAuth state`** after approving the app on GitHub — the sign-in
started on one host and the callback arrived on another, so the browser didn't send
back the state cookie. The frontend starts the sign-in at `http://127.0.0.1:8000`, so
keep `127.0.0.1` (not `localhost`) in both `GITHUB_REDIRECT_URI` and the OAuth App's
callback URL.

**CORS errors in the browser console** — the backend's CORS policy
(`backend/app/main.py`) only allows `http://localhost:5173` by default. Open the
frontend at exactly that address (`http://127.0.0.1:5173` is a different origin and
gets blocked). If Vite starts on a different port (it will if 5173 is already in use —
check the terminal output), either free up port 5173 or add the actual port to
`allow_origins` in `main.py`.
