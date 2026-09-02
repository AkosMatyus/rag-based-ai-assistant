import { useEffect, useRef, useState } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Link,
  Navigate,
  useNavigate,
  useLocation,
} from "react-router-dom";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

type SourceChunk = {
  start_line: number | null;
  end_line: number | null;
  score: number;
  url: string | null;
  content: string;
};

type SourceGroup = {
  file_path: string;
  chunks: SourceChunk[];
};

type ChatResponse = {
  answer: string;
  sources?: SourceGroup[];
};

type RepoIndexStatus = "not_indexed" | "indexing" | "ready" | "error";

type RepoStatusResponse = {
  repo_full_name: string;
  status: RepoIndexStatus;
  file_count: number;
  chunk_count: number;
  indexed_at: string | null;
  error_message: string | null;
};

type LoginResponse = {
  message: string;
  username: string;
  access_token: string;
  access_token_type: string;
};

type GitHubRepo = {
  id: number;
  name: string;
  full_name: string;
  private: boolean;
  html_url: string;
  language: string | null;
  updated_at: string | null;
  affiliation: "owner" | "collaborator";
};

type RepoTab = "own" | "added" | "collaborator";

type SelectedRepo = {
  full_name: string;
};

function getToken() {
  return localStorage.getItem("access_token");
}

function saveToken(token: string) {
  localStorage.setItem("access_token", token);
}

function logout() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("username");
  localStorage.removeItem("selected_repo_full_name");
  localStorage.removeItem("active_repo_tab");
}

function getStoredSelectedRepo(): SelectedRepo | null {
  const fullName = localStorage.getItem("selected_repo_full_name");
  return fullName ? { full_name: fullName } : null;
}

function getStoredActiveTab(): RepoTab {
  const stored = localStorage.getItem("active_repo_tab");
  if (stored === "own" || stored === "added" || stored === "collaborator") {
    return stored;
  }
  return "own";
}

function indexStatusLabel(status?: RepoIndexStatus): string {
  switch (status) {
    case "ready":
      return "Kész";
    case "indexing":
      return "Indexelés folyamatban…";
    case "error":
      return "Hiba az indexelésnél";
    default:
      return "Nincs indexelve";
  }
}

function languageForFile(path: string): string {
  const extension = path.includes(".") ? path.split(".").pop()!.toLowerCase() : "";

  const map: Record<string, string> = {
    py: "python",
    js: "javascript",
    jsx: "jsx",
    ts: "typescript",
    tsx: "tsx",
    java: "java",
    go: "go",
    rs: "rust",
    rb: "ruby",
    php: "php",
    c: "c",
    h: "c",
    cpp: "cpp",
    hpp: "cpp",
    cs: "csharp",
    css: "css",
    scss: "scss",
    html: "markup",
    md: "markdown",
    json: "json",
    yaml: "yaml",
    yml: "yaml",
    sql: "sql",
    sh: "bash",
    bash: "bash",
  };

  return map[extension] ?? "text";
}

function sourceGroupLabel(group: SourceGroup): string {
  const sorted = [...group.chunks].sort(
    (a, b) => (a.start_line ?? 0) - (b.start_line ?? 0)
  );

  const ranges = sorted
    .map((chunk) =>
      chunk.start_line
        ? `${chunk.start_line}-${chunk.end_line ?? chunk.start_line}. sor`
        : null
    )
    .filter((range): range is string => range !== null);

  if (ranges.length === 0) return group.file_path;

  const visible = ranges.slice(0, 2);
  const suffix = ranges.length > 2 ? ", ..." : "";

  return `${group.file_path} (${visible.join(", ")}${suffix})`;
}

function Navbar() {
  const location = useLocation();

  const token = getToken();
  const username = localStorage.getItem("username");

  const isAuthPage =
    location.pathname === "/login" || location.pathname === "/register";

  return (
    <header className="navbar">
      <Link className="logo" to={token ? "/app" : "/"}>
        <span className="logo-mark">R</span>
        <span>RAG Assistant</span>
      </Link>

      <nav className="nav-links" aria-label="Fő navigáció">
        {token && !isAuthPage ? (
          <>
            <Link to="/app">Chat</Link>
            <span className="nav-user">{username}</span>
            <button
              className="nav-button"
              onClick={() => {
                logout();
                window.location.href = "/login";
              }}
            >
              Kilépés
            </button>
          </>
        ) : (
          <>
            <Link to="/">Főoldal</Link>
            <Link to="/login">Bejelentkezés</Link>
            <Link className="nav-register-link" to="/register">
              Regisztráció
            </Link>
          </>
        )}
      </nav>
    </header>
  );
}

function LandingPage() {
  return (
    <main className="page landing-page">
      <section className="landing-hero">
        <div className="eyebrow">Privát kódbázis-asszisztens</div>

        <h1>Kérdezz a repóidról emberi nyelven.</h1>

        <p>
          Jelentkezz be, válassz GitHub repót, és kérdezz rá a projekted
          működésére.
        </p>

        <div className="hero-actions">
          <Link className="primary-link" to="/login">
            Bejelentkezés
          </Link>

          <Link className="secondary-link" to="/register">
            Új fiók létrehozása
          </Link>
        </div>
      </section>

      <section className="feature-grid" aria-label="Funkciók">
        <article className="feature-card">
          <span className="feature-icon">01</span>
          <h2>GitHub kapcsolat</h2>
          <p>
            GitHub login után a backend lekéri az elérhető repókat, a frontend
            pedig egy bal oldali sidebarban mutatja őket.
          </p>
        </article>

        <article className="feature-card">
          <span className="feature-icon">02</span>
          <h2>Repo választás</h2>
          <p>
            A kiválasztott repo neve minden kérdésnél átmegy a backendnek, így
            később a RAG pontosan tudja, hol keressen.
          </p>
        </article>

        <article className="feature-card">
          <span className="feature-icon">03</span>
          <h2>Chat felület</h2>
          <p>
            Letisztult kérdés-válasz felület, forrásokkal. Erre építhető majd
            a valódi kódkeresés, chunkolás és AI válasz.
          </p>
        </article>
      </section>
    </main>
  );
}

function LoginPage() {
  const navigate = useNavigate();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleLogin(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/users/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username,
          password,
        }),
      });

      if (!response.ok) {
        setError("Hibás felhasználónév vagy jelszó.");
        return;
      }

      const data: LoginResponse = await response.json();

      saveToken(data.access_token);
      localStorage.setItem("username", data.username);

      navigate("/app");
    } catch {
      setError("Nem sikerült kapcsolódni a backendhez.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page auth-page">
      <section className="auth-shell">
        <aside className="auth-info-panel">
          <div>
            <div className="eyebrow">Belépés</div>
            <h1>Folytasd a saját RAG asszisztenseddel.</h1>
            <p>
              Jelentkezz be lokális felhasználóval vagy GitHubbal. GitHub login
              után a repóid automatikusan megjelennek a chat oldali sidebarban.
            </p>
          </div>

          <div className="mini-preview-card">
            <span>Aktív munkafolyamat</span>
            <strong>Login → repo választás → chat</strong>
          </div>
        </aside>

        <section className="auth-card">
          <h1>Bejelentkezés</h1>
          <p>Add meg a fiókod adatait.</p>

          <form onSubmit={handleLogin} className="auth-form">
            <label>
              Felhasználónév
              <input
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                placeholder="••••••••"
              />
            </label>

            <label>
              Jelszó
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="••••••••"
              />
            </label>

            {error && <div className="error-box">{error}</div>}

            <button className="ask-button" disabled={loading}>
              {loading ? "Belépés..." : "Belépés"}
            </button>
          </form>

          <div className="auth-divider">
            <span>vagy</span>
          </div>

          <button
            type="button"
            className="github-button"
            onClick={() => {
              window.location.href = `${API_URL}/github/login`;
            }}
          >
            Belépés GitHubbal
          </button>

          <p className="auth-bottom-text">
            Nincs még fiókod? <Link to="/register">Regisztrálj</Link>
          </p>
        </section>
      </section>
    </main>
  );
}

function RegisterPage() {
  const navigate = useNavigate();

  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleRegister(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setSuccess("");

    try {
      const response = await fetch(`${API_URL}/users/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          first_name: firstName,
          last_name: lastName,
          username,
          email,
          password,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        setError(data.detail ?? "Sikertelen regisztráció.");
        return;
      }

      setSuccess("Sikeres regisztráció. Átirányítás a login oldalra...");

      setTimeout(() => {
        navigate("/login");
      }, 900);
    } catch {
      setError("Nem sikerült kapcsolódni a backendhez.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page auth-page">
      <section className="auth-shell">
        <aside className="auth-info-panel">
          <div>
            <div className="eyebrow">Új fiók</div>
            <h1>Hozz létre egy saját felhasználót.</h1>
            <p>
              A regisztráció a saját backend adatbázisodba menti a usert. GitHub
              kapcsolatot később a GitHub bejelentkezés kezeli.
            </p>
          </div>

          <div className="mini-preview-card">
            <span>Backend</span>
            <strong>FastAPI + PostgreSQL + JWT</strong>
          </div>
        </aside>

        <section className="auth-card">
          <h1>Regisztráció</h1>
          <p>Töltsd ki az adatokat az új fiókhoz.</p>

          <form onSubmit={handleRegister} className="auth-form">
            <div className="two-columns">
              <label>
                Keresztnév
                <input
                  value={firstName}
                  onChange={(event) => setFirstName(event.target.value)}
                  placeholder="Ákos"
                />
              </label>

              <label>
                Vezetéknév
                <input
                  value={lastName}
                  onChange={(event) => setLastName(event.target.value)}
                  placeholder="Mátyus"
                />
              </label>
            </div>

            <label>
              Felhasználónév
              <input
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                placeholder="akos123"
              />
            </label>

            <label>
              Email
              <input
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="akos@example.com"
              />
            </label>

            <label>
              Jelszó
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="••••••••"
              />
            </label>

            {error && <div className="error-box">{error}</div>}
            {success && <div className="success-box">{success}</div>}

            <button className="ask-button" disabled={loading}>
              {loading ? "Regisztráció..." : "Regisztráció"}
            </button>
          </form>

          <p className="auth-bottom-text">
            Már van fiókod? <Link to="/login">Lépj be</Link>
          </p>
        </section>
      </section>
    </main>
  );
}

function ChatPage() {
  const token = getToken();

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SourceGroup[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [repos, setRepos] = useState<GitHubRepo[]>([]);
  const [selectedRepo, setSelectedRepo] = useState<SelectedRepo | null>(
    getStoredSelectedRepo
  );
  const [repoError, setRepoError] = useState("");
  const [reposLoading, setReposLoading] = useState(false);

  const [repoStatus, setRepoStatus] = useState<RepoStatusResponse | null>(null);
  const [ingestLoading, setIngestLoading] = useState(false);

  const [activeTab, setActiveTab] = useState<RepoTab>(getStoredActiveTab);

  const [addedRepos, setAddedRepos] = useState<RepoStatusResponse[]>([]);
  const [addedReposLoading, setAddedReposLoading] = useState(false);
  const [addedReposError, setAddedReposError] = useState("");

  const [newRepoInput, setNewRepoInput] = useState("");
  const [addRepoLoading, setAddRepoLoading] = useState(false);
  const [addRepoError, setAddRepoError] = useState("");

  const [selectedGroup, setSelectedGroup] = useState<SourceGroup | null>(null);
  const [selectedChunkIndex, setSelectedChunkIndex] = useState(0);

  useEffect(() => {
    async function loadRepos() {
      if (!token) return;

      setReposLoading(true);
      setRepoError("");

      try {
        const response = await fetch(`${API_URL}/github/repos`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          setRepoError(
            "Nem sikerült lekérni a GitHub repókat. Lehet, hogy nem GitHubbal vagy bejelentkezve."
          );
          return;
        }

        const data: GitHubRepo[] = await response.json();

        setRepos(data);

        setSelectedRepo((current) => {
          if (current) return current;
          return data.length > 0 ? { full_name: data[0].full_name } : null;
        });
      } catch {
        setRepoError("GitHub repo lekérés sikertelen.");
      } finally {
        setReposLoading(false);
      }
    }

    loadRepos();
  }, [token]);

  async function fetchRepoStatus(fullName: string) {
    if (!token) return;

    try {
      const response = await fetch(
        `${API_URL}/rag/status?repo_full_name=${encodeURIComponent(fullName)}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) return;

      const data: RepoStatusResponse = await response.json();
      setRepoStatus(data);
    } catch {
      // Csendben elnyeljük - ez csak egy háttérben futó státusz-lekérdezés.
    }
  }

  // A kiválasztott repót és a fület elmentjük, hogy oldal-újratöltés után
  // ne vesszen el - enélkül a lenti loadRepos mindig az első saját repóra
  // ugrott vissza, ami elfedte a ténylegesen indexelt repódat.
  useEffect(() => {
    if (selectedRepo) {
      localStorage.setItem("selected_repo_full_name", selectedRepo.full_name);
    }
  }, [selectedRepo?.full_name]);

  useEffect(() => {
    localStorage.setItem("active_repo_tab", activeTab);
  }, [activeTab]);

  // Amint másik repót választunk (vagy betöltéskor visszaáll a mentett
  // kiválasztás), azonnal lekérjük annak indexelési státuszát - enélkül a
  // repoStatus a régi (vagy null) értéken ragadt, amíg a pollozás el nem
  // indult, az viszont csak akkor indul, ha a repoStatus MÁR "indexing".
  useEffect(() => {
    setRepoStatus(null);

    if (selectedRepo) {
      fetchRepoStatus(selectedRepo.full_name);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRepo?.full_name]);

  // Amíg az indexelés folyamatban van, néhány másodpercenként újra lekérdezzük.
  useEffect(() => {
    if (repoStatus?.status !== "indexing" || !selectedRepo) return;

    const intervalId = setInterval(() => {
      fetchRepoStatus(selectedRepo.full_name);
    }, 3000);

    return () => clearInterval(intervalId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [repoStatus?.status, selectedRepo?.full_name]);

  async function loadAddedRepos() {
    if (!token) return;

    setAddedReposLoading(true);
    setAddedReposError("");

    try {
      const response = await fetch(`${API_URL}/rag/repos/mine`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        setAddedReposError("Nem sikerült lekérni a hozzáadott repókat.");
        return;
      }

      const data: RepoStatusResponse[] = await response.json();
      setAddedRepos(data);
    } catch {
      setAddedReposError("Hozzáadott repók lekérése sikertelen.");
    } finally {
      setAddedReposLoading(false);
    }
  }

  useEffect(() => {
    loadAddedRepos();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  async function addRepository() {
    if (!newRepoInput.trim() || !token) return;

    setAddRepoLoading(true);
    setAddRepoError("");

    try {
      const response = await fetch(`${API_URL}/rag/repos`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ repo_full_name: newRepoInput.trim() }),
      });

      if (!response.ok) {
        let detail = "Nem sikerült hozzáadni a repót.";
        try {
          const errorBody = await response.json();
          if (errorBody?.detail) detail = errorBody.detail;
        } catch {
          // A válasz nem JSON volt - marad az alapértelmezett üzenet.
        }
        setAddRepoError(detail);
        return;
      }

      const data: RepoStatusResponse = await response.json();
      setNewRepoInput("");
      await loadAddedRepos();
      setSelectedRepo({ full_name: data.repo_full_name });
    } catch {
      setAddRepoError("Nem sikerült kapcsolódni a backendhez.");
    } finally {
      setAddRepoLoading(false);
    }
  }

  async function startIndexing() {
    if (!selectedRepo || !token) return;

    setIngestLoading(true);

    try {
      const response = await fetch(`${API_URL}/rag/ingest`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ repo_full_name: selectedRepo.full_name }),
      });

      if (response.ok) {
        const data: RepoStatusResponse = await response.json();
        setRepoStatus(data);
      }
    } catch {
      // A fenti polling úgyis újrapróbálkozik, itt nincs teendő.
    } finally {
      setIngestLoading(false);
    }
  }

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  async function askQuestion() {
    if (!question.trim()) return;

    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);
    setSelectedGroup(null);
    setSelectedChunkIndex(0);

    try {
      const response = await fetch(`${API_URL}/chat/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          question,
          repo_full_name: selectedRepo?.full_name ?? null,
        }),
      });

      if (!response.ok) {
        let detail = "Nem sikerült választ kapni.";
        try {
          const errorBody = await response.json();
          if (errorBody?.detail) detail = errorBody.detail;
        } catch {
          // A válasz nem JSON volt - marad az alapértelmezett üzenet.
        }
        throw new Error(detail);
      }

      const data: ChatResponse = await response.json();

      setAnswer(data.answer);
      setSources(data.sources ?? []);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Hiba történt. Ellenőrizd, hogy fut-e a FastAPI backend."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.ctrlKey && event.key === "Enter") {
      askQuestion();
    }
  }

  
  function selectSourceGroup(group: SourceGroup) {
    setSelectedGroup(group);
    setSelectedChunkIndex(0);
  }

  const sortedSelectedChunks = selectedGroup
    ? [...selectedGroup.chunks].sort((a, b) => b.score - a.score)
    : [];
  const currentChunk = sortedSelectedChunks[selectedChunkIndex] ?? null;

  const ownRepos = repos.filter((repo) => repo.affiliation === "owner");
  const collaboratorRepos = repos.filter(
    (repo) => repo.affiliation === "collaborator"
  );

  const manuallyAddedRepos = addedRepos.filter(
    (added) => !repos.some((repo) => repo.full_name === added.repo_full_name)
  );

  return (
    <main className="workspace-page">
      <section className="workspace-header">
        <div>
          <p className="workspace-kicker">Repo-alapú kérdezés</p>
          <h1>RAG Chat</h1>
          <p>
            Válassz repót bal oldalt, majd kérdezz a kódbázis működéséről.
          </p>
        </div>

        {selectedRepo && (
          <div className="active-repo-card">
            <span>Aktív repo</span>
            <strong>{selectedRepo.full_name}</strong>

            <div className="index-status-row">
              <span
                className={`status-badge status-${repoStatus?.status ?? "not_indexed"}`}
              >
                {indexStatusLabel(repoStatus?.status)}
              </span>

              <button
                type="button"
                className="index-button"
                onClick={startIndexing}
                disabled={ingestLoading || repoStatus?.status === "indexing"}
              >
                {ingestLoading
                  ? "Indítás..."
                  : repoStatus?.status === "ready"
                  ? "Újraindexelés"
                  : "Indexelés indítása"}
              </button>
            </div>

            {repoStatus?.status === "ready" && (
              <small className="index-meta">
                {repoStatus.file_count} fájl · {repoStatus.chunk_count} chunk
                indexelve
              </small>
            )}

            {repoStatus?.status === "error" && repoStatus.error_message && (
              <small className="index-meta index-meta-error">
                {repoStatus.error_message}
              </small>
            )}
          </div>
        )}
      </section>

      <section className="workspace-layout">
        <aside className="repo-sidebar">
          <div className="sidebar-title-row">
            <div>
              <h2>GitHub repók</h2>
              <p>
                {activeTab === "own" && `${ownRepos.length} saját repo`}
                {activeTab === "collaborator" &&
                  `${collaboratorRepos.length} collaborator repo`}
                {activeTab === "added" && `${manuallyAddedRepos.length} hozzáadott repo`}
              </p>
            </div>
          </div>

          <div className="repo-tabs">
            <button
              type="button"
              className={activeTab === "own" ? "repo-tab active" : "repo-tab"}
              onClick={() => setActiveTab("own")}
            >
              Saját
            </button>
            <button
              type="button"
              className={
                activeTab === "added" ? "repo-tab active" : "repo-tab"
              }
              onClick={() => setActiveTab("added")}
            >
              Hozzáadott
            </button>
            <button
              type="button"
              className={
                activeTab === "collaborator" ? "repo-tab active" : "repo-tab"
              }
              onClick={() => setActiveTab("collaborator")}
            >
              Collaborator
            </button>
          </div>

          {activeTab !== "added" && reposLoading && (
            <p className="repo-muted">Repók betöltése...</p>
          )}

          {activeTab !== "added" && repoError && (
            <p className="repo-error">{repoError}</p>
          )}

          {activeTab === "own" && (
            <div className="repo-list">
              {ownRepos.map((repo) => (
                <button
                  key={repo.id}
                  title={repo.full_name}
                  className={
                    selectedRepo?.full_name === repo.full_name
                      ? "repo-item active"
                      : "repo-item"
                  }
                  onClick={() => setSelectedRepo({ full_name: repo.full_name })}
                >
                  <span>{repo.full_name}</span>
                  <small>
                    {repo.private ? "Privát" : "Publikus"}
                    {repo.language ? ` · ${repo.language}` : ""}
                  </small>
                </button>
              ))}
              {!reposLoading && !repoError && ownRepos.length === 0 && (
                <p className="repo-muted">Nincs saját repód.</p>
              )}
            </div>
          )}

          {activeTab === "collaborator" && (
            <div className="repo-list">
              {collaboratorRepos.map((repo) => (
                <button
                  key={repo.id}
                  title={repo.full_name}
                  className={
                    selectedRepo?.full_name === repo.full_name
                      ? "repo-item active"
                      : "repo-item"
                  }
                  onClick={() => setSelectedRepo({ full_name: repo.full_name })}
                >
                  <span>{repo.full_name}</span>
                  <small>
                    {repo.private ? "Privát" : "Publikus"}
                    {repo.language ? ` · ${repo.language}` : ""}
                  </small>
                </button>
              ))}
              {!reposLoading && !repoError && collaboratorRepos.length === 0 && (
                <p className="repo-muted">Nincs collaborator repód.</p>
              )}
            </div>
          )}

          {activeTab === "added" && (
            <>
              <div className="add-repo-row">
                <input
                  className="add-repo-input"
                  value={newRepoInput}
                  onChange={(event) => setNewRepoInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") addRepository();
                  }}
                  placeholder="pl. facebook/react"
                />
                <button
                  type="button"
                  className="add-repo-button"
                  onClick={addRepository}
                  disabled={addRepoLoading || !newRepoInput.trim()}
                >
                  {addRepoLoading ? "..." : "Hozzáadás"}
                </button>
              </div>

              {addRepoError && <p className="repo-error">{addRepoError}</p>}
              {addedReposLoading && (
                <p className="repo-muted">Betöltés...</p>
              )}
              {addedReposError && (
                <p className="repo-error">{addedReposError}</p>
              )}

              <div className="repo-list">
                {manuallyAddedRepos.map((repo) => (
                  <button
                    key={repo.repo_full_name}
                    title={repo.repo_full_name}
                    className={
                      selectedRepo?.full_name === repo.repo_full_name
                        ? "repo-item active"
                        : "repo-item"
                    }
                    onClick={() =>
                      setSelectedRepo({ full_name: repo.repo_full_name })
                    }
                  >
                    <span>{repo.repo_full_name}</span>
                    <small>{indexStatusLabel(repo.status)}</small>
                  </button>
                ))}
                {!addedReposLoading &&
                  !addedReposError &&
                  manuallyAddedRepos.length === 0 && (
                    <p className="repo-muted">Még nem adtál hozzá repót.</p>
                  )}
              </div>
            </>
          )}
        </aside>

        <section className="chat-panel">
          <div className="chat-panel-header">
            <div>
              <h2>Kérdés a kódbázishoz</h2>
              <p>
                {selectedRepo
                  ? `A kérdés ehhez a repóhoz lesz elküldve: ${selectedRepo.full_name}`
                  : "Válassz ki egy repót, vagy kérdezz repo nélkül."}
                {selectedRepo && repoStatus?.status !== "ready" && (
                  <>
                    {" "}
                    Előbb indítsd el az indexelést a jobb oldali gombbal, és
                    várd meg, amíg elkészül.
                  </>
                )}
              </p>
            </div>
          </div>

          <textarea
            className="question-input"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Például: Hol van az autentikációs logika?"
          />

          <div className="actions">
            <button
              className="ask-button"
              onClick={askQuestion}
              disabled={
                loading ||
                !question.trim() ||
                (!!selectedRepo && repoStatus?.status !== "ready")
              }
            >
              {loading ? "Válasz készül..." : "Kérdés küldése"}
            </button>

            <button
              className="clear-button"
              onClick={() => {
                setQuestion("");
                setAnswer("");
                setSources([]);
                setSelectedGroup(null);
                setSelectedChunkIndex(0);
                setError("");
              }}
            >
              Törlés
            </button>
          </div>

          {loading && (
            <div className="loading-box">
              <div className="loader"></div>
              <span>A backend dolgozik...</span>
            </div>
          )}

          {error && <div className="error-box">{error}</div>}

          {!answer && !loading && !error && (
            <div className="empty-answer">
              <h3>Még nincs válasz</h3>
              <p>
                Írj be egy kérdést, és a backend visszaküldi a választ. Később
                itt jelennek meg a RAG által megtalált források is.
              </p>
            </div>
          )}

          {answer && (
            <div className="answer-box">
              <div className="answer-header">
                <h2>Válasz</h2>
              </div>

              <p>{answer}</p>

              {sources.length > 0 && (
                <div className="sources">
                  <h3>Források</h3>

                  <div className="source-list">
                    {sources.map((group, index) => {
                      const bestScore = Math.max(
                        ...group.chunks.map((chunk) => chunk.score)
                      );

                      return (
                        <button
                          type="button"
                          key={index}
                          className={
                            selectedGroup === group
                              ? "source-pill active"
                              : "source-pill"
                          }
                          onClick={() => selectSourceGroup(group)}
                          title={`Legjobb relevancia: ${Math.round(bestScore * 100)}%`}
                        >
                          {sourceGroupLabel(group)}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
        
        <aside className="source-preview">
          {!selectedGroup && (
            <div className="source-preview-empty">
              <h3>Forrás előnézet</h3>
              <p>
                Kattints egy forrás pillére a válasz alatt, hogy itt
                megjelenjen a pontos kódrészlet.
              </p>
            </div>
          )}

          {selectedGroup && currentChunk && (
            <>
              <div className="source-preview-header">
                <div>
                  <h3>{selectedGroup.file_path}</h3>
                  {currentChunk.start_line && (
                    <p>
                      {currentChunk.start_line}-
                      {currentChunk.end_line ?? currentChunk.start_line}. sor
                      {" · "}
                      {Math.round(currentChunk.score * 100)}% relevancia
                    </p>
                  )}
                </div>

                {currentChunk.url && (
                  <a
                    className="source-preview-link"
                    href={currentChunk.url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    GitHub ↗
                  </a>
                )}
              </div>

              <SyntaxHighlighter
                language={languageForFile(selectedGroup.file_path)}
                style={oneDark}
                showLineNumbers
                startingLineNumber={currentChunk.start_line ?? 1}
                codeTagProps={{ style: { whiteSpace: "pre", display: "block" } }}
                customStyle={{ margin: 0, borderRadius: 12, fontSize: 13, flex: 1 }}
              >
                {currentChunk.content}
              </SyntaxHighlighter>

              {sortedSelectedChunks.length > 1 && (
                <div className="chunk-nav">
                  <button
                    type="button"
                    className="chunk-nav-button"
                    onClick={() => setSelectedChunkIndex((index) => index - 1)}
                    disabled={selectedChunkIndex === 0}
                  >
                    ← Previous chunk
                  </button>

                  <span className="chunk-nav-position">
                    {selectedChunkIndex + 1} / {sortedSelectedChunks.length}
                  </span>

                  <button
                    type="button"
                    className="chunk-nav-button"
                    onClick={() => setSelectedChunkIndex((index) => index + 1)}
                    disabled={selectedChunkIndex === sortedSelectedChunks.length - 1}
                  >
                    Next chunk →
                  </button>
                </div>
              )}
            </>
          )}
        </aside>
      </section>
    </main>
  );
}

function GitHubCallbackPage() {
  const navigate = useNavigate();
  const hasHandledCallback = useRef(false);

  useEffect(() => {
    if (hasHandledCallback.current) {
      return;
    }

    hasHandledCallback.current = true;

    const params = new URLSearchParams(window.location.search);

    const token = params.get("token");
    const username = params.get("username");

    if (token && username) {
      localStorage.setItem("access_token", token);
      localStorage.setItem("username", username);

      navigate("/app", { replace: true });
    } else {
      navigate("/login", { replace: true });
    }
  }, [navigate]);

  return (
    <main className="page auth-page">
      <section className="auth-card callback-card">
        <div className="loader large-loader"></div>
        <h1>GitHub bejelentkezés...</h1>
        <p>Átirányítás folyamatban.</p>
      </section>
    </main>
  );
}

function NotFoundPage() {
  return (
    <main className="page not-found-page">
      <section className="not-found-card">
        <span>404</span>
        <h1>Az oldal nem található.</h1>
        <p>Valószínűleg rossz URL-re navigáltál.</p>

        <Link className="primary-link" to="/">
          Vissza a főoldalra
        </Link>
      </section>
    </main>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Navbar />

      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/app" element={<ChatPage />} />
        <Route path="/github/callback" element={<GitHubCallbackPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;