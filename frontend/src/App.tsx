import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { FormEvent, KeyboardEvent, ReactNode } from "react";
import {
  BrowserRouter,
  Link,
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router-dom";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

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

type Theme = "light" | "dark";

type Turn = {
  id: number;
  repo: string;
  question: string;
  state: "pending" | "done" | "error";
  answer?: string;
  sources?: SourceGroup[];
  error?: string;
};

type ViewerState = {
  turnId: number;
  groupIndex: number;
  filePath: string;
  chunks: SourceChunk[]; // sorted by relevance, best first
  index: number;
};

/* ------------------------------------------------------------------ */
/* Storage + API helpers                                               */
/* ------------------------------------------------------------------ */

function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key: string, value: string | null) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // Storage can be unavailable (private mode); the app still works without it.
  }
}

function getToken() {
  return readStorage("access_token");
}

function logout() {
  writeStorage("access_token", null);
  writeStorage("username", null);
  writeStorage("selected_repo_full_name", null);
  writeStorage("active_repo_tab", null);
}

class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      // FastAPI validation errors: [{ loc, msg, type }, ...]
      return detail
        .map((item: { msg?: string }) => item?.msg)
        .filter(Boolean)
        .join(" ");
    }
  } catch {
    // Not JSON - fall through to the generic message.
  }
  return `Request failed with status ${response.status}.`;
}

async function api<T>(
  path: string,
  options: { method?: string; body?: unknown; token?: string | null } = {}
): Promise<T> {
  const headers: Record<string, string> = {};
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  if (options.token) headers.Authorization = `Bearer ${options.token}`;

  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      method: options.method ?? "GET",
      headers,
      body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
    });
  } catch {
    throw new ApiError(
      `Can't reach the backend at ${API_URL}. Check that the FastAPI server is running.`,
      0
    );
  }

  if (!response.ok) {
    throw new ApiError(await readErrorDetail(response), response.status);
  }

  return (await response.json()) as T;
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

/* ------------------------------------------------------------------ */
/* Formatting helpers                                                  */
/* ------------------------------------------------------------------ */

function splitRepoName(fullName: string): { owner: string; name: string } {
  const slash = fullName.indexOf("/");
  if (slash === -1) return { owner: "", name: fullName };
  return { owner: fullName.slice(0, slash), name: fullName.slice(slash + 1) };
}

function statusLabel(status: RepoIndexStatus): string {
  switch (status) {
    case "ready":
      return "Indexed";
    case "indexing":
      return "Indexing…";
    case "error":
      return "Indexing failed";
    default:
      return "Not indexed";
  }
}

function lineRange(chunk: SourceChunk): string | null {
  if (!chunk.start_line) return null;
  const end = chunk.end_line ?? chunk.start_line;
  return end === chunk.start_line ? `${chunk.start_line}` : `${chunk.start_line}–${end}`;
}

/** "lines 372–396, 402–417" - with ", …" only when there are 3 or more ranges. */
function groupRangesLabel(group: SourceGroup): string | null {
  const ranges = [...group.chunks]
    .sort((a, b) => (a.start_line ?? 0) - (b.start_line ?? 0))
    .map(lineRange)
    .filter((range): range is string => range !== null);

  if (ranges.length === 0) return null;

  const visible = ranges.slice(0, 2).join(", ");
  const suffix = ranges.length > 2 ? ", …" : "";
  const noun = ranges.length === 1 && !ranges[0].includes("–") ? "line" : "lines";
  return `${noun} ${visible}${suffix}`;
}

function languageForFile(path: string): string {
  const filename = path.split("/").pop() ?? path;
  const extension = filename.includes(".") ? filename.split(".").pop()!.toLowerCase() : "";

  const map: Record<string, string> = {
    py: "python",
    js: "javascript",
    mjs: "javascript",
    cjs: "javascript",
    jsx: "jsx",
    ts: "typescript",
    tsx: "tsx",
    java: "java",
    kt: "kotlin",
    go: "go",
    rs: "rust",
    rb: "ruby",
    php: "php",
    c: "c",
    h: "c",
    cpp: "cpp",
    cc: "cpp",
    hpp: "cpp",
    cs: "csharp",
    swift: "swift",
    css: "css",
    scss: "scss",
    html: "markup",
    xml: "markup",
    md: "markdown",
    json: "json",
    yaml: "yaml",
    yml: "yaml",
    toml: "toml",
    sql: "sql",
    sh: "bash",
    bash: "bash",
  };

  return map[extension] ?? "text";
}

/* ------------------------------------------------------------------ */
/* Theme                                                               */
/* ------------------------------------------------------------------ */

const ThemeContext = createContext<{ theme: Theme; toggle: () => void }>({
  theme: "light",
  toggle: () => {},
});

function initialTheme(): Theme {
  const fromDom = document.documentElement.dataset.theme;
  if (fromDom === "light" || fromDom === "dark") return fromDom;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    writeStorage("theme", theme);
  }, [theme]);

  const value = useMemo(
    () => ({ theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")) }),
    [theme]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

function useTheme() {
  return useContext(ThemeContext);
}

/* ------------------------------------------------------------------ */
/* Icons (inline, 16px, stroke = currentColor)                         */
/* ------------------------------------------------------------------ */

function Icon({ children, size = 16 }: { children: ReactNode; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

const IconSun = () => (
  <Icon>
    <circle cx="8" cy="8" r="3" />
    <path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1" />
  </Icon>
);
const IconMoon = () => (
  <Icon>
    <path d="M13.5 9.5A5.5 5.5 0 0 1 6.5 2.5a5.5 5.5 0 1 0 7 7Z" />
  </Icon>
);
const IconSend = () => (
  <Icon>
    <path d="M8 13V3M3.5 7.5 8 3l4.5 4.5" />
  </Icon>
);
const IconRefresh = () => (
  <Icon size={14}>
    <path d="M13.5 8A5.5 5.5 0 1 1 11.9 4.1" />
    <path d="M13.5 2.5v3h-3" />
  </Icon>
);
const IconChevronLeft = () => (
  <Icon size={14}>
    <path d="M10 3.5 5.5 8l4.5 4.5" />
  </Icon>
);
const IconChevronRight = () => (
  <Icon size={14}>
    <path d="M6 3.5 10.5 8 6 12.5" />
  </Icon>
);
const IconExternal = () => (
  <Icon size={14}>
    <path d="M9.5 2.5h4v4M13.5 2.5 7.5 8.5M12 9.5v3a1 1 0 0 1-1 1H3.5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h3" />
  </Icon>
);
const IconClose = () => (
  <Icon>
    <path d="M4 4l8 8M12 4l-8 8" />
  </Icon>
);
const IconMenu = () => (
  <Icon>
    <path d="M2.5 4h11M2.5 8h11M2.5 12h11" />
  </Icon>
);
const IconLock = () => (
  <Icon size={12}>
    <rect x="3.5" y="7" width="9" height="6.5" rx="1" />
    <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" />
  </Icon>
);
const IconFile = () => (
  <Icon size={14}>
    <path d="M9 1.5H4.5a1 1 0 0 0-1 1v11a1 1 0 0 0 1 1h7a1 1 0 0 0 1-1V5Z" />
    <path d="M9 1.5V5h3.5" />
  </Icon>
);
const IconGitHub = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
  </svg>
);

function Spinner() {
  return <span className="spinner" aria-hidden="true" />;
}

/* ------------------------------------------------------------------ */
/* Shared chrome                                                       */
/* ------------------------------------------------------------------ */

function LogoMark() {
  // Three lines of "code" with one highlighted - a retrieved snippet.
  return (
    <svg className="logo-mark" width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
      <rect x="0.5" y="0.5" width="21" height="21" rx="4" fill="var(--ink)" />
      <rect x="5" y="6" width="9" height="2" rx="1" fill="var(--canvas)" opacity="0.55" />
      <rect x="5" y="10" width="12" height="2" rx="1" fill="var(--accent)" />
      <rect x="5" y="14" width="7" height="2" rx="1" fill="var(--canvas)" opacity="0.55" />
    </svg>
  );
}

function Logo({ to }: { to: string }) {
  return (
    <Link className="logo" to={to}>
      <LogoMark />
      <span>RAG Assistant</span>
    </Link>
  );
}

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      className="icon-btn"
      onClick={toggle}
      aria-label={`Switch to ${next} mode`}
      title={`Switch to ${next} mode`}
    >
      {theme === "dark" ? <IconSun /> : <IconMoon />}
    </button>
  );
}

function PublicHeader() {
  const location = useLocation();
  const signedIn = !!getToken();

  return (
    <header className="public-header">
      <Logo to={signedIn ? "/app" : "/"} />
      <nav className="public-nav" aria-label="Main">
        <ThemeToggle />
        {signedIn ? (
          <Link className="btn btn-primary" to="/app">
            Open workspace
          </Link>
        ) : (
          <>
            {location.pathname !== "/login" && (
              <Link className="btn btn-ghost" to="/login">
                Sign in
              </Link>
            )}
            {location.pathname !== "/register" && (
              <Link className="btn btn-primary" to="/register">
                Create account
              </Link>
            )}
          </>
        )}
      </nav>
    </header>
  );
}

/* ------------------------------------------------------------------ */
/* Minimal Markdown renderer for answers                               */
/* Handles fenced code, headings, lists, paragraphs, `code`, **bold**. */
/* Builds React elements directly - no HTML injection.                 */
/* ------------------------------------------------------------------ */

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const parts = text.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/g);
  return parts.map((part, index) => {
    const key = `${keyPrefix}-${index}`;
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={key}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

function renderTextBlocks(text: string, keyPrefix: string): ReactNode[] {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let paragraph: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;

  const flushParagraph = () => {
    if (paragraph.length === 0) return;
    const key = `${keyPrefix}-p${blocks.length}`;
    blocks.push(<p key={key}>{renderInline(paragraph.join(" "), key)}</p>);
    paragraph = [];
  };
  const flushList = () => {
    if (!list) return;
    const key = `${keyPrefix}-l${blocks.length}`;
    const items = list.items.map((item, i) => (
      <li key={`${key}-${i}`}>{renderInline(item, `${key}-${i}`)}</li>
    ));
    blocks.push(list.ordered ? <ol key={key}>{items}</ol> : <ul key={key}>{items}</ul>);
    list = null;
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);

    if (line.trim() === "") {
      flushParagraph();
      flushList();
    } else if (heading) {
      flushParagraph();
      flushList();
      const key = `${keyPrefix}-h${blocks.length}`;
      blocks.push(<h4 key={key}>{renderInline(heading[1], key)}</h4>);
    } else if (bullet || numbered) {
      flushParagraph();
      const ordered = !bullet;
      if (list && list.ordered !== ordered) flushList();
      if (!list) list = { ordered, items: [] };
      list.items.push((bullet ?? numbered)![1]);
    } else if (list && /^\s{2,}\S/.test(raw)) {
      // Indented continuation of the previous list item.
      list.items[list.items.length - 1] += ` ${line.trim()}`;
    } else {
      flushList();
      paragraph.push(line.trim());
    }
  }
  flushParagraph();
  flushList();
  return blocks;
}

function Markdown({ text }: { text: string }) {
  const { theme } = useTheme();
  const segments = text.split(/```([\w+-]*)[^\n]*\n([\s\S]*?)(?:```|$)/g);

  const nodes: ReactNode[] = [];
  for (let i = 0; i < segments.length; i += 3) {
    const prose = segments[i];
    if (prose) nodes.push(...renderTextBlocks(prose, `t${i}`));

    const code = segments[i + 2];
    if (code !== undefined) {
      const language = segments[i + 1] || "text";
      nodes.push(
        <div className="md-code" key={`c${i}`}>
          <SyntaxHighlighter
            language={language}
            style={theme === "dark" ? oneDark : oneLight}
            codeTagProps={{ style: { whiteSpace: "pre", display: "block", fontFamily: "var(--font-mono)" } }}
            customStyle={{
              margin: 0,
              padding: "12px 14px",
              background: "var(--code-bg)",
              fontSize: 12.5,
              borderRadius: 0,
            }}
          >
            {code.replace(/\n$/, "")}
          </SyntaxHighlighter>
        </div>
      );
    }
  }

  return <div className="md">{nodes}</div>;
}

/* ------------------------------------------------------------------ */
/* Landing                                                             */
/* ------------------------------------------------------------------ */

function LandingPage() {
  return (
    <div className="public-page">
      <PublicHeader />

      <main className="landing">
        <section className="landing-intro">
          <h1>Ask questions about a GitHub repository and see the code behind every answer.</h1>
          <p className="landing-lede">
            Index a repository once, then ask in plain language. Answers are written by a
            model that runs on your own machine, and each one points to the exact files and
            lines it was based on.
          </p>
          <div className="landing-actions">
            <Link className="btn btn-primary btn-lg" to="/register">
              Create account
            </Link>
            <Link className="btn btn-secondary btn-lg" to="/login">
              Sign in
            </Link>
          </div>
        </section>

        <figure className="landing-preview" aria-label="Example answer with its source">
          <div className="preview-question">Where is the Session behavior defined?</div>
          <div className="preview-answer">
            <p>
              The <code>Session</code> class is defined in <code>src/requests/sessions.py</code>.
              It keeps settings such as headers, cookies and auth across requests, and{" "}
              <code>session()</code> returns a new instance.
            </p>
            <div className="preview-source">
              <IconFile />
              <span className="path">src/requests/sessions.py</span>
              <span className="ranges">lines 395–420, 908–920</span>
            </div>
          </div>
          <pre className="preview-code">
            <code>
              <span className="ln">908</span>
              {"def session() -> Session:\n"}
              <span className="ln">909</span>
              {'    """Returns a Session for context-management."""\n'}
              <span className="ln">910</span>
              {"    return Session()"}
            </code>
          </pre>
        </figure>
      </main>

      <section className="steps" aria-label="How it works">
        <ol>
          <li>
            <span className="step-number">1</span>
            <h2>Pick a repository</h2>
            <p>Sign in with GitHub to list your own repositories, or add any public one by name.</p>
          </li>
          <li>
            <span className="step-number">2</span>
            <h2>Index it</h2>
            <p>The code is split into chunks and embedded into a local PostgreSQL + pgvector database.</p>
          </li>
          <li>
            <span className="step-number">3</span>
            <h2>Ask and verify</h2>
            <p>Open any source under an answer to read the exact lines the model was given.</p>
          </li>
        </ol>
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Auth pages                                                          */
/* ------------------------------------------------------------------ */

function AuthLayout({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
}) {
  return (
    <div className="public-page">
      <PublicHeader />
      <main className="auth">
        <section className="auth-card">
          <h1>{title}</h1>
          <p className="auth-subtitle">{subtitle}</p>
          {children}
        </section>
      </main>
    </div>
  );
}

function GitHubButton({ label }: { label: string }) {
  return (
    <button
      type="button"
      className="btn btn-secondary btn-block"
      onClick={() => {
        window.location.href = `${API_URL}/github/login`;
      }}
    >
      <IconGitHub />
      {label}
    </button>
  );
}

function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const sessionExpired = new URLSearchParams(location.search).get("expired") === "1";

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  if (getToken()) return <Navigate to="/app" replace />;

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const data = await api<LoginResponse>("/users/login", {
        method: "POST",
        body: { username, password },
      });
      writeStorage("access_token", data.access_token);
      writeStorage("username", data.username);
      navigate("/app");
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 401
          ? "Incorrect username or password."
          : errorMessage(err, "Sign-in failed.")
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthLayout title="Sign in" subtitle="Use your local account or continue with GitHub.">
      {sessionExpired && !error && (
        <div className="notice">Your session expired. Sign in again to continue.</div>
      )}

      <form onSubmit={handleLogin} className="form">
        <label className="field">
          <span>Username</span>
          <input
            className="input"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            autoComplete="username"
            required
          />
        </label>

        <label className="field">
          <span>Password</span>
          <input
            className="input"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            required
          />
        </label>

        {error && <div className="notice notice-error">{error}</div>}

        <button className="btn btn-primary btn-block" disabled={loading}>
          {loading ? "Signing in…" : "Sign in"}
        </button>
      </form>

      <div className="divider">
        <span>or</span>
      </div>

      <GitHubButton label="Continue with GitHub" />

      <p className="auth-footer">
        No account yet? <Link to="/register">Create one</Link>
      </p>
    </AuthLayout>
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

  async function handleRegister(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setSuccess("");

    try {
      await api("/users/register", {
        method: "POST",
        body: { first_name: firstName, last_name: lastName, username, email, password },
      });
      setSuccess("Account created. Taking you to sign in…");
      setTimeout(() => navigate("/login"), 900);
    } catch (err) {
      setError(errorMessage(err, "Couldn't create the account."));
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthLayout title="Create account" subtitle="Stored in your own backend database.">
      <form onSubmit={handleRegister} className="form">
        <div className="field-row">
          <label className="field">
            <span>First name</span>
            <input
              className="input"
              value={firstName}
              onChange={(event) => setFirstName(event.target.value)}
              autoComplete="given-name"
              required
            />
          </label>
          <label className="field">
            <span>Last name</span>
            <input
              className="input"
              value={lastName}
              onChange={(event) => setLastName(event.target.value)}
              autoComplete="family-name"
              required
            />
          </label>
        </div>

        <label className="field">
          <span>Username</span>
          <input
            className="input"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            autoComplete="username"
            minLength={3}
            maxLength={50}
            required
          />
        </label>

        <label className="field">
          <span>Email</span>
          <input
            className="input"
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            autoComplete="email"
            required
          />
        </label>

        <label className="field">
          <span>Password</span>
          <input
            className="input"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="new-password"
            required
          />
        </label>

        {error && <div className="notice notice-error">{error}</div>}
        {success && <div className="notice notice-success">{success}</div>}

        <button className="btn btn-primary btn-block" disabled={loading}>
          {loading ? "Creating account…" : "Create account"}
        </button>
      </form>

      <div className="divider">
        <span>or</span>
      </div>

      <GitHubButton label="Sign up with GitHub" />

      <p className="auth-footer">
        Already have an account? <Link to="/login">Sign in</Link>
      </p>
    </AuthLayout>
  );
}

function GitHubCallbackPage() {
  const navigate = useNavigate();
  const handled = useRef(false);

  useEffect(() => {
    if (handled.current) return;
    handled.current = true;

    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    const username = params.get("username");

    if (token && username) {
      writeStorage("access_token", token);
      writeStorage("username", username);
      navigate("/app", { replace: true });
    } else {
      navigate("/login", { replace: true });
    }
  }, [navigate]);

  return (
    <div className="public-page">
      <main className="auth">
        <section className="auth-card auth-card-center">
          <Spinner />
          <h1>Signing you in with GitHub…</h1>
        </section>
      </main>
    </div>
  );
}

function NotFoundPage() {
  return (
    <div className="public-page">
      <PublicHeader />
      <main className="auth">
        <section className="auth-card">
          <h1>Page not found</h1>
          <p className="auth-subtitle">There's nothing at this address.</p>
          <Link className="btn btn-primary" to="/">
            Go to the home page
          </Link>
        </section>
      </main>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Workspace: repository list                                          */
/* ------------------------------------------------------------------ */

function RepoRow({
  fullName,
  meta,
  status,
  starting,
  error,
  selected,
  onSelect,
  onIndex,
}: {
  fullName: string;
  meta: ReactNode;
  status: RepoIndexStatus;
  starting: boolean;
  error?: string;
  selected: boolean;
  onSelect: () => void;
  onIndex: () => void;
}) {
  const { owner, name } = splitRepoName(fullName);
  const busy = starting || status === "indexing";

  let action: ReactNode;
  if (busy) {
    action = (
      <span className="repo-action repo-action-icon is-busy" role="status" aria-label="Indexing">
        <Spinner />
      </span>
    );
  } else if (status === "ready") {
    action = (
      <button
        type="button"
        className="repo-action repo-action-icon"
        onClick={onIndex}
        aria-label={`Re-index ${fullName}`}
        title="Re-index"
      >
        <IconRefresh />
      </button>
    );
  } else {
    action = (
      <button
        type="button"
        className={`repo-action${status === "error" ? " is-error" : ""}`}
        onClick={onIndex}
        aria-label={`${status === "error" ? "Retry indexing" : "Index"} ${fullName}`}
        title={status === "error" ? "Indexing failed. Try again." : "Index this repository"}
      >
        {status === "error" ? "Retry" : "Index"}
      </button>
    );
  }

  const shownStatus: RepoIndexStatus = busy ? "indexing" : status;

  return (
    <li className={`repo-row${selected ? " is-selected" : ""}`}>
      <button
        type="button"
        className="repo-select"
        onClick={onSelect}
        aria-current={selected ? "true" : undefined}
        title={fullName}
      >
        <span className="repo-name">
          {owner && <span className="repo-owner">{owner}/</span>}
          {name}
        </span>
        <span className="repo-meta">
          <span className={`status status-${shownStatus}`}>
            <span className="status-dot" />
            {statusLabel(shownStatus)}
          </span>
          {meta}
        </span>
        {error && <span className="repo-error">{error}</span>}
      </button>
      {action}
    </li>
  );
}

/* ------------------------------------------------------------------ */
/* Workspace                                                           */
/* ------------------------------------------------------------------ */

const SAMPLE_QUESTIONS = [
  "Where is authentication handled?",
  "How is configuration loaded?",
  "What happens when the application starts?",
  "Which files define the data models?",
];

function WorkspacePage() {
  const navigate = useNavigate();
  const token = getToken();
  const username = readStorage("username") ?? "";

  const [repos, setRepos] = useState<GitHubRepo[]>([]);
  const [reposLoading, setReposLoading] = useState(true);
  const [reposError, setReposError] = useState<string | null>(null);
  const [githubLinked, setGithubLinked] = useState(true);

  // Every repo the user has access to inside the app, with its index status.
  const [tracked, setTracked] = useState<RepoStatusResponse[]>([]);
  const [trackedLoading, setTrackedLoading] = useState(true);
  const [trackedError, setTrackedError] = useState<string | null>(null);

  const [selectedRepo, setSelectedRepo] = useState<string | null>(() =>
    readStorage("selected_repo_full_name")
  );
  const [activeTab, setActiveTab] = useState<RepoTab>(() => {
    const stored = readStorage("active_repo_tab");
    return stored === "added" || stored === "collaborator" ? stored : "own";
  });
  const [filter, setFilter] = useState("");

  const [newRepoInput, setNewRepoInput] = useState("");
  const [addRepoLoading, setAddRepoLoading] = useState(false);
  const [addRepoError, setAddRepoError] = useState("");

  const [startingIndex, setStartingIndex] = useState<Set<string>>(() => new Set());
  const [indexErrors, setIndexErrors] = useState<Record<string, string>>({});

  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [viewer, setViewer] = useState<ViewerState | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const nextTurnId = useRef(1);
  const threadEndRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);

  /** Returns true if the error was an expired/invalid session (and redirects). */
  const handleAuthError = useCallback(
    (err: unknown) => {
      if (err instanceof ApiError && err.status === 401) {
        logout();
        navigate("/login?expired=1", { replace: true });
        return true;
      }
      return false;
    },
    [navigate]
  );

  // GitHub repositories (own + collaborator). Local accounts get a 400 here.
  useEffect(() => {
    if (!token) return;
    let cancelled = false;

    (async () => {
      try {
        const data = await api<GitHubRepo[]>("/github/repos", { token });
        if (cancelled) return;
        setRepos(data);
        setGithubLinked(true);
        setSelectedRepo((current) => current ?? data[0]?.full_name ?? null);
      } catch (err) {
        if (cancelled || handleAuthError(err)) return;
        if (err instanceof ApiError && err.status === 400) {
          setGithubLinked(false);
        } else {
          setReposError(errorMessage(err, "Couldn't load your GitHub repositories."));
        }
      } finally {
        if (!cancelled) setReposLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [token, handleAuthError]);

  const loadTracked = useCallback(
    async (silent: boolean) => {
      if (!token) return;
      try {
        const data = await api<RepoStatusResponse[]>("/rag/repos/mine", { token });
        setTracked(data);
        if (!silent) setTrackedError(null);
      } catch (err) {
        if (handleAuthError(err)) return;
        if (!silent) setTrackedError(errorMessage(err, "Couldn't load your repositories."));
      } finally {
        if (!silent) setTrackedLoading(false);
      }
    },
    [token, handleAuthError]
  );

  useEffect(() => {
    loadTracked(false);
  }, [loadTracked]);

  // While anything is indexing, refresh statuses every few seconds.
  const anyIndexing = tracked.some((repo) => repo.status === "indexing");
  useEffect(() => {
    if (!anyIndexing) return;
    const id = setInterval(() => loadTracked(true), 3000);
    return () => clearInterval(id);
  }, [anyIndexing, loadTracked]);

  useEffect(() => {
    writeStorage("selected_repo_full_name", selectedRepo);
  }, [selectedRepo]);

  useEffect(() => {
    writeStorage("active_repo_tab", activeTab);
  }, [activeTab]);

  // Local accounts have no GitHub list - start them on the "Added" tab.
  useEffect(() => {
    if (!githubLinked && activeTab !== "added") setActiveTab("added");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [githubLinked]);

  const statusByRepo = useMemo(
    () => new Map(tracked.map((repo) => [repo.repo_full_name, repo])),
    [tracked]
  );

  const visibleTurns = useMemo(
    () => turns.filter((turn) => turn.repo === selectedRepo),
    [turns, selectedRepo]
  );

  // Grow the composer with its content (up to the CSS max-height).
  useEffect(() => {
    const el = composerRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [draft]);

  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ block: "end" });
  }, [visibleTurns.length, visibleTurns[visibleTurns.length - 1]?.state]);

  if (!token) return <Navigate to="/login" replace />;

  /* ---------- derived ---------- */

  const statusOf = (fullName: string): RepoIndexStatus =>
    statusByRepo.get(fullName)?.status ?? "not_indexed";

  const selectedInfo = selectedRepo ? statusByRepo.get(selectedRepo) ?? null : null;
  const selectedStatus: RepoIndexStatus =
    selectedRepo && startingIndex.has(selectedRepo)
      ? "indexing"
      : selectedRepo
        ? statusOf(selectedRepo)
        : "not_indexed";

  const anyPending = turns.some((turn) => turn.state === "pending");
  const canAsk = !!selectedRepo && selectedStatus === "ready" && !anyPending;

  const query = filter.trim().toLowerCase();
  const matches = (fullName: string) => !query || fullName.toLowerCase().includes(query);

  const ownRepos = repos.filter((r) => r.affiliation === "owner" && matches(r.full_name));
  const collaboratorRepos = repos.filter(
    (r) => r.affiliation === "collaborator" && matches(r.full_name)
  );
  // "Added" = repos added by name that aren't already in the GitHub lists.
  const addedRepos = tracked.filter(
    (t) => !repos.some((r) => r.full_name === t.repo_full_name) && matches(t.repo_full_name)
  );

  /* ---------- actions ---------- */

  function upsertTracked(info: RepoStatusResponse) {
    setTracked((prev) => {
      const index = prev.findIndex((r) => r.repo_full_name === info.repo_full_name);
      if (index === -1) {
        return [...prev, info].sort((a, b) => a.repo_full_name.localeCompare(b.repo_full_name));
      }
      const next = [...prev];
      next[index] = info;
      return next;
    });
  }

  function selectRepo(fullName: string) {
    setSelectedRepo(fullName);
    setViewer(null);
    setSidebarOpen(false);
  }

  async function indexRepo(fullName: string) {
    setIndexErrors((prev) => {
      const next = { ...prev };
      delete next[fullName];
      return next;
    });
    setStartingIndex((prev) => new Set(prev).add(fullName));

    try {
      const info = await api<RepoStatusResponse>("/rag/ingest", {
        method: "POST",
        token,
        body: { repo_full_name: fullName },
      });
      upsertTracked(info);
    } catch (err) {
      if (handleAuthError(err)) return;
      setIndexErrors((prev) => ({
        ...prev,
        [fullName]: errorMessage(err, "Couldn't start indexing."),
      }));
    } finally {
      setStartingIndex((prev) => {
        const next = new Set(prev);
        next.delete(fullName);
        return next;
      });
    }
  }

  async function addRepository(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const name = newRepoInput.trim();
    if (!name) return;

    setAddRepoLoading(true);
    setAddRepoError("");

    try {
      const info = await api<RepoStatusResponse>("/rag/repos", {
        method: "POST",
        token,
        body: { repo_full_name: name },
      });
      upsertTracked(info);
      setNewRepoInput("");
      setFilter("");
      selectRepo(info.repo_full_name);
    } catch (err) {
      if (handleAuthError(err)) return;
      setAddRepoError(errorMessage(err, "Couldn't add the repository."));
    } finally {
      setAddRepoLoading(false);
    }
  }

  async function ask(text: string) {
    const question = text.trim();
    if (!question || !selectedRepo || !canAsk) return;

    const id = nextTurnId.current++;
    const repo = selectedRepo;
    setTurns((prev) => [...prev, { id, repo, question, state: "pending" }]);
    setDraft("");

    try {
      const data = await api<ChatResponse>("/chat/ask", {
        method: "POST",
        token,
        body: { question, repo_full_name: repo },
      });
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id ? { ...t, state: "done", answer: data.answer, sources: data.sources ?? [] } : t
        )
      );
    } catch (err) {
      if (handleAuthError(err)) return;
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id
            ? { ...t, state: "error", error: errorMessage(err, "Couldn't get an answer.") }
            : t
        )
      );
    }
  }

  function retryTurn(turn: Turn) {
    setTurns((prev) => prev.filter((t) => t.id !== turn.id));
    ask(turn.question);
  }

  function clearConversation() {
    setTurns((prev) => prev.filter((t) => t.repo !== selectedRepo));
    setViewer(null);
  }

  function openSource(turn: Turn, groupIndex: number) {
    const group = turn.sources?.[groupIndex];
    if (!group) return;
    setViewer({
      turnId: turn.id,
      groupIndex,
      filePath: group.file_path,
      chunks: [...group.chunks].sort((a, b) => b.score - a.score),
      index: 0,
    });
  }

  function fillSample(question: string) {
    setDraft(question);
    composerRef.current?.focus();
  }

  function handleComposerKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      ask(draft);
    }
  }

  /* ---------- render pieces ---------- */

  const githubRepoMeta = (repo: GitHubRepo) => (
    <>
      {repo.private && (
        <span className="repo-tag">
          <IconLock />
          Private
        </span>
      )}
      {repo.language && <span className="repo-tag">{repo.language}</span>}
    </>
  );

  const renderGithubList = (list: GitHubRepo[], emptyText: string) => {
    if (!githubLinked) {
      return (
        <div className="list-empty">
          <p>Connect your GitHub account to list the repositories you own or contribute to.</p>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => {
              window.location.href = `${API_URL}/github/login`;
            }}
          >
            <IconGitHub />
            Connect GitHub
          </button>
        </div>
      );
    }
    if (reposLoading) return <p className="list-empty">Loading repositories…</p>;
    if (reposError) return <p className="list-empty list-error">{reposError}</p>;
    if (list.length === 0) {
      return <p className="list-empty">{query ? "No repositories match the filter." : emptyText}</p>;
    }
    return (
      <ul className="repo-list">
        {list.map((repo) => (
          <RepoRow
            key={repo.id}
            fullName={repo.full_name}
            meta={githubRepoMeta(repo)}
            status={statusOf(repo.full_name)}
            starting={startingIndex.has(repo.full_name)}
            error={indexErrors[repo.full_name]}
            selected={selectedRepo === repo.full_name}
            onSelect={() => selectRepo(repo.full_name)}
            onIndex={() => indexRepo(repo.full_name)}
          />
        ))}
      </ul>
    );
  };

  const renderAddedList = () => {
    if (trackedLoading) return <p className="list-empty">Loading repositories…</p>;
    if (trackedError) return <p className="list-empty list-error">{trackedError}</p>;
    if (addedRepos.length === 0) {
      return (
        <p className="list-empty">
          {query
            ? "No repositories match the filter."
            : "Add any repository you can access by its owner/name, for example psf/requests."}
        </p>
      );
    }
    return (
      <ul className="repo-list">
        {addedRepos.map((repo) => (
          <RepoRow
            key={repo.repo_full_name}
            fullName={repo.repo_full_name}
            meta={null}
            status={repo.status}
            starting={startingIndex.has(repo.repo_full_name)}
            error={indexErrors[repo.repo_full_name]}
            selected={selectedRepo === repo.repo_full_name}
            onSelect={() => selectRepo(repo.repo_full_name)}
            onIndex={() => indexRepo(repo.repo_full_name)}
          />
        ))}
      </ul>
    );
  };

  const tabs: { id: RepoTab; label: string; count: number | null }[] = [
    { id: "own", label: "Own", count: githubLinked && !reposLoading ? ownRepos.length : null },
    { id: "added", label: "Added", count: trackedLoading ? null : addedRepos.length },
    {
      id: "collaborator",
      label: "Collaborator",
      count: githubLinked && !reposLoading ? collaboratorRepos.length : null,
    },
  ];

  const selectedParts = selectedRepo ? splitRepoName(selectedRepo) : null;

  const renderChatEmpty = () => {
    if (!selectedRepo) {
      return (
        <div className="chat-empty">
          <h2>Choose a repository</h2>
          <p>Pick one from the list, or add any repository you can access by its owner/name.</p>
        </div>
      );
    }
    if (selectedStatus === "indexing") {
      return (
        <div className="chat-empty">
          <Spinner />
          <h2>Indexing {selectedParts?.name}</h2>
          <p>
            The repository is being downloaded, split into chunks and embedded. This page updates
            on its own when it's ready.
          </p>
        </div>
      );
    }
    if (selectedStatus === "error") {
      return (
        <div className="chat-empty">
          <h2>Indexing failed</h2>
          <p>{selectedInfo?.error_message ?? "Something went wrong while indexing."}</p>
          <button type="button" className="btn btn-primary" onClick={() => indexRepo(selectedRepo)}>
            Try again
          </button>
        </div>
      );
    }
    if (selectedStatus !== "ready") {
      return (
        <div className="chat-empty">
          <h2>Index {selectedParts?.name} to start</h2>
          <p>
            Indexing downloads the repository, splits its files into chunks and stores their
            embeddings. It usually takes a minute or two.
          </p>
          {indexErrors[selectedRepo] && (
            <div className="notice notice-error">{indexErrors[selectedRepo]}</div>
          )}
          <button type="button" className="btn btn-primary" onClick={() => indexRepo(selectedRepo)}>
            Index repository
          </button>
        </div>
      );
    }
    return (
      <div className="chat-empty">
        <h2>Ask about {selectedParts?.name}</h2>
        <p>Answers are based only on code retrieved from this repository.</p>
        <div className="samples">
          {SAMPLE_QUESTIONS.map((sample) => (
            <button key={sample} type="button" className="sample" onClick={() => fillSample(sample)}>
              {sample}
            </button>
          ))}
        </div>
      </div>
    );
  };

  const currentChunk = viewer ? viewer.chunks[viewer.index] : null;
  const viewerRange = currentChunk ? lineRange(currentChunk) : null;

  /* ---------- layout ---------- */

  return (
    <div className="workspace">
      <header className="topbar">
        <button
          type="button"
          className="icon-btn topbar-menu"
          onClick={() => setSidebarOpen(true)}
          aria-label="Show repositories"
        >
          <IconMenu />
        </button>
        <Logo to="/app" />
        <div className="topbar-spacer" />
        <ThemeToggle />
        {username && <span className="topbar-user">{username}</span>}
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={() => {
            logout();
            navigate("/login", { replace: true });
          }}
        >
          Sign out
        </button>
      </header>

      <div className="workspace-body">
        {/* ---------------- Repositories ---------------- */}
        <aside
          className={`pane sidebar${sidebarOpen ? " is-open" : ""}`}
          aria-label="Repositories"
        >
          <div className="pane-head">
            <h2 className="pane-title">Repositories</h2>
            <button
              type="button"
              className="icon-btn sidebar-close"
              onClick={() => setSidebarOpen(false)}
              aria-label="Hide repositories"
            >
              <IconClose />
            </button>
          </div>

          <div className="sidebar-controls">
            <div className="tabs" role="tablist" aria-label="Repository groups">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  role="tab"
                  className="tab"
                  aria-selected={activeTab === tab.id}
                  onClick={() => setActiveTab(tab.id)}
                >
                  {tab.label}
                  {tab.count !== null && <span className="tab-count">{tab.count}</span>}
                </button>
              ))}
            </div>

            {activeTab === "added" && (
              <form className="add-repo" onSubmit={addRepository}>
                <input
                  className="input input-sm"
                  value={newRepoInput}
                  onChange={(event) => setNewRepoInput(event.target.value)}
                  placeholder="owner/repository"
                  aria-label="Repository to add, as owner/name"
                  spellCheck={false}
                />
                <button
                  className="btn btn-secondary btn-sm"
                  disabled={addRepoLoading || !newRepoInput.trim()}
                >
                  {addRepoLoading ? "Adding…" : "Add"}
                </button>
              </form>
            )}
            {activeTab === "added" && addRepoError && (
              <div className="notice notice-error notice-sm">{addRepoError}</div>
            )}

            <input
              className="input input-sm"
              type="search"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
              placeholder="Filter"
              aria-label="Filter repositories"
            />
          </div>

          <div className="sidebar-list" role="tabpanel">
            {activeTab === "own" && renderGithubList(ownRepos, "You don't own any repositories yet.")}
            {activeTab === "collaborator" &&
              renderGithubList(
                collaboratorRepos,
                "No repositories where you're a collaborator or organization member."
              )}
            {activeTab === "added" && renderAddedList()}
          </div>
        </aside>
        {sidebarOpen && (
          <div className="scrim scrim-sidebar" onClick={() => setSidebarOpen(false)} />
        )}

        {/* ---------------- Chat ---------------- */}
        <section className="pane chat" aria-label="Conversation">
          <div className="pane-head chat-head">
            {selectedParts ? (
              <div className="chat-title">
                <h1>
                  {selectedParts.owner && (
                    <span className="repo-owner">{selectedParts.owner}/</span>
                  )}
                  {selectedParts.name}
                </h1>
                <p>
                  <span className={`status status-${selectedStatus}`}>
                    <span className="status-dot" />
                    {statusLabel(selectedStatus)}
                  </span>
                  {selectedStatus === "ready" && selectedInfo && (
                    <span className="chat-stats">
                      {selectedInfo.file_count} files, {selectedInfo.chunk_count} chunks
                    </span>
                  )}
                </p>
              </div>
            ) : (
              <h1 className="pane-title">Conversation</h1>
            )}

            {visibleTurns.length > 0 && (
              <button type="button" className="btn btn-ghost btn-sm" onClick={clearConversation}>
                Clear conversation
              </button>
            )}
          </div>

          <div className="thread">
            <div className="thread-inner">
              {visibleTurns.length === 0
                ? renderChatEmpty()
                : visibleTurns.map((turn) => (
                    <article className="turn" key={turn.id}>
                      <div className="turn-question">{turn.question}</div>

                      <div className="turn-answer">
                        {turn.state === "pending" && (
                          <p className="thinking" role="status">
                            <span className="dots" aria-hidden="true">
                              <span />
                              <span />
                              <span />
                            </span>
                            Searching the code and writing an answer
                          </p>
                        )}

                        {turn.state === "error" && (
                          <div className="notice notice-error">
                            {turn.error}
                            <button
                              type="button"
                              className="btn btn-ghost btn-sm"
                              onClick={() => retryTurn(turn)}
                              disabled={!canAsk}
                            >
                              Try again
                            </button>
                          </div>
                        )}

                        {turn.state === "done" && turn.answer !== undefined && (
                          <>
                            <Markdown text={turn.answer} />

                            {turn.sources && turn.sources.length > 0 && (
                              <div className="sources">
                                <h3 className="sources-title">Sources</h3>
                                <ul className="source-list">
                                  {turn.sources.map((group, index) => {
                                    const active =
                                      viewer?.turnId === turn.id && viewer.groupIndex === index;
                                    const ranges = groupRangesLabel(group);
                                    const best = Math.max(...group.chunks.map((c) => c.score));
                                    return (
                                      <li key={`${group.file_path}-${index}`}>
                                        <button
                                          type="button"
                                          className={`source-chip${active ? " is-active" : ""}`}
                                          onClick={() => openSource(turn, index)}
                                          aria-pressed={active}
                                          title={`Best match ${Math.round(best * 100)}%`}
                                        >
                                          <IconFile />
                                          <span className="path">{group.file_path}</span>
                                          {ranges && <span className="ranges">{ranges}</span>}
                                        </button>
                                      </li>
                                    );
                                  })}
                                </ul>
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </article>
                  ))}
              <div ref={threadEndRef} />
            </div>
          </div>

          <div className="composer-wrap">
            <div className={`composer${canAsk ? "" : " is-disabled"}`}>
              <textarea
                ref={composerRef}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleComposerKey}
                rows={1}
                disabled={!selectedRepo || selectedStatus !== "ready"}
                placeholder={
                  !selectedRepo
                    ? "Choose a repository first"
                    : selectedStatus === "ready"
                      ? `Ask about ${selectedParts?.name}…`
                      : "Index the repository to ask questions"
                }
                aria-label="Your question"
              />
              <button
                type="button"
                className="send-btn"
                onClick={() => ask(draft)}
                disabled={!canAsk || !draft.trim()}
                aria-label="Send question"
                title="Send"
              >
                {anyPending ? <Spinner /> : <IconSend />}
              </button>
            </div>
            <p className="composer-hint">
              Enter sends, Shift+Enter adds a line. Each question is answered on its own.
            </p>
          </div>
        </section>

        {/* ---------------- Source viewer ---------------- */}
        <aside
          className={`pane viewer${viewer ? " is-open" : ""}`}
          aria-label="Source viewer"
        >
          {viewer && currentChunk ? (
            <>
              <div className="pane-head viewer-head">
                <div className="viewer-heading">
                  <h2 className="viewer-path">{viewer.filePath}</h2>
                  <p className="viewer-meta">
                    {viewerRange && (
                      <span>
                        {viewerRange.includes("–") ? "Lines" : "Line"} {viewerRange}
                      </span>
                    )}
                    <span>Relevance {Math.round(currentChunk.score * 100)}%</span>
                  </p>
                </div>
                <div className="viewer-actions">
                  {currentChunk.url && (
                    <a
                      className="icon-btn"
                      href={currentChunk.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label="Open on GitHub"
                      title="Open on GitHub"
                    >
                      <IconExternal />
                    </a>
                  )}
                  <button
                    type="button"
                    className="icon-btn"
                    onClick={() => setViewer(null)}
                    aria-label="Close source"
                    title="Close"
                  >
                    <IconClose />
                  </button>
                </div>
              </div>

              <SourceCode
                key={`${viewer.turnId}-${viewer.groupIndex}-${viewer.index}`}
                path={viewer.filePath}
                chunk={currentChunk}
              />

              {viewer.chunks.length > 1 && (
                <div className="viewer-foot">
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => setViewer({ ...viewer, index: viewer.index - 1 })}
                    disabled={viewer.index === 0}
                  >
                    <IconChevronLeft />
                    Previous
                  </button>
                  <span className="viewer-position">
                    {viewer.index + 1} of {viewer.chunks.length}
                  </span>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => setViewer({ ...viewer, index: viewer.index + 1 })}
                    disabled={viewer.index === viewer.chunks.length - 1}
                  >
                    Next
                    <IconChevronRight />
                  </button>
                </div>
              )}
            </>
          ) : (
            <>
              <div className="pane-head">
                <h2 className="pane-title">Source</h2>
              </div>
              <div className="viewer-empty">
                <p>
                  Select a file under an answer to read the exact code the answer was based on.
                </p>
              </div>
            </>
          )}
        </aside>
        {viewer && <div className="scrim scrim-viewer" onClick={() => setViewer(null)} />}
      </div>
    </div>
  );
}

function SourceCode({ path, chunk }: { path: string; chunk: SourceChunk }) {
  const { theme } = useTheme();
  return (
    <div className="viewer-code">
      <SyntaxHighlighter
        language={languageForFile(path)}
        style={theme === "dark" ? oneDark : oneLight}
        showLineNumbers
        startingLineNumber={chunk.start_line ?? 1}
        lineNumberStyle={{
          minWidth: "3.5em",
          paddingRight: "1em",
          color: "var(--ink-3)",
          userSelect: "none",
        }}
        codeTagProps={{
          style: { whiteSpace: "pre", display: "block", fontFamily: "var(--font-mono)" },
        }}
        customStyle={{
          margin: 0,
          padding: "12px 0",
          minHeight: "100%",
          background: "var(--code-bg)",
          borderRadius: 0,
          fontSize: 12.5,
          lineHeight: 1.6,
        }}
      >
        {chunk.content}
      </SyntaxHighlighter>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* App                                                                 */
/* ------------------------------------------------------------------ */

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/app" element={<WorkspacePage />} />
          <Route path="/github/callback" element={<GitHubCallbackPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
