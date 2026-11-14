import { useEffect, useRef, useState } from "react";

// ─── Constants ────────────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_BASE || "/api";
const MAX_UPLOAD = 4 * 1024 * 1024;

const EXAMPLES = [
  "I'm 45, male, recently diagnosed with type 2 diabetes and high blood pressure. I live in Mumbai.",
  "My doctor thinks I have Crohn's disease. I'm 34, male, not on any medication yet.",
  "I'm a 52-year-old woman with stage 2 breast cancer, currently taking tamoxifen.",
  "I'm 28, female, with PCOS and irregular periods.",
];

const VERDICT = {
  YES: { label: "Good match", cls: "yes", icon: "✓" },
  POSSIBLE: { label: "Possible match", cls: "possible", icon: "?" },
  NO: { label: "Probably not a match", cls: "no", icon: "✕" },
  ERROR: { label: "Couldn't check", cls: "error", icon: "!" },
};

// ─── Utilities ────────────────────────────────────────────────────────────────
function formatTime(iso) {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Escape first, then apply a small, safe subset of markdown.
function parseMarkdown(text) {
  return escapeHtml(text || "")
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
    )
    .replace(/^#{1,3} (.+)$/gm, "<strong>$1</strong>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/(^|\n)[-*] /g, "$1• ")
    .replace(/\n/g, "<br/>");
}

// ─── API layer ────────────────────────────────────────────────────────────────
class AuthError extends Error {
  constructor() {
    super("Your session expired. Please sign in again.");
  }
}

function errorMessage(err, fallback) {
  if (typeof err?.detail === "string") return err.detail;
  if (Array.isArray(err?.detail))
    return err.detail
      .map((e) => `${e.loc?.slice(-1)[0] ?? ""}: ${e.msg}`.replace(/^: /, ""))
      .join(", ");
  return fallback;
}

async function apiLogin(username, password) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(errorMessage(err, "Sign in failed"));
  }
  return res.json();
}

async function apiRegister(username, password, role) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, role }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(errorMessage(err, "Couldn't create your account"));
  }
  return res.json();
}

async function apiSession(token, sessionId) {
  const res = await fetch(`${API_BASE}/chat/sessions/${sessionId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (res.status === 401) throw new AuthError();
  if (!res.ok) return null;
  return res.json();
}

async function* apiChatStream(token, message, sessionId, file) {
  let res;
  if (file) {
    const form = new FormData();
    form.append("file", file);
    form.append("stream", "true");
    if (sessionId) form.append("session_id", sessionId);
    if (message) form.append("note", message);
    res = await fetch(`${API_BASE}/chat/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: form,
    });
  } else {
    res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ message, session_id: sessionId, stream: true }),
    });
  }
  if (res.status === 401) throw new AuthError();
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(errorMessage(err, `Something went wrong (${res.status})`));
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6));
        } catch {
          /* ignore partial lines */
        }
      }
    }
  }
}

// ─── Small pieces ─────────────────────────────────────────────────────────────
function Logo({ size = 32 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="9" fill="var(--brand)" />
      <path
        d="M16 8v16M8 16h16"
        stroke="white"
        strokeWidth="3.2"
        strokeLinecap="round"
      />
      <circle cx="16" cy="16" r="4.5" fill="none" stroke="white" strokeWidth="2" opacity=".55" />
    </svg>
  );
}

function Icon({ name, size = 18 }) {
  const paths = {
    menu: <path d="M3 6h18M3 12h18M3 18h18" />,
    plus: <path d="M12 5v14M5 12h14" />,
    send: <path d="M5 12h14M13 6l6 6-6 6" />,
    clip: (
      <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
    ),
    logout: <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9" />,
    close: <path d="M18 6 6 18M6 6l12 12" />,
    user: (
      <>
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
      </>
    ),
    phone: (
      <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.8 19.8 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.12 4.18 2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.13.96.36 1.9.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.91.34 1.85.57 2.81.7A2 2 0 0 1 22 16.92z" />
    ),
    mail: (
      <>
        <rect x="2" y="4" width="20" height="16" rx="2" />
        <path d="m22 7-10 6L2 7" />
      </>
    ),
    pin: (
      <>
        <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0z" />
        <circle cx="12" cy="10" r="3" />
      </>
    ),
    external: <path d="M15 3h6v6M10 14 21 3M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />,
    shield: <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />,
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}

// ─── Auth screen ──────────────────────────────────────────────────────────────
function AuthScreen({ onAuth }) {
  const [mode, setMode] = useState("register");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("patient");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const data =
        mode === "login"
          ? await apiLogin(username, password)
          : await apiRegister(username, password, role);
      onAuth(data.access_token, data.username, data.role);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-screen">
      <section className="auth-intro">
        <div className="brand">
          <Logo size={40} />
          <div>
            <div className="brand-name">Aria</div>
            <div className="brand-sub">Clinical trial finder</div>
          </div>
        </div>
        <h1>Find clinical trials you may be able to join</h1>
        <p className="lead">
          Tell Aria about your health in your own words. She searches tens of thousands of
          recruiting trials and explains which ones could fit you — and how to contact them.
        </p>
        <ul className="benefits">
          <li>
            <span className="benefit-num">1</span>
            <span>
              <strong>Describe your situation</strong> — no forms or medical jargon needed.
            </span>
          </li>
          <li>
            <span className="benefit-num">2</span>
            <span>
              <strong>Add a doctor's report</strong> if you have one (optional).
            </span>
          </li>
          <li>
            <span className="benefit-num">3</span>
            <span>
              <strong>Get matched trials</strong> with plain-language reasons and contacts.
            </span>
          </li>
        </ul>
      </section>

      <section className="auth-card">
        <div className="auth-tabs" role="tablist">
          <button
            role="tab"
            aria-selected={mode === "register"}
            className={mode === "register" ? "tab active" : "tab"}
            onClick={() => setMode("register")}
          >
            Create account
          </button>
          <button
            role="tab"
            aria-selected={mode === "login"}
            className={mode === "login" ? "tab active" : "tab"}
            onClick={() => setMode("login")}
          >
            Sign in
          </button>
        </div>

        <form onSubmit={submit} className="auth-form">
          <label className="field">
            <span>Username</span>
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="e.g. priya_s"
              autoComplete="username"
              autoFocus
              required
            />
          </label>
          <label className="field">
            <span>Password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={mode === "register" ? "At least 8 characters" : "Your password"}
              autoComplete={mode === "register" ? "new-password" : "current-password"}
              minLength={mode === "register" ? 8 : undefined}
              required
            />
          </label>
          {mode === "register" && (
            <fieldset className="field">
              <span>I am a…</span>
              <div className="segmented">
                <label className={role === "patient" ? "seg active" : "seg"}>
                  <input
                    type="radio"
                    name="role"
                    value="patient"
                    checked={role === "patient"}
                    onChange={() => setRole("patient")}
                  />
                  Patient or caregiver
                </label>
                <label className={role === "doctor" ? "seg active" : "seg"}>
                  <input
                    type="radio"
                    name="role"
                    value="doctor"
                    checked={role === "doctor"}
                    onChange={() => setRole("doctor")}
                  />
                  Doctor or researcher
                </label>
              </div>
            </fieldset>
          )}
          {error && (
            <div className="form-error" role="alert">
              {error}
            </div>
          )}
          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? <span className="spinner" /> : mode === "login" ? "Sign in" : "Get started"}
          </button>
        </form>
        <p className="fine-print">
          <Icon name="shield" size={14} /> No email or real name needed. Uploaded reports are read
          once and never stored. Aria gives information, not medical advice — always talk to your
          doctor before joining a trial.
        </p>
      </section>
    </div>
  );
}

// ─── Profile panel ────────────────────────────────────────────────────────────
function TagRow({ label, items, kind }) {
  if (!items?.length) return null;
  return (
    <div className="profile-row">
      <span className="profile-label">{label}</span>
      <div className="tag-list">
        {items.map((x) => (
          <span key={x} className={`tag tag-${kind}`}>
            {x}
          </span>
        ))}
      </div>
    </div>
  );
}

function ProfilePanel({ profile }) {
  const has =
    profile && (profile.age || profile.sex || profile.location || profile.conditions?.length);
  return (
    <div className="profile-panel">
      <h2 className="panel-title">
        <Icon name="user" size={16} /> What Aria knows about you
      </h2>
      {!has ? (
        <p className="profile-empty">
          As you chat, the details Aria uses to match trials will appear here.
        </p>
      ) : (
        <div className="profile-data">
          <div className="profile-facts">
            {profile.age && (
              <div>
                <span className="profile-label">Age</span>
                <span className="profile-value">{profile.age}</span>
              </div>
            )}
            {profile.sex && (
              <div>
                <span className="profile-label">Sex</span>
                <span className="profile-value cap">{profile.sex}</span>
              </div>
            )}
            {profile.location && (
              <div className="wide">
                <span className="profile-label">Location</span>
                <span className="profile-value">{profile.location}</span>
              </div>
            )}
          </div>
          <TagRow label="Conditions" items={profile.conditions} kind="condition" />
          <TagRow label="Symptoms" items={profile.symptoms} kind="symptom" />
          <TagRow label="Medications" items={profile.medications} kind="med" />
        </div>
      )}
      <div className="tips">
        <h3>Tips for better matches</h3>
        <ul>
          <li>Mention your age, sex and diagnosis</li>
          <li>Say which city you live in to see nearby sites</li>
          <li>Upload a lab or doctor's report with 📎</li>
          <li>Answer Aria's questions — or just say “skip”</li>
        </ul>
      </div>
    </div>
  );
}

// ─── Chat pieces ──────────────────────────────────────────────────────────────
const STAGES = [
  [0, "Reading your message…"],
  [3, "Searching recruiting trials…"],
  [7, "Checking each trial's eligibility rules…"],
  [16, "Writing up what I found…"],
  [30, "Almost there — thanks for your patience…"],
];

function TypingIndicator({ uploading }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(t);
  }, []);
  let label = STAGES.filter(([s]) => elapsed >= s).pop()[1];
  if (uploading && elapsed < 6) label = "Reading your report…";
  return (
    <div className="message assistant" aria-live="polite">
      <div className="avatar aria-avatar">A</div>
      <div className="bubble aria-bubble typing">
        <span className="dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span className="typing-label">{label}</span>
      </div>
    </div>
  );
}

function TrialContacts({ trial }) {
  const people = trial.contacts || [];
  const sites = trial.sites || [];
  if (!people.length && !sites.length) {
    return (
      <p className="contact-none">Contact details are listed on the trial's ClinicalTrials.gov page.</p>
    );
  }
  return (
    <div className="trial-contacts">
      {people.map((c, i) => (
        <div key={i} className="contact-person">
          <span className="contact-name">{c.name || "Study team"}</span>
          <div className="contact-actions">
            {c.phone && (
              <a className="chip-link" href={`tel:${c.phone.replace(/[^\d+]/g, "")}`}>
                <Icon name="phone" size={14} /> {c.phone}
              </a>
            )}
            {c.email && (
              <a
                className="chip-link"
                href={`mailto:${c.email}?subject=${encodeURIComponent(
                  `Question about clinical trial ${trial.trial_id}`,
                )}`}
              >
                <Icon name="mail" size={14} /> {c.email}
              </a>
            )}
          </div>
        </div>
      ))}
      {sites.length > 0 && (
        <div className="contact-sites">
          <Icon name="pin" size={14} />
          <span>
            {sites
              .map((st) => [st.facility, st.city, st.country].filter(Boolean).join(", "))
              .join("  ·  ")}
            {trial.total_sites > sites.length && (
              <em> and {trial.total_sites - sites.length} more locations</em>
            )}
          </span>
        </div>
      )}
    </div>
  );
}

function TrialCards({ trials }) {
  const shown = trials.filter((t) => t.eligibility !== "NO");
  const hidden = trials.length - shown.length;
  const [showAll, setShowAll] = useState(false);
  const list = showAll ? trials : shown;
  return (
    <div className="trial-cards">
      {list.map((t) => {
        const v = VERDICT[t.eligibility] || VERDICT.ERROR;
        return (
          <article key={t.trial_id} className={`trial-card ${v.cls}`}>
            <div className="trial-card-top">
              <span className={`verdict ${v.cls}`}>
                <span className="verdict-icon">{v.icon}</span> {v.label}
              </span>
              <span className="trial-meta">
                {[t.phase && t.phase !== "NA" ? t.phase.replaceAll("PHASE", "Phase ") : null,
                  t.status?.replaceAll("_", " ").toLowerCase()]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
            </div>
            <h4 className="trial-title">{t.title}</h4>
            {t.summary && <p className="trial-summary">{t.summary}</p>}
            {t.eligibility !== "NO" && <TrialContacts trial={t} />}
            <a className="trial-link" href={t.url} target="_blank" rel="noopener noreferrer">
              View full details on ClinicalTrials.gov <Icon name="external" size={14} />
              <span className="trial-id">{t.trial_id}</span>
            </a>
          </article>
        );
      })}
      {hidden > 0 && (
        <button className="link-btn" onClick={() => setShowAll((s) => !s)}>
          {showAll
            ? "Hide trials that probably don't fit"
            : hidden > 1
              ? `Show ${hidden} trials that probably don't fit`
              : "Show 1 trial that probably doesn't fit"}
        </button>
      )}
    </div>
  );
}

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      {!isUser && <div className="avatar aria-avatar">A</div>}
      <div className="bubble-wrapper">
        <div
          className={`bubble ${isUser ? "user-bubble" : "aria-bubble"}`}
          dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.content) }}
        />
        {msg.trials?.length > 0 && <TrialCards trials={msg.trials} />}
        <div className="msg-time">{formatTime(msg.timestamp)}</div>
      </div>
    </div>
  );
}

function WelcomeScreen({ onPick, onUpload }) {
  return (
    <div className="welcome">
      <Logo size={52} />
      <h2>Hi, I'm Aria 👋</h2>
      <p className="welcome-sub">
        Tell me about your health in your own words and I'll look for clinical trials that are
        recruiting now. I'll explain what fits, what to check with your doctor, and who to contact.
      </p>
      <button className="upload-cta" onClick={onUpload}>
        <Icon name="clip" size={18} />
        <span>
          <strong>Have a doctor's report?</strong> Upload a PDF or photo and I'll read it for you
        </span>
      </button>
      <div className="examples">
        <div className="examples-title">Or try an example</div>
        {EXAMPLES.map((s) => (
          <button key={s} className="example" onClick={() => onPick(s)}>
            “{s}”
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── Main app ─────────────────────────────────────────────────────────────────
export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem("aria_token") || "");
  const [username, setUsername] = useState(() => localStorage.getItem("aria_user") || "");
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(() => localStorage.getItem("aria_session") || null);
  const [profile, setProfile] = useState(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 960);
  const [awaitingGapAnswers, setAwaitingGapAnswers] = useState(false);

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const fileRef = useRef(null);

  useEffect(() => {
    if (sessionId) localStorage.setItem("aria_session", sessionId);
    else localStorage.removeItem("aria_session");
  }, [sessionId]);

  // Restore the active conversation after a page reload.
  useEffect(() => {
    if (!token || !sessionId || messages.length) return;
    apiSession(token, sessionId)
      .then((s) => {
        if (!s) return setSessionId(null);
        setProfile(s.patient_profile);
        setAwaitingGapAnswers(s.awaiting_gap_answers);
        setMessages(
          s.messages.map((m, i) => ({
            id: `h${i}`,
            role: m.role,
            content: m.content,
            trials: m.trials,
            timestamp: m.timestamp,
          })),
        );
      })
      .catch((err) => err instanceof AuthError && logout());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, streamingContent, loading]);

  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 180) + "px";
    }
  }, [input]);

  function handleAuth(tok, uname, role) {
    setToken(tok);
    setUsername(uname);
    localStorage.setItem("aria_token", tok);
    localStorage.setItem("aria_user", uname);
    localStorage.setItem("aria_role", role);
  }

  function logout() {
    setToken("");
    setUsername("");
    resetChat();
    ["aria_token", "aria_user", "aria_role", "aria_session"].forEach((k) =>
      localStorage.removeItem(k),
    );
  }

  function resetChat() {
    setMessages([]);
    setSessionId(null);
    setProfile(null);
    setInput("");
    setAwaitingGapAnswers(false);
    if (window.innerWidth <= 960) setSidebarOpen(false);
  }

  function addAssistantNote(content) {
    setMessages((prev) => [
      ...prev,
      { id: Date.now(), role: "assistant", content, timestamp: new Date().toISOString() },
    ]);
  }

  async function sendMessage(text, file = null) {
    if ((!text.trim() && !file) || loading) return;
    setInput("");
    setAwaitingGapAnswers(false);
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        role: "user",
        content: file ? `📄 ${file.name}${text.trim() ? `\n\n${text}` : ""}` : text,
        timestamp: new Date().toISOString(),
      },
    ]);
    setLoading(true);
    setUploading(Boolean(file));
    setStreamingContent("");

    try {
      let full = "";
      let trials = [];
      let done = null;
      for await (const event of apiChatStream(token, text, sessionId, file)) {
        if (event.type === "token") {
          full += event.content;
          setStreamingContent(full);
        } else if (event.type === "done") {
          done = event;
          trials = event.trials || [];
        }
      }
      if (done) {
        if (done.session_id) setSessionId(done.session_id);
        if (done.patient_profile) setProfile(done.patient_profile);
        setAwaitingGapAnswers(Boolean(done.awaiting_gap_answers));
      }
      if (full) {
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now() + 1,
            role: "assistant",
            content: full,
            trials,
            timestamp: new Date().toISOString(),
          },
        ]);
      }
    } catch (err) {
      if (err instanceof AuthError) return logout();
      addAssistantNote(`Sorry — ${err.message}`);
    } finally {
      setLoading(false);
      setUploading(false);
      setStreamingContent("");
      textareaRef.current?.focus();
    }
  }

  function handleFile(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (file.size > MAX_UPLOAD) {
      addAssistantNote(
        `**${file.name}** is larger than 4 MB. Please upload a smaller PDF, or take a photo of the key page.`,
      );
      return;
    }
    sendMessage(input, file);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  if (!token) return <AuthScreen onAuth={handleAuth} />;

  return (
    <div className={`app ${sidebarOpen ? "sidebar-open" : ""}`}>
      <aside className="sidebar" aria-label="Your profile">
        <div className="sidebar-head">
          <div className="brand small">
            <Logo size={30} />
            <div>
              <div className="brand-name">Aria</div>
              <div className="brand-sub">Clinical trial finder</div>
            </div>
          </div>
          <button
            className="icon-btn only-mobile"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close panel"
          >
            <Icon name="close" />
          </button>
        </div>

        <button className="btn-secondary new-chat" onClick={resetChat}>
          <Icon name="plus" size={16} /> New conversation
        </button>

        <ProfilePanel profile={profile} />

        <div className="sidebar-footer">
          <div className="user-chip">
            <span className="user-initial">{username?.[0]?.toUpperCase() || "U"}</span>
            <span className="user-name">{username}</span>
          </div>
          <button className="icon-btn" onClick={logout} title="Sign out" aria-label="Sign out">
            <Icon name="logout" size={18} />
          </button>
        </div>
      </aside>
      <div className="backdrop" onClick={() => setSidebarOpen(false)} />

      <main className="main">
        <header className="topbar">
          <button
            className="icon-btn"
            onClick={() => setSidebarOpen((o) => !o)}
            aria-label="Toggle profile panel"
          >
            <Icon name="menu" />
          </button>
          <div className="topbar-title">
            <span className={`status-dot ${awaitingGapAnswers ? "waiting" : ""}`} />
            {awaitingGapAnswers ? "Aria is waiting for your answers" : "Chat with Aria"}
          </div>
          <span className="topbar-badge">Not medical advice</span>
        </header>

        <div className="messages-area">
          <div className="messages-inner">
            {messages.length === 0 && !loading && (
              <WelcomeScreen
                onPick={(s) => {
                  setInput(s);
                  textareaRef.current?.focus();
                }}
                onUpload={() => fileRef.current?.click()}
              />
            )}
            {messages.map((m) => (
              <MessageBubble key={m.id} msg={m} />
            ))}
            {loading && !streamingContent && <TypingIndicator uploading={uploading} />}
            {streamingContent && (
              <div className="message assistant">
                <div className="avatar aria-avatar">A</div>
                <div className="bubble-wrapper">
                  <div
                    className="bubble aria-bubble streaming"
                    dangerouslySetInnerHTML={{ __html: parseMarkdown(streamingContent) }}
                  />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="input-area">
          {awaitingGapAnswers && (
            <div className="gap-hint">
              Answer whatever you can — or type <strong>skip</strong> to see the results now.
            </div>
          )}
          <div className={`composer ${awaitingGapAnswers ? "gap-mode" : ""}`}>
            <input
              ref={fileRef}
              type="file"
              accept=".pdf,.png,.jpg,.jpeg,.webp,.txt,application/pdf,image/*"
              onChange={handleFile}
              hidden
            />
            <button
              className="icon-btn attach"
              onClick={() => fileRef.current?.click()}
              disabled={loading}
              title="Upload a medical report (PDF or photo, max 4 MB)"
              aria-label="Upload a medical report"
            >
              <Icon name="clip" />
            </button>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                awaitingGapAnswers
                  ? "Type your answers here…"
                  : "Describe your condition, age and where you live…"
              }
              rows={1}
              disabled={loading}
              aria-label="Message Aria"
            />
            <button
              className={`send-btn ${input.trim() && !loading ? "active" : ""}`}
              onClick={() => sendMessage(input)}
              disabled={!input.trim() || loading}
              aria-label="Send"
            >
              {loading ? <span className="spinner" /> : <Icon name="send" />}
            </button>
          </div>
          <p className="input-hint">
            Aria uses live ClinicalTrials.gov data. It can make mistakes — always check with your
            doctor.
          </p>
        </div>
      </main>
    </div>
  );
}
