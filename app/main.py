import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.ingest import ensure_indexed
from app.rag import build_grounded_retriever, make_llm, answer

EMBED_MODEL = os.getenv("EMBED_MODEL", "openai")

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the doc index if this is a fresh environment, then the LLM.
    # Retrievers are built per-request so the user can pick the grounding source.
    ensure_indexed(EMBED_MODEL)
    _state["llm"] = make_llm()
    _state["web_enabled"] = bool(os.getenv("TAVILY_API_KEY"))
    print(f"Ready. embed={EMBED_MODEL} web_enabled={_state['web_enabled']}")
    yield
    _state.clear()


app = FastAPI(title="RAG Assistant", lifespan=lifespan)


class Query(BaseModel):
    question: str
    source: str = "docs"  # "docs" | "web" | "both"


@app.get("/health")
def health():
    return {"status": "ok", "embed_model": EMBED_MODEL,
            "web_enabled": _state.get("web_enabled", False)}


@app.post("/query")
def query(q: Query):
    source = q.source if q.source in ("docs", "web", "both") else "docs"
    # If web isn't configured, fall back to docs so the app never errors.
    if source in ("web", "both") and not _state.get("web_enabled"):
        source = "docs"
    retriever = build_grounded_retriever(source, EMBED_MODEL, k=3)
    text, docs = answer(q.question, retriever, _state["llm"])
    sources = sorted({d.metadata.get("source", "unknown") for d in docs})
    return {"answer": text, "sources": sources, "source_mode": source}


@app.get("/", response_class=HTMLResponse)
def home():
    return _FRONTEND_HTML


_FRONTEND_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RAG Assistant</title>
<style>
  :root { --bg:#0f1117; --card:#1a1d27; --line:#2a2f3d; --text:#e6e9ef;
          --muted:#9aa4b2; --accent:#6ea8fe; --accent2:#3d7bfd; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:760px; margin:0 auto; padding:40px 20px 80px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 28px; font-size:14px; }
  textarea { width:100%; min-height:90px; resize:vertical; padding:12px 14px;
             background:var(--card); color:var(--text); border:1px solid var(--line);
             border-radius:10px; font:inherit; }
  .row { display:flex; align-items:center; gap:14px; margin:14px 0 0; flex-wrap:wrap; }
  .seg { display:inline-flex; background:var(--card); border:1px solid var(--line);
         border-radius:10px; overflow:hidden; }
  .seg button { background:transparent; color:var(--muted); border:0; padding:9px 16px;
                cursor:pointer; font:inherit; }
  .seg button.active { background:var(--accent2); color:#fff; }
  .go { margin-left:auto; background:var(--accent2); color:#fff; border:0;
        padding:10px 22px; border-radius:10px; cursor:pointer; font:inherit; font-weight:600; }
  .go:disabled { opacity:.5; cursor:default; }
  .label { color:var(--muted); font-size:13px; }
  .answer { margin-top:26px; background:var(--card); border:1px solid var(--line);
            border-radius:12px; padding:18px 20px; white-space:pre-wrap; display:none; }
  .answer.show { display:block; }
  .sources { margin-top:14px; font-size:13px; color:var(--muted); }
  .sources a { color:var(--accent); word-break:break-all; }
  .tag { display:inline-block; font-size:11px; padding:2px 8px; border-radius:999px;
         background:#243; color:#9f9; margin-left:8px; }
  .spin { color:var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <h1>RAG Assistant</h1>
  <p class="sub">Ask a question. Choose where the answer is grounded: your document corpus, live web search, or both.</p>

  <textarea id="q" placeholder="e.g. What does the response_model parameter do?"></textarea>

  <div class="row">
    <span class="label">Grounding:</span>
    <div class="seg" id="seg">
      <button data-s="docs" class="active">Docs</button>
      <button data-s="web">Web</button>
      <button data-s="both">Both</button>
    </div>
    <button class="go" id="go">Ask</button>
  </div>

  <div class="answer" id="answer"></div>
  <div class="sources" id="sources"></div>
</div>

<script>
  let source = "docs";
  const seg = document.getElementById("seg");
  seg.addEventListener("click", (e) => {
    if (e.target.tagName !== "BUTTON") return;
    [...seg.children].forEach(b => b.classList.remove("active"));
    e.target.classList.add("active");
    source = e.target.dataset.s;
  });

  const go = document.getElementById("go");
  const ansEl = document.getElementById("answer");
  const srcEl = document.getElementById("sources");

  async function ask() {
    const question = document.getElementById("q").value.trim();
    if (!question) return;
    go.disabled = true;
    ansEl.className = "answer show";
    ansEl.innerHTML = '<span class="spin">Thinking…</span>';
    srcEl.innerHTML = "";
    try {
      const r = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, source }),
      });
      const data = await r.json();
      ansEl.innerHTML = escapeHtml(data.answer) +
        '<span class="tag">' + data.source_mode + '</span>';
      if (data.sources && data.sources.length) {
        srcEl.innerHTML = "<strong>Sources:</strong> " + data.sources.map(s =>
          s.startsWith("http")
            ? '<a href="' + s + '" target="_blank">' + s + '</a>'
            : escapeHtml(s)
        ).join(" &middot; ");
      }
    } catch (err) {
      ansEl.textContent = "Error: " + err.message;
    } finally {
      go.disabled = false;
    }
  }
  function escapeHtml(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

  go.addEventListener("click", ask);
  document.getElementById("q").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
  });
</script>
</body>
</html>"""