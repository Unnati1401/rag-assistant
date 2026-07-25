import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from app.ingest import ensure_indexed
from app.rag import (
    build_retriever,
    build_hybrid_retriever,
    build_hybrid_rerank_retriever,
    make_llm,
    answer,
)

EMBED_MODEL = os.getenv("EMBED_MODEL", "openai")
# Retriever strategy: "hybrid" (light, no torch - default for deploy),
# "hybrid_rerank" (full, needs torch), or "vector".
RETRIEVER = os.getenv("RETRIEVER", "hybrid")

_state = {}


def _make_retriever():
    if RETRIEVER == "hybrid":
        return build_hybrid_retriever(EMBED_MODEL, k=3)
    elif RETRIEVER == "hybrid_rerank":
        return build_hybrid_rerank_retriever(EMBED_MODEL, top_k=3)
    elif RETRIEVER == "vector":
        return build_retriever(EMBED_MODEL, k=3)
    raise ValueError(f"Unknown RETRIEVER: {RETRIEVER!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the index if this is a fresh environment, then the retriever + llm.
    ensure_indexed(EMBED_MODEL)
    _state["retriever"] = _make_retriever()
    _state["llm"] = make_llm()
    print(f"Ready. embed={EMBED_MODEL} retriever={RETRIEVER}")
    yield
    _state.clear()


app = FastAPI(title="RAG Assistant", lifespan=lifespan)


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "embed_model": EMBED_MODEL, "retriever": RETRIEVER}


@app.post("/query")
def query(q: Query):
    text, docs = answer(q.question, _state["retriever"], _state["llm"])
    sources = sorted({os.path.basename(d.metadata.get("source", "unknown")) for d in docs})
    return {"answer": text, "sources": sources}