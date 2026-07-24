import os
from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import build_retriever, make_llm, answer

EMBED_MODEL = os.getenv("EMBED_MODEL", "openai")

retriever = build_retriever(EMBED_MODEL, k=4)
llm = make_llm()

app = FastAPI()


class Query(BaseModel):
    question: str


@app.post("/query")
def query(q: Query):
    text, docs = answer(q.question, retriever, llm)
    sources = sorted({os.path.basename(d.metadata.get("source", "unknown")) for d in docs})
    return {"answer": text, "sources": sources}