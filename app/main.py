import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("docs")

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
def query(q: Query):
    q_emb = client.embeddings.create(
        model="text-embedding-3-small", input=[q.question]
    ).data[0].embedding

    results = collection.query(query_embeddings=[q_emb], n_results=4)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = "\n\n".join(chunks)

    prompt = (
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {q.question}"
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "answer": completion.choices[0].message.content,
        "sources": sorted(set(sources)),
    }