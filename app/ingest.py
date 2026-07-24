import os, glob, uuid
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("docs")

def chunk_text(text, size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def main():
    files = glob.glob("data/*.md")
    docs, metas, ids = [], [], []
    for path in files:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for i, ch in enumerate(chunk_text(text)):
            docs.append(ch)
            metas.append({"source": os.path.basename(path), "chunk": i})
            ids.append(str(uuid.uuid4()))

    resp = client.embeddings.create(model="text-embedding-3-small", input=docs)
    embeddings = [d.embedding for d in resp.data]

    collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    print(f"Indexed {len(docs)} chunks from {len(files)} files.")

if __name__ == "__main__":
    main()