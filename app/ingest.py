import os, glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "openai")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def get_embeddings(kind):
    """model -> (embeddings, collection_name). Mirrors rag.get_embeddings."""
    if kind == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small"), "docs_openai"
    elif kind == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings
        return (
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            "docs_minilm",
        )
    else:
        raise ValueError(f"Unknown EMBED_MODEL: {kind!r} (use 'openai' or 'hf')")


def load_docs(data_dir="data"):
    """Dispatch each file to the right loader based on its extension."""
    docs = []
    for path in glob.glob(os.path.join(data_dir, "*")):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".md", ".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext in (".html", ".htm"):
            loader = BSHTMLLoader(path, open_encoding="utf-8")
        else:
            print(f"Skipping unsupported file: {path}")
            continue
        docs.extend(loader.load())
    return docs


def ingest(kind=EMBED_MODEL, persist_directory="./chroma_db"):
    """Load, chunk, embed, and store the corpus. Resets the collection first so
    re-running does not pile up duplicates. Returns the number of chunks."""
    embeddings, collection_name = get_embeddings(kind)
    docs = load_docs()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)

    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vs.reset_collection()
    vs.add_documents(chunks)
    print(f"[{kind}] Indexed {len(chunks)} chunks into '{collection_name}'.")
    return len(chunks)


def ensure_indexed(kind=EMBED_MODEL, persist_directory="./chroma_db"):
    """Index only if the collection is empty. Used at app startup so a fresh
    container (with no persisted DB) builds its index once, from the corpus."""
    embeddings, collection_name = get_embeddings(kind)
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    count = vs._collection.count()
    if count == 0:
        print(f"Collection '{collection_name}' empty; indexing corpus...")
        ingest(kind, persist_directory)
    else:
        print(f"Collection '{collection_name}' already has {count} chunks; skipping ingest.")


if __name__ == "__main__":
    ingest()