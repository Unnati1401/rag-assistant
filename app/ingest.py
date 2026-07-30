import os, glob, argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "openai")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
OKF_COLLECTION = "docs_okf"


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
    """Dispatch each top-level file to the right loader by extension.
    Directories (e.g. data/okf) are skipped here and handled by the OKF loader."""
    docs = []
    for path in glob.glob(os.path.join(data_dir, "*")):
        if os.path.isdir(path):
            continue
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
    """Load, chunk, embed, and store the raw corpus. Resets the collection first."""
    embeddings, collection_name = get_embeddings(kind)
    docs = load_docs()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)

    vs = Chroma(collection_name=collection_name, embedding_function=embeddings,
                persist_directory=persist_directory)
    vs.reset_collection()
    vs.add_documents(chunks)
    print(f"[{kind}] Indexed {len(chunks)} chunks into '{collection_name}'.")
    return len(chunks)

def ingest_okf(kind="openai", okf_dir="data/okf", persist_directory="./chroma_db"):
    """Load OKF entries into their own collection (docs_okf), preserving the
    structured metadata (category, confidence, okf_id) on each chunk. Kept
    separate from the raw-doc collection for clean A/B comparison."""
    from app.okf import load_okf_docs

    embeddings, _ = get_embeddings(kind)
    docs = load_okf_docs(okf_dir)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)

    vs = Chroma(collection_name=OKF_COLLECTION, embedding_function=embeddings,
                persist_directory=persist_directory)
    vs.reset_collection()
    vs.add_documents(chunks)
    print(f"[okf] Indexed {len(chunks)} OKF chunks into '{OKF_COLLECTION}'.")
    return len(chunks)


def ensure_indexed(kind=EMBED_MODEL, persist_directory="./chroma_db"):
    """Index the raw corpus only if its collection is empty (used at app startup)."""
    embeddings, collection_name = get_embeddings(kind)
    vs = Chroma(collection_name=collection_name, embedding_function=embeddings,
                persist_directory=persist_directory)
    if vs._collection.count() == 0:
        print(f"Collection '{collection_name}' empty; indexing corpus...")
        ingest(kind, persist_directory)
    else:
        print(f"Collection '{collection_name}' already populated; skipping ingest.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--okf", action="store_true", help="ingest OKF entries into docs_okf")
    parser.add_argument("--model", default=EMBED_MODEL, help="embedding model: openai or hf")
    args = parser.parse_args()

    if args.okf:
        ingest_okf(args.model)
    else:
        ingest(args.model)