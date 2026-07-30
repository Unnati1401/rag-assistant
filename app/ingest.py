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


def load_docs(data_dir="data", exclude_dirs=("okf",)):
    """Recursively load supported files from data_dir and its subfolders
    (e.g. data/k8s). Excluded subdirectories (default: okf) are handled
    separately by the OKF loader / its own collection."""
    docs = []
    for root, dirs, files in os.walk(data_dir):
        # prune excluded subdirectories in-place so os.walk skips them
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for fname in sorted(files):
            path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".md", ".txt"):
                loader = TextLoader(path, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext in (".html", ".htm"):
                loader = BSHTMLLoader(path, open_encoding="utf-8")
            else:
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


import hashlib, json

MANIFEST_PATH = "./chroma_db/ingest_manifest.json"
SUPPORTED_EXT = (".md", ".txt", ".pdf", ".html", ".htm")


def _loader_for(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".md", ".txt"):
        return TextLoader(path, encoding="utf-8")
    if ext == ".pdf":
        return PyPDFLoader(path)
    if ext in (".html", ".htm"):
        return BSHTMLLoader(path, open_encoding="utf-8")
    return None


def _iter_files(data_dir="data", exclude_dirs=("okf",)):
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in SUPPORTED_EXT:
                yield os.path.join(root, fname)


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _load_manifest(path=MANIFEST_PATH):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest, path=MANIFEST_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def incremental_ingest(kind=EMBED_MODEL, data_dir="data", persist_directory="./chroma_db"):
    """Only (re)embed files that are new or changed since the last run, and drop
    chunks for deleted files. Tracks content hashes in a manifest. This is the
    production ingestion path - avoids re-embedding the whole corpus each time."""
    embeddings, collection_name = get_embeddings(kind)
    vs = Chroma(collection_name=collection_name, embedding_function=embeddings,
                persist_directory=persist_directory)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    manifest = _load_manifest()
    current = {p: _file_hash(p) for p in _iter_files(data_dir)}

    new_or_changed = [p for p, h in current.items() if manifest.get(p) != h]
    deleted = [p for p in manifest if p not in current]

    # Remove old chunks for changed + deleted files (delete-before-add = no dupes).
    for p in new_or_changed + deleted:
        try:
            vs.delete(where={"source": p})
        except Exception:
            pass

    added = 0
    for p in new_or_changed:
        loader = _loader_for(p)
        if loader is None:
            continue
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = p  # normalize so per-file delete is reliable
        chunks = splitter.split_documents(docs)
        if chunks:
            vs.add_documents(chunks)
            added += len(chunks)

    _save_manifest(current)
    result = {"changed_or_new": len(new_or_changed), "deleted": len(deleted),
              "chunks_added": added, "total_files": len(current)}
    print(f"Incremental ingest: {result}")
    return result


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