import os, glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# Which embedding model to use: "openai" or "hf". Reads from env, defaults to openai.
EMBED_MODEL = os.getenv("EMBED_MODEL", "openai")


def get_embeddings(kind):
    """Return (embeddings, collection_name).

    Each model gets its own collection because vector dimensions differ
    (OpenAI text-embedding-3-small = 1536, MiniLM = 384) and a Chroma
    collection is locked to a single dimension. Separate collections are
    what let Week 4 compare the two models head-to-head.
    """
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
        loaded = loader.load()
        docs.extend(loaded)
        print(f"Loaded {len(loaded)} doc(s) from {os.path.basename(path)}")
    return docs


def main():
    embeddings, collection_name = get_embeddings(EMBED_MODEL)

    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    # Clean slate so re-running does not pile up duplicate chunks.
    vs.reset_collection()
    vs.add_documents(chunks)

    print(
        f"[{EMBED_MODEL}] Indexed {len(chunks)} chunks "
        f"from {len(docs)} loaded docs into '{collection_name}'."
    )


if __name__ == "__main__":
    main()