import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "openai")

# Chunk config must match ingest.py so BM25 lines up with the stored vectors.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_embeddings(kind):
    """model -> (embeddings, collection_name)."""
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
        raise ValueError(f"Unknown embedding kind: {kind!r} (use 'openai' or 'hf')")


def build_retriever(kind=DEFAULT_EMBED_MODEL, k=3, persist_directory="./chroma_db"):
    """Plain vector retriever. Used by the eval scripts for baseline comparison."""
    embeddings, collection_name = get_embeddings(kind)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def _bm25_over_corpus(k):
    """BM25 keyword retriever built over the same chunks that were embedded."""
    from langchain_community.retrievers import BM25Retriever
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from app.ingest import load_docs

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(load_docs())
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k
    return bm25


def build_hybrid_retriever(kind=DEFAULT_EMBED_MODEL, k=3, persist_directory="./chroma_db"):
    """Hybrid (vector + BM25 keyword) fused. No PyTorch -> deploy-friendly.
    Measured recall 1.000 / precision 0.747 / MRR 0.950 on the 50-question set."""
    from langchain_classic.retrievers import EnsembleRetriever

    embeddings, collection_name = get_embeddings(kind)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vector = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25 = _bm25_over_corpus(k)
    return EnsembleRetriever(retrievers=[vector, bm25], weights=[0.5, 0.5])


def build_hybrid_rerank_retriever(
    kind=DEFAULT_EMBED_MODEL, top_k=3, wide_k=10, persist_directory="./chroma_db"
):
    """Full config: hybrid + cross-encoder reranker. Best metrics but pulls in
    PyTorch, so used locally, not in the light deployment.
    Measured recall 1.000 / precision 0.820 / MRR 0.943."""
    from langchain_classic.retrievers import (
        EnsembleRetriever,
        ContextualCompressionRetriever,
    )
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    embeddings, collection_name = get_embeddings(kind)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    wide_vector = vectorstore.as_retriever(search_kwargs={"k": wide_k})
    bm25 = _bm25_over_corpus(wide_k)
    hybrid = EnsembleRetriever(retrievers=[wide_vector, bm25], weights=[0.5, 0.5])
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_k)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=hybrid)


def make_llm(model="gpt-4o-mini", temperature=0):
    return ChatOpenAI(model=model, temperature=temperature)


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question using ONLY the "
        "context provided. If the answer is not in the context, say you "
        "don't know. Be concise.",
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def format_context(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def answer(question, retriever, llm):
    """Retrieve, generate, and return (answer_text, retrieved_docs)."""
    docs = retriever.invoke(question)
    context = format_context(docs)
    messages = _PROMPT.invoke({"context": context, "question": question})
    text = llm.invoke(messages).content
    return text, docs