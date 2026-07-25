"""
Compare retrieval strategies: vector (baseline) vs hybrid vs reranker vs both.

All configs evaluated at k=3 on the 50-question golden set, holding embedding
model (openai) and chunk config constant. Self-contained: uses the existing
docs_openai Chroma collection for vector search and rebuilds chunks for BM25.

Run from the project root:
    uv run python -m eval.retriever_compare
"""

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from app.rag import get_embeddings
from app.ingest import load_docs
from eval.retrieval_eval import load_golden, per_query_metrics

MODEL = "openai"
K = 3
CHUNK_SIZE = 800
OVERLAP = 100
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def score(retriever, golden, k=K):
    recalls, precisions, rrs = [], [], []
    for item in golden:
        docs = retriever.invoke(item["question"])
        sources = [os.path.basename(d.metadata.get("source", "unknown")) for d in docs]
        recall, precision, rr = per_query_metrics(sources, item["expected_sources"], k)
        recalls.append(recall)
        precisions.append(precision)
        rrs.append(rr)
    n = len(golden)
    return sum(recalls) / n, sum(precisions) / n, sum(rrs) / n


def main():
    golden = load_golden()
    embeddings, collection_name = get_embeddings(MODEL)
    print(f"Loaded {len(golden)} golden questions | model={MODEL} k={K}\n")

    # --- vector retriever (existing collection) ---
    vs = Chroma(collection_name=collection_name, embedding_function=embeddings,
                persist_directory="./chroma_db")
    vector = vs.as_retriever(search_kwargs={"k": K})

    # --- BM25 keyword retriever (rebuild chunks to match ingest) ---
    docs = load_docs()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP).split_documents(docs)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = K

    # --- hybrid = vector + keyword, fused ---
    hybrid = EnsembleRetriever(retrievers=[vector, bm25], weights=[0.5, 0.5])

    # --- reranker over a wider vector net (top-10 -> top-3) ---
    print(f"Loading cross-encoder reranker ({RERANK_MODEL})...")
    ce = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
    reranker = CrossEncoderReranker(model=ce, top_n=K)
    wide_vector = vs.as_retriever(search_kwargs={"k": 10})
    rerank = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=wide_vector)

    # --- hybrid + rerank (rerank a wide hybrid net) ---
    wide_bm25 = BM25Retriever.from_documents(chunks); wide_bm25.k = 10
    wide_hybrid = EnsembleRetriever(
        retrievers=[vs.as_retriever(search_kwargs={"k": 10}), wide_bm25],
        weights=[0.5, 0.5])
    hybrid_rerank = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=wide_hybrid)

    configs = [
        ("vector", vector),
        ("hybrid", hybrid),
        ("rerank", rerank),
        ("hybrid+rerank", hybrid_rerank),
    ]

    rows = []
    for name, r in configs:
        print(f"=== {name} ===")
        rows.append((name,) + score(r, golden))

    print(f"\n{'config':<15} {'recall@k':<10} {'precision@k':<12} {'MRR':<6}")
    print("-" * 45)
    for name, rec, prec, mrr in rows:
        print(f"{name:<15} {rec:<10.3f} {prec:<12.3f} {mrr:<6.3f}")


if __name__ == "__main__":
    main()