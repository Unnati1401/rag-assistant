"""
Chunk-size sweep (retrieval metrics).

Chunk size is fixed at ingest time, so each size needs its own re-ingest. This
script is self-contained: for each chunk size it re-chunks the corpus, ingests
into a TEMPORARY collection (sweep_*), and runs the retrieval metrics. It does
not touch your main collections or pipeline files.

Only chunk_size varies; embedding model (openai), overlap (100), and k (3) are
held constant so the experiment has a single variable.

Run from the project root:
    uv run python -m eval.chunk_sweep
    uv run python -m eval.chunk_sweep --sizes 400 800 1200 --k 3
"""

import os
import argparse

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.rag import get_embeddings
from app.ingest import load_docs
from eval.retrieval_eval import load_golden, per_query_metrics

OVERLAP = 100
MODEL = "openai"


def evaluate_chunk_size(chunk_size, k, embeddings, docs, golden):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=OVERLAP
    )
    chunks = splitter.split_documents(docs)

    collection = f"sweep_{MODEL}_cs{chunk_size}"
    vs = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )
    vs.reset_collection()
    vs.add_documents(chunks)
    retriever = vs.as_retriever(search_kwargs={"k": k})

    recalls, precisions, rrs = [], [], []
    for item in golden:
        result_docs = retriever.invoke(item["question"])
        retrieved_sources = [
            os.path.basename(d.metadata.get("source", "unknown")) for d in result_docs
        ]
        recall, precision, rr = per_query_metrics(
            retrieved_sources, item["expected_sources"], k
        )
        recalls.append(recall)
        precisions.append(precision)
        rrs.append(rr)

    n = len(golden)
    return {
        "chunk_size": chunk_size,
        "n_chunks": len(chunks),
        "recall@k": sum(recalls) / n,
        "precision@k": sum(precisions) / n,
        "MRR": sum(rrs) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 800, 1200])
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    golden = load_golden()
    embeddings, _ = get_embeddings(MODEL)
    docs = load_docs()
    print(f"Loaded {len(golden)} golden questions | model={MODEL} k={args.k} overlap={OVERLAP}\n")

    results = []
    for cs in args.sizes:
        print(f"=== chunk_size={cs} (ingesting...) ===")
        results.append(evaluate_chunk_size(cs, args.k, embeddings, docs, golden))

    print(f"\n{'chunk_size':<11} {'n_chunks':<9} {'recall@k':<10} {'precision@k':<12} {'MRR':<6}")
    print("-" * 50)
    for r in results:
        print(f"{r['chunk_size']:<11} {r['n_chunks']:<9} {r['recall@k']:<10.3f} "
              f"{r['precision@k']:<12.3f} {r['MRR']:<6.3f}")


if __name__ == "__main__":
    main()