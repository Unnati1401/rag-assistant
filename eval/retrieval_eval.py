import os
import json
import argparse
from app.rag import build_retriever


def load_golden(path="eval/golden.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def per_query_metrics(retrieved_sources, expected_sources, k):
    """retrieved_sources: ordered list of source basenames (top-k).
    expected_sources: list of relevant source basenames."""
    expected = set(expected_sources)
    topk = retrieved_sources[:k]
    relevant = [s for s in topk if s in expected]

    recall = len(set(relevant) & expected) / len(expected) if expected else 0.0
    precision = len(relevant) / len(topk) if topk else 0.0
    rr = 0.0
    for i, s in enumerate(topk):
        if s in expected:
            rr = 1.0 / (i + 1)
            break
    return recall, precision, rr


def evaluate(model, k, golden, verbose=False):
    retriever = build_retriever(model, k=k)
    recalls, precisions, rrs = [], [], []

    for item in golden:
        docs = retriever.invoke(item["question"])
        retrieved_sources = [
            os.path.basename(d.metadata.get("source", "unknown")) for d in docs
        ]
        recall, precision, rr = per_query_metrics(
            retrieved_sources, item["expected_sources"], k
        )
        recalls.append(recall)
        precisions.append(precision)
        rrs.append(rr)

        if verbose:
            hit = "OK " if recall > 0 else "MISS"
            print(f"  [{hit}] q{item['id']}: expected {item['expected_sources']}, "
                  f"got {retrieved_sources}")

    n = len(golden)
    return {
        "model": model,
        "k": k,
        "recall@k": sum(recalls) / n,
        "precision@k": sum(precisions) / n,
        "MRR": sum(rrs) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["openai", "hf"])
    parser.add_argument("--ks", nargs="+", type=int, default=[4])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    golden = load_golden()
    print(f"Loaded {len(golden)} golden questions.\n")

    results = []
    for model in args.models:
        for k in args.ks:
            print(f"=== model={model}, k={k} ===")
            res = evaluate(model, k, golden, verbose=args.verbose)
            results.append(res)
            print()

    # summary table
    print(f"{'model':<8} {'k':<3} {'recall@k':<10} {'precision@k':<12} {'MRR':<6}")
    print("-" * 42)
    for r in results:
        print(f"{r['model']:<8} {r['k']:<3} {r['recall@k']:<10.3f} "
              f"{r['precision@k']:<12.3f} {r['MRR']:<6.3f}")


if __name__ == "__main__":
    main()