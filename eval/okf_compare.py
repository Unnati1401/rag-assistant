"""
Phase 3 experiment: does OKF structure improve retrieval?

Compares plain vector retrieval over the raw-doc corpus (docs_openai) against
retrieval over the atomic OKF entries (docs_okf), on the subset of golden
questions whose topic has an OKF entry.

Because OKF entries use different filenames than the raw docs, expected sources
are remapped per topic. This is a fair A/B of STRUCTURE (verbose raw pages vs
atomic OKF entries) on the same questions.

Also demonstrates metadata-filtered retrieval (category filter) as a capability.

Run from the project root:
    uv run python -m eval.okf_compare
"""

import os
from app.rag import build_retriever, build_okf_retriever
from eval.retrieval_eval import load_golden, per_query_metrics

# Map raw-doc expected sources -> OKF okf_id (the reliable identifier;
# on-disk OKF filenames may differ, so we score OKF retrieval on okf_id).
RAW_TO_OKF = {
    "path-params.md": "path-parameters",
    "query-params.md": "query-parameters",
    "body.md": "request-body",
    "response-model.md": "response-model",
}


def score(retriever, golden, source_map=None, match_field="source", k=3):
    recalls, precisions, rrs = [], [], []
    for item in golden:
        expected = item["expected_sources"]
        if source_map:
            expected = [source_map[s] for s in expected]
        docs = retriever.invoke(item["question"])
        if match_field == "okf_id":
            got = [d.metadata.get("okf_id", "") for d in docs]
        else:
            got = [os.path.basename(d.metadata.get("source", "")) for d in docs]
        rec, prec, rr = per_query_metrics(got, expected, k)
        recalls.append(rec); precisions.append(prec); rrs.append(rr)
    n = len(golden)
    return sum(recalls)/n, sum(precisions)/n, sum(rrs)/n


def main():
    golden = load_golden()
    # keep only questions whose (single) expected source has an OKF entry
    subset = [g for g in golden
              if all(s in RAW_TO_OKF for s in g["expected_sources"])]
    print(f"Comparing on {len(subset)} of {len(golden)} questions "
          f"(topics with an OKF entry).\n")

    raw = build_retriever("openai", k=3)                 # docs_openai
    okf = build_okf_retriever("openai", k=3)             # docs_okf, no filter

    r_raw = score(raw, subset, source_map=None)
    r_okf = score(okf, subset, source_map=RAW_TO_OKF, match_field="okf_id")

    print(f"{'corpus':<14} {'recall@k':<10} {'precision@k':<12} {'MRR':<6}")
    print("-" * 44)
    print(f"{'raw docs':<14} {r_raw[0]:<10.3f} {r_raw[1]:<12.3f} {r_raw[2]:<6.3f}")
    print(f"{'okf':<14} {r_okf[0]:<10.3f} {r_okf[1]:<12.3f} {r_okf[2]:<6.3f}")

    # --- capability demo: category-filtered retrieval ---
    print("\nMetadata-filter demo (category='routing'):")
    filtered = build_okf_retriever("openai", k=4, category="routing")
    docs = filtered.invoke("How do I declare parameters?")
    for d in docs:
        print(f"  {d.metadata.get('okf_id',''):<18} category={d.metadata.get('category','')}")


if __name__ == "__main__":
    main()