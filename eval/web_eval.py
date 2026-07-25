"""
Grounding-mode comparison: docs vs web vs both.

Uses questions that REQUIRE current/web information not in the local corpus.
On these, docs-only should correctly refuse (grounded behavior, not failure),
while web/both should answer with cited URLs. Metrics:

  - answered_rate:  fraction that gave a substantive answer (vs "I don't know")
  - avg_faithfulness (1-5): over answered questions, is the answer grounded in
                            the retrieved context? (LLM-as-judge, JSON mode)
  - url_source_rate: fraction whose sources include at least one URL (web-grounded)
  - avg_sources:    average number of sources returned

Requires OPENAI_API_KEY and TAVILY_API_KEY.

Run from the project root:
    uv run python -m eval.web_eval
    uv run python -m eval.web_eval --modes docs web both --verbose
"""

import json
import re
import argparse

from app.rag import build_grounded_retriever, make_llm, answer, format_context


def load_golden(path="eval/web_golden.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


REFUSAL_MARKERS = [
    "don't know", "do not know", "not in the context", "cannot find",
    "no information", "not provided", "isn't in the context", "is not in the context",
    "unable to find", "does not contain",
]


def is_answered(text):
    t = text.lower()
    return not any(m in t for m in REFUSAL_MARKERS)


JUDGE_SYSTEM = (
    "You evaluate whether an answer is grounded in the provided context. "
    "Given a QUESTION, the retrieved CONTEXT, and the SYSTEM answer, score "
    "faithfulness (1-5): 5 = every claim is supported by the context; "
    "3 = mostly grounded, minor unsupported detail; 1 = largely unsupported. "
    'Respond with JSON only: {"faithfulness": <int 1-5>, "reasoning": "<one short sentence>"}'
)


def parse_json(raw):
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s != -1 and e != -1:
        cleaned = cleaned[s:e + 1]
    return json.loads(cleaned)


def judge_faithfulness(question, context, system_answer, judge):
    msgs = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nSYSTEM answer:\n{system_answer}"},
    ]
    try:
        r = parse_json(judge.invoke(msgs).content)
        return int(r["faithfulness"]), r.get("reasoning", "")
    except Exception as e:
        return None, f"PARSE_ERROR: {e}"


def evaluate(mode, golden, llm, judge, verbose=False):
    retriever = build_grounded_retriever(mode, k=3)
    answered, faith, url_hits, n_sources = [], [], [], []

    for item in golden:
        text, docs = answer(item["question"], retriever, llm)
        context = format_context(docs)
        sources = [d.metadata.get("source", "") for d in docs]

        ans = is_answered(text)
        answered.append(1 if ans else 0)
        url_hits.append(1 if any(str(s).startswith("http") for s in sources) else 0)
        n_sources.append(len(sources))
        if ans:
            f, reason = judge_faithfulness(item["question"], context, text, judge)
            if f is not None:
                faith.append(f)

        if verbose:
            mark = "ANSWERED" if ans else "refused "
            print(f"  q{item['id']} [{mark}]: {text[:80].strip()}...")

    n = len(golden)
    return {
        "mode": mode,
        "answered_rate": sum(answered) / n,
        "avg_faithfulness": (sum(faith) / len(faith)) if faith else 0.0,
        "url_source_rate": sum(url_hits) / n,
        "avg_sources": sum(n_sources) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["docs", "web", "both"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    golden = load_golden()
    llm = make_llm()
    judge = make_llm().bind(response_format={"type": "json_object"})
    print(f"Loaded {len(golden)} web-requiring questions.\n")

    results = []
    for mode in args.modes:
        print(f"=== mode={mode} ===")
        results.append(evaluate(mode, golden, llm, judge, verbose=args.verbose))
        print()

    print(f"{'mode':<6} {'answered':<10} {'faithfulness':<13} {'url_sources':<12} {'avg_sources':<11}")
    print("-" * 54)
    for r in results:
        print(f"{r['mode']:<6} {r['answered_rate']:<10.2f} {r['avg_faithfulness']:<13.2f} "
              f"{r['url_source_rate']:<12.2f} {r['avg_sources']:<11.1f}")


if __name__ == "__main__":
    main()