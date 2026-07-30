"""
Answer-quality evaluation using LLM-as-judge.

For each golden question it runs the full RAG pipeline (retrieve + generate),
then asks a judge model to score the generated answer on two dimensions:

  - correctness:  does the answer match the reference answer? (1-5)
  - faithfulness: is the answer grounded in the retrieved context, i.e.
                  not hallucinated? (1-5)

The judge runs in OpenAI JSON mode (response_format=json_object) so its output
is always valid JSON and free-text reasoning can never break parsing.

Run from the project root:
    uv run python -m eval.answer_eval
    uv run python -m eval.answer_eval --models openai hf --verbose
    uv run python -m eval.answer_eval --judge-model gpt-4o   # stronger judge
"""

import json
import re
import argparse

from app.rag import build_retriever, make_llm, answer, format_context


def load_golden(path="eval/golden.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


JUDGE_SYSTEM = (
    "You are a strict but fair evaluator of question-answering systems. "
    "You will be given a QUESTION, a REFERENCE answer (the known-correct "
    "answer), the CONTEXT that was retrieved, and the SYSTEM answer produced "
    "by the model under test. Score the SYSTEM answer on two dimensions:\n\n"
    "1. correctness (1-5): How well does the SYSTEM answer match the REFERENCE "
    "answer in meaning? 5 = fully correct and complete; 3 = partially correct "
    "or missing detail; 1 = wrong or irrelevant.\n"
    "2. faithfulness (1-5): Is the SYSTEM answer supported by the CONTEXT? "
    "5 = every claim is grounded in the context; 3 = mostly grounded with minor "
    "unsupported detail; 1 = largely fabricated / not in the context.\n\n"
    "Respond with a JSON object in exactly this shape: "
    '{"correctness": <int 1-5>, "faithfulness": <int 1-5>, '
    '"reasoning": "<one short sentence>"}. '
    "Keep reasoning brief and do not include line breaks in it."
)

JUDGE_TEMPLATE = (
    "QUESTION:\n{question}\n\n"
    "REFERENCE answer:\n{reference}\n\n"
    "CONTEXT that was retrieved:\n{context}\n\n"
    "SYSTEM answer to evaluate:\n{system_answer}"
)


def parse_judge_json(raw):
    """Extract the JSON object from a judge response. With JSON mode on, `raw`
    is already valid JSON; the fence-stripping is a harmless safety net."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def judge_answer(question, reference, context, system_answer, judge_llm):
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        reference=reference,
        context=context,
        system_answer=system_answer,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    raw = judge_llm.invoke(messages).content
    try:
        result = parse_judge_json(raw)
        return (
            int(result["correctness"]),
            int(result["faithfulness"]),
            result.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return None, None, f"PARSE_ERROR: {e} | raw={raw[:200]}"


def evaluate(model, judge_model, golden, verbose=False):
    retriever = build_retriever(model, k=4)
    answer_llm = make_llm(model="gpt-4o-mini", temperature=0)
    # JSON mode guarantees the judge returns parseable JSON.
    judge_llm = make_llm(model=judge_model, temperature=0).bind(
        response_format={"type": "json_object"}
    )

    correctness_scores, faithfulness_scores = [], []
    parse_errors = []

    for item in golden:
        text, docs = answer(item["question"], retriever, answer_llm)
        context = format_context(docs)
        c, f, reasoning = judge_answer(
            item["question"], item["reference_answer"], context, text, judge_llm
        )

        if c is not None:
            correctness_scores.append(c)
            faithfulness_scores.append(f)
        else:
            parse_errors.append(item["id"])

        if verbose:
            print(f"  q{item['id']}: correctness={c} faithfulness={f}")
            print(f"      system: {text[:120].strip()}...")
            print(f"      judge:  {reasoning[:120]}")

    n = len(correctness_scores) or 1
    return {
        "model": model,
        "judge": judge_model,
        "avg_correctness": sum(correctness_scores) / n,
        "avg_faithfulness": sum(faithfulness_scores) / n,
        "n_scored": len(correctness_scores),
        "n_total": len(golden),
        "parse_errors": parse_errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["openai", "hf"])
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--golden", default="eval/golden.json")
    args = parser.parse_args()

    golden = load_golden(args.golden)
    print(f"Loaded {len(golden)} golden questions. Judge: {args.judge_model}\n")

    results = []
    for model in args.models:
        print(f"=== embedding model={model} ===")
        res = evaluate(model, args.judge_model, golden, verbose=args.verbose)
        results.append(res)
        if res["parse_errors"]:
            print(f"  parse errors on q: {res['parse_errors']}")
        print()

    print(f"{'model':<8} {'correctness':<13} {'faithfulness':<14} {'scored':<8}")
    print("-" * 45)
    for r in results:
        print(f"{r['model']:<8} {r['avg_correctness']:<13.2f} "
              f"{r['avg_faithfulness']:<14.2f} {r['n_scored']}/{r['n_total']}")


if __name__ == "__main__":
    main()