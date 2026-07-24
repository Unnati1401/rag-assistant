# Experiment Log — RAG Assistant

A running record of changes and their measured effect. Each entry: what was
changed, the numbers before/after, and what it means. This is the artifact that
proves the project was engineered against measurements rather than eyeballed.

---

## Experiment 1 — Baseline retrieval: OpenAI vs. local MiniLM embeddings

**Date:** Week 4, initial run
**Golden set:** 10 FastAPI questions (`eval/golden.json`)
**Config:** k=4, RecursiveCharacterTextSplitter (chunk_size=800, overlap=100),
Chroma, one collection per embedding model.

**Command:**
```
uv run python -m eval.retrieval_eval --verbose
```

**Results (retrieval metrics):**

| model  | k | recall@k | precision@k | MRR   |
|--------|---|----------|-------------|-------|
| openai | 4 | 0.800    | 0.650       | 0.800 |
| hf     | 4 | 1.000    | 0.650       | 0.883 |

(openai = `text-embedding-3-small`, 1536-dim; hf = `all-MiniLM-L6-v2`, 384-dim)

**What happened:**
- The free local MiniLM model outperformed paid OpenAI embeddings on this
  corpus: higher recall (1.000 vs 0.800) and higher MRR (0.883 vs 0.800).
- OpenAI missed 2 of 10 questions:
  - q6 ("what library validates request body data?" -> Pydantic, `body.md`):
    retrieved `path-params.md` + `features.html`, missed `body.md`.
  - q9 ("what interactive docs does FastAPI provide?" -> `first-steps.md`):
    retrieved `path-params.md` + `features.html`, missed `first-steps.md`.
  - Both are questions where the answer term does not lexically echo the
    question wording (harder semantic match). MiniLM handled both correctly.
- The off-topic distractor `attention.pdf` never appeared in any top-4 result,
  so misses are FastAPI docs being confused with each other, not PDF pollution.

**Interpretation / hypotheses:**
- MiniLM is fine-tuned specifically for sentence-similarity, which suits
  short-question-to-passage matching on clean text; OpenAI's model is
  general-purpose. "Paid/bigger" does not automatically mean better retrieval
  on a specific domain corpus.
- CAVEAT: 10 questions is a small, noisy sample. Do not over-conclude "MiniLM
  is better." Need a larger golden set (target 25-30) to be confident.


## Experiment 2 — Answer-quality baseline (LLM-as-judge)

**Golden set:** 10 questions | **Judge:** gpt-4o-mini | **Answer model:** gpt-4o-mini | **k=4**

**Command:**
```
uv run python -m eval.answer_eval --verbose
```

**Results (answer quality, 1-5):**

| model  | correctness | faithfulness | scored |
|--------|-------------|--------------|--------|
| openai | 4.00        | 4.40         | 10/10  |
| hf     | 4.40        | 4.60         | 10/10  |

**What happened:**
- hf scores higher on both dimensions, consistent with its stronger retrieval
  in Experiment 1 (better chunks in -> better answers out).
- Faithfulness >= correctness for both models = answers are grounded in the
  retrieved context, not hallucinated. Grounding is working.

**Interpretation:**
- All scores are high (4.0-4.6), i.e. the metrics are saturated. With only 10
  fairly easy questions there is little room to discriminate. Need a larger,
  harder golden set before these numbers can meaningfully move.

## Experiment 3 — Corrected baseline on 25-question set

**Golden set:** 25 questions (corrected) | **Judge:** gpt-4o-mini (JSON mode) | **k=4**

**Fixes applied since Exp 2:**
- Judge now runs in OpenAI JSON mode -> no more parse failures (was 24/25, now 25/25).
- Corrected 2 ground-truth questions found during eval:
  - q24: stale reference (`uvicorn ...`) -> docs now recommend `fastapi dev`. Model was right.
  - q25: over-specified reference (claimed HTTP 422) -> docs describe a validation
    error but never state the code. Loosened reference to match the docs.

**Retrieval metrics:**

| model  | k | recall@k | precision@k | MRR   |
|--------|---|----------|-------------|-------|
| openai | 4 | 0.920    | 0.770       | 0.900 |
| hf     | 4 | 1.000    | 0.690       | 0.913 |

**Answer quality (1-5):**

| model  | correctness | faithfulness | scored |
|--------|-------------|--------------|--------|
| openai | 4.36        | 4.64         | 25/25  |
| hf     | 4.40        | 4.56         | 25/25  |

**Key observations:**
- Clear retrieval tradeoff: hf has higher recall (1.00 vs 0.92) but openai has
  higher precision (0.77 vs 0.69). hf finds the right doc more often; openai
  pulls fewer off-target chunks.
- Answer scores are near-tied, i.e. the answer LLM (gpt-4o-mini) compensates for
  moderate retrieval differences on this corpus.
- Correctness rose vs Exp 2 (4.25/4.29 -> 4.36/4.40) purely from fixing the two
  bad reference answers. Lesson: bad ground truth silently depresses metrics.
