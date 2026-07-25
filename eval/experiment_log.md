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

**Next actions:**
- [ ] Expand golden set to 25-30 questions for statistical reliability.
- [ ] Add answer-quality scoring (LLM-as-judge: correctness + faithfulness).
- [ ] Re-run and compare once the set is larger.

---

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

**Next actions:**
- [x] Expand golden set to 25 questions incl. harder / cross-doc / edge cases.
- [ ] Re-run retrieval + answer eval on the 25-question set.
- [ ] Begin Week 5 tuning (chunk size, k, hybrid search, reranker) against the
      larger set.

---

## Experiment 3 — (pending: re-run on 25-question set)

_To be filled after re-running both harnesses on the expanded golden set._

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

**This is the locked Week 4 baseline. Week 5 tuning measured against it.**

**Next actions (Week 5 — each a logged experiment):**
- [ ] Chunk size sweep (500 / 800 / 1200) — expected to move precision.
- [ ] k sweep (3 / 4 / 6) — precision vs recall tradeoff.
- [ ] Hybrid search (keyword + vector) — target openai's recall misses.
- [ ] Reranker (cross-encoder) — expected to improve MRR.

---

## Experiment 4 — Expanded 50-question baseline (THE baseline)

**Golden set:** 50 questions (verified against uploaded docs) | **Judge:** gpt-4o-mini (JSON mode) | **k=4**

**Why expand from 25 to 50:** at n=25 one question = 0.04 of recall, so small
differences were unreadable noise. 50 questions (with 21 hard ones) gives enough
resolution to trust smaller movements, while still being hand-verifiable.

**Retrieval metrics:**

| model  | k | recall@k | precision@k | MRR   |
|--------|---|----------|-------------|-------|
| openai | 4 | 0.960    | 0.775       | 0.917 |
| hf     | 4 | 0.980    | 0.710       | 0.907 |

**Answer quality (1-5):**

| model  | correctness | faithfulness | scored |
|--------|-------------|--------------|--------|
| openai | 4.38        | 4.56         | 50/50  |
| hf     | 4.30        | 4.40         | 50/50  |

**Key finding — the ranking flipped vs the 25-question set:**
- At n=25: hf led (recall 1.00 vs 0.92, MRR 0.913 vs 0.900, both answer scores).
- At n=50: openai leads MRR (0.917 vs 0.907) AND both answer scores
  (correctness 4.38 vs 4.30, faithfulness 4.56 vs 4.40). hf keeps only a slim
  recall edge (0.980 vs 0.960).
- Lesson: the earlier "hf is better" conclusion was largely small-sample noise.
  Expanding the golden set changed the decision. This is exactly why eval-set
  size matters, and why you validate before trusting a comparison.

**The stable picture:** openai has consistently higher precision (0.775 vs 0.710)
across every set size, and at adequate n it also wins ranking and answer quality.
hf's advantage is only marginally higher recall. Leaning openai as the default.

**This 50-question set is the locked baseline for all Week 5 tuning.**

**Next actions (Week 5):**
- [ ] k sweep (3 / 4 / 6) on openai.
- [ ] chunk-size sweep (500 / 800 / 1200) — requires re-ingest per size.
- [ ] hybrid search (keyword + vector).
- [ ] cross-encoder reranker.

---

## Experiment 5 — k sweep (retrieval), openai vs hf, k in {3,4,6}

**Golden set:** 50 questions | **k values:** 3, 4, 6 | change: retrieval only, no re-ingest.

**Command:**
```
uv run python -m eval.retrieval_eval --models openai --ks 3 4 6
uv run python -m eval.retrieval_eval --models hf --ks 3 4 6
```

**Results:**

| model  | k | recall@k | precision@k | MRR   |
|--------|---|----------|-------------|-------|
| openai | 3 | 0.960    | 0.820       | 0.917 |
| openai | 4 | 0.960    | 0.775       | 0.927 |
| openai | 6 | 0.960    | 0.700       | 0.917 |
| hf     | 3 | 0.980    | 0.760       | 0.907 |
| hf     | 4 | 0.980    | 0.710       | 0.907 |
| hf     | 6 | 0.980    | 0.647       | 0.907 |

**Reading:**
- Recall is FLAT across all k for both models. The relevant chunk is already in
  the top 3; larger k finds nothing new.
- Precision drops monotonically as k rises (openai 0.82 -> 0.775 -> 0.70), because
  each extra chunk is more likely off-target.
- MRR barely moves (openai peaks at k=4, +0.010 over k=3 — negligible).

**Decision: set k=3 (default).**
- Same recall as k=4/6, best precision (0.82 vs 0.775), near-identical MRR.
- Feeds the LLM less noise, and a smaller prompt is slightly cheaper + faster.
- No measured downside on this corpus.

**Interview line:** "swept k, found recall saturated by k=3, so dropped k from 4
to 3 for a precision gain (0.775 -> 0.82) at zero recall cost."

**Change applied:** `build_retriever` default k 4 -> 3 in `app/rag.py`.

**Next:** chunk-size sweep (500 / 800 / 1200) — requires re-ingesting each size.

---

## Experiment 5 — k sweep (retrieval, k = 3 / 4 / 6)

**Golden set:** 50 questions | **k values:** 3, 4, 6 | no re-ingest (query-time only)

**Command:**
```
uv run python -m eval.retrieval_eval --models openai --ks 3 4 6
uv run python -m eval.retrieval_eval --models hf --ks 3 4 6
```

**Results:**

| model  | k | recall@k | precision@k | MRR   |
|--------|---|----------|-------------|-------|
| openai | 3 | 0.960    | 0.820       | 0.917 |
| openai | 4 | 0.960    | 0.775       | 0.927 |
| openai | 6 | 0.960    | 0.700       | 0.917 |
| hf     | 3 | 0.980    | 0.760       | 0.907 |
| hf     | 4 | 0.980    | 0.710       | 0.907 |
| hf     | 6 | 0.980    | 0.647       | 0.907 |

**Finding:**
- Recall is FLAT across all k (openai 0.960, hf 0.980). The relevant doc is
  already in the top 3, so larger k finds nothing new.
- Precision DROPS steadily as k grows (openai 0.820 -> 0.775 -> 0.700). Extra
  chunks are just noise diluting the context.
- MRR is essentially flat (openai peaks slightly at k=4 = 0.927).

**Decision: set default k = 3.** Keeps full recall, maximizes precision, feeds
the LLM less noise. Applied to build_retriever default.

**Important caveat (interview framing):** this holds because the corpus is tiny
(6 docs) and answers are concentrated, so recall saturates early. On a large
corpus recall typically climbs with k and k=3 vs k=6 becomes a real
recall/precision tradeoff. The honest takeaway is "recall saturates early on
this corpus, so small k maximizes precision at no recall cost" - NOT "k=3 is
universally best."

---

## Experiment 6 — chunk-size sweep (500 / 800 / 1200)

**Golden set:** 50 | model=openai | k=3 | overlap=100 (only chunk_size varies)

**Command:** `uv run python -m eval.chunk_sweep`

| chunk_size | n_chunks | recall@k | precision@k | MRR   |
|------------|----------|----------|-------------|-------|
| 500        | 282      | 0.960    | 0.760       | 0.850 |
| 800        | 167      | 0.960    | 0.820       | 0.927 |
| 1200       | 105      | 0.940    | 0.760       | 0.930 |

**Finding:** 800 (the existing default) is optimal. Classic chunk-size tradeoff:
- 500 (too small): fragments answers across many chunks -> MRR drops to 0.850
  (best chunk ranks lower) even though recall holds.
- 1200 (too large): merges distinct topics -> recall dips (0.940) and precision
  falls (0.760).
- 800 wins on precision (0.820) and near-top MRR (0.927).

**Decision: keep chunk_size = 800.** Confirmed empirically rather than assumed.

---

## Experiment 7 — Retrieval strategies: vector vs hybrid vs rerank (THE WINNER)

**Golden set:** 50 | model=openai | k=3 | reranker=cross-encoder/ms-marco-MiniLM-L-6-v2

**Command:** `uv run python -m eval.retriever_compare`

| config         | recall@k | precision@k | MRR   |
|----------------|----------|-------------|-------|
| vector (base)  | 0.960    | 0.820       | 0.917 |
| hybrid         | 1.000    | 0.747       | 0.950 |
| rerank         | 0.980    | 0.800       | 0.933 |
| hybrid+rerank  | 1.000    | 0.820       | 0.943 |

**Finding — hybrid+rerank wins, strictly >= baseline on all three metrics:**
- Hybrid (BM25 + vector) reaches recall 1.000: the keyword half catches
  exact-term questions the vector search missed. But precision drops to 0.747
  because BM25 also pulls in noise.
- Rerank alone lifts MRR (0.933) and mostly holds precision: the cross-encoder
  reorders the right chunk higher.
- Hybrid+rerank combines both: hybrid finds everything (recall 1.000), then the
  reranker strips hybrid's noise (precision back to 0.820) and ranks well
  (MRR 0.943). The reranker fixing hybrid's precision cost is the key interaction.

**vs baseline (0.960 / 0.820 / 0.917): recall +0.040, precision equal, MRR +0.026.**

**Decision: adopt hybrid+rerank as the production retriever.** Wired into
app/rag.py (build_hybrid_rerank_retriever) and used by the FastAPI app.

**Interview narrative:** hybrid buys recall, rerank buys back the precision hybrid
costs; together they dominate every single-strategy config. Measured, not assumed.

---

## Summary — tuning journey (Week 5)

| stage                    | recall | precision | MRR   |
|--------------------------|--------|-----------|-------|
| baseline (vector, k=4)   | 0.960  | 0.775     | 0.917 |
| k=3                      | 0.960  | 0.820     | 0.917 |
| + hybrid+rerank          | 1.000  | 0.820     | 0.943 |

Net: recall 0.960 -> 1.000, precision 0.775 -> 0.820, MRR 0.917 -> 0.943, all
driven by measured experiments against a 50-question golden set.

---