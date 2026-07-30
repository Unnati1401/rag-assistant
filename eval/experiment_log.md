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

**Important caveat:** this holds because the corpus is tiny
(6 docs) and answers are concentrated, so recall saturates early. On a large
corpus recall typically climbs with k and k=3 vs k=6 becomes a real
recall/precision tradeoff. The honest takeaway is "recall saturates early on
this corpus, so small k maximizes precision at no recall cost" - NOT "k=3 is
universally best."

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


## Summary — tuning journey (Week 5)

| stage                    | recall | precision | MRR   |
|--------------------------|--------|-----------|-------|
| baseline (vector, k=4)   | 0.960  | 0.775     | 0.917 |
| k=3                      | 0.960  | 0.820     | 0.917 |
| + hybrid+rerank          | 1.000  | 0.820     | 0.943 |

Net: recall 0.960 -> 1.000, precision 0.775 -> 0.820, MRR 0.917 -> 0.943, all
driven by measured experiments against a 50-question golden set.

---

## Experiment 8 — Multi-source grounding: docs vs web vs both

**Feature:** user-selectable grounding source (docs / web / both) on `/query`,
mirroring how Gemini/ChatGPT expose web grounding as a user control.
**Eval set:** 10 questions requiring current/web info NOT in the local corpus
(`eval/web_golden.json`). **Web:** Tavily. **Judge:** gpt-4o-mini (JSON mode).

**Command:** `uv run python -m eval.web_eval --verbose`

| mode | answered_rate | avg_faithfulness | url_source_rate | avg_sources |
|------|---------------|------------------|-----------------|-------------|
| docs | 0.20          | 5.00             | 0.00            | 5.8         |
| web  | 0.90          | 5.00             | 1.00            | 3.0         |
| both | 0.80          | 5.00             | 1.00            | 8.8         |

**Finding — the core contrast validates the feature:**
- docs-only correctly REFUSES 8/10 (answers aren't in the corpus). Refusal is
  correct grounded behavior, not failure - no hallucination.
- web answers 90% with a URL source on every answer, faithfulness 5.00.
- Faithfulness = 5.00 across all modes: whenever the system answered, it stayed
  grounded in retrieved context. No fabrication.

**Honest observations:**
- The 2 questions docs answered (q9, q10, FastAPI Cloud) are actually mentioned
  in first-steps.md, so they weren't purely web-requiring.
- web refused q8 (alternatives to FastAPI): search didn't surface a clean list,
  so the model refused rather than inventing - good behavior.
- both underperformed web on answered_rate (0.80 vs 0.90) and gave one muddled
  answer (q4): fusing doc + web context can dilute focus. More sources (8.8) is
  not always better.
- Web answers were only as accurate as the search results: some stale facts
  surfaced (e.g. an outdated minimum Python version). Web grounding inherits the
  quality of the underlying search index - a real limitation, not a bug in the
  pipeline.

**Takeaway:** docs grounding refuses safely; web grounding answers with
citations; both trades focus for coverage. Exposing the choice to the user
(rather than auto-routing) avoids mis-route risk and matches production
assistants - the measured comparison is what justifies offering all three.

---

## Experiment 9 — OKF structured knowledge: does it improve retrieval?
 
**Feature:** adopted Google's Open Knowledge Format (OKF, 2026) — atomic Markdown
entries with YAML frontmatter (id, category, confidence, source) — in a separate
`docs_okf` collection, plus metadata-filtered retrieval via Chroma where-clauses.
**Compared on:** 40/50 golden questions whose topic has an OKF entry. OKF scored
on `okf_id` (reliable identifier).
 
**Command:** `uv run python -m eval.okf_compare`
 
| corpus   | recall@k | precision@k | MRR   |
|----------|----------|-------------|-------|
| raw docs | 0.975    | 0.842       | 0.946 |
| okf      | 0.800    | 0.267       | 0.775 |
 
**Finding — OKF underperformed raw docs at this corpus size (honest negative):**
- With only 4 short OKF entries, all terse FastAPI-parameter text, their
  embeddings cluster tightly in vector space, so semantic ranking is near-random
  (precision 0.267 ~ 1-of-4 relevant). Example: "how do you declare a request
  body?" ranked response-model above request-body.
- Raw docs win because verbose full-page text gives more distinguishing signal
  per topic to embed against.
**What DID work — metadata filtering (the capability OKF unlocks):**
- Category filter (`category=routing`) returned exactly the routing entries,
  confirming Chroma where-clause filtering over OKF metadata works.
**Interpretation:** OKF's atomic structure needs corpus SCALE to pay off. At 4
entries it hurts ranking; at scale, metadata filtering narrows the search space
BEFORE semantic ranking, which is precisely what mitigates the clustering problem
seen here. Structure alone is not enough - the metadata filtering is the lever,
and it only bites on a large, categorized corpus.
 
**Takeaway (interview framing):** implemented OKF + metadata-filtered retrieval,
measured it honestly, and interpreted a negative result - OKF's value is
scale-dependent, not automatic.

## Experiment 10 — Distractor robustness: FastAPI eval on the expanded corpus
 
After adding ~29 Kubernetes docs (~6 -> ~36 documents), re-ran the FastAPI
50-question eval to measure whether the new cross-domain distractors hurt
retrieval on the original questions.
 
**Retrieval (openai):**
 
| setting            | recall@k | precision@k | MRR   |
|--------------------|----------|-------------|-------|
| baseline (pre-K8s, k=3) | 0.960 | 0.820   | 0.917 |
| expanded (k=3)     | 0.960    | 0.787       | 0.927 |
| expanded (k=4)     | 0.960    | 0.750       | 0.927 |
 
**Answer quality (expanded):** correctness 4.80, faithfulness 4.84 (50/50).
(Higher than Exp 4's 4.38/4.56 - reflects accumulated golden-answer fixes and
the JSON-mode judge, not the K8s docs helping FastAPI answers.)
 
**Finding:** expanding the corpus 6x cost only 0.033 precision at k=3
(0.820 -> 0.787); recall held at 0.960 and MRR rose slightly. The K8s distractors
rarely displace the correct FastAPI chunk - the two domains separate cleanly in
embedding space. Retrieval is robust to cross-domain distractors.
 
**Caveat:** this only evaluates the FastAPI half. The K8s docs themselves are
still untested - a dedicated K8s golden set is needed to evaluate the new domain
(next).
 
---
 
 ## Experiment 11 — Kubernetes domain evaluation (new corpus)
 
Built a dedicated K8s golden set to evaluate the expanded domain (the FastAPI
golden set doesn't cover Kubernetes). Added a `--golden` flag to the eval scripts
to target different question sets.
 
**Golden set:** 26 K8s questions (`eval/k8s_golden.json`) — 5 easy, 10 medium,
11 hard, including 5 multi-doc questions (answer spans two docs) and cross-topic
disambiguation (e.g. Deployment vs StatefulSet).
 
**Command:**
```
uv run python -m eval.retrieval_eval --models openai --ks 3 --golden eval/k8s_golden.json
uv run python -m eval.answer_eval --models openai --golden eval/k8s_golden.json
```
 
**Results:**
 
| set                    | recall@k | precision@k | MRR   | correctness | faithfulness |
|------------------------|----------|-------------|-------|-------------|--------------|
| 18 easy/medium (initial) | 1.000  | 0.870       | 0.907 | 5.00        | 5.00         |
| 26 incl. hard/multi-doc  | 0.942  | 0.872       | 0.936 | 5.00        | 5.00         |
 
**Findings:**
- On single-topic questions, retrieval is near-perfect: K8s docs are each a
  distinct, self-contained concept, so they separate cleanly in embedding space
  (unlike the more overlapping FastAPI param docs). Well-separated topics retrieve
  better than semantically overlapping ones - a real retrieval insight.
- Adding hard/multi-doc questions dropped recall 1.000 -> 0.942: the multi-doc
  questions require BOTH relevant sources in the top-3, which is genuinely harder.
  This is the eval doing its job - discriminating rather than saturating.
- Precision rose slightly (0.870 -> 0.872) because multi-doc questions have two
  relevant sources, so more of the top-3 is on-target.
**Honest caveat:** answer quality is a flat 5.00 across easy and hard sets. The
LLM knows Kubernetes well and questions map to clear docs, so the answer-quality
metric is not discriminating here - retrieval metrics are the more informative
signal for this domain. The K8s set (26) is also smaller than the FastAPI set (50).
 
**Combined story (Exp 10 + 11):** the system scaled to a 6x-larger, two-domain
corpus with no quality loss - FastAPI retrieval held (precision -0.03) and the new
K8s domain evaluates strongly. Retrieval is robust to cross-domain distractors and
generalizes to a new domain.
 
---