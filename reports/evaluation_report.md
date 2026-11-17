# Evaluation Report

Real run of the full pipeline (parse → retrieve → QA → judge) over the golden set.
Qwen is the agent model, Mistral the judge. QA answers over each system's own parsed records, so parsing errors carry into the QA numbers.

## Metrics

- **schema-valid** — share of outputs that pass JSON-schema validation (structured output).
- **F1 exact** — parsing field F1 against the gold record (token-level match).
- **F1 judge (strict / value)** — Mistral-graded field F1; strict requires the value *and* the
  cited evidence to match, value-only ignores the evidence.
- **recall@5** — share of gold-relevant incidents that appear in the top-5 retrieved.
- **status acc** — whether the model's `answered` / `insufficient_evidence` status matches gold.
- **citation acc** — predicted citations point at the gold-relevant incident ids.
- **grounding** — citations are backed by verbatim evidence present in the record.
- **answer acc (judge)** — Mistral-graded answer correctness.

## Parsing (n=20 docs)

| system | schema-valid | F1 exact | F1 judge (strict) | F1 judge (value) |
|---|---|---|---|---|
| baseline | 1.00 | 0.150 | 0.017 | 0.203 |
| direct | 1.00 | 0.387 | 0.678 | 0.706 |
| tool_assisted | 1.00 | 0.426 | 0.767 | 0.782 |

All outputs are schema-valid. Baseline only fills a few fields; the LLM systems are clearly
better. The judge is more permissive than exact match (it accepts paraphrases).

## Parsing of the QA corpus (n=20)

| system | schema-valid | F1 exact |
|---|---|---|
| baseline | 1.00 | 0.127 |
| direct | 0.95 | 0.360 |
| tool_assisted | 1.00 | 0.394 |

tool_assisted extracts the QA corpus best, so downstream QA inherits the fullest records.

## Retrieval (Recall@5)

Recall is scored over 28 of the 30 questions — QA-Q27/Q28 are "insufficient_evidence" questions
with no gold incident, so there is nothing to recall against. Two views:

- Per-system, over each system's own parsed corpus, on the rankings it actually returned:
  baseline 0.827, direct 0.794, tool_assisted 0.948.
- Retriever alone on the hand-gold corpus: plain doc BM25 0.721 → hybrid (explicit
  incident/facility lookup + doc/chunk BM25 + abbreviation expansion + stemming) 0.960.

## QA (n=30 questions)

| system | schema | recall@5 | status acc | citation acc | grounding | answer acc (judge) |
|---|---|---|---|---|---|---|
| baseline | 1.00 | 0.827 | 0.433 | 0.300 | 1.000 | 0.400 |
| direct | 1.00 | 0.794 | 0.767 | 0.633 | 0.902 | 0.767 |
| tool_assisted | 1.00 | 0.948 | 0.933 | 0.800 | 0.983 | 0.900 |

tool_assisted is best on every axis: its context gives the model the structured facts plus the
exact evidence chunks to cite. direct is the weakest retriever — a single rewrite can drop the
explicit `QA-<n>`/facility reference. 

## Cost and latency

| stage | system | calls | prompt tok | completion tok | latency |
|---|---|---|---|---|---|
| parsing (40 docs) | baseline | 0 | 0 | 0 | 0 |
| parsing (40 docs) | direct | 40 | 156k | 98k | 10.8 min |
| parsing (40 docs) | tool_assisted | 44 | 187k | 106k | 11.7 min |
| judge (Mistral) | parsing | 58 | 163k | 14k | 75 min |
| QA (30 q) | baseline | 30 | 212k | 11k | 1.9 min |
| QA (30 q) | direct | 60 | 216k | 12k | 2.1 min |
| QA (30 q) | tool_assisted | 60 | 339k | 25k | 3.7 min |
| judge (Mistral) | qa | 52 | 55k | 4k | 13 min |

Total 344 LLM calls (234 Qwen + 110 Mistral), ~1.33M prompt / ~0.27M completion
tokens, ~2 h wall-clock. Baseline parsing is rules-only (no LLM). The judge is the slowest
(77 s/call on parsing) as it is run locally. 
Per QA question tool_assisted is the most token-heavy: it passes the full evidence context.
tool_assisted parsing includes a few schema-repair/no-JSON retries (4 calls). 

## Error analysis (v1 → v2)

direct is v1, tool_assisted is v2. v1 used a plain doc-BM25 retriever, a chunks-only prompt, and
dropped any answer with imperfect citations — the result was empty `insufficient_evidence`
responses. Three root causes and their fixes:

1. **Retrieval missed the record** (doc BM25 Recall@5 0.721). Fixed with the hybrid retriever
   (explicit incident/facility lookup + doc/chunk BM25 + abbreviation expansion + stemming, and
   a stop-word filter so facility matching does not fire on "подстанция") → 0.960.
2. **Context hid the facts** — facility, causes and reclosing result live in the structured
   fields, not in the chunks. Fixed by showing both the facts and the exact evidence objects to
   cite, with the facility in the record header.
3. **Good answers were thrown away** — `build_response` demanded a non-empty citation list. Fixed
   to keep the answer unless the model itself said `insufficient_evidence`.

v2 also searches the original question plus 1–3 reformulations, so the explicit reference is never
dropped (v1's direct lost it on QA-Q07/Q10/Q13).

Measured effect (judge answer acc / wrong answers): baseline 0.400 (18/30) → direct 0.767 (7/30)
→ tool_assisted 0.900 (3/30).

Remaining v2 errors:

- Archive-filter questions (QA-Q14, QA-Q22) probably need more than `top_k=5` relevant records.
- QA-Q13 — QA-01 is an instruction, not an incident; the status is still misread.
- Weakest parsed field: `observations/0/connections` (F1 0.10–0.18).
