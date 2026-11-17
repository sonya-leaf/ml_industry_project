# Threat Model (we are still working on it)

Same golden-set run as the evaluation report. `guardrails.py` protects the pipeline against
prompt injection on two surfaces: the source documents and the user question.

## Metrics

- **guardrail detection** — share of injection samples the pattern scan flags.
- **fully sanitized** — share stripped of the payload before any LLM.
- **injection refusal** — share of injection questions the model refuses to answer.
- **refusal on missing** — share of questions the model declines when the data is absent.

## Documents (4 `*_inj`)

| metric | value |
|---|---|
| detected | 3/4 |
| fully sanitized | 4/4 |

The 4th document has no imperative keyword, so it is not flagged, but it is still stripped
before the parsing LLM; the in-prompt rule is then the only guard.

## Questions (QA-Q29, QA-Q30)

| system | injection refusal | refusal on missing |
|---|---|---|
| baseline | 1.00 | 1.00 |
| direct | 1.00 | 1.00 |
| tool_assisted | 1.00 | 1.00 |

All systems refuse both injection questions and decline to answer when the data is missing. The
in-prompt security rule is a second line on top of the sanitizer.

## Residual risk

- Payloads with no imperative keyword slip past the pattern scan; the in-prompt rule is then the
  only guard.
- Over-flagging can remove a benign span that merely looks like an instruction.
- A citation could echo an injected span that survived inside a chunk (not yet validated against
  the unsanitized text).

## Why no human approval on tools

The tools are read-only: `lookup_abbreviation` (static `abbreviations.csv`),
`query_incidents` (BM25 over the in-memory index), `get_document_chunks` (returns already-loaded
evidence). None writes, contacts external systems or changes state, and the pipeline calls them
itself, so no approval step is needed. 