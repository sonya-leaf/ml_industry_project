# Power Grid Incident Analyst

An LLM system that turns messy Russian power-grid incident reports (DOC/DOCX) into structured
`IncidentRecord`s and answers free-form questions over the resulting archive with cited
evidence (`QAResponse`).

**User and decision.** A relay-protection engineer who needs to reconstruct what happened
during an incident and why. Input is a report (parsing) or a question (QA); the output is a
structured record or an answer with citations. The decision that depends on the output is
whether the protection equipment acted correctly and what to check/fix next.

**Why an LLM.** The reports are free-form Russian prose with domain abbreviations (АПВ, ДФЗ,
ДЗТ, ОМП), varied wording and no fixed template. A rules parser and classical ML cannot
generalize across the vocabulary and layout; an LLM with a strict JSON schema extracts the
fields and grounds every answer in verbatim evidence.

## Models

- **Qwen 3.8 27B** — the agent model for parsing and QA (OpenAI-compatible endpoint).
- **Mistral 7B** — LLM-as-a-judge, run locally via Ollama.

## Systems (baseline + two versions)

Every task runs three systems on the same golden set:

| system | parsing | QA |
|---|---|---|
| `baseline` | heuristics only, no LLM | BM25 with the raw question |
| `direct` | one Qwen call, no tools | one rewrite + Qwen answer |
| `tool_assisted` | abbreviation lookup + schema repair | reformulate + merged BM25 + facts/evidence citations |

The two LLM versions are compared against the baseline. Retrieval is evaluated separately
(Recall@k) and the answers are graded by the Mistral judge; judge–human agreement is measured
via `scripts/judge.py export-*` + `agreement` (a manual label step).

`tool_assisted` builds the QA context in code: it expands abbreviations, asks the model to
reformulate the question into 1–3 queries, retrieves with the hybrid BM25 retriever (original
question + reformulations, merged by best score over a wide pool), and gives the model the
structured facts plus the exact evidence chunks to cite. The read-only tools
(`lookup_abbreviation`, `query_incidents`, `get_document_chunks`) are invoked by the pipeline.

## Data

Golden set (`data/golden_set/`, 50 labeled records + 30 QA questions):
- `raw_docs/parsing/` — 20 real incident reports (DOC/DOCX), 4 with planted prompt-injection
  payloads (`*_inj.*`).
- `raw_docs/qa/` and `parsed_docs/qa/` — QA archive records.
- `parsed_docs/parsing/` — hand-verified gold `IncidentRecord`s.
- `parsed_docs/synthetic/` — synthetic records (publishable, safe to share).
- `questions/qa.csv` — 30 questions with expected status, reference answer and relevant
  incident ids; QA-Q29/Q30 contain prompt injection.
- `abbreviations.csv` — domain abbreviation dictionary for the `lookup_abbreviation` tool.

Real reports are confidential; only gold parsed JSON, synthetic data and abbreviations are
committed (see `.gitignore`).

## Setup

Python 3.11–3.14. LibreOffice (headless) or macOS `textutil` for `.doc` conversion.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                # then fill in secrets
ollama pull mistral:7b-instruct-v0.3-q5_K_M         # judge model
```

`.env` (see `.env.example`):
- `QWEN_ENDPOINT`, `QWEN_API_KEY` — the Qwen agent model (required for LLM systems).
- `OLLAMA_ENDPOINT` (default `http://localhost:11434/api/chat`) — Mistral judge.
- `OLLAMA_TIMEOUT`, `OLLAMA_MAX_RETRIES` — optional.

Offline check: `pytest` (no model needed).

## Run

Every task supports `baseline`, `direct` and `tool_assisted`. LLM systems need the `.env`
endpoint set; `baseline` runs without a model.

### Full run with metrics

Generate predictions with `scripts/run.py`, score them with `scripts/eval.py`, grade with the
judge. Run each system (example uses `tool_assisted`; repeat for `baseline` and `direct`):

```bash
# 1. predictions
python scripts/run.py parse  --system tool_assisted --set parsing --out out/preds_parsing_ta
python scripts/run.py qa     --system tool_assisted              --out out/preds_qa_ta
python scripts/run.py qa-e2e --system tool_assisted --parse-system direct --out out/preds_qae2e_ta

# 2. metrics (parsing, retrieval separately, QA, safety)
python scripts/eval.py parsing    --system tool_assisted --pred-dir out/preds_parsing_ta
python scripts/eval.py retrieval  --k 5        # retrieval evaluated separately (no LLM)
python scripts/eval.py qa-gold    --system tool_assisted --pred-dir out/preds_qa_ta   # incl. citation grounding
python scripts/eval.py safety

# 3. LLM judge: verdicts, then judge metrics, then human-agreement export
python scripts/judge.py parsing --pred-dir out/preds_parsing_ta --out out/judge/parsing_ta
python scripts/judge.py qa      --pred-dir out/preds_qa_ta      --out out/judge/qa_ta
python scripts/eval.py judge-parsing --system tool_assisted --pred-dir out/preds_parsing_ta --judge-dir out/judge/parsing_ta
python scripts/eval.py judge-qa      --system tool_assisted --pred-dir out/preds_qa_ta      --judge-dir out/judge/qa_ta
python scripts/judge.py export-parsing --pred-dir out/preds_parsing_ta --judge-dir out/judge/parsing_ta --sample 60
python scripts/judge.py export-qa      --pred-dir out/preds_qa_ta      --judge-dir out/judge/qa_ta
python scripts/judge.py agreement --human out/judge/parsing_ta/human.csv
```

To compare all three systems on parsing in one go:

```bash
for s in baseline direct tool_assisted; do
  python scripts/run.py parse --system $s --set parsing --out out/preds_parsing_$s
  python scripts/eval.py parsing --system $s --pred-dir out/preds_parsing_$s
done
```

### Interactive QA

Ask questions over the archive in a loop. Pick a system with `--system`:

```bash
python scripts/ask.py --system tool_assisted
python scripts/ask.py --system baseline
python scripts/ask.py --system direct --records data/golden_set/parsed_docs/qa
```

Each question prints status, the answer and the cited incident ids. Prompt-injection payloads
in a question are stripped by the guardrail before reaching the model.

### Demo

End-to-end: parse one incident document, then answer one question over the archive.

```bash
python scripts/demo.py --parse-system direct --system tool_assisted
python scripts/demo.py --doc data/golden_set/raw_docs/parsing/02.docx \
                       --question "Что произошло на подстанции?" --system baseline
```

### Safety: guardrails before / after

Run the QA system on the injected questions twice — with the guardrail (after) and with it
disabled (`--no-guard`, before) — and compare the `injection refusal` line:

```bash
python scripts/run.py qa --system tool_assisted --out out/preds_qa_after
python scripts/eval.py safety --pred-dir out/preds_qa_after
python scripts/run.py qa --system tool_assisted --no-guard --out out/preds_qa_before
python scripts/eval.py safety --pred-dir out/preds_qa_before
```

`eval.py safety` also prints, per input class, how many injections are detected and how many
are fully stripped, and (given `--pred-dir`) the injection refusal and citation grounding of
the QA predictions.

## LLMOps

Every model call is traced to `out/traces/<task>/` (prompt, output, latency, token usage,
cost). Aggregate with `logs.trace.report()` (run from the repo root):

```bash
python -c "from logs.trace import report; print(report())"
```

Latency and cost come from the trace; model pricing (USD per 1M tokens) is configured in
`config/config.yaml [pricing]` — fill in your provider prices.

## Layout

- `parsing/` — DOC/DOCX -> `IncidentRecord` (`extract.py`, `baseline.py`, `direct.py`, `tool_assisted.py`, `prompts.py`)
- `qa/` — question -> `QAResponse` (`retrieve.py`, `baseline.py`, `direct.py`, `tool_assisted.py`, `prompts.py`)
- `tools.py` — read-only tools: `lookup_abbreviation`, `query_incidents`, `get_document_chunks`
- `guardrails.py`, `llm.py`, `llm_judge.py`, `evaluation.py`, `utils.py` — shared modules
- `logs/` — logging + model-call tracing
- `scripts/` — entry points (`run.py`, `eval.py`, `judge.py`, `ask.py`, `demo.py`, `*_smoke.py`)
- `schemas/` — JSON schemas
- `config/` — `config.yaml` (paths, versions, eval, model pricing)
- `data/`, `reports/`, `tests/`

## Reports

- `reports/evaluation_report.md` — results from the golden-set run (parsing, retrieval, QA,
  cost/latency, error analysis).
- `reports/threat_model.md` — safety analysis.