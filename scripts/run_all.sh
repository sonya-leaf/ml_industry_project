#!/usr/bin/env bash
# Run parsing + QA for all systems with the project's own commands, score with eval.py and
# the LLM judge, and collect every metric into one table.
#
#   bash scripts/run_all.sh [OUT_DIR, default out/compare_sh/<date_time>]                  (Windows: "C:\Program Files\Git\bin\bash.exe" scripts/run_all.sh)
#   SYSTEMS="direct tool_assisted" bash scripts/run_all.sh
#
# Uses .venv's Python if present (override: PYTHON=...). Needs QWEN_* in .env and Ollama
# running (judge). Outputs:
#   OUT_DIR/results.csv   section,system,metric,value
#   OUT_DIR/results.txt   full eval/judge printouts (per-field P/R/F1 included)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${PYTHON:-}" ]; then
  for PYTHON in .venv/Scripts/python.exe .venv/bin/python python; do
    command -v "$PYTHON" > /dev/null && break
  done
fi
python() { "$PYTHON" "$@"; }

OUT=${1:-out/compare_sh/$(date +%Y%m%d_%H%M)}
SYSTEMS=${SYSTEMS:-"baseline direct tool_assisted"}
LOG=$OUT/results.txt
CSV=$OUT/results.csv
mkdir -p "$OUT"
: > "$LOG"
echo "section,system,metric,value" > "$CSV"

# ev SECTION SYSTEM <eval.py args...>: print to the log, add "label: number" lines to the csv
ev() {
  local sec=$1 sys=$2
  shift 2
  echo "== $sec [$sys]" | tee -a "$LOG"
  python scripts/eval.py "$@" | tee -a "$LOG" | awk -v sec="$sec" -v sys="$sys" '
    !/P=/ && match($0, /[0-9]+\.[0-9]+/) {
      m = substr($0, 1, RSTART - 1); gsub(/^ +|[ :]+$/, "", m)
      print sec "," sys ",\"" m "\"," substr($0, RSTART, RLENGTH)
    }' >> "$CSV"
}

ev retrieval all retrieval --k 5
ev guardrail all safety

for S in $SYSTEMS; do
  P=$OUT/$S
  python scripts/run.py parse --system "$S" --set parsing --out "$P/parsing"
  python scripts/run.py parse --system "$S" --set qa --out "$P/qa_records"
  python scripts/run.py qa --system "$S" --records "$P/qa_records" --out "$P/qa"

  if [ "$S" = baseline ]; then
    ev parsing "$S" parsing --system baseline --set parsing
    ev qa_records "$S" parsing --system baseline --set qa
  else
    ev parsing "$S" parsing --system "$S" --set parsing --pred-dir "$P/parsing"
    ev qa_records "$S" parsing --system "$S" --set qa --pred-dir "$P/qa_records"
  fi
  ev qa "$S" qa-gold --system "$S" --pred-dir "$P/qa" --records "$P/qa_records"

  python scripts/judge.py parsing --pred-dir "$P/parsing" --out "$P/judge_parsing" --force
  ev judge_parsing "$S" judge-parsing --system "$S" --pred-dir "$P/parsing" --judge-dir "$P/judge_parsing"
  python scripts/judge.py qa --pred-dir "$P/qa" --out "$P/judge_qa" --force
  ev judge_qa "$S" judge-qa --system "$S" --pred-dir "$P/qa" --judge-dir "$P/judge_qa"
done

echo
column -s, -t "$CSV" 2>/dev/null || cat "$CSV"
echo "results: $CSV"
