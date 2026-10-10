#!/usr/bin/env bash
# Run from any directory. Prefer the workspace's prepared environment.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-../.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON=python3; fi
mode="${1:-selftest}"
selftest() {
  "$PYTHON" -m unittest -v test_hedge_core
  "$PYTHON" level5_hedge.py --selftest
  "$PYTHON" test_plumbing.py
}
need_model() {
  : "${OPT_BASE_URL:=${HEATROD_BASE_URL:-}}"
  if [ -z "$OPT_BASE_URL" ]; then
    echo "Set OPT_BASE_URL to your model endpoint, or use ./run_all.sh offline" >&2
    exit 1
  fi
  export OPT_BASE_URL
}
case "$mode" in
  selftest) selftest ;;
  offline)
    mkdir -p artifacts
    "$PYTHON" agent.py --offline --all --log artifacts/offline.jsonl ;;
  real)
    need_model
    selftest
    mkdir -p artifacts
    for i in 1 2 3; do
      "$PYTHON" agent.py --all --tag baseline --seed "$i" --log artifacts/attempts.jsonl
    done ;;
  experiments)
    need_model
    mkdir -p artifacts
    for i in 1 2 3; do
      "$PYTHON" agent.py --all --no-tools --tag no-tools --seed "$i" --log artifacts/attempts.jsonl
      "$PYTHON" agent.py --all --feedback score --tag score-only --seed "$i" --log artifacts/attempts.jsonl
      "$PYTHON" agent.py --all --hint 0 --tag no-hint --seed "$i" --log artifacts/attempts.jsonl
    done ;;
  *) echo "usage: $0 {selftest|offline|real|experiments}" >&2; exit 2 ;;
esac
