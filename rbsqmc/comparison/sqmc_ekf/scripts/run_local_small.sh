#!/usr/bin/env bash
# Run the SQMC–EKF comparison locally with config_local_small.json.
#
# Runs both methods end-to-end on CPU into a timestamped output directory,
# then validates the combined artifacts. Use --methods ekf|sqmc for a partial
# run, and --study to follow up with the F8 seed/particle sensitivity study.
#
# Usage:
#   ./run_local_small.sh [--study] [--seeds 0 1 2] [--particle-counts 128 512 2048]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="rbsqmc/comparison/sqmc_ekf/scripts/config/config_local_small.json"
DATA="rbsqmc/data/results.csv"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_BASE="${OUT_BASE:-rbsqmc/comparison/sqmc_ekf/outputs_local}"

STUDY=0
SEEDS=(0 1 2)
COUNTS=(128 512 2048)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --study) STUDY=1; shift ;;
    --seeds) shift; SEEDS=("$@"); break ;;
    --particle-counts) shift; COUNTS=("$@"); break ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "== Running local comparison (config_local_small.json) =="
"$PYTHON" -m rbsqmc.comparison.sqmc_ekf.run \
  --config "$CONFIG" \
  --data "$DATA" \
  --output-dir "$OUT_BASE"

RUN_DIR="$OUT_BASE"
echo "== Comparison complete: $RUN_DIR =="

if [[ "${STUDY:-0}" == "1" ]]; then
  echo "== Running F8 seed/particle study =="
  "$PYTHON" -m rbsqmc.comparison.sqmc_ekf.scripts.seed_particle_study \
    --run-dir "$RUN_DIR" \
    --data "$DATA" \
    --seeds "${SEEDS[@]}" \
    --particle-counts "${COUNTS[@]}"
fi