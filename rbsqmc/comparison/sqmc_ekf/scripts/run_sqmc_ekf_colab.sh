#!/usr/bin/env bash
# Activate the desired virtual environment first; colab is discovered on PATH.
# Falls back to python3 when no venv is active (plain `python` may not exist).
#
# The real (default) comparison runs:
#   1. EKF locally on CPU, then
#   2. SQMC on a Colab GPU, then
#   3. combines the two partials locally.
#
# Usage:
#   run_sqmc_ekf_colab.sh [--config overrides.json]   # EKF local + SQMC GPU (real run)
#   run_sqmc_ekf_colab.sh --smoke [--config overrides.json]  # same, small config
#   run_sqmc_ekf_colab.sh --local [--smoke]   # everything on this machine's CPU (smoke only)
#   run_sqmc_ekf_colab.sh --resume <output-dir>       # reconnect to a detached run
#
# The Colab session is preserved when local monitoring disconnects; reconnect
# with --resume to collect the results without restarting training.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Prefer the repo's .venv so the launcher never silently falls back to a
# system Python that lacks the project's dependencies (jax, etc.).
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
if [ -n "${PYTHON:-}" ]; then
    INTERPRETER="$PYTHON"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    INTERPRETER="$REPO_ROOT/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    INTERPRETER="python"
else
    INTERPRETER="python3"
fi
# The launcher runs as a script, so sys.path[0] is the scripts directory and
# the repo root is NOT importable: its artifact validation imports rbsqmc.*
# lazily and would raise ModuleNotFoundError. Export PYTHONPATH so the
# launcher process (and every child it spawns) can resolve the package.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec "$INTERPRETER" "$SCRIPT_DIR/run_sqmc_ekf_local.py" "$@"
