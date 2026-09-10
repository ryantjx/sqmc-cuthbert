#!/usr/bin/env bash
# Orchestrator: run the rbsqmc model_unbiased pipeline with a Colab GPU for the
# optimization phase and local compute for filtering, plotting, and prediction.
#
# This script runs LOCALLY. It:
#   1. Validates the config, bootstrap, and validator.
#   2. Creates a timestamped run directory
#      (rbsqmc/outputs/train_model_gpu_YYYYMMDD_HHmm/).
#   3. Launches the optimization-only phase on a Colab GPU via `colab run`.
#   4. Downloads the optimization artifacts from the session.
#   5. Stops the Colab session (server no longer needed).
#   6. Regenerates the logZ / gradient-norm curves locally from the summary.
#   7. Runs the filter phase locally, then the predict phase locally.
#   8. Validates all downloaded + generated artifacts.
#
# Usage:
#   bash rbsqmc/comparison/sqmc_smc/run_model_unbiased_colab.sh
#   bash rbsqmc/comparison/sqmc_smc/run_model_unbiased_colab.sh --dry-run
#
# Environment overrides:
#   GPU_TYPE       e.g. A100, T4 (default: from config)
#   COLAB_TIMEOUT  seconds (default: from config)
#   SESSION        colab session name (default: rbsqmc_model_unbiased_gpu)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
CONFIG="${HERE}/config/model_unbiased_gpu_config.json"
BOOTSTRAP="${HERE}/run_model_unbiased_gpu.py"
VALIDATOR="${HERE}/validate_model_unbiased_outputs.py"
TRAIN_MODEL="rbsqmc/src/model/train_model_gpu.py"
DRY_RUN=0
SESSION_LAUNCHED=0

progress() {
    local stream="$1"; shift
    local outer inner
    outer="$(date '+%Y-%m-%d %H:%M:%S')"
    inner="$(date '+%H:%M:%S')"
    printf '[%s] %s: [%s] %s\n' "${outer}" "${stream}" "${inner}" "$*" | tee -a "${RUN_LOG:-/dev/null}"
}

read_config() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "${CONFIG}" "$1"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        progress ERR "required command not found: $1"
        return 1
    }
}

require_file() {
    local file="$1" label="$2"
    [[ -s "${file}" ]] || {
        progress ERR "missing ${label}: ${file}"
        return 1
    }
}

download_required() {
    local file="$1"
    local attempt retries=3
    progress OUT "downloading required artifact ${file}"
    mkdir -p "$(dirname "${LOCAL_OUTPUTS}/${file}")"
    for attempt in $(seq 1 "${retries}"); do
        colab download -s "${SESSION}" "${REMOTE_OUTPUTS}/${file}" "${LOCAL_OUTPUTS}/${file}" >/dev/null 2>&1
        if [[ -s "${LOCAL_OUTPUTS}/${file}" ]]; then
            return 0
        fi
        if [[ ${attempt} -lt ${retries} ]]; then
            progress ERR "download of ${file} failed (attempt ${attempt}/${retries}); retrying in 5s"
            sleep 5
        fi
    done
    progress ERR "required artifact is unavailable or empty after ${retries} attempts: ${file}"
    return 1
}

stop_colab() {
    if [[ "${SESSION_LAUNCHED}" -eq 1 ]]; then
        progress OUT "stopping Colab session ${SESSION}"
        colab stop -s "${SESSION}" >/dev/null 2>&1 || true
    fi
}

cleanup() {
    local status=$?
    trap - EXIT
    stop_colab
    exit "${status}"
}

validate_local_outputs() {
    python3 "${VALIDATOR}" "${LOCAL_OUTPUTS}"
}

# Generate the optimization logZ + gradient-norm curves locally from the
# optimization_summary.json downloaded from the GPU.
regen_curves() {
    (cd "${REPO_ROOT}" && python3 - "${LOCAL_OUTPUTS}" <<'PY'
import json, os, sys
sys.path.insert(0, os.getcwd())
from rbsqmc.src.utils.graphic import (
    plot_logmarginal_history_train_test,
    plot_gradient_norm_curve,
)
out = sys.argv[1]
with open(os.path.join(out, "optimization_summary.json")) as f:
    summary = json.load(f)
plot_logmarginal_history_train_test(
    train_logz_history=summary["train_logZ_history"],
    train_match_count=summary.get("train_match_count"),
    test_logz_history=summary["test_logZ_history"],
    test_match_count=summary.get("test_match_count"),
    save_path=os.path.join(out, "optimization_logZ_curve.png"),
)
plot_gradient_norm_curve(
    grad_norm_history=summary["gradient_norm_history"],
    save_path=os.path.join(out, "gradient_norm_curve.png"),
)
PY
    )
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    if [[ "${1:-}" == "--dry-run" ]]; then
        DRY_RUN=1
    elif [[ $# -gt 0 ]]; then
        progress ERR "unsupported argument: $1"
        return 2
    fi

    require_command python3
    require_file "${CONFIG}" "config"
    require_file "${BOOTSTRAP}" "bootstrap script"
    require_file "${VALIDATOR}" "validator script"
    require_file "${REPO_ROOT}/${TRAIN_MODEL}" "train_model_gpu.py"

    # Activate the local virtualenv so python3 / colab use the right packages.
    if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
        progress OUT "activating virtualenv: ${REPO_ROOT}/.venv"
        # shellcheck source=/dev/null
        source "${REPO_ROOT}/.venv/bin/activate"
    else
        progress OUT "no .venv found at ${REPO_ROOT}/.venv — using system python3"
    fi

    local gpu_type timeout base_output_dir
    gpu_type="${GPU_TYPE:-$(read_config gpu_type)}"
    timeout="${COLAB_TIMEOUT:-$(read_config colab_timeout)}"
    base_output_dir="$(read_config output_dir)"
    SESSION="${SESSION:-rbsqmc_model_unbiased_gpu}"

    # Timestamped per-run output directory (TASK step 1), as a slash-delimited
    # subfolder under the base output dir: .../train_model_gpu/YYYYMMDD_HHmm/.
    RUN_SUFFIX="$(date '+%Y%m%d_%H%M')"
    RUN_DIR="${REPO_ROOT}/${base_output_dir}/${RUN_SUFFIX}"
    LOCAL_OUTPUTS="${RUN_DIR}"
    REMOTE_OUTPUTS="/content/rbsqmc/${base_output_dir}/${RUN_SUFFIX}"

    # Ensure the local run dir exists (for the run log + downloaded artifacts).
    mkdir -p "${LOCAL_OUTPUTS}"
    RUN_LOG="${LOCAL_OUTPUTS}/run_$(date '+%Y%m%d_%H%M%S').log"
    progress OUT "logging this run to ${RUN_LOG}"

    # Dry-run validation of the bootstrap against the committed config.
    python3 "${BOOTSTRAP}" --config "${CONFIG}" --dry-run >/dev/null

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf 'colab run --gpu %q --keep --timeout %q --session %q %q --config %q --output-dir %q\n' \
            "${gpu_type}" "${timeout}" "${SESSION}" "${BOOTSTRAP}" \
            "$(basename "${CONFIG}")" "${REMOTE_OUTPUTS}"
        return 0
    fi

    require_command colab
    trap cleanup EXIT

    progress OUT "launching rbsqmc model_unbiased optimization on Colab GPU=${gpu_type}"
    SESSION_LAUNCHED=1
    # --keep keeps the VM alive after the script finishes so artifacts can be
    # downloaded. Run in the background and tee colab's output into the log.
    colab run --gpu "${gpu_type}" --keep --timeout "${timeout}" \
        --session "${SESSION}" "${BOOTSTRAP}" \
        --config "$(basename "${CONFIG}")" --output-dir "${REMOTE_OUTPUTS}" \
        > >(tee -a "${RUN_LOG}") 2>&1 &
    local colab_pid=$!

    progress OUT "waiting for optimization to complete..."
    wait "${colab_pid}"
    local colab_exit=$?
    if [[ ${colab_exit} -ne 0 ]]; then
        progress ERR "colab run failed (exit ${colab_exit}) — optimization did not complete."
        return 1
    fi
    progress OUT "optimization completed, downloading artifacts immediately"

    # Download only the optimization artifacts (filter/predict run locally).
    local required=(
        params_unbiased.json
        run_config.json
        optimization_summary.json
        optimization_logZ_curve.png
        gradient_norm_curve.png
    )
    local file
    for file in "${required[@]}"; do
        download_required "${file}"
    done

    # Stop the Colab session (server no longer needed; TASK step 4).
    stop_colab
    SESSION_LAUNCHED=0

    # Regenerate the curves locally from the downloaded summary (TASK step 5).
    progress OUT "regenerating optimization curves locally from summary"
    regen_curves

    # ------------------------------------------------------------------
    # Local filter phase (TASK steps 6-7).
    # ------------------------------------------------------------------
    progress OUT "running local filter phase"
    local filter_dir="${LOCAL_OUTPUTS}/filtered"
    (cd "${REPO_ROOT}" && python3 -m rbsqmc.src.model.train_model_gpu filter \
        --params "${LOCAL_OUTPUTS}/params_unbiased.json" \
        --config "${CONFIG}" \
        --out "${filter_dir}")

    # ------------------------------------------------------------------
    # Local predict phase (TASK steps 8-9).
    # ------------------------------------------------------------------
    progress OUT "running local predict phase"
    local predict_dir="${LOCAL_OUTPUTS}/predict"
    (cd "${REPO_ROOT}" && python3 -m rbsqmc.src.model.train_model_gpu predict \
        --params "${LOCAL_OUTPUTS}/params_unbiased.json" \
        --config "${CONFIG}" \
        --out "${predict_dir}")

    # ------------------------------------------------------------------
    # Validate everything.
    # ------------------------------------------------------------------
    progress OUT "validating outputs"
    validate_local_outputs
    progress OUT "rbsqmc model_unbiased Colab acceptance completed successfully"
}

main "$@"
