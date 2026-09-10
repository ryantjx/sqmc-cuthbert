#!/usr/bin/env bash
# Launch a Colab GPU, shallow-clone the public rbsqmc repository, run only the
# Hilbert-sort benchmark, download the required artifacts, and stop the session.
# Uses the shared QMC+hilbert runner with --module hilbert_sort.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../.." && pwd)"
CONFIG="${HERE}/config/hilbert_sort_gpu_benchmark_config.json"
REMOTE_RUNNER="${REPO_ROOT}/sqmc/sqmc/scripts/run_qmc_benchmarks_gpu.py"
LOCAL_OUTPUTS="${REPO_ROOT}/sqmc/hilbert_sort/outputs"
SESSION_LAUNCHED=0

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

read_config() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
        "${CONFIG}" "$1"
}

require_file() {
    [[ -s "$1" ]] || {
        log "Required file is missing or empty: $1"
        return 1
    }
}

stop_session() {
    if [[ "${SESSION_LAUNCHED}" -eq 1 ]]; then
        log "Stopping Colab session ${SESSION}"
        colab stop --session "${SESSION}" >/dev/null 2>&1 || true
        SESSION_LAUNCHED=0
    fi
}

cleanup() {
    local status=$?
    trap - EXIT
    stop_session
    exit "${status}"
}

download_required() {
    local filename="$1"
    log "Downloading ${filename}"
    colab download --session "${SESSION}" \
        "${REMOTE_OUTPUTS}/${filename}" "${LOCAL_OUTPUTS}/${filename}"
    require_file "${LOCAL_OUTPUTS}/${filename}"
}

main() {
    local dry_run=0
    if [[ "${1:-}" == "--dry-run" ]]; then
        dry_run=1
    elif [[ $# -gt 0 ]]; then
        log "Unsupported argument: $1"
        return 2
    fi

    command -v python3 >/dev/null
    require_file "${CONFIG}"
    require_file "${REMOTE_RUNNER}"

    GPU_TYPE="${GPU_TYPE:-$(read_config gpu_type)}"
    COLAB_TIMEOUT="${COLAB_TIMEOUT:-$(read_config colab_timeout)}"
    SESSION="${SESSION:-$(read_config session)}"
    REMOTE_OUTPUTS="/content/rbsqmc/$(read_config remote_output_dir)"
    CONFIG_JSON="$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",", ":")))' "${CONFIG}")"

    if [[ "${dry_run}" -eq 1 ]]; then
        log "GPU=${GPU_TYPE}, timeout=${COLAB_TIMEOUT}, session=${SESSION}"
        log "Remote source: $(read_config repo_url), branch=$(read_config repo_branch)"
        python3 "${REMOTE_RUNNER}" --config-json "${CONFIG_JSON}" --dry-run --module hilbert_sort
        return 0
    fi

    command -v colab >/dev/null
    mkdir -p "${LOCAL_OUTPUTS}"
    trap cleanup EXIT

    # Avoid accidentally attaching to stale state from an earlier failed run.
    colab stop --session "${SESSION}" >/dev/null 2>&1 || true
    log "Cloning the public repository and running the Hilbert-sort benchmark on GPU=${GPU_TYPE}"
    SESSION_LAUNCHED=1
    colab run --gpu "${GPU_TYPE}" --keep --timeout "${COLAB_TIMEOUT}" \
        --session "${SESSION}" "${REMOTE_RUNNER}" \
        --config-json "${CONFIG_JSON}" --module hilbert_sort

    local required_outputs=(
        "hilbert_sort_benchmark_gpu.png"
        "hilbert_sort_benchmark.csv"
        "hilbert_sort_benchmark_by_algorithm_gpu.png"
        "run_metadata.json"
        "run_config.json"
    )
    local output
    for output in "${required_outputs[@]}"; do
        download_required "${output}"
    done

    stop_session
    trap - EXIT
    log "Hilbert-sort GPU benchmark artifacts downloaded to ${LOCAL_OUTPUTS}"
}

main "$@"
