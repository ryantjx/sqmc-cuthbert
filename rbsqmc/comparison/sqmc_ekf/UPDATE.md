# `run_sqmc_ekf_colab.sh` — full breakdown

The script is a thin bash wrapper whose entire job is to pick the right
Python interpreter and hand off to
`rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_local.py`, which does all
the work. This document describes both layers: the shell entry point and
the pipeline it drives.

---

## 1. The shell script, line by line

```bash
#!/usr/bin/env bash
# Activate the desired virtual environment first; colab is discovered on PATH.
# Falls back to python3 when no venv is active (plain `python` may not exist).
```

The header comment describes the *old* behaviour; since commit `a5edc2b` the
script actively prefers the repo's virtual environment, so no prior
`source .venv/bin/activate` is needed.

```bash
set -euo pipefail
```

Strict mode: abort on any non-zero exit (`-e`), error on undefined
variables (`-u`), and fail a pipeline if any command in it fails (`-o
pipefail`). Without this, a silently failing setup step could cascade.

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
```

Resolves the directory containing the script itself (not the caller's
working directory), so the script works when invoked from anywhere.

```bash
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
```

Walks four levels up (`scripts/ → sqmc_ekf/ → comparison/ → rbsqmc/ →
repository root`). This is the repository root that contains `.venv/`,
`rbsqmc/` (the package), and `rbsqmc/data/results.csv`.

```bash
if [ -n "${PYTHON:-}" ]; then
    INTERPRETER="$PYTHON"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    INTERPRETER="$REPO_ROOT/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    INTERPRETER="python"
else
    INTERPRETER="python3"
fi
exec "$INTERPRETER" "$SCRIPT_DIR/run_sqmc_ekf_local.py" "$@"
```

The interpreter-resolution cascade, in priority order:

| Priority | Source | Rationale |
| --- | --- | --- |
| 1 | `$PYTHON` env var | explicit override for CI or unusual setups |
| 2 | `<repo>/.venv/bin/python` | the project's dependencies (jax, optax, …) live here; works from any shell without activation |
| 3 | `python` on PATH | legacy fallback |
| 4 | `python3` on PATH | last resort |

`exec` replaces the shell process with the Python process, so the Python
program inherits the PID, environment, and — importantly — the exit code
and signal disposition of the shell. All arguments (`"$@"`) are forwarded
unchanged.

**Why priority 2 was added:** the launcher previously fell back to
`python`/`python3`, which silently picked a system interpreter (e.g.
Homebrew Python 3.14) that has no jax. The local EKF stage then failed
instantly with a bare `CalledProcessError`, and the run directory was left
behind as a timestamp-collision blocker. Preferring `.venv` makes the
launcher work from any shell.

---

## 2. What the Python launcher does with the arguments

`run_sqmc_ekf_local.py` accepts:

| Flag | Meaning |
| --- | --- |
| `--config <overrides.json>` | JSON merged over the base profile (`config/config_gpu.json`) |
| `--resume <output-dir>` | reconnect to a detached run; uses the saved configuration, nothing else may accompany it |
| `--local` | run everything on this machine's CPU (smoke/small runs only) |
| `--smoke` | shrink the config (≤3 epochs, ≤2 replicas, ≤64 particles) |
| `--methods ekf,sqmc` | method subset for `--local` mode |
| `--dry-run` | print the resolved config and output path, then exit |
| `--no-stream-logs` | do not mirror per-epoch remote progress locally |

Before anything runs, `resolve()` builds the effective configuration:

1. Reads `config/config_gpu.json` as the base profile.
2. Merges the `--config` overrides on top.
3. Applies environment overrides (`GPU_TYPE`, `COLAB_TIMEOUT`, `SESSION`,
   `REPO_BRANCH`).
4. Pins `source_commit` to the current `git rev-parse HEAD` and
   `repo_branch` to the current branch — the exact commit must be **pushed**
   to that branch before provisioning, because the Colab VM reconstructs the
   source from it.
5. Assigns `run_id = DDMMYYYY_HHMM` (UTC) and a unique
   `session_name = <session>_<run_id>_<10-hex>`. The output directory is
   `rbsqmc/comparison/sqmc_ekf/outputs/<run_id>`; a collision (same minute)
   aborts with `UTC timestamp collision`.

---

## 3. The default (hybrid) pipeline: three stages

With no flags, the launcher runs the real comparison in three stages,
tracked in `outputs/<run_id>/status.json` under `stages`:

```
stages: { ekf: complete, sqmc: complete, combine: complete }
```

### Stage 1 — EKF locally on CPU → `output/ekf/`

```
python -m rbsqmc.comparison.sqmc_ekf.run --config <config> --data <results.csv> \
    --methods ekf --output-dir <output>/ekf
```

Runs on this machine's CPU (fast: ~1 s compile, ~13 s execution). The
environment pins `JAX_ENABLE_X64=true` and `RBSQMC_PLATFORM=cpu` so the
platform is explicit rather than inherited. This stage produces the EKF
partial run: training, prediction, metrics, plots, and its
`fitted_params.json` checkpoint.

### Stage 2 — SQMC on a Colab GPU → `output/sqmc/`

Driven by the `Launcher` class, which performs six steps:

1. **Verify the source commit** is pushed to `repo_branch` (the launcher
   refuses to provision otherwise).
2. **Provision** a Colab session of the configured GPU type (`A100`) with a
   unique session name.
3. **Upload a git bundle** of the pinned commit plus the bootstrap script;
   the VM verifies the bundle checksum, checks out the exact commit,
   installs any missing pinned dependencies, asserts the GPU is visible,
   and generates/verifies the Sobol data.
4. **Download and verify root metadata** (config digest, hardware
   provenance).
5. **Dispatch the run**: on the VM, `run_sqmc_ekf_gpu.py` sets
   `RBSQMC_PLATFORM=cuda`, `JAX_ENABLE_X64=true`, and
   `PYTHONPATH=<repo root>` (so the artifact validator can import the
   `rbsqmc` namespace package from any import site), then runs the
   comparison with `--methods sqmc`, validates the artifacts, and archives
   the run and root bundles.
6. **Download and verify** the run bundle (checksum + manifest), then stop
   the session.

Per-epoch progress is mirrored locally from the remote log while training
runs. The local process is only a *monitor*: the training itself happens on
the VM and survives a local disconnect.

### Stage 3 — Combine locally → `output/combined/`

```
python -m rbsqmc.comparison.sqmc_ekf.run --config <config> --data <results.csv> \
    --combine <output>/ekf <output>/sqmc --output-dir <output>/combined
```

Merges the two partial runs into a complete comparison (combined report,
plots, CSVs, metadata) and validates the merged artifacts. Both partials
must share the same scientific configuration and the same frozen dataset.

---

## 4. Failure handling and reconnection

The launcher distinguishes three failure classes, with different
consequences:

| Scenario | What happens to the results | What happens to the session |
| --- | --- | --- |
| **Training completes, validation fails** | **Salvaged**: the launcher downloads the run bundle with `--partial` (skipping the failed validation gate, but still verifying the archive checksum and manifest) | stopped — nothing left to keep alive |
| **Training crashes** | nothing to salvage | stopped |
| **Local monitoring disconnects** (Ctrl-C, sleep, network) | training continues on the VM | **left running** for reconnection |

The GPU worker reports which kind of failure occurred
(`failure_kind: "validation"` when `results/run_metadata.json` exists,
i.e. training finished and artifacts were exported; `"training"`
otherwise), and the launcher acts accordingly. The salvage path exists
because a validation failure previously destroyed hours of completed GPU
training: the worker archived the results, but nothing downloaded them
before the session stopped.

### Reconnecting

If local monitoring ends while the session is alive, the launcher prints:

```
Reconnect with: bash .../run_sqmc_ekf_colab.sh --resume <output-dir>
```

`--resume` re-attaches to the saved worker endpoint, waits for training to
finish, downloads and verifies the artifacts, and stops the session. It
reuses the saved configuration (a config mismatch is rejected), requires
the EKF stage to have completed, and never re-dispatches the worker (so
training cannot be started twice). A session that was already **stopped**
(`shutdown: verified_stopped`) cannot be reconnected.

---

## 5. Other modes

- `--local` — runs EKF **and** SQMC sequentially on this machine's CPU,
  then combines. Intended for smoke/small runs; the full SQMC stage is far
  too slow on CPU. `--methods` selects a subset (a single-method run
  persists that method's artifacts and skips the combined report).
- `--smoke` — caps `n_epochs` at 3, `n_reps` at 2, and `n_particles` at 64
  for a fast end-to-end check of the whole pipeline.
- `--dry-run` — prints the effective configuration (including the pinned
  `source_commit`, `run_id`, and `session_name`) and the output path
  without provisioning anything.

---

## 6. Quick reference

```bash
# Real comparison: EKF local + SQMC on Colab GPU + combine
bash rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_colab.sh \
    --config rbsqmc/comparison/sqmc_ekf/scripts/config/config_gpu.json

# Fast end-to-end pipeline check
bash rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_colab.sh --smoke

# Everything on this machine (CPU)
bash rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_colab.sh --local --smoke

# Reconnect to a detached run
bash rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_colab.sh --resume \
    rbsqmc/comparison/sqmc_ekf/outputs/<run_id>

# Preview the effective configuration without running
bash rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_colab.sh \
    --config <config.json> --dry-run
```

Prerequisites: the `colab` CLI on PATH, the pinned `source_commit` pushed
to `repo_branch`, and (for the local stages) a Python environment with the
project's dependencies — which the script now locates in `.venv`
automatically.

Outputs land in `rbsqmc/comparison/sqmc_ekf/outputs/<run_id>/`:
`ekf/`, `sqmc/`, `combined/` (each with `results/` and `images/`), plus
`status.json` (stage tracking), `comparison_config.json` (the frozen
effective configuration), and `logs.txt` (mirrored progress).