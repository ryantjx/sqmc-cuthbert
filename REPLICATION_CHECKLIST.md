# Replication Checklist — `sqmc` and `rbsqmc`

This document is an extensive, step-by-step checklist for verifying that the
code transferred to `ryantjx/sqmc-cuthbert` is fully replicable. It covers the
two packages `sqmc/` (the shared SQMC library and CPU/GPU benchmark suite) and
`rbsqmc/` (the football application). Work through the sections in order; tick
each box and record the result (pass/fail + any error) in the notes column.

> **Conventions.** All commands run from the **repository root** with the
> virtual environment active. `RBSQMC_PLATFORM` selects the JAX platform
> (default `cpu`; set `RBSQMC_PLATFORM=cuda` on a GPU host). GPU steps require
> a JAX CUDA backend and are marked **[GPU]**.

---

## 0. Environment and data setup

| # | Check | Command / action | Result |
| --- | --- | --- | --- |
| 0.1 | Python version ≥ 3.10 | `python --version` | |
| 0.2 | Virtual environment created | `python -m venv .venv && source .venv/bin/activate` | |
| 0.3 | Dependencies installed | `pip install -r requirements.txt` | |
| 0.4 | Sobol' direction table generated | `python sqmc/qmc/_generate_sobol_data.py` | |
| 0.5 | `_sobol_direction_numbers.npz` exists | `ls sqmc/qmc/_sobol_direction_numbers.npz` | |
| 0.6 | Football results data present | `ls rbsqmc/data/results.csv` | |
| 0.7 | Parquet cache present (or download once) | `ls rbsqmc/data/results.parquet` — if absent, run the data download (`download=True` in `rbsqmc/src/data/data.py`) or copy `results.csv` → `results.parquet` | |
| 0.8 | Both packages importable from root | `python -c "import sqmc.qmc.qmc, sqmc.sqmc.sqmc, sqmc.hilbert_sort.hilbert_sort, rbsqmc.src.data.data"` | |

---

## 1. Unit tests

| # | Check | Command | Result |
| --- | --- | --- | --- |
| 1.1 | SQMC core tests | `python -m pytest sqmc/tests -q` | |
| 1.2 | SQMC comparison tests | `python -m pytest sqmc/comparison/tests -q` | |
| 1.3 | rbsqmc core tests | `python -m pytest rbsqmc/tests -q` | |
| 1.4 | rbsqmc EKF comparison tests | `python -m pytest rbsqmc/comparison/sqmc_ekf/tests -q` | |

**Expected:** all tests pass. Record the pass/fail counts for each.

---

## 2. SQMC CPU/GPU benchmark suite (`sqmc/comparison`)

These three benchmarks reproduce the Chapter 2 performance results. Run the
CPU smoke variants first, then the full profile (optionally on GPU).

### 2.1 QMC generation benchmark

| # | Check | Command | Result |
| --- | --- | --- | --- |
| 2.1.1 | CPU smoke run completes | `python -m sqmc.comparison.benchmark_qmc --platforms cpu --dimensions 2 --n-values 8 16 --repeats 2 --warmups 0` | |
| 2.1.2 | Output directory created | `ls sqmc/comparison/outputs/` (new timestamped dir) | |
| 2.1.3 | `results.json` / `results.csv` written | check inside the new output dir | |
| 2.1.4 | `cpu_gpu_comparison.json` written | check inside the new output dir | |
| 2.1.5 | `runtime.png` figure written | check inside the new output dir | |
| 2.1.6 | `status.json` shows success | `cat <outdir>/status.json` | |
| 2.1.7 | **[GPU]** Full profile runs | `python -m sqmc.comparison.benchmark_qmc --platforms cpu gpu --sequences sobol halton --dimensions 2 5 10 --n-values 128 512 2048 8192 32768 --modes fresh --scramble --implementations jax scipy --repeats 7 --warmups 2` | |

### 2.2 Hilbert sorting benchmark

| # | Check | Command | Result |
| --- | --- | --- | --- |
| 2.2.1 | CPU smoke run completes | `python -m sqmc.comparison.benchmark_hilbert_sort --platforms cpu --dimensions 2 --n-values 8 16 --repeats 2 --warmups 0` | |
| 2.2.2 | Output directory created | `ls sqmc/comparison/outputs/` | |
| 2.2.3 | `results.json` / `results.csv` written | check inside the new output dir | |
| 2.2.4 | `runtime.png` figure written | check inside the new output dir | |
| 2.2.5 | `status.json` shows success | `cat <outdir>/status.json` | |
| 2.2.6 | **[GPU]** Full profile runs | `python -m sqmc.comparison.benchmark_hilbert_sort --platforms cpu gpu --sequences sobol halton --distribution normal --dimensions 2 5 10 --n-values 128 256 512 2048 8192 --repeats 7 --warmups 2` | |

### 2.3 SQMC accuracy benchmark

| # | Check | Command | Result |
| --- | --- | --- | --- |
| 2.3.1 | CPU smoke run completes | `python -m sqmc.comparison.benchmark_sqmc --platforms cpu --dimensions 2 --particle-counts 8 16 --n-steps 3 --selection-reps 2 --validation-reps 3 --bootstrap-reps 50 --budget-seconds 0.000000001 1 --repeats 2 --warmups 0` | |
| 2.3.2 | Output directory created | `ls sqmc/comparison/outputs/` | |
| 2.3.3 | `results.json` / `results.csv` written | check inside the new output dir | |
| 2.3.4 | `accuracy_records.json` written | check inside the new output dir | |
| 2.3.5 | `budget_summary.json` written | check inside the new output dir | |
| 2.3.6 | `status.json` shows success | `cat <outdir>/status.json` | |
| 2.3.7 | **[GPU]** Full profile runs | `python -m sqmc.comparison.benchmark_sqmc --platforms cpu gpu --dimensions 2 5 --particle-counts 128 256 512 1024 2048 --n-steps 100 --budget-seconds 0.01 0.05 0.1 --datasets 1 --selection-reps 8 --validation-reps 16 --repeats 7 --warmups 2 --bootstrap-reps 500` | |

### 2.4 Colab launcher (optional, full chapter profile)

| # | Check | Command | Result |
| --- | --- | --- | --- |
| 2.4.1 | Dry-run validates config | `sqmc/comparison/scripts/run_comparison_colab.sh --dry-run` | |
| 2.4.2 | Full comparison runs | `sqmc/comparison/scripts/run_comparison_colab.sh` | |
| 2.4.3 | All three stage outputs downloaded & validated | check `sqmc/comparison/outputs/DDMMYYYY_HHMM/` for `qmc/`, `hilbert_sort/`, `sqmc/` subdirs | |

---

## 3. Football model: RB-SQMC vs factorial EKF (`rbsqmc/comparison/sqmc_ekf`)

This reproduces the Chapter 3 performance comparison.

| # | Check | Command | Result |
| --- | --- | --- | --- |
| 3.1 | Local smoke run completes | `python -m rbsqmc.comparison.sqmc_ekf.run --config rbsqmc/comparison/sqmc_ekf/scripts/config/config_smoke.json --data rbsqmc/data/results.csv --smoke` | |
| 3.2 | Output directory created | `ls rbsqmc/comparison/sqmc_ekf/outputs/` (new `DDMMYYYY_HHMM` dir) | |
| 3.3 | `ekf/`, `sqmc/`, `combined/` subdirs present | `ls <outdir>/` | |
| 3.4 | Artifacts validated | look for `OK: comparison artifacts validated` in the run output | |
| 3.5 | `run_config.json` written per method | `ls <outdir>/combined/results/{ekf,sqmc}/run_config.json` | |
| 3.6 | Plots written | `ls <outdir>/images/` (rankings, correlation, timeseries, predictions) | |
| 3.7 | **[GPU]** Full run via Colab launcher | `bash rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_colab.sh` | |
| 3.8 | Resume path works (if interrupted) | `bash rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_colab.sh --resume <outdir>` | |

---

## 4. Football model: RB-SMC vs RB-SQMC training (`rbsqmc/comparison/sqmc_smc`)

> **Note:** `compare_smc_sqmc` has no `--help` flag and no smoke mode — running
> it launches a full 100-epoch training run. Use a reduced config or expect a
> long run.

| # | Check | Command | Result |
| --- | --- | --- | --- |
| 4.1 | Module imports without error | `python -c "import rbsqmc.comparison.sqmc_smc.compare_smc_sqmc"` | |
| 4.2 | Full comparison runs (long) | `python -m rbsqmc.comparison.sqmc_smc.compare_smc_sqmc` | |
| 4.3 | Output written to `rbsqmc/outputs/compare/` | `ls rbsqmc/outputs/compare/` | |
| 4.4 | Train/test logZ histories + gradient norms written | check for `logz_history.csv` / plots in the output dir | |

---

## 5. Standalone RB-SMC pipeline (`rbsqmc/src/model/rbsmc`)

| # | Check | Command | Result |
| --- | --- | --- | --- |
| 5.1 | `optimize` phase runs | `python -m rbsqmc.src.model.rbsmc.train_model_gpu --config <config.json> optimize` | |
| 5.2 | `filter` phase runs | `python -m rbsqmc.src.model.rbsmc.train_model_gpu --config <config.json> filter` | |
| 5.3 | `predict` phase runs | `python -m rbsqmc.src.model.rbsmc.train_model_gpu --config <config.json> predict` | |
| 5.4 | `all` phase runs end-to-end | `python -m rbsqmc.src.model.rbsmc.train_model_gpu --config <config.json> all` | |
| 5.5 | Config resolution works (env var) | `RBSQMC_CONFIG=<config.json> python -m rbsqmc.src.model.rbsmc.train_model_gpu all` | |

---

## 6. Output verification

| # | Check | Command | Result |
| --- | --- | --- | --- |
| 6.1 | Every run dir has `config.json` | `find sqmc/comparison/outputs rbsqmc/comparison/sqmc_ekf/outputs -name config.json` | |
| 6.2 | Every run dir has `metadata.json` (provenance) | `find ... -name metadata.json` | |
| 6.3 | Every run dir has `status.json` | `find ... -name status.json` | |
| 6.4 | Validate an sqmc output dir | `python sqmc/comparison/scripts/validate_artifacts.py <outdir>` | |
| 6.5 | Validate an EKF output dir | `python rbsqmc/comparison/sqmc_ekf/scripts/validate_sqmc_ekf_outputs.py <outdir>` | |

---

## 7. Cross-cutting checks

| # | Check | Command / action | Result |
| --- | --- | --- | --- |
| 7.1 | No hard-coded absolute paths in source | `grep -rn "/Users/\|/home/\|C:\\\\" sqmc rbsqmc --include=*.py` (should be empty) | |
| 7.2 | No leftover references to the old repo name | `grep -rn "ryantjx/rbsqmc" sqmc rbsqmc --include=*.py --include=*.json --include=*.sh` | |
| 7.3 | `RBSQMC_PLATFORM` respected in both filters | `grep -rn "RBSQMC_PLATFORM" rbsqmc/src/model` | |
| 7.4 | Sobol' data file regenerable | re-run `python sqmc/qmc/_generate_sobol_data.py` and confirm no error | |
| 7.5 | `requirements.txt` matches imports | `pip install -r requirements.txt` succeeds in a clean env | |
| 7.6 | Git-ignored data files documented | `results.parquet` and `_sobol_direction_numbers.npz` are not committed | |

---

## 8. Summary

| Section | Pass | Fail | Notes |
| --- | --- | --- | --- |
| 0. Environment & data | | | |
| 1. Unit tests | | | |
| 2. SQMC benchmarks | | | |
| 3. RB-SQMC vs EKF | | | |
| 4. RB-SMC vs RB-SQMC | | | |
| 5. Standalone pipeline | | | |
| 6. Output verification | | | |
| 7. Cross-cutting | | | |

**Overall verdict:** [ ] Fully replicable   [ ] Partially replicable   [ ] Not replicable

**Blocking issues found:** (list any failures and the fix required)
