# RB-SQMC: Rao-Blackwellised Sequential quasi-Monte Carlo

This repository contains the implementation and experiments for the dissertation
*High-performance sequential quasi-Monte Carlo and its application to football
match modelling* (ST980). It is organised as two Python packages:

- **`sqmc/`** — the shared high-performance SQMC library: low-discrepancy
  sequence generators (Sobol', Halton), the SQMC particle filter, Hilbert
  space-filling-curve sorting, and a CPU/GPU benchmark suite.
- **`rbsqmc/`** — the football application: a Rao-Blackwellised state-space
  model for international football results, with RB-SMC, RB-SQMC and factorial
  EKF filters, parameter estimation, prediction, and comparison pipelines.

`rbsqmc/` imports `sqmc.qmc` and `sqmc.hilbert_sort` at runtime, so both
packages must be importable from the repository root.

## Repository structure

```
sqmc/
├── qmc/                  # Sobol' and Halton generators (JAX), scrambling
├── sqmc/                 # SQMC filter (build_filter) + SMC baseline wrapper
├── hilbert_sort/         # Hilbert-curve index computation and sorting
├── comparison/           # CPU/GPU benchmark suite (QMC, Hilbert, SQMC)
└── tests/                # Unit tests for the three core modules

rbsqmc/
├── src/
│   ├── data/             # Data loading, bivariate-Poisson likelihood, EKF dataset
│   ├── model/
│   │   ├── rbsmc/        # RB-SMC filter, optimisation, train/predict pipeline
│   │   ├── rbsqmc/       # RB-SQMC filter, optimisation, prediction
│   │   └── ekf/          # Factorial EKF baseline (replicates cuthberto-carlos)
│   └── utils/            # Types, helpers, plotting, stable linear algebra
├── comparison/
│   ├── sqmc_smc/         # RB-SMC vs RB-SQMC training comparison
│   └── sqmc_ekf/         # RB-SQMC vs factorial EKF comparison (main experiment)
├── data/                 # results.csv, team sets, fixtures
├── outputs/              # Run artefacts (timestamped directories)
└── tests/                # Unit tests
```

## Installation

Python 3.10+ is required. The experiments were run with Python 3.13 and the
pinned dependencies in `requirements.txt` (JAX 0.11.0, cuthbert 0.0.14,
cuthbertlib 0.0.15, optax 0.2.8, NumPy 2.2.6, SciPy 1.18.0).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install cuthbertlib==0.0.15 ghq==0.0.5 chex==0.1.92
```

For GPU runs, install the CUDA build of JAX instead of the CPU build:

```bash
pip install "jax[cuda12]==0.11.0"
```

The football code selects its JAX platform through the `RBSQMC_PLATFORM`
environment variable (default `cpu`; set `RBSQMC_PLATFORM=cuda` on a GPU host).

### Data files

Two data files are not committed and must be created on a fresh clone:

- `sqmc/qmc/_sobol_direction_numbers.npz` — the Sobol' direction-number table,
  generated from the committed Joe–Kuo source table:

  ```bash
  python sqmc/qmc/_generate_sobol_data.py
  ```

- `rbsqmc/data/results.parquet` — the parquet cache of the international
  football results. The committed `rbsqmc/data/results.csv` is the frozen
  source used by the EKF comparison; the parquet cache is used by the football
  training pipeline. On a fresh clone, either run the data download once
  (`download=True` in `rbsqmc/src/data/data.py`) or copy `results.csv` to
  `results.parquet` if the frozen data set is intended.

## Replication

All commands are run from the repository root with the virtual environment
active. Each experiment writes a timestamped directory under its `outputs/`
folder containing `config.json`, `metadata.json`, `status.json`, results and
figures.

### 1. Unit tests

```bash
python -m pytest sqmc/tests -q
python -m pytest rbsqmc/tests -q
python -m pytest sqmc/comparison/tests -q
python -m pytest rbsqmc/comparison/sqmc_ekf/tests -q
```

### 2. CPU/GPU performance comparison (Chapter 2)

Three standalone benchmarks compare the shared algorithms on CPU and GPU.
Full documentation is in `sqmc/comparison/COMPARISON.md`. CPU-only smoke runs:

```bash
python -m sqmc.comparison.benchmark_qmc \
  --platforms cpu --dimensions 2 --n-values 8 16 --repeats 2 --warmups 0

python -m sqmc.comparison.benchmark_hilbert_sort \
  --platforms cpu --dimensions 2 --n-values 8 16 --repeats 2 --warmups 0

python -m sqmc.comparison.benchmark_sqmc \
  --platforms cpu --dimensions 2 --particle-counts 8 16 --n-steps 3 \
  --selection-reps 2 --validation-reps 3 --bootstrap-reps 50 \
  --budget-seconds 0.000000001 1 --repeats 2 --warmups 0
```

The full chapter profile (dimensions 2–60, counts up to 131072, seven timed
repetitions, both backends) is defined in
`sqmc/comparison/scripts/config/comparison_config.json` and is executed
through the Colab launcher, which provisions an A100 session, runs the three
benchmarks sequentially, and verifies each stage's artefacts:

```bash
sqmc/comparison/scripts/run_comparison_colab.sh --dry-run
sqmc/comparison/scripts/run_comparison_colab.sh
```

The three benchmarks can also be run directly on a CPU/GPU host with the
`--platforms cpu gpu` flags shown in `COMPARISON.md`.

### 3. Football model: RB-SQMC vs factorial EKF (Chapter 3)

The main comparison trains the correlated RB-SQMC model and the factorial EKF
baseline on identical data splits and optimiser settings, then evaluates
predictions. The authoritative configuration is
`rbsqmc/comparison/sqmc_ekf/scripts/config/config_gpu.json` (512 particles,
50 epochs, learning rate 0.02, seed 0, 15 RQMC replicas, Gauss–Hermite degree
32).

Local smoke run (CPU, small subset):

```bash
python -m rbsqmc.comparison.sqmc_ekf.run \
  --config rbsqmc/comparison/sqmc_ekf/scripts/config/config_smoke.json \
  --data rbsqmc/data/results.csv --smoke
```

Full run (RB-SQMC on GPU, EKF on CPU, combined and validated):

```bash
bash rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_colab.sh
```

Outputs are written to
`rbsqmc/comparison/sqmc_ekf/outputs/DDMMYYYY_HHMM/{ekf,sqmc,combined}/`. The
authoritative run reported in the dissertation is `09092026_1431` at source
commit `336842f`. An interrupted run can be resumed without retraining:

```bash
bash rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_colab.sh \
  --resume rbsqmc/comparison/sqmc_ekf/outputs/DDMMYYYY_HHMM
```

### 4. Football model: RB-SMC vs RB-SQMC training

```bash
python -m rbsqmc.comparison.sqmc_smc.compare_smc_sqmc
```

This runs both filters under the shared configuration in
`rbsqmc/comparison/sqmc_smc/config/model_unbiased_gpu_config.json` and writes
train/test log-likelihood histories, gradient norms and prediction evaluation
to `rbsqmc/outputs/compare/`.

### 5. Standalone football pipeline

The RB-SMC pipeline supports `optimize`, `filter`, `predict` and `all` phases,
with configuration from `--config`, the `RBSQMC_CONFIG` environment variable,
or the repository default:

```bash
python -m rbsqmc.src.model.rbsmc.train_model_gpu --config <config.json> all
```

### 6. Verification

Each run directory contains `status.json` (completion state), `config.json`
(the effective configuration), `metadata.json` (device, software versions, git
commit and source hashes) and the results/figures for that stage. The Colab
launchers validate every downloaded archive (sizes, hashes, provenance and
complete parameter grids) before proceeding. To validate an existing output
directory:

```bash
python rbsqmc/comparison/sqmc_ekf/scripts/validate_sqmc_ekf_outputs.py <output_dir>
python sqmc/comparison/scripts/validate_artifacts.py <output_dir>
```

## Documentation

- `sqmc/comparison/COMPARISON.md` — the CPU/GPU benchmark suite: commands,
  budget semantics, outputs and Colab launcher.
- `rbsqmc/comparison/sqmc_ekf/README.md` — the RB-SQMC vs EKF comparison:
  background, model, evaluation axes and Colab orchestration.
- `sqmc/qmc/QMC.md` — QMC theory notes (Sobol', Halton, scrambling).
- `sqmc/hilbert_sort/HILBERT_SORT.md` — Hilbert sorting references.
- `rbsqmc/MODEL_OLD.md` — the RB-SQMC model specification.

## References

- Gerber, M., & Chopin, N. (2015). Sequential quasi Monte Carlo. *Journal of the Royal Statistical Society: Series B*, 77(3), 509–579.
- Chopin, N., & Gerber, M. (2017). Sequential quasi-Monte Carlo: Introduction for Non-Experts, Dimension Reduction, Application to Partly Observed Diffusion Processes. arXiv:1706.05305.
- Duffield, S., Power, S., & Rimella, L. (2024). A state-space perspective on modelling and inference for online skill rating. *Journal of the Royal Statistical Society: Series C*, 73(5), 1262–1282.
