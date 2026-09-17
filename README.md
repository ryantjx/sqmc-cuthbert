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

<!-- `rbsqmc/` imports `sqmc.qmc` and `sqmc.hilbert_sort` at runtime, so both
packages must be importable from the repository root. -->

<!-- ## The `sqmc` package and `cuthbert`

The `sqmc` package is a JAX implementation of sequential quasi-Monte Carlo
(SQMC) that is designed to be incorporated into the `cuthbert` library
(`state-space-models/cuthbert`). All three components — the SQMC filter, the
low-discrepancy sequence generators (`qmc`) and the Hilbert space-filling-curve
sorting (`hilbert_sort`) — are intended to be upstreamed into `cuthbert`.

- **`sqmc/sqmc/sqmc.py`** — the SQMC filter. `build_filter(...)` assembles a
  `cuthbert.inference.Filter` from `init_transform`, `propagate_transform`,
  `log_potential`, the particle count and a QMC engine. The three filter steps
  (`init_prepare`, `filter_prepare`, `filter_combine`) implement SQMC's
  deterministic resampling and propagation: generate `N` RQMC points of
  dimension `1 + d`, Hilbert-sort the previous particles, sort the RQMC points
  by their first coordinate, select ancestors by inverse-CDF, propagate
  deterministically, reweight and update the log-normalising constant. It is
  implemented as a `cuthbert.inference.Filter` and mirrors the interface of
  `cuthbert.smc.particle_filter`, so it can be dropped into any pipeline that
  already uses `cuthbert`'s filtering API.
- **`sqmc/sqmc/smc.py`** — a thin wrapper around
  `cuthbert.smc.particle_filter` (stochastic propagation, systematic
  resampling) that mirrors the SQMC interface, providing the SMC baseline for
  benchmarking.
- **`sqmc/qmc/`** — the low-discrepancy sequence generators (Sobol', Halton,
  scrambling) that SQMC relies on. See `sqmc/qmc/QMC.md` for the theory notes.
- **`sqmc/hilbert_sort/`** — Hilbert space-filling-curve index computation and
  sorting. See `sqmc/hilbert_sort/HILBERT_SORT.md` for references.

The SQMC filter is a work in progress towards being implemented as a
`cuthbert.inference.Filter`; the current `build_filter` already returns a
`Filter` and is exercised by the unit tests in `sqmc/tests/`. -->

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

## Installation and Results Replication

Python 3.10+ is required. The experiments were run with Python 3.13 and the
pinned dependencies in `requirements.txt` (JAX 0.11.0, cuthbert 0.0.14,
cuthbertlib 0.0.15, optax 0.2.8, NumPy 2.2.6, SciPy 1.18.0).

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

For GPU runs, install the CUDA build of JAX instead of the CPU build:

```bash
uv pip install "jax[cuda12]==0.11.0"
```

The football code selects its JAX platform through the `RBSQMC_PLATFORM`
environment variable (default `cpu`; set `RBSQMC_PLATFORM=cuda` on a GPU host).
Sanity check the installation with
`python -m pytest sqmc/tests rbsqmc/tests sqmc/comparison/tests rbsqmc/comparison/sqmc_ekf/tests -q`.

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

### Replication

All commands are run from the repository root with the virtual environment
active. Each experiment is launched by a single bash script and writes a
timestamped directory under its `outputs/` folder containing `config.json`,
`metadata.json`, `status.json`, results and figures. Both launchers use a
Colab GPU session by default: the `colab` CLI must be on `PATH` and the source
commit pushed, so the local and remote branches point at the same commit.
Where a local GPU is available, each experiment can instead be run directly.

#### SQMC CPU/GPU performance comparison (Chapter 2)

The Colab launcher provisions an A100 session, runs the three benchmarks
sequentially with the full chapter profile from
`sqmc/comparison/scripts/config/comparison_config.json`, and verifies each
stage's artefacts:

```bash
sqmc/comparison/scripts/run_comparison_colab.sh --dry-run
sqmc/comparison/scripts/run_comparison_colab.sh
```

On a local GPU host, run the three benchmarks directly with the same profile
(full documentation in `sqmc/comparison/COMPARISON.md`):

```bash
python -m sqmc.comparison.benchmark_qmc \
  --platforms cpu gpu --sequences sobol halton \
  --dimensions 2 5 10 30 60 --n-values 128 256 512 2048 8192 32768 \
  --repeats 10 --warmups 2 --seed 42

python -m sqmc.comparison.benchmark_hilbert_sort \
  --platforms cpu gpu --sequences sobol halton \
  --dimensions 2 5 10 30 60 --n-values 128 256 512 2048 8192 32768 \
  --repeats 10 --warmups 2 --seed 42

python -m sqmc.comparison.benchmark_sqmc \
  --platforms cpu gpu --dimensions 2 5 10 30 60 \
  --particle-counts 128 256 512 1024 2048 \
  --budget-seconds 0.01 0.05 0.1 --n-steps 100 \
  --selection-reps 8 --validation-reps 16 --bootstrap-reps 500 \
  --repeats 10 --warmups 2 --seed 42
```

#### Football model: RB-SQMC vs factorial EKF (Chapter 3)

The default launcher trains the factorial EKF baseline on the local CPU and
RB-SQMC on a Colab GPU under the authoritative configuration
`rbsqmc/comparison/sqmc_ekf/scripts/config/config_gpu.json` (512 particles,
50 epochs, learning rate 0.02, seed 0, 15 RQMC replicas, Gauss–Hermite degree
32), then combines and validates the two partial runs:

```bash
bash rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_colab.sh --dry-run
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

On a local GPU host, run both methods directly (`run.py` requires
`RBSQMC_PLATFORM=cuda` to match the JAX backend):

```bash
RBSQMC_PLATFORM=cuda python -m rbsqmc.comparison.sqmc_ekf.run \
  --config rbsqmc/comparison/sqmc_ekf/scripts/config/config_gpu.json \
  --data rbsqmc/data/results.csv
```

Without a GPU, exercise the full pipeline on CPU with the small smoke
configuration:

```bash
bash rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_colab.sh --local --smoke
```

Each output directory contains `status.json`, the effective `config.json`,
`metadata.json` (device, software versions, git commit and source hashes) and
the results/figures for that stage; the launchers validate every downloaded
archive before proceeding. An existing directory can be validated with:

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

## References

- Gerber, M., & Chopin, N. (2015). Sequential quasi Monte Carlo. *Journal of the Royal Statistical Society: Series B*, 77(3), 509–579.
- Chopin, N., & Gerber, M. (2017). Sequential quasi-Monte Carlo: Introduction for Non-Experts, Dimension Reduction, Application to Partly Observed Diffusion Processes. arXiv:1706.05305.
- Duffield, S., Power, S., & Rimella, L. (2024). A state-space perspective on modelling and inference for online skill rating. *Journal of the Royal Statistical Society: Series C*, 73(5), 1262–1282.
