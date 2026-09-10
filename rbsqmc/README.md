# `rbsqmc` — Rao-Blackwellised SQMC for football match modelling

The football application package of the repository. It implements a
Rao-Blackwellised state-space model for international football results, with
RB-SMC, RB-SQMC and factorial EKF filters, parameter estimation, prediction,
and comparison pipelines. It imports the QMC generation and Hilbert-sorting
modules from the sibling `sqmc` package at runtime, so both packages must be
importable from the repository root.

## Model

The latent attack/defence strengths of `M` teams, `X_t^m = (X_t^{m,att},
X_t^{m,def})`, evolve as an Ornstein–Uhlenbeck process with a shared time
delta. Observed goals follow a bivariate Poisson distribution. The initial
covariance is the Kronecker product `Σ_0 = Γ_0 ⊗ B`, where `Γ_0` is the
between-team correlation matrix and `B` the per-team `2 × 2` covariance. Since
only the two teams in a match enter the likelihood, the remaining latent states
are represented analytically as a Gaussian conditional (Rao-Blackwellisation).
See `MODEL_OLD.md` for the full specification.

## Repository layout

```text
src/
├── data/
│   ├── data.py             # Data loading, filtering, JAX tensor construction
│   ├── bivariate_poisson.py# Bivariate-Poisson log-likelihood
│   └── data_ekf.py         # Frozen per-row dataset shared with the EKF
├── model/
│   ├── rbsmc/              # RB-SMC filter, optimisation, train/predict pipeline
│   ├── rbsqmc/             # RB-SQMC filter, optimisation, prediction
│   └── ekf/                # Factorial EKF baseline (replicates cuthberto-carlos)
└── utils/
    ├── type.py             # NamedTuple type definitions
    ├── helpers.py          # Params encode/decode, save/load, diagnostics
    ├── graphic.py          # Plotting helpers
    └── stats.py            # Stable Cholesky, Gaussian-Kron logpdf/sampler

comparison/
├── sqmc_smc/               # RB-SMC vs RB-SQMC training comparison
└── sqmc_ekf/               # RB-SQMC vs factorial EKF comparison (main experiment)

data/                       # results.csv, team sets, fixtures
outputs/                    # Run artefacts (timestamped directories)
tests/                      # Unit tests
```

## Filters

- **RB-SMC** (`src/model/rbsmc/`) — Rao-Blackwellised SMC filter built on
  `cuthbert`, with autodiff resampling, Adam + cosine-schedule optimisation,
  and a train/test/predict pipeline (`train_model_gpu.py`).
- **RB-SQMC** (`src/model/rbsqmc/`) — the SQMC analogue: deterministic
  propagation driven by scrambled Sobol' points and Hilbert-sorted inverse-CDF
  ancestor selection. The core is `model_rbsqmc.py`; `train_model_rbsqmc.py`
  and `predict_rbsqmc.py` provide optimisation and one-step-ahead prediction.
- **EKF** (`src/model/ekf/`) — a factorial extended Kalman filter replicating
  the `cuthberto-carlos` baseline, used as the comparison reference.

## Running

All commands run from the repository root with the virtual environment active.
The JAX platform is selected by the `RBSQMC_PLATFORM` environment variable
(default `cpu`; set `RBSQMC_PLATFORM=cuda` on a GPU host).

### Tests

```bash
python -m pytest rbsqmc/tests -q
```

### RB-SQMC vs factorial EKF (main comparison)

```bash
python -m rbsqmc.comparison.sqmc_ekf.run \
  --config rbsqmc/comparison/sqmc_ekf/scripts/config/config_gpu.json \
  --data rbsqmc/data/results.csv
```

See `comparison/sqmc_ekf/README.md` for the full comparison spec, evaluation
axes and Colab orchestration.

### RB-SMC vs RB-SQMC training

```bash
python -m rbsqmc.comparison.sqmc_smc.compare_smc_sqmc
```

### Standalone RB-SMC pipeline

```bash
python -m rbsqmc.src.model.rbsmc.train_model_gpu --config <config.json> all
```

Phases are `optimize`, `filter`, `predict` and `all`; configuration comes from
`--config`, the `RBSQMC_CONFIG` environment variable, or the repository default
(`rbsqmc/comparison/sqmc_smc/config/model_unbiased_gpu_config.json`).

## Data

The runtime data source is `data/results.csv` (the frozen international
football results) and its parquet cache `data/results.parquet`. The parquet
cache is git-ignored; on a fresh clone either run the data download once
(`download=True` in `src/data/data.py`) or copy `results.csv` to
`results.parquet`. Team sets are defined in `data/worldcup2026.json` and
related files.

## Dependencies

See `pyproject.toml` / `requirements.txt`: `jax[cuda12]==0.11.0`,
`cuthbert==0.0.14`, `optax==0.2.8`, NumPy, SciPy, pandas, pyarrow, matplotlib,
tqdm. The EKF comparison additionally needs `cuthbertlib==0.0.15` and
`ghq==0.0.5` (Gauss–Hermite quadrature).
