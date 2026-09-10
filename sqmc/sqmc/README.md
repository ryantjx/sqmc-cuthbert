# Sequential Quasi-Monte Carlo (SQMC) filter (`sqmc/sqmc`)

A JAX implementation of the Sequential quasi-Monte Carlo (SQMC) particle filter,
compatible with `cuthbert` and `cuthbertlib`. It also provides a bootstrap
particle filter (SMC) baseline wrapper.

## Overview

SQMC replaces the stochastic resampling and propagation of a standard particle
filter with deterministic, low-discrepancy (QMC) counterparts. At each time
step the filter:

1. Generates `N` RQMC points of dimension `1 + d`.
2. Hilbert-sorts the previous `N` particles to obtain Hilbert-ordered weights.
3. Sorts the RQMC points by the first coordinate.
4. Selects ancestors by inverse-CDF using the first coordinates and the
   Hilbert-ordered weights.
5. Propagates the ancestors through a deterministic function
   `propagate_transform` (instead of the stochastic `propagate_sample`) using
   the remaining `d` coordinates.
6. Reweights the observation and updates the log-normalising constant.

## Public API

- `build_filter(...)` — assembles a `cuthbert.inference.Filter` from
  `init_transform`, `propagate_transform`, `log_potential`, the particle count,
  and a QMC engine.
- `init_prepare` / `filter_prepare` / `filter_combine` — the three filter steps;
  `filter_combine` implements the SQMC resampling (sort RQMC by first
  coordinate, Hilbert-sort particles, inverse-CDF ancestor selection,
  deterministic propagation, reweighting, log-normalising-constant update).
- `resample_from_uniform(sorted_uniforms, logits)` — pure-JAX inverse-CDF
  ancestor selection. This avoids a numba 0.66.0 typing regression in
  `cuthbertlib.resampling.utils.inverse_cdf`; the pure-JAX path is what the GPU
  branch of `inverse_cdf` uses and is correct for sorted uniforms.
- `_sample_points(qmc, key, n)` — fresh scrambled Sobol' engine per call when
  scrambling is enabled.

The SMC baseline is in `smc.py`: `build_filter(...)` wraps
`cuthbert.smc.particle_filter` with systematic resampling, mirroring the SQMC
interface.

## Usage

```python
from sqmc.sqmc.sqmc import build_filter

filter_ = build_filter(
    init_transform=...,
    propagate_transform=...,
    log_potential=...,
    n_particles=512,
    qmc=...,
)
```

A `main()` demo runs a 1D random-walk SQMC filter:

```bash
python -m sqmc.sqmc.sqmc
```

## Tests

```bash
python -m pytest sqmc/tests/test_sqmc.py -q
```

## References

- Gerber, M., & Chopin, N. (2015). Sequential quasi-Monte Carlo. *Journal of the Royal Statistical Society: Series B*, 77(3), 509–579.
- Gerber, M., & Chopin, N. (2019). Sequential quasi-Monte Carlo smoothing. *Journal of the American Statistical Association*, 114(525), 1550–1567.
- Chopin, N., & Papaspiliopoulos, O. (2020). *An Introduction to Sequential Monte Carlo.* Springer.
