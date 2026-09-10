# Quasi-Monte Carlo point generation (`sqmc/qmc`)

Low-discrepancy sequence generators in JAX, used as the sampling engine for the
SQMC filter. This module provides the `QMC` interface and the `Halton` and
`Sobol` implementations.

## Overview

Quasi-Monte Carlo (QMC) methods produce an array of `n × d` points in
`[0, 1]^d` that are more evenly distributed than random points, with fewer
gaps and clusters. This is quantified by discrepancy measures; by the
Koksma–Hlawka inequality, low discrepancy reduces a bound on integration
error, so averaging a function over `n` QMC points can achieve an integration
error close to `O(n^{-1})` for well-behaved functions.

The implementation follows `scipy.stats.qmc` and `QuasiMonteCarlo.jl`, but is
written in JAX so that generation is differentiable and runs on CPU or GPU.

## Public API

- `QMC` — abstract base class with `sample(n, *, state)` and an explicit
  `QMCState` (a `next_index` counter carried through `jax.lax.scan`).
- `Halton(QMC)` — radical-inverse sequence in prime bases, with optional Owen
  scrambling (random digit permutations plus tail correction).
- `Sobol(QMC)` — Joe–Kuo direction numbers, with optional LMS+shift scrambling
  (left linear matrix scramble plus digital shift, matching SciPy). Uses a
  parallel XOR prefix scan (`_sobol_sample_batched`).
- `normal_coordinates(u)` — standard-normal quantile with open-interval
  clipping `[2^-31, 1-2^-31]` for a finite inverse-CDF.

## Data file

The Sobol' generator reads the direction-number table
`_sobol_direction_numbers.npz`. This file is git-ignored and is generated from
the committed Joe–Kuo source table `new-joe-kuo-6.21201`:

```bash
python sqmc/qmc/_generate_sobol_data.py
```

## Usage

```python
from sqmc.qmc.qmc import Sobol, Halton

sobol = Sobol(d=5, scramble=True)
points = sobol.sample(512)   # (512, 5) scrambled Sobol' points in [0,1]^5
```

A `main()` demo runs a 1D random-walk SQMC filter:

```bash
python -m sqmc.qmc.qmc
```

## Tests

```bash
python -m pytest sqmc/tests/test_qmc.py -q
```

## References

- Owen, A. B. (2019). *Monte Carlo Book: the Quasi-Monte Carlo parts.*
- Niederreiter, H. (1992). *Random number generation and quasi-Monte Carlo methods.* SIAM.
- Dick, J., Kuo, F. Y., & Sloan, I. H. (2013). High-dimensional integration: the quasi-Monte Carlo way. *Acta Numerica*, 22, 133–288.
- Hickernell, F. J. (2014). Koksma–Hlawka inequality. *Wiley StatsRef.*
