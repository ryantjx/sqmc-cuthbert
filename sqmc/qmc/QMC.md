# Quasi Monte Carlo (QMC)

## Low Discrepancy Sequences 

Quasirandom Numbers Application:

1. Finding derivative of function with small amount of noise
2. Allow higher order moments to be estimated more accurately

https://en.wikipedia.org/wiki/Low-discrepancy_sequence

## Deterministic QMC Sampling

**Deterministic QMC Sampling**: Fully deterministic low-discrepancy sequences. always produce the same sequence of points for a given dimension and number of points.

- Halton: Uses coprime bases (e.g., 2, 3, 5, 7...) and radical inverse functions
- Sobol: Uses direction numbers and Gray code sequences in base 2
- Hammersley: Combines a Halton sequence with evenly spaced points on one axis

### Halton



A randomized Halton algorithm in R - https://arxiv.org/abs/1706.02808

tensorflow implementation - https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/mcmc/sample_halton_sequence
### Sobol

Wikipedia - https://en.wikipedia.org/wiki/Sobol_sequence

Direction numbers - precomputed from [Joe & Kuo's](https://web.maths.unsw.edu.au/~fkuo/sobol/)

Sobol sequence - https://web.maths.unsw.edu.au/~fkuo/sobol/



## Randomized QMC Sampling

**Randomized QMC Sampling**: Stochastic methods with randomness. Generates different sequences of points for the same dimension and number of points, while still maintaining low-discrepancy properties.

- Latin Hypercube: Divides the unit hypercube into equal subintervals and randomly samples within each subinterval
- Randomized Halton: Applies a random shift to the Halton sequence to reduce correlation and improve uniformity

## References

Adrien's JAX issue - https://github.com/jax-ml/jax/issues/8807

Scipy implementation - https://docs.scipy.org/doc/scipy/reference/stats.qmc.html

sciml library - https://docs.sciml.ai/QuasiMonteCarlo/stable/samplers/