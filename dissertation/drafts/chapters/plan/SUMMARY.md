# Summary of the CPU/GPU comparison results

The experiments in [Section 2.3](../2_high_performance_sqmc.tex) evaluate QMC generation, Hilbert sorting and complete SQMC filtering using the verified A100 run [`08092026_0048`](../../../../sqmc/comparison/outputs/08092026_0048/). The comparison uses double precision and includes fresh scrambling on both CPU and GPU. SciPy CPU is an additional reference for QMC generation; Hilbert sorting and SQMC compare the same shared JAX implementation on the two devices.

**QMC generation.** With 131,072 points, JAX GPU is **4.42–31.05× faster than SciPy CPU for Sobol’** and **86.57–103.48× faster for Halton** across the five dimensions. Against JAX CPU, the corresponding speedups are **5.34–45.59×** and **8.82–12.29×**. Every timed call includes fresh scrambling and point generation. SciPy also includes Python generator construction, while JAX compilation is excluded, so the SciPy-relative ratios reflect implementation differences as well as hardware. Small point sets can favour CPU execution.

**Hilbert sorting.** The GPU achieves speedups of **79.42–107.93× for Sobol’ inputs** and **78.30–109.14× for Halton inputs** at 131,072 points. All 70 matched CPU/GPU configurations produce identical permutations from identical normal-transformed inputs. Input generation is excluded from sorting time, and the CPU is faster at the smallest tested count, 128 points, in every dimension.

**SQMC filtering.** At a fixed horizon of **100 filtering updates** and **2,048 particles**, GPU execution is **6.23–13.37× faster**, with the same validation RMSE to the displayed precision. The largest absolute RMSE difference across all 25 matched CPU/GPU configurations is approximately **2.22 × 10⁻¹⁶**. The primary benefit is therefore the same measured accuracy in less time. As a secondary result, a 100 ms median-runtime budget allows the GPU to use 2,048 particles in every dimension, while the CPU uses 128–512 particles, yielding CPU/GPU error ratios of **1.34–3.73**.

## First measured GPU advantage

The table gives the **smallest tested count N for which GPU median runtime is lower than JAX CPU median runtime**, at each dimension. N denotes points for QMC/Hilbert and particles for SQMC; every SQMC trajectory processes 100 observations. All larger tested counts also favour GPU in these comparisons.

| Dimension d | QMC: Sobol’ | QMC: Halton | Hilbert: Sobol’ inputs | Hilbert: Halton inputs | SQMC: scrambled Sobol’ |
|---:|---:|---:|---:|---:|---:|
| 2 | 8,192 | 32,768 | 2,048 | 2,048 | 256 |
| 5 | 8,192 | 32,768 | 2,048 | 512 | 128 |
| 10 | 2,048 | 32,768 | 512 | 512 | 128 |
| 30 | 512 | 8,192 | 512 | 256 | 128 |
| 60 | 128 | 2,048 | 256 | 256 | 128 |

Against **SciPy CPU**, freshly scrambled **Sobol’ first favours JAX GPU at N = 128**, and **Halton at N = 2,048**, in every tested dimension. These are separate library comparisons; SciPy is not a Hilbert or SQMC baseline.

These counts are measured grid points, not exact crossover thresholds. Where GPU already wins at N = 128, the crossover may lie below the tested range. The Hilbert advantage for Halton inputs at d = 5, N = 512 is marginal (**1.03×**). For SQMC, the first GPU-favouring configuration gives **1.56×** speedup at d = 2, N = 256; at N = 128, speedups are **1.42×, 1.25×, 2.61× and 4.57×** for d = 5, 10, 30 and 60, respectively. GPU speedup generally grows with the amount of work, while higher dimensions can make acceleration worthwhile at smaller counts.

These results are conditional on one hardware session and one observation dataset per dimension. Steady-state timings exclude JAX compilation and input transfers, and median budgets are not hard latency guarantees. High-dimensional filtering remains difficult: at dimension 60, validation RMSE is approximately **1.79** even with 2,048 particles. The experiment establishes SQMC CPU/GPU acceleration, rather than SQMC’s statistical efficiency relative to SMC.
