# GPU Resampling and Why Hilbert Sort Bypasses the Problem

## The GPU resampling problem

In SMC, the propagation and weighting steps parallelize trivially: they are
independent operations on each particle. The **resampling step does not** — it is
a *collective* operation that requires a sum (or prefix-scan) across all particle
weights, forcing thread synchronization. On a GPU, where threads in a warp must
execute in lockstep and global communication is expensive, this collective
operation is the bottleneck.

The standard inverse-CDF resamplers (multinomial, stratified, systematic) all
share this structure:

1. Compute the cumulative distribution function (CDF) of the weights,
   $c_i = \sum_{j \le i} w_j$ — an **inclusive prefix sum**.
2. Draw $N$ uniforms $u^n \sim \mathcal{U}[0,1]$.
3. Map each $u^n$ to an ancestor via the inverse CDF,
   $a^n = \min\{i : c_i \ge u^n\}$ (a binary search, $O(N \log N)$).

The prefix sum is the culprit: it is a collective operation that accumulates
rounding error and requires inter-thread communication.

---

## How the literature addresses it

Four papers define the landscape of GPU resampling. They fall into two strategies:
**avoid the collective sum** or **parallelize the collective sum**.

| Paper | Strategy | Bias? | Key result |
|-------|----------|-------|------------|
| Murray, Lee & Jacob (2013) | Avoid collective sum (Metropolis/rejection) | Metropolis biased, rejection unbiased | Inverse-CDF resamplers biased in single precision for $N \gtrsim 2^{18}$ |
| Murray (2012) | Metropolis resampler | Biased (finite $B$) | Faster at low weight variance |
| McAlinn & Nakatsuma (2012) | Parallelize the CDF (cut-point method) | Unbiased | ~30× full-cycle, ~10× resampling speedup |
| Schieffer et al. (2023) | Half-precision + algorithmic fixes | Empirical accuracy loss | 1.5–4.6× speedup; resampling is the bottleneck |

### Murray, Lee & Jacob (2013) — the definitive study

The most important reference. It classifies resamplers into two families:

- **Inverse-CDF (cumulative)**: multinomial, stratified, systematic. All require
  the inclusive prefix sum.
- **Pairwise (no collective)**: Metropolis and rejection. Only compute *ratios*
  of weights, independently per thread.

**Key theoretical results:**

- **Unbiasedness condition**:
  $$\mathbb{E}(o_t^i \mid \mathbf{w}_{t-1}) = \frac{N w_{t-1}^i}{\sum_j w_{t-1}^j}.$$
  This ensures unbiased marginal-likelihood estimates.

- **Metropolis resampler is biased** for finite steps $B$ (it is a Markov chain
  run $B$ steps). They bound the total-variation distance:
  $$\|P^B(i,\cdot) - \pi(\cdot)\|_{\text{TV}} \le (1-\beta)^B, \qquad
    \beta = \min_i \frac{1}{N}\sum_j \frac{w_j}{w_i} \ge \frac{1}{N}.$$
  Choosing $B \ge B^* = \log\epsilon / \log(1-\beta)$ ensures TV distance $\le \epsilon$.

- **Rejection resampler is unbiased**; expected parallel complexity $O(\log N / p)$
  where $p$ is the acceptance probability.

**The precision/bias finding (the key one):** In **single precision**, the
standard inverse-CDF resamplers become **numerically biased for
$N \gtrsim 2^{18}$–$2^{19}$** (hundreds of thousands of particles), because the
**prefix sum accumulates rounding error**. The Metropolis/rejection resamplers do
**not** share this instability (they only compute ratios). Double precision fixes
it, but at ~2× slower throughput.

**Empirical decision rules:**
- GPU not worth it for resampling with fewer than $2^{10}$ particles.
- Systematic resampler is a good all-round candidate.
- Metropolis/rejection are faster at **low weight variance**.
- CPU↔GPU memory transfer does not change the decision boundaries much.

### Murray (2012) — the Metropolis resampler

The precursor to the above. The Metropolis resampler needs only **pairwise weight
ratios**, computed independently by threads, avoiding the collective sum. It is
tunable via the number of iterations $B$. Faster than stratified/multinomial when
the variance of importance weights is modest. Recommended for performance-critical
contexts (PMCMC, real-time). **Caveat:** it is biased for finite $B$ — a
speed-vs-bias trade-off.

### McAlinn & Nakatsuma (2012) — parallelize the CDF

The *opposite* strategy: parallelize the CDF itself rather than avoid it. A
**cut-point method** (Chen & Asau 1974) partitions the CDF so each thread finds
its ancestor in parallel. Combined with a parallel prefix-sum, the *entire*
particle filter runs on the GPU with **no CPU↔GPU data transfer** during the loop.

Results: ~30× speedup over sequential CPU for the full particle-learning cycle;
~10× on the resampling step alone. Per-step breakdown: CDF 248×, propagation 45×,
resampling 12×. GPU double-precision still 5–10× faster than CPU single-precision.

**Relevance:** This shows the CDF prefix-sum *can* be made GPU-parallel — but it
still relies on the collective operation that SQMC's Hilbert sort replaces. Its
per-step breakdown is a good template for a per-step time-breakdown plot.

### Schieffer et al. (2023) — half-precision

The empirical half-precision study. It does **not** make a theoretical bias
statement — it is an engineering study. It reports:
- **1.5–2× speedup** (half vs. single), **2.5–4.6×** (half vs. double).
- "Relatively small loss of accuracy" — measured empirically on object tracking.
- Algorithmic changes (log-sum-exp normalization, moving denominators into
  squares) to mitigate numerical instability.

They use **systematic resampling** (an inverse-CDF method). The *naive* half-precision
resampling kernel was slow (FP16 pipeline only 12% utilized); after optimization
(reducing reciprocal/casting ops) it reached 51% utilization and **3.0× / 2.7×**
speedup over double/single for the resampling kernel.

**Relevance:** The empirical companion to Murray et al.'s theoretical bias claim.
Shows reduced precision *can* be used for speed, but **requires algorithmic
redesign** to stay stable. Confirms that double precision is the safe default.

---

## Why Hilbert sort bypasses the problem

SQMC's resampling is **fundamentally different** from SMC's random inverse-CDF
resampling — and this is a structural advantage on the GPU.

In **SMC**, resampling draws *random* uniforms $u^n$ and inverts the CDF. The
collective prefix-sum is unavoidable, and it is where the reduced-precision bias
(per Murray et al.) and the synchronization bottleneck (per all four papers)
originate.

In **SQMC**, resampling is **deterministic**:

1. **Hilbert-sort** the particles (embarrassingly parallel, bit-level).
2. Apply an **inverse-CDF transform to the first coordinate of the QMC points**
   (fixed, low-discrepancy points — not random uniforms).

The expensive collective step in SMC (the CDF prefix-sum) is replaced by the
**Hilbert sort** — exactly the bit-level, embarrassingly parallel primitive that
runs fast on GPUs. This means SQMC:

- **Avoids the collective prefix-sum** entirely (no thread synchronization
  bottleneck).
- **Does not inherit the reduced-precision bias** of random inverse-CDF
  resampling (the QMC points are fixed and low-discrepancy, not accumulated
  through a rounding-error-prone prefix sum).
- **Is exact** (unlike the Metropolis resampler, which is biased for finite $B$).

This is the novel contribution to position against the GPU-resampling
literature: the four papers above all work *within* the SMC resampling paradigm
(either avoiding or parallelizing the collective sum), whereas SQMC's Hilbert
sort **sidesteps the problem entirely** by making resampling deterministic and
embarrassingly parallel.

---

## Practical implications for the implementation

1. **The Hilbert sort is the resampling enabler** — it is the GPU-friendly
   replacement for the collective CDF scan. This is a strong point to make in the
   chapter.
2. **Watch precision** — both Murray et al. (theoretically) and Schieffer et al.
   (empirically) warn that weight/CDF accumulation in reduced precision is biased.
   `hilbert_sort.py` already enables `jax_enable_x64` (double precision) for
   exactly this reason. If FP16/FP32 is tried later for speed, the same
   algorithmic care Schieffer et al. describe will be needed.
3. **If a random-resampling fallback is ever needed** (e.g., for an SMC
   comparison baseline), Murray et al. suggest Metropolis/rejection resamplers
   over inverse-CDF for GPU speed.

---

## Source papers

- Murray, L. M., Lee, A., & Jacob, P. E. (2013). *Parallel resampling in the
  particle filter.* arXiv:1301.4019.
- Murray, L. M. (2012). *GPU acceleration of the particle filter: the Metropolis
  resampler.* arXiv:1202.6163.
- McAlinn, K., & Nakatsuma, T. (2012). *Parallel Resampling for Fully
  Parallelized Particle Filters.* arXiv:1212.1639.
- Schieffer, G., Pornthisan, N., de Medeiros, D. A., Markidis, S., Wahlgren, J.,
  & Peng, I. (2023). *Boosting the Performance of Object Tracking with a
  Half-Precision Particle Filter on GPU.* arXiv:2308.00763.
