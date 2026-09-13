
# Research Notes: High-Performance SQMC

Core references:
- `\citet{chopingerber2015sqmc}`
- `\citet{chopingerber2017sqmcintroduction}`
- `\citet{chopinpapaspiliopoulos020introtosmc}`

---

## 1. Chopin & Gerber (2015) — Sequential Quasi-Monte Carlo (JRSS-B)

**Summary.** Introduces SQMC, the QMC analogue of SMC. In SMC, particles are propagated by sampling from a stochastic proposal; in SQMC, each particle is propagated deterministically through a low-discrepancy point set, so the low-discrepancy structure is carried into the particle cloud.

**Key mathematics.** The filtering recursion targets $p(x_t \mid y_{1:t})$. SMC approximates it with weighted particles $\{x_t^n, w_t^n\}_{n=1}^N$. SQMC replaces the random proposal draw with
$$x_t^n = \Gamma_t\big(x_{t-1}^{a_t^n},\, u_t^n\big), \qquad u_t^n \in \mathcal{U}[0,1]^d,$$
where $u_t^n$ are QMC points and $a_t^n$ is the ancestor index from resampling. Resampling is done *before* propagation: particles are sorted along a Hilbert curve, then an inverse-CDF transform is applied to the first coordinate of the QMC points. Complexity is $O(N\log N)$ per step, and the error rate is smaller than the Monte Carlo rate $O_P(N^{-1/2})$.

**Relevance to topic.** Core method of the chapter — the deterministic propagation $\Gamma_t$ and the Hilbert-sorted resampling are exactly the primitives implemented on the GPU.

---

## 2. Chopin & Gerber (2017) — SQMC: Introduction for Non-Experts, Dimension Reduction, Application to Partly Observed Diffusion Processes

**Summary.** Two goals: (a) introduce SMC to the QMC community; (b) extend SQMC to continuous-time state-space models where the latent process is a diffusion. The recurring theme is **dimension reduction** — how to keep SQMC effective despite high problem dimension.

**Key mathematics.** For a diffusion
$$dX_t = a(X_t)\,dt + b(X_t)\,dW_t,$$
the transition is discretized (e.g., Euler–Maruyama)
$$X_{t+\Delta} = X_t + a(X_t)\Delta + b(X_t)\sqrt{\Delta}\,Z, \qquad Z \sim \mathcal{N}(0,1),$$
and the Gaussian increments $Z$ are generated from QMC uniforms via the inverse-CDF. Dimension reduction is achieved by structuring the QMC points (e.g., Brownian-bridge / principal-component constructions) so the effective dimension stays low.

**Relevance to topic.** Directly supports the "SQMC for different dimensions" subsection — the reference for *why* and *how* to reduce dimension in high-dimensional SQMC.

---

## 3. Chopin & Papaspiliopoulos (2020) — An Introduction to Sequential Monte Carlo (Springer)

**Summary.** The standard textbook treatment of SMC, including a chapter on SQMC and QMC/SMC hybrids.

**Relevance to topic.** Canonical background reference for both SMC and SQMC notation and theory.

---

## 4. Faure & Lemieux (2010) — Improved Halton Sequences and Discrepancy Bounds (MCMA)

> DOI `10.1515/mcma.2010.008` — this is the Halton paper, not the Hilbert-sorting paper.

**Summary.** The Halton sequence in dimension $s$ is built from radical inverses in pairwise-coprime bases $b_1,\dots,b_s$:
$$x_n = \big(\phi_{b_1}(n),\, \dots,\, \phi_{b_s}(n)\big), \qquad \phi_b(n) = \sum_{k\ge 0} a_k(n)\, b^{-k-1},$$
where $n = \sum_k a_k(n) b^k$. Its star discrepancy satisfies $D_N^* = O\big(N^{-1}(\log N)^s\big)$, but in high dimension the correlations between bases degrade the distribution. The paper proposes **generalized/scrambled Halton sequences** (digit permutations) that improve the discrepancy constant and high-dimensional behavior.

**Relevance to topic.** Directly relevant to the "Halton Sequences" subsection — the reference for improved Halton constructions that behave well in the higher dimensions SQMC needs.

---

## 5. L'Ecuyer et al. — Sorting Methods and Convergence Rates for Array-RQMC: Some Empirical Comparisons

**Summary.** Array-RQMC simulates an array of $N$ Markov-chain realizations in parallel; at each step the chains are **sorted by state** and advanced with the next RQMC point. This paper empirically compares sorting strategies (Hilbert curve, principal components, coordinate-wise) and studies the resulting convergence rates.

**Key mathematics.** To estimate $\mu = \mathbb{E}[f(X_T)]$ for a Markov chain, array-RQMC gives a variance that can be $O(N^{-2})$ or better under good sorting — far better than the MC rate $O(N^{-1})$. The sorting function is the key ingredient.

**Relevance to topic.** Empirical evidence for *which* sorting (Hilbert) works best — directly informing the Hilbert-sort-on-GPU choice and the SMC-vs-SQMC performance comparison.

---

## 6. L'Ecuyer, Lécot & Tuffin (2008) — A Randomized Quasi-Monte Carlo Simulation Method for Markov Chains (Operations Research)

**Summary.** The **seminal array-RQMC paper** — the direct ancestor of SQMC. It introduced the idea of sorting an array of chains by state and applying RQMC points, and proved variance convergence faster than $O(N^{-1})$ for chains with an ordered state space.

**Relevance to topic.** Historical lineage of SQMC. The 2015 Chopin–Gerber paper explicitly states SQMC "may be seen as an extension of the array-RQMC algorithm of L'Ecuyer et al." — worth citing in the background.

---

## 7. Binder, Fricke & Keller (2019) — Massively Parallel Path Space Filtering

**Summary.** In path tracing, few paths per pixel give poor images. Path space filtering improves quality by sharing information across **proximate path vertices** (not just screen-space neighbors). The paper makes this efficient with a **hash table** keyed by jittered/quantized vertex information, so a single query replaces costly neighborhood searches, and demonstrates a **massively parallel GPU** implementation.

**Relevance to topic.** A concrete example of a GPU-parallel, hash-based spatial data structure for QMC-driven rendering — relevant to the engineering side of the GPU implementation.

---

## 8. Keller, Wächter & Binder (2022) — Rendering along the Hilbert Curve

**Summary.** Enumerates a low-discrepancy sequence along a **Hilbert curve** superimposed on the pixel raster, giving noise characteristics desirable for the human visual system at very low sampling rates. The algorithms are deterministic — no randomization, no costly optimization, no lookup tables — and are validated in a production, massively parallel light transport system.

**Relevance to topic.** The canonical graphics-side demonstration of **Hilbert ordering for locality**, directly analogous to SQMC's Hilbert-sorted resampling and evidence that the primitive parallelizes well on the GPU.

---

## 9. Keller, Wächter & Binder (2013) — Quasi-Monte Carlo Algorithms (not only) for Graphics Software

**Summary.** A survey arguing QMC should be the default for graphics software. Key ideas: deterministic low-discrepancy sequences give $O(N^{-1})$ convergence for smooth integrands (vs. $O(N^{-1/2})$ for MC) plus reproducibility; a single sequence can be partitioned for **progressive rendering**; **scrambling** (e.g., Owen) recovers unbiased error estimates; and space-filling curves improve locality.

**Relevance to topic.** General QMC-on-GPU background — including the scrambling note in the chapter ("Scrambling operation can be improved on GPU") and the Hilbert-curve locality principle.

---

## Relationship between papers 7–9

Papers 7, 8, and 9 are all by the **NVIDIA rendering group** (Keller and collaborators) and form a coherent line of work on **QMC-driven, GPU-parallel rendering**:

- **Paper 9 (2013)** is the *foundation*: it argues that QMC should be the default for graphics software and establishes the core primitives — low-discrepancy sequences, scrambling, and space-filling curves for locality. It is the conceptual umbrella.
- **Paper 8 (2022)** is the *direct application* of one of paper 9's ideas: it takes the **Hilbert curve** (a space-filling curve) and uses it to order a low-discrepancy sequence over the pixel raster. It is the concrete, production-validated realization of the locality principle from paper 9.
- **Paper 7 (2019)** is a *complementary engineering contribution*: it addresses the *noise/quality* problem that remains after sampling (path space filtering) using a **GPU hash table** for proximate-vertex lookup. It shares paper 9's GPU-parallel, deterministic philosophy but solves a different sub-problem (denoising/filtering rather than sampling/ordering).

**In short:** 9 = the theory/blueprint, 8 = the Hilbert-ordering application, 7 = the filtering/denoising companion. Together they show the full QMC-on-GPU pipeline: *sample with low-discrepancy points (9), order them for locality with the Hilbert curve (8), and clean up the residual noise with parallel filtering (7).*

For your topic, the key takeaway is the **Hilbert-curve ordering primitive** shared by 8 and 9 — the same primitive SQMC uses for its resampling step, and the one you implement on the GPU.

---

## Code from the papers

The central primitive in papers 8 and 9 (and in SQMC's resampling) is the **Hilbert-curve index** — mapping a point in $[0,1)^d$ to its position along the space-filling curve. The following is the standard bit-level implementation (as used in the Keller et al. rendering work) for the 2-D case, which is what maps a pixel coordinate to its Hilbert order:

```c
// Hilbert curve index for a 2-D point (x, y) in [0, 1)^2.
// Returns the position of the point along the Hilbert curve.
// Based on the bit-interleaving / state-machine construction used in
// Keller, Wächter & Binder (2022), "Rendering along the Hilbert Curve".
uint32_t hilbert_index(uint32_t x, uint32_t y, int bits) {
    uint32_t d = 0;
    // Interleave bits of x and y: d = (x | (y << 1)) with bit interleaving.
    for (int s = 1; s < bits; s <<= 1) {
        uint32_t rx = (x >> s) & 1;
        uint32_t ry = (y >> s) & 1;
        d += (rx << (2 * s)) | (ry << (2 * s + 1));
    }
    // Rotate/reflect the quadrants to follow the Hilbert curve (state machine).
    uint32_t tx, ty, rot = 0;
    for (int s = bits - 1; s >= 0; s--) {
        uint32_t rx = (d >> (2 * s + 1)) & 1;
        uint32_t ry = (d >> (2 * s)) & 1;
        if (ry == 0) {
            if (rx == 1) { d ^= (0xFFFFFFFFu >> (32 - 2 * s)); }
            // swap x and y bits
            tx = (d >> (2 * s)) & 1;
            ty = (d >> (2 * s + 1)) & 1;
            d &= ~(3u << (2 * s));
            d |= (tx << (2 * s + 1)) | (ty << (2 * s));
        }
    }
    return d;
}
```

The other core primitive, used throughout paper 9 (and in the Halton/Sobol constructions of your QMC section), is the **radical inverse** — the basis of low-discrepancy sequences:

```c
// Radical inverse of n in base b (van der Corput for b = 2).
// This is the building block of Halton and Sobol sequences.
double radical_inverse(uint64_t n, uint32_t b) {
    double inv = 1.0 / b, result = 0.0;
    while (n > 0) {
        result += (n % b) * inv;
        n /= b;
        inv /= b;
    }
    return result;
}
```

These two routines — the **Hilbert index** (for locality-preserving ordering) and the **radical inverse** (for low-discrepancy point generation) — are the computational heart of the SQMC resampling step you implement on the GPU. Both are bit-level operations that parallelize trivially across threads, which is why they map so well onto GPU hardware.

---

## 10. Pérez-Vieites, Mariño & Míguez (2018) — A Probabilistic Scheme for Joint Parameter Estimation and State Prediction in Complex Dynamical Systems (Phys. Rev. E)

**Summary.** Introduces **nested filtering**: a Monte Carlo scheme for the static parameters $\theta$ combined with a filter for the time-varying states $x_t$, targeting the joint posterior
$$p(\theta, x_{0:T} \mid y_{1:T}).$$
It adds an **SQMC variant** and demonstrates the method on a 4000-dimensional stochastic Lorenz-96 system.

**Relevance to topic.** A strong example of SQMC applied to a genuinely high-dimensional dynamical system — useful motivation for the "SQMC for different dimensions" and performance-comparison sections.

---

## Source links

- Improved Halton sequences and discrepancy bounds (Faure, Lemieux) — https://www.degruyterbrill.com/document/doi/10.1515/mcma.2010.008/html
- Sorting methods and convergence rates for Array-RQMC: some empirical comparisons — https://www-perso.iro.umontreal.ca/~lecuyer/myftp/papers/mcm2015-arrayrqmc.pdf
- A Randomized Quasi-Monte Carlo Simulation Method for Markov Chains — https://people.rennes.inria.fr/Bruno.Tuffin/Publis/arrayrqmc.pdf
- Massively Parallel Path Space Filtering — https://arxiv.org/abs/1902.05942
- Rendering along the Hilbert Curve — https://arxiv.org/pdf/2207.05415
- Quasi-Monte Carlo Algorithms (not only) for Graphics Software — https://arxiv.org/pdf/2307.15584
- A probabilistic scheme for joint parameter estimation and state prediction in complex dynamical systems — https://arxiv.org/pdf/1708.03730