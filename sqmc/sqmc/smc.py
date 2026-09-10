"""Bootstrap particle filter (SMC) baseline for the SQMC comparison.

This module provides a thin wrapper around ``cuthbert.smc.particle_filter``
that mirrors the SQMC filter interface in ``sqmc.sqmc.sqmc`` so that the two
algorithms can be benchmarked against each other on identical models.

The SMC baseline uses stochastic propagation (``propagate_sample``) and
systematic resampling, in contrast to SQMC's deterministic propagation from
low-discrepancy point sets with Hilbert-ordered resampling.
"""

from __future__ import annotations

from functools import partial
from typing import Protocol

import cuthbert
from cuthbert.smc.particle_filter import ParticleFilterState
from cuthbert.inference import Filter
from cuthbertlib.types import Array, ArrayTree, ArrayTreeLike, KeyArray, ScalarArray
from cuthbertlib.resampling import systematic

from cuthbert.smc.types import InitSample, LogPotential, PropagateSample


class InitSample(Protocol):
    def __call__(self, key: KeyArray, model_inputs: ArrayTreeLike) -> ArrayTree:
        ...


class PropagateSample(Protocol):
    def __call__(
        self, key: KeyArray, state: ArrayTreeLike, model_inputs: ArrayTreeLike
    ) -> ArrayTree:
        ...


def build_filter(
    init_sample: InitSample,
    propagate_sample: PropagateSample,
    log_potential: LogPotential,
    n_filter_particles: int,
    resampling_fn=systematic.resampling,
) -> Filter:
    """Build a bootstrap particle filter compatible with the SQMC benchmark.

    Args:
        init_sample: Samples from the initial distribution ``M_0(x_0)``.
        propagate_sample: Samples from the Markov kernel ``M_t(x_t | x_{t-1})``.
        log_potential: Computes ``log G_t(x_{t-1}, x_t)``.
        n_filter_particles: Number of particles.
        resampling_fn: Resampling algorithm (default: systematic).

    Returns:
        Filter object for the particle filter.
    """
    return Filter(
        init_prepare=partial(
            cuthbert.smc.particle_filter.init_prepare,
            init_sample=init_sample,
            n_filter_particles=n_filter_particles,
        ),
        filter_prepare=partial(
            cuthbert.smc.particle_filter.filter_prepare,
            init_sample=init_sample,
            n_filter_particles=n_filter_particles,
        ),
        filter_combine=partial(
            cuthbert.smc.particle_filter.filter_combine,
            propagate_sample=propagate_sample,
            log_potential=log_potential,
            resampling_fn=resampling_fn,
        ),
        associative=False,
    )


__all__ = ["InitSample", "PropagateSample", "build_filter"]