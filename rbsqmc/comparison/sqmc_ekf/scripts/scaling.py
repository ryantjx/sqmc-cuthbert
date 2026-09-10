"""Shared per-match observation scaling for the SQMC filter.

Training, prediction, and ranking must use the *same* observation model.
``Methods.filter`` (training) applies a learned ``friendly_scale`` to friendly
fixtures and a configured baseline to all other matches. Prediction and ranking
previously omitted these scales, silently defaulting to ones inside
``run_filter_sqmc``. This module centralises the construction so every path
uses identical per-match scales.

The friendly flags must come from ``dataset.inputs.friendly`` (a
``MatchInputs`` field), not from ``dataset.sqmc`` (a ``FootballResults`` has no
``friendly`` field). Both are built from the same ``frame`` in
``data_ekf.load_dataset`` and are row-aligned one-match-per-row.
"""

import jax.numpy as jnp


def sqmc_match_scales(friendly, friendly_scale, match_scale=1.0):
    """Map ``(T,)`` friendly flags to ``(T, 1)`` per-match observation scales.

    Preserves the observation model currently used during SQMC training:
    friendlies use the learned ``friendly_scale``, all other matches use the
    configured ``match_scale`` baseline. The two are *not* multiplied for
    friendlies; doing so would change the training model.

    Args:
        friendly: ``(T,)`` boolean array, one flag per match row.
        friendly_scale: learned positive scalar for friendly fixtures.
        match_scale: configured baseline for non-friendly fixtures (default 1).

    Returns:
        ``(T, 1)`` float array of per-match scales, aligned with the
        comparison dataset's one-match-per-row order.
    """
    return jnp.where(
        jnp.asarray(friendly, dtype=bool), friendly_scale, match_scale
    )[:, None]
