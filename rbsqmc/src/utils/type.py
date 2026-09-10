import jax
from typing import NamedTuple

class Matches(NamedTuple):
    home_id: jax.Array
    away_id: jax.Array
    home_score: jax.Array
    away_score: jax.Array

class FootballResults(NamedTuple):
    # match_index_id : jax.Array
    date: jax.Array # (T,)
    timestamp: jax.Array # (T,)
    timestamp_prev: jax.Array # (T,)
    matches: Matches # (T, M), where M is the max number of matches per day
    match_mask: jax.Array # (T, M), boolean mask indicating valid matches
class RBPFState(NamedTuple):
    x: jax.Array

class RBPFFootballResults(NamedTuple):
    timestamp: jax.Array
    timestamp_prev: jax.Array
    matches: Matches
    match_mask: jax.Array
    gamma: jax.Array #  Covariance at end of day t, shape (T, N, N)
    gamma_pred: jax.Array # Covariance at start of day, (T, N, N)
    gamma_observed: jax.Array # 2x2 covariance for teams in match j before conditioning, (T, M, 2, 2)
    kalman_gain: jax.Array # Kalman gain, (T, M, N, 2)
    
class EMParams(NamedTuple):
    """All parameters that EM optimizes."""
    mean_0: jax.Array      # (M, 2)
    gamma_0: jax.Array
    B: jax.Array         # (2, 2)
    kappa: jax.Array      # scalar
    alpha: jax.Array      # scalar
    beta: jax.Array       # scalar

class RawEMParams(NamedTuple):
    gamma_0_chol: jax.Array
    b_chol_raw: jax.Array    # (2, 2) free factor -> decodes to det(B)=1 SPD
    kappa_raw: jax.Array
    alpha: jax.Array
    beta: jax.Array