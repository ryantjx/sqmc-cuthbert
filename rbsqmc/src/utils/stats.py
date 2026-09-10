import jax.numpy as jnp
import jax

def _stable_cholesky(matrix: jnp.ndarray, jitter: float = 1e-6) -> jnp.ndarray:
    """Cholesky robust to tiny negative eigenvalues from float32 roundoff.

    The covariance matrices from compute_gamma_trajectory are structurally PSD
    (unobserved block is PD, observed block is exactly 0), but float32 noise
    can leave the zero eigenvalues at ~-5e-10, which makes
    jnp.linalg.cholesky silently return NaN. A small diagonal ridge scaled to
    the matrix norm dominates this noise without affecting the signal.
    """
    matrix = 0.5 * (matrix + matrix.T)
    diag = jnp.diagonal(matrix, axis1=-2, axis2=-1)
    scale = jnp.maximum(jnp.max(diag, axis=-1, keepdims=True), 1.0)
    n = matrix.shape[-1]
    # Broadcast jitter to the matrix's batch shape.
    eye = jnp.eye(n, dtype=matrix.dtype)
    if matrix.ndim > 2:
        jitter_diag = (jitter * scale)[..., None] * eye
    else:
        jitter_diag = (jitter * scale) * eye
    return jnp.linalg.cholesky(matrix + jitter_diag)


def gaussian_kron_logpdf(x, mean, gamma, B):
    M, K = gamma.shape[0], B.shape[0]
    residual = x - mean                          # (..., M, 2)

    L_gamma = _stable_cholesky(gamma)         # (M, M)
    L_B = _stable_cholesky(B)                 # (2, 2)

    # log|Gamma (x) B| = K log|Gamma| + M log|B|
    log_det = (
        K * 2.0 * jnp.sum(jnp.log(jnp.diag(L_gamma)))
        + M * 2.0 * jnp.sum(jnp.log(jnp.diag(L_B)))
    )

    # Whiten: solve L_gamma Z = R over the M axis, then L_B W = Z^T over the K axis.
    # quad = tr(B^{-1} R^T Gamma^{-1} R) = sum(W^2)
    Z = jax.scipy.linalg.solve_triangular(L_gamma, residual, lower=True)  # (..., M, 2)
    Z_T = jnp.swapaxes(Z, -1, -2)                # (..., 2, M)
    W = jax.scipy.linalg.solve_triangular(L_B, Z_T, lower=True)            # (..., 2, M)
    quad = jnp.sum(W * W, axis=(-1, -2))        # (...,)

    return -0.5 * (M * K * jnp.log(2.0 * jnp.pi) + log_det + quad)


def sample_multivariate_normal_kron(key, mean, gamma, B):
    L_gamma = _stable_cholesky(gamma)         # (M, M)
    L_B = _stable_cholesky(B)                 # (2, 2)
    noise = jax.random.normal(key, mean.shape)   # (M, 2)
    # mean + L_gamma @ noise @ L_B^T
    return mean + L_gamma @ noise @ L_B.T