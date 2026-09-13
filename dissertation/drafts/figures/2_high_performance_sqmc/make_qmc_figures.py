"""Generate QMC-vs-MC point-set figures for the dissertation.

Figure 1: 256 points in [0,1)^2, independent uniform random (left) vs the
first 256 non-origin points of the Sobol' sequence (right).

Figure 2: 2D projections of a 256-point set in higher dimensions, comparing
independent uniform random points with the Sobol' sequence. The first two
coordinates are shown for d in {2, 5, 10, 30}.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

rng = np.random.default_rng(0)
N = 256


def sobol_points(d, n):
    """First n non-origin points of the (unscrambled) Sobol' sequence in dim d."""
    sampler = qmc.Sobol(d=d, scramble=False)
    return sampler.random(n + 1)[1:]  # drop the origin


# ---------------------------------------------------------------------------
# Figure 1: QMC vs MC in two dimensions
# ---------------------------------------------------------------------------
mc2 = rng.random((N, 2))
sobol2 = sobol_points(2, N)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
for ax, pts, title in zip(axes, (mc2, sobol2),
                          ("Monte Carlo", "Sobol' sequence")):
    ax.scatter(pts[:, 0], pts[:, 1], s=8, c="k", linewidths=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("$u_1$")
    ax.set_ylabel("$u_2$")
fig.tight_layout()
fig.savefig("qmc_vs_mc.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: QMC vs MC in higher dimensions (2D projections)
# ---------------------------------------------------------------------------
dims = [2, 5, 10, 30]
fig, axes = plt.subplots(2, len(dims), figsize=(12, 6), sharex=True, sharey=True)

for j, d in enumerate(dims):
    mc = rng.random((N, d))
    sobol = sobol_points(d, N)

    # Top row: Monte Carlo; bottom row: Sobol'
    for row, pts, label in ((0, mc, "MC"), (1, sobol, "Sobol'")):
        ax = axes[row, j]
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c="k", linewidths=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        if row == 0:
            ax.set_title(f"$d={d}$")
        if j == 0:
            ax.set_ylabel(label)

fig.suptitle("First two coordinates of 256 points")
fig.tight_layout()
fig.savefig("qmc_vs_mc_highdim.png", dpi=300)
plt.close(fig)

print("Wrote qmc_vs_mc.png and qmc_vs_mc_highdim.png")
