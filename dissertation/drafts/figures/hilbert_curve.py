"""Generate Hilbert space-filling curve figures for the dissertation.

Figure: the first iterates H_1, H_2, H_3 of the 2D Hilbert curve, showing how
the curve fills the unit square at increasing resolution.
"""
import numpy as np
import matplotlib.pyplot as plt


def _rot(n, x, y, rx, ry):
    """Rotate/flip a quadrant of the Hilbert curve (standard d2xy helper)."""
    if ry == 0:
        if rx == 1:
            x[0] = n - 1 - x[0]
            y[0] = n - 1 - y[0]
        x[0], y[0] = y[0], x[0]


def d2xy(n, d):
    """Return the (x, y) coordinates of distance d along the n x n 2D Hilbert
    curve, with n a power of two. Canonical algorithm."""
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        xv, yv = [x], [y]
        _rot(s, xv, yv, rx, ry)
        x, y = xv[0], yv[0]
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_path(order):
    """Coordinates (N, 2) of the points along the 2D Hilbert curve of the
    given order, in visitation order, scaled to [0, 1]^2."""
    n = 1 << order
    N = n * n
    pts = np.empty((N, 2))
    for i in range(N):
        x, y = d2xy(n, i)
        pts[i, 0] = (x + 0.5) / n
        pts[i, 1] = (y + 0.5) / n
    return pts


def draw_order(ax, order):
    pts = hilbert_path(order)
    ax.plot(pts[:, 0], pts[:, 1], lw=0.8, color="k")
    ax.plot(pts[0, 0], pts[0, 1], "o", ms=4, color="C0")   # start
    ax.plot(pts[-1, 0], pts[-1, 1], "o", ms=4, color="C3")  # end
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"$H_{order}$")


fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
for order, ax in zip((1, 2, 3, 4, 5, 6), axes.ravel()):
    draw_order(ax, order)
fig.tight_layout()
fig.savefig("hilbert_curve.png", dpi=300)
plt.close(fig)
print("Wrote hilbert_curve.png")
