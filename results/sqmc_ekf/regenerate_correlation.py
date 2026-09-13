"""Regenerate the SQMC prior-correlation figure from the fitted gamma_0.

The dissertation figure should show the *prior* between-team correlations
carried by Gamma_0 (the learned initial covariance), not the final-state
gamma_T which is near-zero after Kalman conditioning. This script reads the
fitted params checkpoint, computes the correlation matrix from gamma_0, and
writes the top-N positive/negative pair bar chart plus a JSON of the pairs.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FITTED = os.path.join(HERE, "sqmc", "results", "sqmc", "fitted_params.json")
OUT_PNG = os.path.join(HERE, "combined", "images", "sqmc_correlation_topn_bar.png")
OUT_JSON = os.path.join(HERE, "combined", "images", "sqmc_correlation_topn_bar.json")
TOP_N = 5


def main() -> None:
    with open(FITTED) as f:
        d = json.load(f)

    gamma_0 = np.asarray(d["constrained"]["model"]["gamma_0"], dtype=float)
    team_map = d["team_id_to_name"]
    teams = [team_map[str(i)] for i in range(len(team_map))]

    std = np.sqrt(np.diag(gamma_0))
    std_safe = np.where(std > 1e-10, std, 1.0)
    corr = np.clip(gamma_0 / np.outer(std_safe, std_safe), -1, 1)

    num_teams = len(teams)
    iu = np.triu_indices(num_teams, k=1)
    vals = corr[iu]
    pairs = [(int(i), int(j)) for i, j in zip(*iu)]

    order = np.argsort(vals)[::-1]
    pos_idx = order[vals[order] > 1e-8][:TOP_N]
    neg_order = np.argsort(vals)
    neg_idx = neg_order[vals[neg_order] < -1e-8][:TOP_N]

    def label_for(i, j):
        return f"{teams[i]} - {teams[j]}"

    pos_labels = [label_for(*pairs[k]) for k in pos_idx[::-1]]
    pos_values = [vals[k] for k in pos_idx[::-1]]
    neg_labels = [label_for(*pairs[k]) for k in neg_idx[::-1]]
    neg_values = [vals[k] for k in neg_idx[::-1]]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(16, max(6, 0.5 * max(len(pos_idx), len(neg_idx))))
    )
    if neg_values:
        ax_right.barh(neg_labels, neg_values, color="red")
    ax_right.set_xlim(-1, 0)
    ax_right.axvline(0, color="black", linewidth=1)
    ax_right.set_xlabel("Correlation")
    ax_right.set_title(f"Top {len(neg_idx)} Negative Correlations (Prior $\\Gamma_0$)")
    ax_right.grid(True, axis="x", alpha=0.3)

    if pos_values:
        ax_left.barh(pos_labels, pos_values, color="blue")
    ax_left.set_xlim(0, 1)
    ax_left.axvline(0, color="black", linewidth=1)
    ax_left.set_xlabel("Correlation")
    ax_left.set_title(f"Top {len(pos_idx)} Positive Correlations (Prior $\\Gamma_0$)")
    ax_left.grid(True, axis="x", alpha=0.3)

    if not pos_values:
        ax_left.set_title("No significant positive correlations found")
    if not neg_values:
        ax_right.set_title("No significant negative correlations found")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUT_PNG}")
    plt.close(fig)

    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "positive": [
                    {"pair": label, "correlation": float(val)}
                    for label, val in zip(pos_labels, pos_values)
                ],
                "negative": [
                    {"pair": label, "correlation": float(val)}
                    for label, val in zip(neg_labels, neg_values)
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved correlation data to {OUT_JSON}")


if __name__ == "__main__":
    main()
