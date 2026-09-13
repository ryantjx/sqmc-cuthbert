"""Publication figures from the saved 7 September 2026 CPU/GPU comparison.

No experiments are rerun. Inputs are the supplied results/budget JSON files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sqmc-dissertation-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

REPO = Path(__file__).resolve().parents[4]
DIMS = [2, 5, 10, 30, 60]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.run_dir, args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42})
    sources = ["comparison_config.json", "qmc/results.json", "qmc/scipy_comparison.json", "qmc/cpu_gpu_comparison.json", "hilbert_sort/cpu_gpu_comparison.json",
               "sqmc/results.json", "sqmc/budget_summary.json"]
    config = read(root / "comparison_config.json")
    provenance = {"run_id": config["run_id"], "source_commit": config["source_commit"],
                  "input_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
                  "analysis_sha256": {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in [Path(__file__).resolve(), REPO / "sqmc/comparison/plotting.py"]},
                  "figures": []}

    def save(fig, stem):
        filename = f"{stem}_{config['run_id']}"
        fig.savefig(output / (filename + ".pdf"), bbox_inches="tight")
        fig.savefig(output / (filename + ".png"), dpi=220, bbox_inches="tight")
        provenance["figures"].append(filename + ".pdf")
        plt.close(fig)

    import sys
    sys.path.insert(0, str(REPO))
    from sqmc.comparison.plotting import qmc_figure, hilbert_figure
    save(qmc_figure(read(root / "qmc/results.json")), "comparison_qmc")
    save(hilbert_figure(read(root / "hilbert_sort/cpu_gpu_comparison.json")), "comparison_hilbert")

    rows = read(root / "sqmc/results.json")
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.5), layout="constrained", sharey=True)
    for dimension, ax in zip(DIMS, axes):
        for backend, color, marker in [("cpu", "#0072B2", "o"), ("gpu", "#D55E00", "s")]:
            data = sorted((v for v in rows if v["dimension"] == dimension and v["backend"] == backend), key=lambda v: v["n"])
            ax.plot([v["n"] for v in data], [1000*v["median_seconds"] for v in data], color=color, marker=marker, markersize=3, label=backend.upper())
            ax.fill_between([v["n"] for v in data], [1000*v["q25_seconds"] for v in data], [1000*v["q75_seconds"] for v in data], color=color, alpha=.2)
        ax.set(xscale="log", yscale="log", title=f"d = {dimension}", xlabel="Particles N")
        ax.xaxis.set_major_locator(FixedLocator([128, 512, 2048]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: rf"$2^{{{int(np.log2(x))}}}$"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", alpha=.15)
    axes[0].set_ylabel("100-step trajectory time (ms)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    save(fig, "comparison_sqmc_runtime")

    budgets = read(root / "sqmc/budget_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3), layout="constrained", sharey=True)
    for ax, budget in zip(axes, [.05, .1]):
        for backend, color, marker, offset in [("cpu", "#0072B2", "o", -.06), ("gpu", "#D55E00", "s", .06)]:
            data = sorted((v for v in budgets if v["budget_seconds"] == budget and v["backend"] == backend and v["status"] == "selected"), key=lambda v: v["dimension"])
            x = np.asarray([DIMS.index(v["dimension"]) for v in data], dtype=float) + offset
            means = np.asarray([v["validation_rmse"] for v in data])
            bounds = np.asarray([v["validation_ci95"] for v in data])
            ax.errorbar(x, means, yerr=[means-bounds[:, 0], bounds[:, 1]-means], color=color, marker=marker, markersize=4, capsize=3, label=backend.upper())
        ax.set(title=f"Median runtime budget: {1000*budget:g} ms", xlabel="State dimension d", yscale="log", xticks=range(5), xticklabels=DIMS)
        ax.grid(True, which="major", alpha=.15)
    axes[0].text(.97, .97, "CPU infeasible\nat d = 60", transform=axes[0].transAxes,
                 ha="right", va="top", color="#0072B2", fontsize=7,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1})
    axes[0].set_ylabel("Held-out normalised filtering RMSE")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    save(fig, "comparison_sqmc_budget")
    (output / f"comparison_figures_{config['run_id']}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Generated", *provenance["figures"], sep="\n")


if __name__ == "__main__":
    main()
