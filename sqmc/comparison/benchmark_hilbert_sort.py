"""Compare shared Hilbert sorting on identical QMC-generated CPU/GPU inputs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import numpy as np
from sqmc.comparison import common
from sqmc.qmc.qmc import normal_coordinates
from sqmc.hilbert_sort.hilbert_sort import hilbert_sort


def generate_points(sequence, dimension, n, seed, distribution):
    # One shared host array is transferred to both devices; generation is untimed.
    with jax.default_device(jax.devices("cpu")[0]):
        generator = common.engine(sequence, dimension, jax.random.key(seed), True)
        points = generator.sample(n)
        if distribution == "normal":
            points = normal_coordinates(points)
        return np.asarray(jax.block_until_ready(points))


def validate_permutation(permutation, n):
    permutation = np.asarray(permutation)
    if permutation.shape != (n,) or not np.array_equal(np.sort(permutation), np.arange(n)):
        raise RuntimeError("Hilbert output is not a permutation of all input rows.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_arguments(parser)
    parser.add_argument("--sequences", nargs="+", choices=["sobol", "halton"], default=["sobol"])
    parser.add_argument("--distribution", choices=["uniform", "normal"], default="normal")
    args = parser.parse_args(argv)
    with common.run_directory("hilbert_sort", args) as output:
        common.validate_grid(args.dimensions, args.n_values, max_dimension=62,
                             power_two="sobol" in args.sequences)
        selected = common.devices(args.platforms)
        common.provenance(output, selected)
        rows, comparisons = [], []
        case = 0
        for sequence in dict.fromkeys(args.sequences):
            for dimension in args.dimensions:
                for n in args.n_values:
                    start = time.perf_counter()
                    points = generate_points(sequence, dimension, n, args.seed, args.distribution)
                    generation = time.perf_counter() - start
                    digest = hashlib.sha256(points.tobytes()).hexdigest()
                    np.savez_compressed(output / f"{sequence}_d{dimension}_n{n}_inputs.npz", points=points)
                    permutations = {}
                    order = list(selected)
                    if case % 2:
                        order.reverse()
                    case += 1
                    for position, backend in enumerate(order):
                        device = selected[backend]
                        with jax.default_device(device):
                            runner = jax.jit(hilbert_sort)
                            inputs = common.prepare_inputs([(points,)], device)
                            timing, result = common.timed(runner, inputs, args.warmups, args.repeats)
                        common.assert_device(result, device)
                        validate_permutation(result, n)
                        permutations[backend] = np.asarray(result)
                        row = dict(sequence=sequence, distribution=args.distribution, dimension=dimension,
                                   n=n, backend=backend, order=position, input_sha256=digest,
                                   input_generation_seconds=generation, **timing)
                        rows.append(row)
                        common.write_json(output / "results.json", rows)
                        common.write_csv(output / "results.csv", rows)
                        print(f"{sequence}/{args.distribution} d={dimension} N={n} {backend}: {timing['median_seconds']:.6f}s")
                    if set(permutations) == {"cpu", "gpu"}:
                        pair = {row["backend"]: row for row in rows[-2:]}
                        comparisons.append(dict(sequence=sequence, dimension=dimension, n=n,
                            cpu_over_gpu=pair["cpu"]["median_seconds"] / pair["gpu"]["median_seconds"],
                            identical_permutation=bool(np.array_equal(permutations["cpu"], permutations["gpu"])),
                            differing_positions=int(np.count_nonzero(permutations["cpu"] != permutations["gpu"]))))
                        common.write_json(output / "cpu_gpu_comparison.json", comparisons)
        common.write_json(output / "cpu_gpu_comparison.json", comparisons)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sqmc.comparison.plotting import hilbert_figure
        if comparisons:
            figure = hilbert_figure(comparisons)
            figure.savefig(output / "runtime.png", dpi=220, bbox_inches="tight")
            figure.savefig(output / "runtime.pdf", bbox_inches="tight")
            plt.close(figure)
        else:
            common.plot_lines(rows, output, "runtime.png", group_fields=["sequence", "dimension", "backend"],
                              x="n", y="median_seconds", ylabel="Median sorting time (seconds)")


if __name__ == "__main__":
    main()
