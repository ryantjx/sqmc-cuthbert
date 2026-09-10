"""Fresh-scramble generation with shared JAX CPU/GPU and SciPy CPU engines."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc as scipy_qmc
from sqmc.comparison import common


def make_runner(sequence, n, dimension, scramble, mode, device, seed):
    if not scramble or mode != "fresh":
        raise ValueError("This benchmark requires fresh scrambling.")
    with jax.default_device(device):
        @jax.jit
        def run(index, key):
            generator = common.engine(sequence, dimension, key, True)
            return generator.sample(n)
    return run


def make_scipy_runner(sequence, n, dimension):
    def run(seed):
        # RNG creation, generator construction, scrambling and sampling are timed.
        rng = np.random.default_rng(seed)
        if sequence == "sobol":
            generator = scipy_qmc.Sobol(dimension, scramble=True, bits=30,
                                        rng=rng, optimization=None)
            return generator.random_base2(n.bit_length() - 1)
        generator = scipy_qmc.Halton(dimension, scramble=True, rng=rng, optimization=None)
        return generator.random(n, workers=1)
    return run


def scipy_timed(runner, seeds, warmups, repeats):
    start = time.perf_counter()
    result = runner(seeds[0])
    first = time.perf_counter() - start
    for i in range(warmups):
        runner(seeds[i % len(seeds)])
    samples = []
    for seed in seeds[:repeats]:
        start = time.perf_counter()
        result = runner(seed)
        samples.append(time.perf_counter() - start)
    return dict(first_call_seconds=first, samples_seconds=samples,
                median_seconds=float(np.median(samples)),
                q25_seconds=float(np.quantile(samples, .25)),
                q75_seconds=float(np.quantile(samples, .75))), result


def comparisons(rows):
    groups = {}
    fields = ("sequence", "dimension", "n", "mode")
    for row in rows:
        group = groups.setdefault(tuple(row[k] for k in fields), {})
        identity = (row["implementation"], row["backend"])
        if identity in group:
            raise ValueError("Duplicate QMC implementation record")
        group[identity] = row["median_seconds"]
    jax_pairs, scipy_pairs = [], []
    for key, times in groups.items():
        identity = dict(zip(fields, key))
        cpu, gpu, scipy = (times.get(k) for k in (("jax", "cpu"), ("jax", "gpu"), ("scipy", "cpu")))
        if cpu is not None and gpu is not None:
            jax_pairs.append(identity | dict(cpu_seconds=cpu, gpu_seconds=gpu, cpu_over_gpu=cpu/gpu))
        for backend, value in (("cpu", cpu), ("gpu", gpu)):
            if scipy is not None and value is not None:
                scipy_pairs.append(identity | dict(jax_backend=backend, scipy_seconds=scipy,
                                                   jax_seconds=value, scipy_over_jax=scipy/value))
    return jax_pairs, scipy_pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_arguments(parser)
    parser.add_argument("--sequences", nargs="+", choices=["sobol", "halton"], default=["sobol", "halton"])
    parser.add_argument("--modes", nargs="+", choices=["fresh"], default=["fresh"])
    parser.add_argument("--scramble", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--implementations", nargs="+", choices=["jax", "scipy"], default=["jax", "scipy"])
    args = parser.parse_args(argv)
    if not args.scramble:
        parser.error("Fresh-scramble comparison requires --scramble.")
    if args.implementations != ["jax", "scipy"]:
        parser.error("Both implementations must be requested in order: jax scipy.")
    with common.run_directory("qmc", args) as output:
        common.validate_grid(args.dimensions, args.n_values, max_dimension=10000,
                             power_two="sobol" in args.sequences)
        selected = common.devices(args.platforms)
        common.provenance(output, selected)
        import json
        metadata = json.loads((output / "metadata.json").read_text())
        metadata.update(schema_version=2, qmc_contract_version=2,
                        scrambling="fresh on every execution; index zero; Sobol bits=30",
                        timing_boundaries={"jax": "compiled scrambling and sampling; compilation and input transfers excluded",
                                           "scipy": "RNG and engine construction, scrambling and sampling; Halton workers=1"})
        common.write_json(output / "metadata.json", metadata)
        rows = []
        case = 0
        for sequence in dict.fromkeys(args.sequences):
            for dimension in args.dimensions:
                for n in args.n_values:
                    implementations = [("jax", b) for b in selected] + [("scipy", "cpu")]
                    shift = case % len(implementations)
                    order = implementations[shift:] + implementations[:shift]
                    case += 1
                    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(
                        [args.seed, int(sequence == "halton"), dimension, n]).spawn(args.repeats)]
                    for position, (implementation, backend) in enumerate(order):
                        start = time.perf_counter()
                        if implementation == "jax":
                            device = selected[backend]
                            runner = make_runner(sequence, n, dimension, True, "fresh", device, args.seed)
                            setup = time.perf_counter() - start
                            with jax.default_device(device):
                                inputs = common.prepare_inputs([(jnp.uint32(0), jax.random.key(seed)) for seed in seeds], device)
                                timing, points = common.timed(runner, inputs, args.warmups, args.repeats)
                            common.assert_device(points, device)
                        else:
                            runner = make_scipy_runner(sequence, n, dimension)
                            setup = time.perf_counter() - start
                            timing, points = scipy_timed(runner, seeds, args.warmups, args.repeats)
                        host = np.asarray(points)
                        if host.shape != (n, dimension) or host.dtype != np.float64 or not np.isfinite(host).all() or not ((host >= 0) & (host < 1)).all():
                            raise RuntimeError("QMC output failed dtype/shape/range validation.")
                        row = dict(sequence=sequence, dimension=dimension, n=n, mode="fresh", scramble=True,
                                   implementation=implementation, backend=backend, order=position,
                                   repetition_seeds=seeds, setup_seconds=setup, **timing)
                        rows.append(row)
                        common.write_json(output / "results.json", rows)
                        common.write_csv(output / "results.csv", rows)
                        print(f"{sequence} d={dimension} N={n} fresh {implementation} {backend}: {timing['median_seconds']:.6f}s", flush=True)
        jax_pairs, scipy_pairs = comparisons(rows)
        common.write_json(output / "cpu_gpu_comparison.json", jax_pairs)
        common.write_json(output / "scipy_comparison.json", scipy_pairs)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sqmc.comparison.plotting import qmc_figure
        figure = qmc_figure(rows)
        figure.savefig(output / "runtime.png", dpi=220, bbox_inches="tight")
        figure.savefig(output / "runtime.pdf", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
