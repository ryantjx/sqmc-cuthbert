"""Compare SQMC accuracy on CPU and GPU at fixed per-trajectory runtime budgets."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
from sqmc.comparison import common
from sqmc.qmc.qmc import Sobol, normal_coordinates
from sqmc.sqmc import sqmc


def kalman(observations):
    """Independent NumPy reference: X0~N(0,I), Q=.25 I, R=I, Y1..YT."""
    mean = np.zeros(observations.shape[1])
    variance = np.ones_like(mean)
    means, variances, log_likelihood = [], [], 0.0
    for observation in observations:
        prediction_variance = variance + .25
        innovation_variance = prediction_variance + 1
        residual = observation - mean
        log_likelihood -= .5 * np.sum(np.log(2 * np.pi * innovation_variance) + residual**2 / innovation_variance)
        gain = prediction_variance / innovation_variance
        mean = mean + gain * residual
        variance = (1 - gain) * prediction_variance
        means.append(mean.copy())
        variances.append(variance.copy())
    return np.asarray(means), np.asarray(variances), float(log_likelihood)


def generate_dataset(seed, dimension, steps, dataset):
    rng = np.random.default_rng(np.random.SeedSequence([seed, dimension, dataset, 0]))
    initial = rng.normal(size=dimension)
    latent = initial + np.cumsum(.5 * rng.normal(size=(steps, dimension)), axis=0)
    return latent + rng.normal(size=latent.shape)


def trajectory_key(seed, dimension, n, dataset, replicate, stream):
    key = jax.random.key(seed)
    for value in (stream, dimension, n, dataset, replicate):
        key = jax.random.fold_in(key, value)
    return key


def make_runner(n, dimension, steps):
    def initial(u, inputs):
        return normal_coordinates(u)

    def propagate(u, previous, inputs):
        return previous + .5 * normal_coordinates(u)

    def potential(previous, state, inputs):
        return -.5 * jnp.sum((inputs["y"] - state)**2 + np.log(2 * np.pi))

    # Only the shared scrambled Sobol path is used. No deterministic mutable
    # sampler is captured by the scan, and sample counts are validated in main.
    generator = Sobol(d=dimension + 1, scramble=True, start_index=0, dtype=jnp.float64)
    filter_ = sqmc.build_filter(initial, propagate, potential, n, generator)

    @jax.jit
    def run(observations, key):
        initial_key, update_key = jax.random.split(key)
        step_keys = jax.random.split(update_key, steps + 1)
        state = filter_.init_prepare({"y": observations[0]}, key=initial_key)
        # Shared combine consumes the previous state's key; do not reuse the
        # initialization key for the first observation's randomization.
        state = state._replace(key=step_keys[0])

        def step(previous, inputs):
            observation, next_key = inputs
            prepared = filter_.filter_prepare({"y": observation}, key=next_key)
            updated = filter_.filter_combine(previous, prepared)
            mean = jnp.sum(jax.nn.softmax(updated.log_weights)[:, None] * updated.particles, axis=0)
            return updated, mean

        state, means = jax.lax.scan(step, state, (observations, step_keys[1:]))
        return means, state.log_normalizing_constant

    return run


def summarize_accuracy(records, bootstrap_reps, seed):
    """Equal-size datasets; hierarchical bootstrap keeps time/coordinates together."""
    ids = sorted({record["dataset"] for record in records})
    squared = np.asarray([[r["normalized_mse"] for r in records if r["dataset"] == dataset] for dataset in ids])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap_reps):
        datasets = rng.integers(0, len(ids), size=len(ids))
        sample = [rng.choice(squared[i], size=squared.shape[1], replace=True).mean() for i in datasets]
        draws.append(np.sqrt(np.mean(sample)))
    return {"rmse": float(np.sqrt(squared.mean())),
            "rmse_ci95": [float(v) for v in np.quantile(draws, [.025, .975])],
            "mean_loglik_error": float(np.mean([r["loglik_error"] for r in records]))}


def budget_summary(rows, dimensions, platforms, budgets):
    """Select by pilot error and measured median time; report held-out accuracy."""
    summaries = []
    for dimension in dimensions:
        for budget in budgets:
            for backend in platforms:
                eligible = [r for r in rows if r["dimension"] == dimension and r["backend"] == backend
                            and r["median_seconds"] <= budget]
                result = dict(dimension=dimension, budget_seconds=budget, backend=backend)
                if not eligible:
                    result["status"] = "no_measured_configuration_within_budget"
                else:
                    chosen = min(eligible, key=lambda r: (r["selection_accuracy"]["rmse"], r["median_seconds"], r["n"]))
                    result.update(status="selected", n=chosen["n"], median_seconds=chosen["median_seconds"],
                                  q75_seconds=chosen["q75_seconds"], selection_rmse=chosen["selection_accuracy"]["rmse"],
                                  validation_rmse=chosen["validation_accuracy"]["rmse"],
                                  validation_ci95=chosen["validation_accuracy"]["rmse_ci95"],
                                  measured_fraction_within_budget=float(np.mean(np.asarray(chosen["samples_seconds"]) <= budget)))
                summaries.append(result)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_arguments(parser, counts=False)
    parser.set_defaults(dimensions=[2, 5])
    parser.add_argument("--particle-counts", nargs="+", type=common.positive, default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--budget-seconds", nargs="+", type=float, default=[.01, .05, .1])
    parser.add_argument("--n-steps", type=common.positive, default=100)
    parser.add_argument("--datasets", type=common.positive, default=1)
    parser.add_argument("--selection-reps", type=common.positive, default=8)
    parser.add_argument("--validation-reps", type=common.positive, default=16)
    parser.add_argument("--bootstrap-reps", type=common.positive, default=500)
    args = parser.parse_args(argv)
    with common.run_directory("sqmc", args) as output:
        common.validate_grid(args.dimensions, args.particle_counts, max_dimension=62, power_two=True)
        if any(not np.isfinite(b) or b <= 0 for b in args.budget_seconds):
            raise ValueError("Runtime budgets must be finite and positive.")
        selected = common.devices(args.platforms)
        common.provenance(output, selected)
        rows, records = [], []
        case = 0
        for dimension in args.dimensions:
            observations = [generate_dataset(args.seed, dimension, args.n_steps, i) for i in range(args.datasets)]
            references = [kalman(y) for y in observations]
            np.savez_compressed(output / f"reference_d{dimension}.npz", observations=observations,
                                means=[r[0] for r in references], variances=[r[1] for r in references],
                                log_likelihoods=[r[2] for r in references])
            for n in args.particle_counts:
                order = list(selected)
                if case % 2:
                    order.reverse()
                case += 1
                for position, backend in enumerate(order):
                    device = selected[backend]
                    with jax.default_device(device):
                        runner = make_runner(n, dimension, args.n_steps)
                        timing_inputs = common.prepare_inputs([
                            (observations[r % args.datasets], trajectory_key(args.seed, dimension, n, r % args.datasets, r, 30))
                            for r in range(args.repeats)], device)
                        timing, result = common.timed(runner, timing_inputs, args.warmups, args.repeats)
                        common.assert_device(result, device)
                        accuracies = {}
                        for phase, repetitions, stream in (("selection", args.selection_reps, 10), ("validation", args.validation_reps, 20)):
                            phase_records, estimates = [], []
                            for dataset, (y, reference) in enumerate(zip(observations, references)):
                                truth_mean, truth_variance, truth_loglik = reference
                                for replicate in range(repetitions):
                                    key = trajectory_key(args.seed, dimension, n, dataset, replicate, stream)
                                    inputs = common.prepare_inputs([(y, key)], device)[0]
                                    mean, loglik = jax.block_until_ready(runner(*inputs))
                                    common.assert_device((mean, loglik), device)
                                    host_mean, host_loglik = np.asarray(mean), float(loglik)
                                    if host_mean.shape != (args.n_steps, dimension) or not np.isfinite(host_mean).all() or not np.isfinite(host_loglik):
                                        raise RuntimeError("SQMC produced invalid means or likelihood.")
                                    record = dict(backend=backend, dimension=dimension, n=n, phase=phase,
                                                  dataset=dataset, replicate=replicate, key_data=np.asarray(jax.random.key_data(key)).tolist(),
                                                  normalized_mse=float(np.mean((host_mean - truth_mean)**2 / truth_variance)),
                                                  loglik_error=host_loglik - truth_loglik)
                                    phase_records.append(record)
                                    records.append(record)
                                    estimates.append(host_mean)
                            accuracies[phase] = summarize_accuracy(phase_records, args.bootstrap_reps, args.seed)
                            np.savez_compressed(output / f"{backend}_d{dimension}_n{n}_{phase}.npz", means=estimates)
                            common.write_json(output / "accuracy_records.json", records)
                    row = dict(backend=backend, dimension=dimension, n=n, order=position, **timing,
                               selection_accuracy=accuracies["selection"], validation_accuracy=accuracies["validation"],
                               validation_rmse=accuracies["validation"]["rmse"])
                    rows.append(row)
                    common.write_json(output / "results.json", rows)
                    common.write_csv(output / "results.csv", rows)
                    common.write_json(output / "budget_summary.json", budget_summary(rows, args.dimensions, selected, args.budget_seconds))
                    print(f"SQMC {backend} d={dimension} N={n}: {timing['median_seconds']:.6f}s, held-out RMSE={row['validation_rmse']:.6g}")
        summary = budget_summary(rows, args.dimensions, selected, args.budget_seconds)
        common.write_json(output / "budget_summary.json", summary)
        common.write_csv(output / "budget_summary.csv", summary)
        comparisons = []
        for dimension in args.dimensions:
            for budget in args.budget_seconds:
                pair = {r["backend"]: r for r in summary if r["dimension"] == dimension and r["budget_seconds"] == budget and r["status"] == "selected"}
                if "cpu" in pair and "gpu" in pair:
                    cpu, gpu = pair["cpu"], pair["gpu"]
                    comparisons.append(dict(dimension=dimension, budget_seconds=budget,
                        cpu_n=cpu["n"], gpu_n=gpu["n"], cpu_validation_rmse=cpu["validation_rmse"],
                        gpu_validation_rmse=gpu["validation_rmse"],
                        cpu_over_gpu_error=cpu["validation_rmse"] / gpu["validation_rmse"] if gpu["validation_rmse"] else None))
        common.write_json(output / "cpu_gpu_comparison.json", comparisons)
        for entry in summary:
            print(f"Budget result: {entry}")
        common.plot_lines(rows, output, "runtime.png", group_fields=["dimension", "backend"], x="n", y="median_seconds",
                          ylabel="Median SQMC trajectory time (seconds)", xlabel="Particles N")
        common.plot_lines(rows, output, "accuracy_vs_runtime.png", group_fields=["dimension", "backend"],
                          x="median_seconds", y="validation_rmse", xlabel="Median SQMC trajectory time (seconds)",
                          ylabel="Held-out normalized filtering RMSE")
        feasible = [row for row in summary if row["status"] == "selected"]
        if feasible:
            common.plot_lines(feasible, output, "accuracy_at_budget.png", group_fields=["dimension", "backend"],
                              x="budget_seconds", y="validation_rmse", xlabel="Per-trajectory runtime budget (seconds)",
                              ylabel="Selected configuration: held-out normalized filtering RMSE")


if __name__ == "__main__":
    main()
