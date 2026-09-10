"""Benchmark the Hilbert-sort implementations and create line charts.

The benchmark compares:

* the optimized JAX implementation in ``sqmc.hilbert_sort.hilbert_sort``;
* Adrien's archived JAX implementation (disabled; kept for reference); and
* ``particles.hilbert.hilbert_sort`` (Numba on CPU).

JAX compilation and host/device transfers are excluded from timed regions.
The benchmark always creates a CPU chart. If JAX reports an available GPU, it
also benchmarks the optimized implementation there and creates a GPU chart; the
Particles CPU result is retained as a reference line in the GPU chart.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

# Keep Matplotlib's cache in a writable location on sandboxed systems.
_MPL_CACHE = Path(tempfile.gettempdir()) / "sqmc-matplotlib-cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sqmc.hilbert_sort.archive import hilbert_adrien  # noqa: E501  # (Adrien JAX baseline commented out)
from sqmc.hilbert_sort.hilbert_sort import hilbert_sort as optimized_hilbert_sort


_DEFAULT_N_VALUES = (100, 300, 1_000, 3_000, 10_000, 30_000, 100_000)
_DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class Timing:
    """Summary of repeated runtime measurements."""

    median_seconds: float
    lower_quartile_seconds: float
    upper_quartile_seconds: float


@dataclass(frozen=True)
class BenchmarkRow:
    """One method's timing at one particle count and execution backend."""

    backend: str
    method: str
    number_of_particles: int
    timing: Timing


def _load_particles_hilbert_sort() -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Prefer the installed Particles package and fall back to its local copy."""
    try:
        from particles.hilbert import hilbert_sort as particles_hilbert_sort

        return particles_hilbert_sort, "installed particles.hilbert"
    except ImportError:
        from sqmc.hilbert_sort.archive.hilbert_particles import (
            hilbert_sort as particles_hilbert_sort,
        )

        return particles_hilbert_sort, "vendored archive/hilbert_particles.py"


def _gpu_devices() -> list[jax.Device]:
    """Return available JAX GPU devices without failing on CPU-only installs."""
    try:
        return list(jax.devices("gpu"))
    except RuntimeError:
        return []


def _summarize(samples: Sequence[float]) -> Timing:
    values = np.asarray(samples, dtype=np.float64)
    return Timing(
        median_seconds=float(np.median(values)),
        lower_quartile_seconds=float(np.quantile(values, 0.25)),
        upper_quartile_seconds=float(np.quantile(values, 0.75)),
    )


def _time_numpy_function(
    function: Callable[[np.ndarray], np.ndarray],
    points: np.ndarray,
    warmups: int,
    repeats: int,
) -> Timing:
    # The first call performs Numba compilation and is intentionally not timed.
    result = function(points)
    if np.asarray(result).shape != (points.shape[0],):
        raise RuntimeError("Particles Hilbert sort returned an invalid shape.")

    for _ in range(warmups):
        result = function(points)
        if np.asarray(result).shape != (points.shape[0],):
            raise RuntimeError("Particles Hilbert sort returned an invalid shape.")

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(points)
        elapsed = time.perf_counter() - start
        if np.asarray(result).shape != (points.shape[0],):
            raise RuntimeError("Particles Hilbert sort returned an invalid shape.")
        samples.append(elapsed)

    return _summarize(samples)


def _time_jax_function(
    function: Callable[[jax.Array], jax.Array],
    points: np.ndarray,
    device: jax.Device,
    warmups: int,
    repeats: int,
) -> Timing:
    """Time a compiled JAX function with data already resident on ``device``."""
    compiled = jax.jit(function)
    device_points = jax.device_put(jnp.asarray(points), device=device)

    # The first call performs compilation and is intentionally not timed.
    compiled(device_points).block_until_ready()
    for _ in range(warmups):
        compiled(device_points).block_until_ready()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        compiled(device_points).block_until_ready()
        samples.append(time.perf_counter() - start)

    return _summarize(samples)


def _format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _benchmark_particles(
    inputs: dict[int, np.ndarray],
    particles_sort: Callable[[np.ndarray], np.ndarray],
    warmups: int,
    repeats: int,
) -> dict[int, Timing]:
    timings = {}
    print("\nParticles/Numba (CPU)")
    for number_of_particles, points in inputs.items():
        timing = _time_numpy_function(
            particles_sort,
            points,
            warmups=warmups,
            repeats=repeats,
        )
        timings[number_of_particles] = timing
        print(
            f"  n={number_of_particles:>8,}: "
            f"{_format_seconds(timing.median_seconds)}"
        )
    return timings


def _benchmark_jax_backend(
    inputs: dict[int, np.ndarray],
    device: jax.Device,
    warmups: int,
    repeats: int,
) -> dict[str, dict[int, Timing]]:
    methods = {
        "Optimized JAX": optimized_hilbert_sort,
        # "Adrien JAX": hilbert_adrien.hilbert_sort,
    }
    results: dict[str, dict[int, Timing]] = {}

    for method_name, function in methods.items():
        print(f"\n{method_name} ({device.platform.upper()}: {device})")
        method_timings = {}
        for number_of_particles, points in inputs.items():
            timing = _time_jax_function(
                function,
                points,
                device=device,
                warmups=warmups,
                repeats=repeats,
            )
            method_timings[number_of_particles] = timing
            print(
                f"  n={number_of_particles:>8,}: "
                f"{_format_seconds(timing.median_seconds)}"
            )
        results[method_name] = method_timings

    return results


def _rows_for_backend(
    backend: str,
    n_values: Sequence[int],
    particles_timings: dict[int, Timing],
    jax_timings: dict[str, dict[int, Timing]],
) -> list[BenchmarkRow]:
    rows = []
    for method_name, timings in jax_timings.items():
        rows.extend(
            BenchmarkRow(
                backend=backend,
                method=method_name,
                number_of_particles=n,
                timing=timings[n],
            )
            for n in n_values
        )
    rows.extend(
        BenchmarkRow(
            backend=backend,
            method="Particles Numba (CPU)",
            number_of_particles=n,
            timing=particles_timings[n],
        )
        for n in n_values
    )
    return rows


def _plot_rows(
    rows: Sequence[BenchmarkRow],
    dimension: int,
    backend: str,
    output_directory: Path,
) -> Path:
    """Create the requested particle-count versus runtime line chart."""
    figure, axis = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)
    styles = {
        "Optimized JAX": ("o", "-"),
        # "Adrien JAX": ("s", "-"),
        "Particles Numba (CPU)": ("^", "--"),
    }

    for method in styles:
        method_rows = sorted(
            (row for row in rows if row.method == method),
            key=lambda row: row.number_of_particles,
        )
        particles = np.asarray(
            [row.number_of_particles for row in method_rows],
            dtype=np.int64,
        )
        medians = np.asarray(
            [row.timing.median_seconds for row in method_rows],
            dtype=np.float64,
        )
        lower = np.asarray(
            [row.timing.lower_quartile_seconds for row in method_rows],
            dtype=np.float64,
        )
        upper = np.asarray(
            [row.timing.upper_quartile_seconds for row in method_rows],
            dtype=np.float64,
        )
        log_particles = np.log10(particles)
        log_medians = np.log10(medians)
        log_lower = np.log10(lower)
        log_upper = np.log10(upper)
        marker, line_style = styles[method]
        (line,) = axis.plot(
            log_particles,
            log_medians,
            marker=marker,
            linestyle=line_style,
            linewidth=2.0,
            markersize=5.5,
            label=method,
        )
        axis.fill_between(
            log_particles,
            log_lower,
            log_upper,
            color=line.get_color(),
            alpha=0.14,
            linewidth=0,
        )

    axis.set_xlabel(r"$\log_{10}(n)$")
    axis.set_ylabel(r"$\log_{10}(t / \mathrm{s})$")
    axis.set_title(
        f"Hilbert-sort runtime, d={dimension} ({backend.upper()})\n"
        "Compilation and data transfer excluded; shaded region is the IQR"
    )
    axis.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )
    axis.grid(False, which="minor")
    axis.legend(frameon=False)

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"hilbert_sort_benchmark_{backend}.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def _write_csv(rows: Sequence[BenchmarkRow], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "backend",
                "method",
                "number_of_particles",
                "median_seconds",
                "lower_quartile_seconds",
                "upper_quartile_seconds",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.backend,
                    row.method,
                    row.number_of_particles,
                    row.timing.median_seconds,
                    row.timing.lower_quartile_seconds,
                    row.timing.upper_quartile_seconds,
                ]
            )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark optimized JAX, Adrien JAX, and Particles Hilbert sort."
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        help="Particle dimension (default: 2).",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=list(_DEFAULT_N_VALUES),
        help="Particle counts used on the x-axis.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="Timed repetitions per point (default: 7).",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Untimed calls after compilation (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIRECTORY,
    )
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if not 2 <= arguments.dimension <= 62:
        raise ValueError("dimension must be in [2, 62].")
    if not arguments.n_values or any(n <= 0 for n in arguments.n_values):
        raise ValueError("all n-values must be positive.")
    if len(set(arguments.n_values)) != len(arguments.n_values):
        raise ValueError("n-values must not contain duplicates.")
    if arguments.repeats <= 0:
        raise ValueError("repeats must be positive.")
    if arguments.warmups < 0:
        raise ValueError("warmups must be non-negative.")


def main() -> None:
    arguments = _parse_arguments()
    _validate_arguments(arguments)
    n_values = tuple(sorted(arguments.n_values))

    particles_sort, particles_source = _load_particles_hilbert_sort()
    cpu_device = jax.devices("cpu")[0]
    gpu_devices = _gpu_devices()

    print("Hilbert-sort benchmark")
    print(f"  dimension: {arguments.dimension}")
    print(f"  n values: {list(n_values)}")
    print(f"  repeats: {arguments.repeats}")
    print(f"  JAX x64 enabled: {jax.config.jax_enable_x64}")
    print(f"  CPU device: {cpu_device}")
    print(f"  Particles source: {particles_source}")
    print(f"  GPU devices: {gpu_devices if gpu_devices else 'none'}")

    rng = np.random.default_rng(arguments.seed)
    inputs = {
        n: rng.standard_normal((n, arguments.dimension)).astype(np.float64)
        for n in n_values
    }

    particles_timings = _benchmark_particles(
        inputs,
        particles_sort=particles_sort,
        warmups=arguments.warmups,
        repeats=arguments.repeats,
    )

    all_rows = []
    cpu_jax_timings = _benchmark_jax_backend(
        inputs,
        device=cpu_device,
        warmups=arguments.warmups,
        repeats=arguments.repeats,
    )
    cpu_rows = _rows_for_backend(
        "cpu",
        n_values,
        particles_timings,
        cpu_jax_timings,
    )
    all_rows.extend(cpu_rows)
    cpu_chart = _plot_rows(
        cpu_rows,
        dimension=arguments.dimension,
        backend="cpu",
        output_directory=arguments.output_dir,
    )
    print(f"\nSaved CPU chart: {cpu_chart}")

    if gpu_devices:
        gpu_jax_timings = _benchmark_jax_backend(
            inputs,
            device=gpu_devices[0],
            warmups=arguments.warmups,
            repeats=arguments.repeats,
        )
        gpu_rows = _rows_for_backend(
            "gpu",
            n_values,
            particles_timings,
            gpu_jax_timings,
        )
        all_rows.extend(gpu_rows)
        gpu_chart = _plot_rows(
            gpu_rows,
            dimension=arguments.dimension,
            backend="gpu",
            output_directory=arguments.output_dir,
        )
        print(f"Saved GPU chart: {gpu_chart}")
    else:
        print("No JAX GPU detected; skipped the GPU benchmark and chart.")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = arguments.output_dir / "hilbert_sort_benchmark.csv"
    _write_csv(all_rows, csv_path)
    print(f"Saved timings: {csv_path}")


if __name__ == "__main__":
    main()
