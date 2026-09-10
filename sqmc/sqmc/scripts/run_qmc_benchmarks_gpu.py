"""Clone rbsqmc and run its Hilbert-sort and QMC benchmarks on Colab.

The local orchestrator submits only this entry point and its compact JSON
configuration. This script shallow-clones the public repository, installs
missing non-JAX dependencies, generates the ignored Sobol runtime data file,
verifies that JAX is using a GPU, runs both benchmarks, and records metadata.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_DIR = Path("/content/rbsqmc")
REPO_URL = "https://github.com/ryantjx/rbsqmc.git"
DEFAULT_CONFIG_NAME = "qmc_gpu_benchmark_config.json"
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "numba": "numba",
}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    log("Running: " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def install_missing_packages() -> None:
    """Install only unavailable benchmark dependencies; preserve Colab JAX."""
    try:
        importlib.import_module("jax")
    except Exception as error:
        raise RuntimeError(
            "JAX is unavailable. A Colab GPU runtime with its preinstalled "
            "CUDA-enabled JAX is required; this script will not replace it."
        ) from error

    missing = []
    for module, package in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(package)

    if missing:
        log(f"Installing missing packages: {missing}")
        run([sys.executable, "-m", "pip", "install", *missing])
    else:
        log("All benchmark dependencies are already available")


def clone_repository(repo_url: str, branch: str) -> Path:
    """Create a shallow sparse checkout containing only ``sqmc/``."""
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(
        [
            "git",
            "-c",
            "http.version=HTTP/1.1",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--single-branch",
            "--branch",
            branch,
            repo_url,
            str(REPO_DIR),
        ]
    )
    run(
        ["git", "sparse-checkout", "set", "sqmc"],
        cwd=REPO_DIR,
    )
    return REPO_DIR


def validate_config(config: dict) -> dict:
    """Validate and normalize the benchmark configuration."""
    for section in ("hilbert_sort", "qmc"):
        if section not in config:
            raise ValueError(f"missing config section: {section}")
        values = config[section]
        dimensions = values.get("dimensions")
        if not dimensions or any(int(d) <= 0 for d in dimensions):
            raise ValueError(f"{section}.dimensions must contain positive integers")
        if len(set(int(d) for d in dimensions)) != len(dimensions):
            raise ValueError(f"{section}.dimensions must not contain duplicates")
        if not values["n_values"] or any(int(n) <= 0 for n in values["n_values"]):
            raise ValueError(f"{section}.n_values must contain positive integers")
        if int(values["repeats"]) <= 0:
            raise ValueError(f"{section}.repeats must be positive")
        if int(values["warmups"]) < 0:
            raise ValueError(f"{section}.warmups must be non-negative")
    return config


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return validate_config(json.load(source))


def benchmark_command(
    module: str,
    values: dict,
    dimension: int,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        module,
        "--dimension",
        str(dimension),
        "--n-values",
        *(str(n) for n in values["n_values"]),
        "--repeats",
        str(values["repeats"]),
        "--warmups",
        str(values["warmups"]),
        "--seed",
        str(values["seed"]),
        "--output-dir",
        str(output_dir),
    ]
    if module == "sqmc.qmc.benchmark_qmc":
        command.append(
            "--scramble" if values.get("scramble", True) else "--no-scramble"
        )
    return command


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_series(axis, rows, x_field: str, label_field: str, styles: dict) -> None:
    """Plot median and IQR values after explicitly applying log10."""
    import numpy as np

    for label, (marker, line_style) in styles.items():
        series = sorted(
            (row for row in rows if row[label_field] == label),
            key=lambda row: int(row[x_field]),
        )
        counts = np.asarray([int(row[x_field]) for row in series])
        medians = np.asarray([float(row["median_seconds"]) for row in series])
        lower = np.asarray(
            [float(row["lower_quartile_seconds"]) for row in series]
        )
        upper = np.asarray(
            [float(row["upper_quartile_seconds"]) for row in series]
        )
        log_counts = np.log10(counts)
        (line,) = axis.plot(
            log_counts,
            np.log10(medians),
            marker=marker,
            linestyle=line_style,
            linewidth=2.0,
            markersize=5.0,
            label=label,
        )
        axis.fill_between(
            log_counts,
            np.log10(lower),
            np.log10(upper),
            color=line.get_color(),
            alpha=0.14,
            linewidth=0,
        )

    axis.set_xlabel(r"$\log_{10}(n)$")
    axis.set_ylabel(r"$\log_{10}(t / \mathrm{s})$")
    axis.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.6)
    axis.grid(False, which="minor")
    axis.legend(frameon=False)


def create_hilbert_summary(
    dimension_rows: dict[int, list[dict[str, str]]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    dimensions = list(dimension_rows)
    figure, axes = plt.subplots(
        len(dimensions),
        1,
        figsize=(10.0, 4.3 * len(dimensions)),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
    )
    styles = {
        "Optimized JAX": ("o", "-"),
        "Adrien JAX": ("s", "-"),
        "Particles Numba (CPU)": ("^", "--"),
    }
    aggregate_rows = []
    for row_index, dimension in enumerate(dimensions):
        rows = dimension_rows[dimension]
        for row in rows:
            aggregate_rows.append({"dimension": str(dimension), **row})
        gpu_rows = [row for row in rows if row["backend"] == "gpu"]
        axis = axes[row_index, 0]
        plot_series(axis, gpu_rows, "number_of_particles", "method", styles)
        axis.set_title(f"d={dimension}")

    figure.suptitle(
        "Hilbert-sort GPU runtime by dimension\n"
        "Compilation and data transfer excluded; shaded region is the IQR"
    )
    figure.savefig(
        output_dir / "hilbert_sort_benchmark_gpu.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    write_csv_rows(
        aggregate_rows,
        [
            "dimension",
            "backend",
            "method",
            "number_of_particles",
            "median_seconds",
            "lower_quartile_seconds",
            "upper_quartile_seconds",
        ],
        output_dir / "hilbert_sort_benchmark.csv",
    )


def create_qmc_summary(
    dimension_rows: dict[int, list[dict[str, str]]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    dimensions = list(dimension_rows)
    figure, axes = plt.subplots(
        len(dimensions),
        2,
        figsize=(13.0, 4.2 * len(dimensions)),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
    )
    styles = {"QMC JAX": ("o", "-"), "SciPy CPU": ("s", "--")}
    aggregate_rows = []
    for row_index, dimension in enumerate(dimensions):
        rows = dimension_rows[dimension]
        aggregate_rows.extend(rows)
        for column_index, sequence in enumerate(("Halton", "Sobol")):
            gpu_rows = [
                row
                for row in rows
                if row["backend"] == "gpu" and row["sequence"] == sequence
            ]
            axis = axes[row_index, column_index]
            plot_series(
                axis,
                gpu_rows,
                "number_of_samples",
                "implementation",
                styles,
            )
            axis.set_title(f"d={dimension} — {sequence}")

    scramble = next(iter(dimension_rows.values()))[0]["scramble"] == "True"
    scramble_label = "scrambled" if scramble else "unscrambled"
    figure.suptitle(
        f"QMC GPU sampling runtime by dimension ({scramble_label})\n"
        "Compilation, engine initialization, and transfer excluded; "
        "shaded region is the IQR"
    )
    figure.savefig(
        output_dir / "qmc_benchmark_gpu.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    write_csv_rows(
        aggregate_rows,
        [
            "backend",
            "sequence",
            "implementation",
            "dimension",
            "scramble",
            "number_of_samples",
            "median_seconds",
            "lower_quartile_seconds",
            "upper_quartile_seconds",
        ],
        output_dir / "qmc_benchmark.csv",
    )


def plot_dimension_comparison(
    axis,
    dimension_rows: dict[int, list[dict[str, str]]],
    row_filter,
    x_field: str,
) -> None:
    """Plot one line per dimension, using a consistent color assignment."""
    import matplotlib.pyplot as plt
    import numpy as np

    colors = plt.get_cmap("tab10").colors
    for index, (dimension, rows) in enumerate(dimension_rows.items()):
        series = sorted(
            (row for row in rows if row_filter(row)),
            key=lambda row: int(row[x_field]),
        )
        counts = np.asarray([int(row[x_field]) for row in series])
        medians = np.asarray([float(row["median_seconds"]) for row in series])
        lower = np.asarray(
            [float(row["lower_quartile_seconds"]) for row in series]
        )
        upper = np.asarray(
            [float(row["upper_quartile_seconds"]) for row in series]
        )
        log_counts = np.log10(counts)
        color = colors[index % len(colors)]
        axis.plot(
            log_counts,
            np.log10(medians),
            color=color,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label=f"d={dimension}",
        )
        axis.fill_between(
            log_counts,
            np.log10(lower),
            np.log10(upper),
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    axis.set_xlabel(r"$\log_{10}(n)$")
    axis.set_ylabel(r"$\log_{10}(t / \mathrm{s})$")
    axis.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.6)
    axis.grid(False, which="minor")
    axis.legend(frameon=False)


def create_hilbert_algorithm_summary(
    dimension_rows: dict[int, list[dict[str, str]]],
    output_dir: Path,
) -> None:
    """Create a 1x3 algorithm-first Hilbert comparison."""
    import matplotlib.pyplot as plt

    methods = ("Optimized JAX", "Adrien JAX", "Particles Numba (CPU)")
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 5.3),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for axis, method in zip(axes, methods):
        plot_dimension_comparison(
            axis,
            dimension_rows,
            lambda row, selected=method: (
                row["backend"] == "gpu" and row["method"] == selected
            ),
            "number_of_particles",
        )
        axis.set_title(method)

    figure.suptitle(
        "Hilbert-sort runtime across dimensions (GPU comparison)\n"
        "Compilation and data transfer excluded; shaded region is the IQR"
    )
    figure.savefig(
        output_dir / "hilbert_sort_benchmark_by_algorithm_gpu.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def create_qmc_algorithm_summary(
    dimension_rows: dict[int, list[dict[str, str]]],
    output_dir: Path,
) -> None:
    """Create a 1x2 sequence-first QMC dimension comparison."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    dimensions = list(dimension_rows)
    colors = plt.get_cmap("tab10").colors
    implementations = {
        "QMC JAX": ("-", "o"),
        "SciPy CPU": ("--", "s"),
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.4),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    for axis, sequence in zip(axes, ("Halton", "Sobol")):
        for dimension_index, dimension in enumerate(dimensions):
            color = colors[dimension_index % len(colors)]
            for implementation, (line_style, marker) in implementations.items():
                series = sorted(
                    (
                        row
                        for row in dimension_rows[dimension]
                        if row["backend"] == "gpu"
                        and row["sequence"] == sequence
                        and row["implementation"] == implementation
                    ),
                    key=lambda row: int(row["number_of_samples"]),
                )
                counts = np.asarray(
                    [int(row["number_of_samples"]) for row in series]
                )
                medians = np.asarray(
                    [float(row["median_seconds"]) for row in series]
                )
                lower = np.asarray(
                    [float(row["lower_quartile_seconds"]) for row in series]
                )
                upper = np.asarray(
                    [float(row["upper_quartile_seconds"]) for row in series]
                )
                log_counts = np.log10(counts)
                axis.plot(
                    log_counts,
                    np.log10(medians),
                    color=color,
                    linestyle=line_style,
                    marker=marker,
                    linewidth=2.0,
                    markersize=4.5,
                )
                axis.fill_between(
                    log_counts,
                    np.log10(lower),
                    np.log10(upper),
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                )

        axis.set_xlabel(r"$\log_{10}(n)$")
        axis.set_ylabel(r"$\log_{10}(t / \mathrm{s})$")
        axis.set_title(sequence)
        axis.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )
        axis.grid(False, which="minor")

    dimension_handles = [
        Line2D(
            [0],
            [0],
            color=colors[index % len(colors)],
            marker="o",
            linewidth=2.0,
            label=f"d={dimension}",
        )
        for index, dimension in enumerate(dimensions)
    ]
    implementation_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=line_style,
            marker=marker,
            linewidth=2.0,
            label=implementation,
        )
        for implementation, (line_style, marker) in implementations.items()
    ]
    axes[0].legend(
        handles=dimension_handles + implementation_handles,
        frameon=False,
        ncol=2,
    )

    scramble = next(iter(dimension_rows.values()))[0]["scramble"] == "True"
    scramble_label = "scrambled" if scramble else "unscrambled"
    figure.suptitle(
        f"QMC runtime across dimensions ({scramble_label}, GPU comparison)\n"
        "Dimension is color; implementation is line style; shaded region is IQR"
    )
    figure.savefig(
        output_dir / "qmc_benchmark_by_algorithm_gpu.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Run SQMC comparison benchmarks on a Colab GPU."
    )
    argument_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_NAME,
        help="Config filename under sqmc/sqmc/scripts/config after cloning.",
    )
    argument_parser.add_argument(
        "--config-json",
        help="Compact config supplied by the local orchestrator.",
    )
    argument_parser.add_argument(
        "--module",
        choices=("qmc", "hilbert_sort"),
        action="append",
        default=[],
        help=(
            "Restrict the benchmark to one or more modules. May be given "
            "repeatedly. When omitted, both qmc and hilbert_sort run."
        ),
    )
    argument_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print commands without requiring a GPU.",
    )
    return argument_parser


def _selected_modules(arguments: argparse.Namespace) -> tuple[str, ...]:
    """Return the modules to benchmark, honouring the ``--module`` filter."""
    return tuple(dict.fromkeys(arguments.module)) if arguments.module else ("qmc", "hilbert_sort")


MODULE_ARTIFACTS = {
    "hilbert_sort": (
        "hilbert_sort_benchmark_gpu.png",
        "hilbert_sort_benchmark.csv",
        "hilbert_sort_benchmark_by_algorithm_gpu.png",
    ),
    "qmc": (
        "qmc_benchmark_gpu.png",
        "qmc_benchmark.csv",
        "qmc_benchmark_by_algorithm_gpu.png",
    ),
}


# ---------------------------------------------------------------------------
# Hardware spec capture
#
# Recorded per run so the result is auditable and not confounded by a mismatch
# in machine (e.g. different RAM or CPU model) rather than by the algorithm.
# ---------------------------------------------------------------------------
def _memory_bytes() -> int | None:
    """Total physical RAM in bytes, via psutil or platform-specific fallback."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
        return int(out)
    except Exception:
        return None


def _gpu_devices() -> list:
    """List JAX GPU devices, tolerating builds without a GPU backend."""
    try:
        import jax

        return [str(d) for d in jax.devices("gpu")]
    except Exception:
        return []


def _cpu_model() -> str | None:
    """Return the host CPU model when the platform exposes it."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _gpu_specs() -> list[dict]:
    """Return actual NVIDIA model/memory/driver metadata when available."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    devices = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            devices.append(
                {"name": fields[0], "memory_mib": fields[1], "driver": fields[2]}
            )
    return devices


def capture_hardware() -> dict:
    """Return a dict describing the current host and its accelerators."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "node": platform.node(),
        "memory_bytes": _memory_bytes(),
        "gpu_devices": _gpu_devices(),
        "gpu_specs": _gpu_specs(),
    }


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.config_json:
        config = validate_config(json.loads(arguments.config_json))
    else:
        config_path = Path(arguments.config)
        if not config_path.is_absolute():
            local_candidate = Path(__file__).resolve().parent / "config" / config_path
            config_path = (
                local_candidate
                if local_candidate.exists()
                else Path.cwd() / arguments.config
            )
        config = load_config(config_path)

    output_dir = Path(config["remote_output_dir"])

    modules = _selected_modules(arguments)

    if arguments.dry_run:
        for module in modules:
            if module == "hilbert_sort":
                command = "sqmc.hilbert_sort.benchmark_hilbert_sort"
                section = config["hilbert_sort"]
            else:
                command = "sqmc.qmc.benchmark_qmc"
                section = config["qmc"]
            dimension_dir = f"by_dimension/{module}"
            for dimension_value in section["dimensions"]:
                print(
                    " ".join(
                        benchmark_command(
                            command,
                            section,
                            int(dimension_value),
                            output_dir / dimension_dir / f"d{dimension_value}",
                        )
                    )
                )
        return 0

    install_missing_packages()
    repo_root = clone_repository(
        config.get("repo_url", REPO_URL),
        config.get("repo_branch", "main"),
    )
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    # The compressed table is intentionally ignored by Git. Recreate it from
    # the checked-in Joe--Kuo source before importing sqmc.qmc.qmc.
    direction_numbers = repo_root / "sqmc/qmc/_sobol_direction_numbers.npz"
    if not direction_numbers.exists():
        run(
            [
                sys.executable,
                "sqmc/qmc/_generate_sobol_data.py",
                "--verify-scipy",
            ],
            cwd=repo_root,
        )

    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/sqmc-matplotlib")

    import jax
    import matplotlib
    import numba
    import numpy
    import scipy

    gpu_devices = list(jax.devices("gpu"))
    if not gpu_devices:
        raise RuntimeError(f"No JAX GPU is active; available devices: {jax.devices()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"JAX devices: {jax.devices()}")

    hilbert_rows = {}
    if "hilbert_sort" in modules:
        for dimension_value in config["hilbert_sort"]["dimensions"]:
            dimension = int(dimension_value)
            dimension_output = (
                output_dir / "by_dimension/hilbert_sort" / f"d{dimension}"
            )
            run(
                benchmark_command(
                    "sqmc.hilbert_sort.benchmark_hilbert_sort",
                    config["hilbert_sort"],
                    dimension,
                    dimension_output,
                ),
                cwd=repo_root,
            )
            hilbert_rows[dimension] = read_csv_rows(
                dimension_output / "hilbert_sort_benchmark.csv"
            )

    qmc_rows = {}
    if "qmc" in modules:
        for dimension_value in config["qmc"]["dimensions"]:
            dimension = int(dimension_value)
            dimension_output = output_dir / "by_dimension/qmc" / f"d{dimension}"
            run(
                benchmark_command(
                    "sqmc.qmc.benchmark_qmc",
                    config["qmc"],
                    dimension,
                    dimension_output,
                ),
                cwd=repo_root,
            )
            qmc_rows[dimension] = read_csv_rows(
                dimension_output / "qmc_benchmark.csv"
            )

    if "hilbert_sort" in modules:
        create_hilbert_summary(hilbert_rows, output_dir)
        create_hilbert_algorithm_summary(hilbert_rows, output_dir)
    if "qmc" in modules:
        create_qmc_summary(qmc_rows, output_dir)
        create_qmc_algorithm_summary(qmc_rows, output_dir)

    required_outputs = list(
        name for module in modules for name in MODULE_ARTIFACTS[module]
    )
    missing_outputs = [
        name for name in required_outputs if not (output_dir / name).is_file()
    ]
    if missing_outputs:
        raise RuntimeError(f"Benchmarks did not create: {missing_outputs}")

    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "numba": numba.__version__,
        "modules": list(modules),
        "config": config,
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # A separate run-config record audits the benchmark parameters together
    # with the captured hardware specs (RAM, CPU, GPU) for the run, so the
    # results are reproducible and not confounded by a machine mismatch.
    run_config = {
        "config": config,
        "modules": list(modules),
        "hardware": capture_hardware(),
    }
    run_config_path = output_dir / "run_config.json"
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    log(f"GPU benchmark(s) '{'/'.join(modules)}' completed; outputs are in {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"{type(error).__name__}: {error}")
        raise
