"""Shared logging, timing, and provenance for CPU/GPU comparisons."""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sqmc-comparison-mpl"))
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[2]


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def add_common_arguments(parser, *, counts=True):
    parser.add_argument("--platforms", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"])
    parser.add_argument("--dimensions", nargs="+", type=positive, default=[2, 5, 10])
    if counts:
        parser.add_argument("--n-values", nargs="+", type=positive, default=[128, 512, 2048, 8192, 32768])
    parser.add_argument("--repeats", type=positive, default=7)
    parser.add_argument("--warmups", type=nonnegative, default=2)
    parser.add_argument("--seed", type=nonnegative, default=42)
    parser.add_argument("--output-dir", type=Path)


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


@contextmanager
def run_directory(name, args):
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output_dir or Path(__file__).parent / "outputs" / f"{name}_{timestamp}_{uuid.uuid4().hex[:8]}"
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use a fresh output directory; {output} is not empty.")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "logs.txt").open("a", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            status = {"benchmark": name, "status": "running", "started_utc": timestamp}
            write_json(output / "status.json", status)
            write_json(output / "config.json", vars(args))
            print(f"{name}: {output}", flush=True)
            try:
                yield output
            except BaseException as error:
                status.update(status="failed", error=f"{type(error).__name__}: {error}")
                traceback.print_exc()
                raise
            else:
                status["status"] = "complete"
                print("Benchmark completed.", flush=True)
            finally:
                status["total_wall_seconds"] = time.perf_counter() - started
                status["finished_utc"] = datetime.now(timezone.utc).isoformat()
                write_json(output / "status.json", status)


def devices(platforms):
    if len(set(platforms)) != len(platforms):
        raise ValueError("Platforms must be unique.")
    result = {}
    for name in platforms:
        try:
            available = jax.devices(name)
        except RuntimeError as error:
            raise RuntimeError(f"Requested {name} is unavailable. Use --platforms cpu for a CPU-only run.") from error
        if not available:
            raise RuntimeError(f"Requested {name} is unavailable.")
        result[name] = available[0]
    return result


def validate_grid(dimensions, counts, *, max_dimension, power_two=False):
    if len(set(dimensions)) != len(dimensions) or len(set(counts)) != len(counts):
        raise ValueError("Dimensions and sample counts must be unique.")
    if any(not 1 <= d <= max_dimension for d in dimensions):
        raise ValueError(f"Dimensions must be in [1, {max_dimension}].")
    if any(n < 1 or n > 2**30 for n in counts):
        raise ValueError("Counts must be in [1, 2**30].")
    if power_two and any(n & (n - 1) for n in counts):
        raise ValueError("Balanced Sobol/SQMC counts must be powers of two.")


def provenance(output, selected_devices):
    sources = ["sqmc/qmc/qmc.py", "sqmc/qmc/_sobol_direction_numbers.npz",
               "sqmc/hilbert_sort/hilbert_sort.py", "sqmc/sqmc/sqmc.py"]
    sources += [str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob("*.py")]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    versions = {}
    for package in ("jax", "jaxlib", "numpy", "scipy", "cuthbert", "cuthbertlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(output / "metadata.json", {
        "schema_version": 1, "python": sys.version, "system": platform.platform(),
        "cpu_count": os.cpu_count(), "versions": versions,
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "source_sha256": hashes, "x64": jax.config.jax_enable_x64,
        "devices": {name: {"kind": dev.device_kind, "id": dev.id, "platform": dev.platform}
                    for name, dev in selected_devices.items()},
        "timing": "device-resident, synchronized outputs; raw steady-state samples; first call includes tracing/compilation/execution",
    })


def prepare_inputs(arguments, device):
    return [jax.block_until_ready(jax.device_put(args, device)) for args in arguments]


def timed(function, inputs, warmups, repeats):
    """Inputs must already be resident and synchronized on the desired device."""
    start = time.perf_counter()
    result = jax.block_until_ready(function(*inputs[0]))
    first = time.perf_counter() - start
    for i in range(warmups):
        jax.block_until_ready(function(*inputs[i % len(inputs)]))
    samples = []
    for i in range(repeats):
        start = time.perf_counter()
        result = jax.block_until_ready(function(*inputs[i % len(inputs)]))
        samples.append(time.perf_counter() - start)
    return {"first_call_seconds": first, "samples_seconds": samples,
            "median_seconds": float(np.median(samples)),
            "q25_seconds": float(np.quantile(samples, .25)),
            "q75_seconds": float(np.quantile(samples, .75))}, result


def assert_device(result, device):
    for leaf in jax.tree.leaves(result):
        if leaf.devices() != {device}:
            raise RuntimeError(f"Output was not computed on requested device {device}.")


def engine(sequence, dimension, key, scramble=True):
    from sqmc.qmc.qmc import Halton, Sobol
    cls = {"sobol": Sobol, "halton": Halton}[sequence]
    return cls(d=dimension, key=key, scramble=scramble, start_index=0, dtype=jnp.float64)


def plot_lines(rows, output, filename, *, group_fields, x, y, ylabel, xlabel="Samples N", target=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    groups = sorted({tuple(row[field] for field in group_fields) for row in rows})
    for group in groups:
        values = sorted((row for row in rows if tuple(row[f] for f in group_fields) == group), key=lambda row: row[x])
        ax.plot([v[x] for v in values], [v[y] for v in values], marker="o", label=" / ".join(map(str, group)))
    if target is not None:
        ax.axhline(target, color="black", linestyle="--", label=f"Target {target:g}")
    ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel=ylabel)
    ax.legend(fontsize="small")
    fig.savefig(output / filename, dpi=160)
    plt.close(fig)
