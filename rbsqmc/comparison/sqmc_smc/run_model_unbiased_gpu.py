"""Bootstrap a Colab GPU and run the rbsqmc model_unbiased optimization.

This script runs inside the Colab VM (uploaded by ``colab run``). It:
  1. Clones/updates the repository (or uses an uploaded copy with --no-clone).
  2. Installs any missing runtime dependencies (skip-if-present).
  3. Loads the config JSON (model_unbiased_gpu_config.json).
  4. Asserts a GPU is active.
  5. Runs the optimization-only phase via ``train_model_gpu.main(['optimize'])``
     from ``rbsqmc.src.model.rbsmc.train_model_gpu``.

Only the backward-gradient optimization runs on the GPU; filtering, plotting,
and prediction run locally (driven by the orchestrator).

Usage (local dry-run):
    python rbsqmc/comparison/sqmc_smc/run_model_unbiased_gpu.py \
        --config rbsqmc/comparison/sqmc_smc/config/model_unbiased_gpu_config.json --dry-run

Usage (Colab, via the orchestrator):
    bash rbsqmc/comparison/sqmc_smc/run_model_unbiased_colab.sh
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_DIR = Path("/content/rbsqmc")
REPO_URL = "https://github.com/ryantjx/rbsqmc.git"

# Packages the pipeline needs at runtime, mapped to their pip package names
# (module -> package). We only ever install a package if its module is NOT
# importable in the active environment, and we never install ``jax``/``jaxlib``:
# Colab ships a working CUDA JAX and re-pinning it (e.g. from requirements.txt)
# can leave a broken mix of jax vs jax._src that breaks imports like
# ``from jax.scipy.linalg import ...``.
REQUIRED_PACKAGES = {
    "cuthbert": "cuthbert",
    "cuthbertlib": "cuthbertlib",
    "optax": "optax",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
}

# Modules Colab is expected to already provide; we probe these but never try to
# reinstall them (installing jax/jaxlib can corrupt Colab's CUDA setup).
PROTECTED_MODULES = ("jax",)


def log(message: str, *, stream: str = "OUT") -> None:
    outer = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inner = datetime.now().strftime("%H:%M:%S")
    print(f"[{outer}] {stream}: [{inner}] {message}", flush=True)


def run(command, *, cwd=None, forward_raw=False) -> None:
    log("Running: " + " ".join(map(str, command)))
    process = subprocess.Popen(
        command, cwd=cwd, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in process.stdout:
        if forward_raw:
            print(line, end="", flush=True)
        else:
            log(line.rstrip("\n"))
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}:\n"
            + " ".join(map(str, command))
        )


def is_colab() -> bool:
    return Path("/content").exists()


def resolve_python(repo_root: Path) -> str:
    """Return the Python interpreter to use for the pipeline.

    Prefer the repository's virtualenv when present and local; on Colab fall
    back to the active interpreter (``sys.executable``).
    """
    if not is_colab():
        candidates = [
            repo_root / ".venv" / "bin" / "python",
            repo_root.parent / ".venv" / "bin" / "python",
            Path(sys.executable),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return sys.executable
    return sys.executable


def _missing_packages() -> list[str]:
    """Return the pip package names we need to install.

    A package is installed only if its module is not importable AND it is not a
    protected module (jax). Colab already provides a working JAX; we never
    reinstall it.
    """
    missing = []
    for module, package in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
        except Exception:
            if module in PROTECTED_MODULES:
                log(f"module {module} not importable but protected; trusting Colab's install")
            else:
                missing.append(package)
    return missing


# ---------------------------------------------------------------------------
# Bootstrap: clone repo + install deps
# ---------------------------------------------------------------------------
def bootstrap(no_clone: bool = False) -> Path:
    """Clone/update the repository and install missing runtime dependencies.

    Uses Colab's pre-installed Python environment (which already ships JAX,
    NumPy, SciPy, Matplotlib, etc.). Only missing packages are installed via
    pip, by name, and never ``jax``/``jaxlib`` — so we do not disturb Colab's
    CUDA/JAX setup.

    Returns the repository root.
    """
    if no_clone:
        if not (REPO_DIR / "rbsqmc").is_dir():
            raise FileNotFoundError(
                f"--no-clone: expected repo at {REPO_DIR} but rbsqmc/ is missing"
            )
        log(f"Using uploaded repo at {REPO_DIR} (no clone)")
    elif (REPO_DIR / ".git").is_dir():
        log(f"Using existing checkout at {REPO_DIR}")
        run(["git", "-c", "http.version=HTTP/1.1", "pull", "--ff-only"], cwd=REPO_DIR)
    else:
        last_error = None
        for attempt in range(1, 4):
            if REPO_DIR.exists():
                shutil.rmtree(REPO_DIR)
            try:
                run([
                    "git", "-c", "http.version=HTTP/1.1", "clone",
                    "--depth", "1", "--single-branch", "--branch", "main",
                    REPO_URL, str(REPO_DIR),
                ])
                last_error = None
                break
            except RuntimeError as error:
                last_error = error
                log(f"clone attempt {attempt}/3 failed")
        if last_error is not None:
            raise last_error

    # Install only the missing runtime packages into the active environment, by
    # name, so pip does not try to reconcile (downgrade) Colab's working jax.
    missing = _missing_packages()
    if missing:
        log(f"Installing missing packages (modules not importable): {missing}")
        run([
            sys.executable, "-m", "pip", "install", *missing,
        ])
    else:
        log("All required modules already importable; skipping pip install")
    return REPO_DIR


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(repo_root: Path, override: str | None = None) -> dict:
    """Load and validate the model_unbiased configuration.

    Reuses ``train_model.load_config`` (which merges the config JSON over the
    pipeline's own defaults) and then applies the GPU/orchestrator-only
    validation (positive particle/epoch counts and learning rate).
    """
    if override and not Path(override).is_absolute() and not Path(override).parent.parts:
        override = str(repo_root / "rbsqmc/comparison/sqmc_smc/config" / override)
    # Lazy import: the local dry-run may run under a system python without jax,
    # so we only import the pipeline module once we actually need its config.
    # The script may be executed by path (sys.path[0] = scripts/), so ensure the
    # repo root is importable.
    sys.path.insert(0, str(repo_root))
    from rbsqmc.src.model.rbsmc.train_model_gpu import load_config as _load_train_config
    config = _load_train_config(override)

    positive_integer = (
        "n_particles", "max_goals", "n_epochs", "colab_timeout",
    )
    for key in positive_integer:
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if float(config.get("learning_rate", 0)) <= 0:
        raise ValueError("learning_rate must be positive")
    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Colab GPU bootstrap for the rbsqmc model_unbiased optimization"
    )
    p.add_argument("--config", help="Optional config JSON override")
    p.add_argument("--output-dir", help="Override the config output_dir (repo-relative remote path)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-clone", action="store_true",
                   help="Skip git clone; use the repo already uploaded to /content/rbsqmc")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)

    if args.dry_run:
        config = load_config(Path.cwd(), args.config)
        print(f"model_unbiased (rbsqmc): n_particles={config['n_particles']}, "
              f"n_epochs={config['n_epochs']}, lr={config['learning_rate']}, "
              f"n_reps={config['n_reps']}")
        return 0

    repo_root = bootstrap(no_clone=args.no_clone) if is_colab() else Path(__file__).resolve().parents[3]
    config = load_config(repo_root, args.config)
    py = resolve_python(repo_root)
    log(f"Using Python interpreter: {py}")

    # On Colab the repo is a fresh clone, so the local results.parquet cache is
    # absent. Force data download for the optimize phase.
    if is_colab():
        config["download"] = True

    # Verify GPU using the selected Python.
    run([
        py, "-c",
        "import jax; print('JAX devices:', jax.devices()); "
        "assert jax.default_backend() == 'gpu', 'Colab GPU is not active'",
    ], cwd=repo_root)

    # The optimization child process reloads its config from RBSQMC_CONFIG. If
    # --output-dir was given (per-run timestamped dir), write a resolved config
    # on the VM (so the child writes outputs there) and point RBSQMC_CONFIG at it.
    if args.output_dir:
        config["output_dir"] = args.output_dir
        resolved_path = repo_root / args.output_dir / "resolved_run_config.json"
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(json.dumps(config, indent=2))
        log(f"Wrote resolved config to {resolved_path}")
        config_path = resolved_path
    else:
        config_path = Path(args.config) if args.config else repo_root / "rbsqmc/comparison/sqmc_smc/config/model_unbiased_gpu_config.json"
        if not config_path.is_absolute():
            config_path = repo_root / "rbsqmc/comparison/sqmc_smc/config" / config_path

    log(f"output_dir={config['output_dir']}")
    log(f"Writing remote outputs to {repo_root / config['output_dir']}")

    os.environ["RBSQMC_CONFIG"] = str(config_path)
    os.environ.setdefault("RBSQMC_PLATFORM", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rbsqmc_matplotlib")
    os.environ["MPLBACKEND"] = "Agg"

    # Optimization-only phase: the train_model_gpu optimize subcommand reads
    # RBSQMC_CONFIG and writes params_unbiased.json + optimization_summary.json
    # + curves into output_dir.
    run([
        py, "-c",
        "import sys; sys.path.insert(0, '.'); "
        "from rbsqmc.src.model.rbsmc.train_model_gpu import main; "
        "sys.exit(main(['optimize']))",
    ], cwd=repo_root, forward_raw=True)

    log("GPU model_unbiased optimization completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"{type(error).__name__}: {error}", stream="ERR")
        raise
