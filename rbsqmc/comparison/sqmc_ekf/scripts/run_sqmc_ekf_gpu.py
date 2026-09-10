"""Colab bootstrap and run worker for the SQMC–EKF comparison.

This file is submitted by ``colab run`` (which supplies no ``__file__``) and
runs inside the Colab VM. It:
  - ``provision``: waits for the verified source bundle.
  - ``setup``: pins the exact source commit from the uploaded git bundle,
    installs missing deps, asserts a GPU, and writes ``run_config.json``
    (compute setup) into the VM run directory.
  - ``run``: executes the comparison, storing config/settings/results in the
    VM run directory, then archives ``run`` and ``root`` bundles.

No third-party imports before setup.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import traceback

REPO = Path("/content/rbsqmc")


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def writelines(self, lines):
        for line in lines:
            self.write(line)


def run(command, *, timeout=None, process_log=None):
    print("Running: " + repr(command), flush=True)
    child = subprocess.Popen(command, cwd=REPO if REPO.exists() else None,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, start_new_session=True)
    expired = threading.Event()

    def kill():
        expired.set()
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(timeout, kill) if timeout else None
    if timer:
        timer.start()
    try:
        for line in child.stdout:
            print(line, end="", flush=True)
            if process_log:
                process_log.write(line)
                process_log.flush()
        code = child.wait()
        if expired.is_set():
            raise TimeoutError(f"Execution exceeded {timeout} seconds")
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        kill()
        child.wait()
        raise
    finally:
        if timer:
            timer.cancel()
        child.stdout.close()


def setup(config, root):
    # Fresh session, fresh directory; never delete an existing checkout.
    if REPO.exists():
        raise FileExistsError(REPO)
    run(["git", "init", str(REPO)])
    run(["git", "remote", "add", "origin", config["repo_url"]])
    bundle = root / "source.bundle"
    with bundle.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != config["source_bundle_sha256"]:
            raise RuntimeError("Uploaded source bundle checksum mismatch")
    run(["git", "bundle", "verify", str(bundle)])
    run(["git", "fetch", str(bundle), "refs/heads/" + config["repo_branch"]])
    run(["git", "checkout", "--detach", "FETCH_HEAD"])
    # Only materialize the directories the comparison needs. The monorepo also
    # tracks archive/, papers/, dissertation/ and generated images/data that
    # are irrelevant to the run; a sparse checkout keeps the VM working tree
    # small and avoids shipping those bytes over the bundle transport. The
    # exact source_commit is still pinned (HEAD is unchanged by sparse
    # checkout), so the run remains reproducible.
    run(["git", "sparse-checkout", "init", "--cone"])
    run(["git", "sparse-checkout", "set", "rbsqmc", "sqmc"])
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if actual != config["source_commit"]:
        raise RuntimeError("Remote checkout does not match source commit")
    sys.path.insert(0, str(REPO / "rbsqmc/comparison/sqmc_ekf/scripts"))
    from sqmc_ekf_protocol import write_json

    installed = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    if not all(importlib.util.find_spec(module) for module in ("jax", "jaxlib")):
        raise RuntimeError("A preinstalled GPU-enabled JAX runtime is required")
    protected = {name: version for name, version in installed.items()
                 if name.lower().startswith(("jax", "nvidia-"))}
    constraints = root / "jax_constraints.txt"
    constraints.write_text("\n".join(f"{name}=={version}" for name, version in protected.items()) + "\n")
    missing = [name for name in ("numpy", "scipy", "matplotlib", "cuthbert", "cuthbertlib", "ghq", "optax", "pandas", "pyarrow")
               if importlib.util.find_spec(name) is None]
    if missing:
        run([sys.executable, "-m", "pip", "install", "-c", str(constraints), *missing])
    if any(importlib.metadata.version(name) != version for name, version in protected.items()):
        raise RuntimeError("Preinstalled JAX/CUDA packages changed")

    # The Sobol direction numbers are generated data (gitignored); build them
    # from the tracked Joe--Kuo table so the SQMC module can import.
    run([sys.executable, "sqmc/qmc/_generate_sobol_data.py", "--verify-scipy"])

    import jax
    import jaxlib
    jax.config.update("jax_enable_x64", True)
    devices = {backend: [{"kind": d.device_kind, "id": d.id, "platform": d.platform} for d in jax.devices(backend)] for backend in ("cpu", "gpu")}
    if not all(devices.values()):
        raise RuntimeError("CPU and GPU are both required")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()
    if config["gpu"].lower() not in gpu.lower():
        raise RuntimeError(f"Requested {config['gpu']}, provisioned {gpu}")
    cpuinfo = Path("/proc/cpuinfo").read_text()
    info = {"source_commit": actual, "run_id": config["run_id"], "python": sys.version,
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "cpu_models": sorted({line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")}),
            "host_memory": Path("/proc/meminfo").read_text(), "gpu_name_memory_MiB_driver": gpu,
            "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
            "devices": devices, "jax": jax.__version__, "jaxlib": jaxlib.__version__,
            "versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
            "precision": {"jax_enable_x64": jax.config.jax_enable_x64, "jax_default_matmul_precision": jax.config.jax_default_matmul_precision},
            "prng": {"jax_default_prng_impl": str(jax.config.jax_default_prng_impl)},
            "environment": {k: os.environ.get(k) for k in ("JAX_ENABLE_X64", "XLA_PYTHON_CLIENT_PREALLOCATE", "MPLBACKEND", "MPLCONFIGDIR")}}
    from sqmc_ekf_protocol import config_digest
    info["config_sha256"] = config_digest(config)
    write_json(root / "run_config.json", info)
    print("Hardware: " + gpu, flush=True)


def run_comparison(config, root, methods):
    from sqmc_ekf_protocol import now, read_json, validate_run, write_json, make_archive
    status_path = root / "remote_status.json"
    status = read_json(status_path) if status_path.exists() else {"setup": "complete", "run": {"execution": "pending"}}
    run_status = status["run"]
    if run_status["execution"] != "pending":
        raise RuntimeError("Run already attempted; use a fresh run")
    run_status.update(execution="running", started_utc=now())
    write_json(status_path, status)
    try:
        with (root / "run_process_logs.txt").open("w") as process_log:
            run([
                sys.executable, "-u", "-m", "rbsqmc.comparison.sqmc_ekf.run",
                "--config", str(root / "comparison_config.json"),
                "--data", str(REPO / "rbsqmc" / "data" / "results.csv"),
                "--output-dir", str(root),
                "--methods", "both" if set(methods) == {"ekf", "sqmc"} else methods[0],
            ], timeout=config["colab_timeout"], process_log=process_log)
        validate_run(root, config, methods=methods)
        run_status["execution"] = "complete"
    except BaseException as error:
        # Distinguish "training finished but the validation gate failed" from
        # "training crashed": in the former case the results/images exist on
        # the VM and are archived below, so the launcher can salvage them.
        training_completed = (root / "results" / "run_metadata.json").exists()
        run_status.update(
            execution="failed",
            error=f"{type(error).__name__}: {error}",
            failure_kind="validation" if training_completed else "training",
            results_exported=training_completed,
        )
        traceback.print_exc()
        raise
    finally:
        run_status["finished_utc"] = now()
        write_json(status_path, status)
        make_archive(root, "run", config)
        run_status["archive_ready"] = True
        write_json(status_path, status)
        make_archive(root, "root", config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["provision", "setup", "run"], required=True)
    parser.add_argument("--config-json")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--methods", default="ekf,sqmc",
                        help="Comma-separated methods to run (default: ekf,sqmc).")
    args = parser.parse_args()
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    if not methods or not set(methods) <= {"ekf", "sqmc"}:
        raise ValueError("--methods must be a non-empty subset of {ekf, sqmc}")
    config = json.loads(args.config_json) if args.config_json else json.loads(args.config.read_text())
    root = Path("/content/sqmc-ekf-" + config["run_id"])
    root.mkdir(exist_ok=args.action != "provision")
    # RBSQMC_PLATFORM must be set before the comparison imports the model
    # modules, which otherwise pin jax_platforms to cpu (audit finding P1).
    # PYTHONPATH makes the rbsqmc namespace package importable to every
    # subprocess regardless of import site: the artifact validator imports
    # rbsqmc.* lazily inside function bodies, which fails when only the
    # scripts directory is on sys.path.
    os.environ.update(JAX_ENABLE_X64="true", XLA_PYTHON_CLIENT_PREALLOCATE="false", MPLBACKEND="Agg", MPLCONFIGDIR="/tmp/sqmc-matplotlib", PYTHONUNBUFFERED="1", RBSQMC_PLATFORM="cuda", PYTHONPATH=str(REPO))
    (root / "comparison_config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (root / "remote_logs.txt").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            if args.action == "provision":
                print("Provisioned; waiting for the verified source bundle.", flush=True)
                return
            if args.action == "setup":
                try:
                    setup(config, root)
                except BaseException:
                    traceback.print_exc()
                    raise
            sys.path.insert(0, str(REPO / "rbsqmc/comparison/sqmc_ekf/scripts"))
            from sqmc_ekf_protocol import validate_config, make_archive, write_json
            validate_config(config)
            if args.action == "setup":
                # The orchestrator polls remote_status.json and downloads the
                # root bundle right after setup, so write the initial status
                # and archive the metadata (config, run_config, status, logs).
                write_json(root / "remote_status.json",
                           {"setup": "complete", "run": {"execution": "pending"}})
                make_archive(root, "root", config)
            if args.action == "run":
                run_comparison(config, root, methods)


if __name__ == "__main__":
    main()
