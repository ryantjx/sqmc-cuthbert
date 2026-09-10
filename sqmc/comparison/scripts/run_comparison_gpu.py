"""Colab bootstrap and stage worker. No third-party imports before setup.

This file can be submitted by `colab run` (which supplies no __file__).
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
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if actual != config["source_commit"]:
        raise RuntimeError("Remote checkout does not match source commit")
    sys.path.insert(0, str(REPO / "sqmc/comparison/scripts"))
    from comparison_protocol import write_json
    # Constrain the entire preinstalled JAX/CUDA stack; pip may add missing
    # dependencies but cannot replace the working accelerator installation.
    installed = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    if not all(importlib.util.find_spec(module) for module in ("jax", "jaxlib")):
        raise RuntimeError("A preinstalled GPU-enabled JAX runtime is required")
    protected = {name: version for name, version in installed.items()
                 if name.lower().startswith(("jax", "nvidia-"))}
    constraints = root / "jax_constraints.txt"
    constraints.write_text("\n".join(f"{name}=={version}" for name, version in protected.items()) + "\n")
    missing = [name for name in ("numpy", "scipy", "matplotlib", "numba", "cuthbert", "cuthbertlib") if importlib.util.find_spec(name) is None]
    if missing:
        run([sys.executable, "-m", "pip", "install", "-c", str(constraints), *missing])
    if any(importlib.metadata.version(name) != version for name, version in protected.items()):
        raise RuntimeError("Preinstalled JAX/CUDA packages changed")
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
    from comparison_protocol import config_digest
    info["config_sha256"] = config_digest(config)
    write_json(root / "run_config.json", info)
    print("Hardware: " + gpu, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["provision", "setup", "stage", "snapshot"], required=True)
    parser.add_argument("--config-json")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage", choices=["qmc", "hilbert_sort", "sqmc"])
    args = parser.parse_args()
    config = json.loads(args.config_json) if args.config_json else json.loads(args.config.read_text())
    root = Path("/content/sqmc-comparison-" + config["run_id"])
    root.mkdir(exist_ok=args.action != "provision")
    os.environ.update(JAX_ENABLE_X64="true", XLA_PYTHON_CLIENT_PREALLOCATE="false", MPLBACKEND="Agg", MPLCONFIGDIR="/tmp/sqmc-matplotlib", PYTHONUNBUFFERED="1")
    (root / "comparison_config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Keep redirected worker output separate from the canonical remote log.
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
            sys.path.insert(0, str(REPO / "sqmc/comparison/scripts"))
            from comparison_protocol import STAGES, make_archive, now, read_json, stage_command, validate_config, validate_stage, write_json
            validate_config(config)
            status_path = root / "remote_status.json"
            status = read_json(status_path) if status_path.exists() else {"setup": "complete", "stages": {s: {"execution": "pending"} for s in STAGES}}
            if args.action == "stage":
                stage = args.stage
                if status["stages"][stage]["execution"] != "pending":
                    raise RuntimeError("Stage already attempted; use a fresh run")
                stage_status = status["stages"][stage]
                stage_status.update(execution="running", started_utc=now())
                write_json(status_path, status)
                try:
                    with (root / f"{stage}_process_logs.txt").open("w") as process_log:
                        run(stage_command(config, stage, root / stage, sys.executable),
                            timeout=config["colab_timeout"], process_log=process_log)
                    validate_stage(root / stage, stage, config)
                    stage_status["execution"] = "complete"
                except BaseException as error:
                    stage_status.update(execution="failed", error=f"{type(error).__name__}: {error}")
                    traceback.print_exc()
                    raise
                finally:
                    (root / stage).mkdir(exist_ok=True)
                    # Native stderr and crashes are retained even before Python logging starts.
                    process_path = root / f"{stage}_process_logs.txt"
                    if process_path.exists():
                        (root / stage / "process_logs.txt").write_bytes(process_path.read_bytes())
                        if not (root / stage / "logs.txt").exists():
                            (root / stage / "logs.txt").write_bytes(process_path.read_bytes())
                    stage_status["finished_utc"] = now()
                    write_json(status_path, status)
                    make_archive(root, stage, config)
                    stage_status["archive_ready"] = True
                    write_json(status_path, status)
                    make_archive(root, "root", config)
            else:
                write_json(status_path, status)
                if args.stage and (root / args.stage).exists():
                    make_archive(root, args.stage, config)
                make_archive(root, "root", config)


def interrupted(signum, frame):
    raise InterruptedError(f"Worker received signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    main()
