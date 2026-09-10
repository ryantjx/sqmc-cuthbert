"""Local Colab lifecycle, incremental downloads and recovery (stdlib only)."""
import argparse
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid

from comparison_protocol import (STAGES, REMOTE_REPO, ROOT_FILES, config_digest, digest,
                                 now, read_json, stage_command, unpack_verified,
                                 validate_config, validate_stage, write_json)

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[2]


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


def command(argv, timeout=600):
    print("+ " + shlex.join(map(str, argv)), flush=True)
    child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, start_new_session=True)
    expired = threading.Event()

    def kill():
        expired.set()
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(timeout, kill)
    timer.start()
    lines = []
    try:
        for line in child.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        code = child.wait()
        if expired.is_set():
            raise TimeoutError(f"Local command exceeded {timeout}s: {argv[0:2]}")
        if code:
            raise subprocess.CalledProcessError(code, argv)
        return "".join(lines)
    except BaseException:
        kill()
        child.wait()
        raise
    finally:
        timer.cancel()
        child.stdout.close()


def resolve(path=None, environ=None, repo=REPO, clock=None):
    environ = os.environ if environ is None else environ
    config = read_json(SCRIPTS / "config/comparison_config.json")
    if path:
        overrides = read_json(path)
        for key, value in overrides.items():
            if key in STAGES and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    for env, key in (("GPU_TYPE", "gpu"), ("COLAB_TIMEOUT", "colab_timeout"), ("SESSION", "session")):
        if env in environ:
            config[key] = float(environ[env]) if key == "colab_timeout" else environ[env]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    config.setdefault("source_commit", git("rev-parse", "HEAD"))
    config.setdefault("repo_branch", git("branch", "--show-current"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", config["repo_branch"]) or ".." in config["repo_branch"]:
        raise ValueError("A valid pushed repo_branch is required")
    timestamp = (clock or datetime.now(timezone.utc)).strftime("%d%m%Y_%H%M")
    # Runtime identifiers cannot be reused from a saved config.
    config["run_id"] = timestamp
    config["session_name"] = f"{config['session']}_{timestamp}_{uuid.uuid4().hex[:10]}"
    config["resolved_utc"] = now()
    config["source_transport"] = "colab_git_bundle"
    config.pop("source_bundle_sha256", None)
    validate_config(config)
    return config


def session_map(text):
    return dict(re.findall(r"^\[([^\]]+)\]\s+(\S+)\s+\|\s+Hardware:", text, re.MULTILINE))


class Launcher:
    def __init__(self, config, output, *, run=command, sleep=time.sleep):
        self.config, self.output, self.run, self.sleep = config, Path(output), run, sleep
        self.colab = shutil.which("colab") or "colab"
        self.session = config["session_name"]
        self.remote = Path("/content/sqmc-comparison-" + config["run_id"])
        self.attempted = False
        self.endpoint = None
        self.current = None
        self.log_offset = 0
        self.status = {"status": "running", "run_id": config["run_id"], "source_commit": config["source_commit"],
                       "config_sha256": config_digest(config), "started_utc": now(),
                       "session": self.session, "shutdown": "pending", "secondary_errors": [],
                       "stages": {s: {"execution": "pending", "download": "pending"} for s in STAGES}}

    def save(self):
        write_json(self.output / "status.json", self.status)

    def call(self, *args, timeout=None):
        return self.run([self.colab, *map(str, args)], timeout=timeout or self.config["transfer_timeout"])

    def sessions(self):
        return session_map(self.call("sessions"))

    def execute(self, code, timeout=None):
        with tempfile.TemporaryDirectory(prefix="sqmc-dispatch-") as temp:
            path = Path(temp) / "dispatch.py"
            path.write_text(code)
            duration = timeout or self.config["transfer_timeout"]
            return self.call("exec", "--session", self.session, "--timeout", duration, "--file", path, timeout=duration + 30)

    def prepare_source(self, directory):
        ref = "refs/heads/" + self.config["repo_branch"]
        sha = self.run(["git", "-C", str(REPO), "rev-parse", ref], timeout=60).strip()
        if sha != self.config["source_commit"]:
            raise RuntimeError("Local source branch does not match the pushed commit")
        # Upload only committed objects; never include the working tree/index.
        bundle = Path(directory) / "source.bundle"
        self.run(["git", "-C", str(REPO), "bundle", "create", str(bundle), ref], timeout=120)
        self.config["source_bundle_sha256"] = digest(bundle)
        self.status["config_sha256"] = config_digest(self.config)
        write_json(self.output / "comparison_config.json", self.config)
        self.save()
        return bundle

    def setup_from_bundle(self, bundle):
        self.call("upload", "--session", self.session, bundle, self.remote / "source.bundle")
        self.call("upload", "--session", self.session, SCRIPTS / "run_comparison_gpu.py", self.remote / "bootstrap.py")
        args = [str(self.remote / "bootstrap.py"), "--action", "setup", "--config", str(self.remote / "comparison_config.json")]
        self.execute(f"import subprocess, sys\nsubprocess.run([sys.executable, *{args!r}], check=True)\n", timeout=self.config["setup_timeout"])

    def worker_command(self, action, stage=None):
        args = ["/usr/bin/python3", str(REMOTE_REPO / "sqmc/comparison/scripts/run_comparison_gpu.py"),
                "--action", action, "--config", str(self.remote / "comparison_config.json")]
        if stage:
            args += ["--stage", stage]
        return args

    def start_stage(self, stage):
        args = self.worker_command("stage", stage)
        # Use the actual Colab kernel interpreter, never a presumed venv path.
        response = self.execute("import subprocess, sys\nfrom pathlib import Path\n"
                     f"args = {args!r}\nargs[0] = sys.executable\n"
                     f"with open({str(self.remote / (stage + '_worker.log'))!r}, 'w') as log:\n"
                     "    worker = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)\n"
                     f"Path({str(self.remote / (stage + '.pid'))!r}).write_text(str(worker.pid))\n"
                     "print('Started worker', worker.pid)\n")
        if not re.search(r"Started worker \d+", response):
            raise RuntimeError("Colab did not acknowledge starting the stage worker")

    def remote_json(self, name):
        with tempfile.TemporaryDirectory(prefix="sqmc-status-") as temp:
            target = Path(temp) / name
            self.call("download", "--session", self.session, self.remote / name, target)
            return read_json(target)

    def stream_log(self):
        with tempfile.TemporaryDirectory(prefix="sqmc-log-") as temp:
            target = Path(temp) / "remote_logs.txt"
            self.call("download", "--session", self.session, self.remote / "remote_logs.txt", target)
            data = target.read_bytes()
            print(data[self.log_offset:].decode(errors="replace"), end="", flush=True)
            self.log_offset = len(data)
            (self.output / "remote_logs.txt").write_bytes(data)

    def wait_stage(self, stage):
        # Worker enforces execution timeout; allow extra time for final archive.
        deadline = time.monotonic() + self.config["colab_timeout"] + self.config["transfer_timeout"]
        while time.monotonic() < deadline:
            state = self.remote_json("remote_status.json")["stages"][stage]
            self.stream_log()
            if state.get("archive_ready"):
                self.status["stages"][stage].update(state)
                self.save()
                if state["execution"] != "complete":
                    raise RuntimeError(f"{stage}: {state.get('error', state['execution'])}")
                return
            self.sleep(15)
        raise TimeoutError(f"{stage} worker did not finish and archive within its deadline")

    def download(self, kind, *, partial=False):
        if kind in STAGES:
            if self.status["stages"][kind]["download"] == "complete":
                raise RuntimeError("Refusing to overwrite a complete download")
            self.status["stages"][kind]["download"] = "downloading"
            self.save()
        with tempfile.TemporaryDirectory(prefix="sqmc-download-") as temp:
            temp = Path(temp)
            archive, checksum = temp / f"{kind}.tar.gz", temp / f"{kind}.tar.gz.sha256"
            self.call("download", "--session", self.session, self.remote / checksum.name, checksum)
            self.call("download", "--session", self.session, self.remote / archive.name, archive)
            unpack_verified(archive, checksum, temp / "verified", kind, self.config)
            verified = temp / "verified"
            if kind == "root":
                if read_json(verified / "comparison_config.json") != self.config:
                    raise ValueError("Remote effective configuration mismatch")
                if not partial:
                    hardware = read_json(verified / "run_config.json")
                    if hardware["source_commit"] != self.config["source_commit"] or hardware["config_sha256"] != config_digest(self.config):
                        raise ValueError("Remote hardware provenance mismatch")
                for name in ROOT_FILES | {"root_manifest.json"}:
                    if (verified / name).exists():
                        shutil.copyfile(verified / name, self.output / name)
            else:
                if not partial:
                    validate_stage(verified / kind, kind, self.config)
                    from validate_artifacts import validate_numerical
                    validate_numerical(verified / kind, kind, self.config)
                # Replace only this run's stage after all validation; recovery
                # cannot overwrite a successfully validated stage.
                if self.status["stages"][kind].get("download") == "complete":
                    raise RuntimeError("Refusing to overwrite a complete download")
                shutil.copytree(verified / kind, self.output / kind, dirs_exist_ok=True)
                shutil.copyfile(verified / f"{kind}_manifest.json", self.output / f"{kind}_manifest.json")
                self.status["stages"][kind].update(download="partial" if partial else "complete", downloaded_utc=now())
                self.save()
        print(f"Downloaded and verified {kind}{' (partial)' if partial else ''}.", flush=True)

    def secondary(self, label, action):
        try:
            action()
        except Exception as error:
            message = f"{label}: {type(error).__name__}: {error}"
            print(message, flush=True)
            self.status["secondary_errors"].append(message)
            self.save()

    def recover(self):
        if self.current:
            args = self.worker_command("snapshot", self.current)
            # Stop only the worker PID created in this owned session/run. Its
            # signal handler kills the benchmark child and archives partial data.
            self.secondary("Cancel/snapshot", lambda: self.execute(
                "import os, signal, subprocess, sys, time\nfrom pathlib import Path\n"
                f"p = Path({str(self.remote / (self.current + '.pid'))!r})\n"
                "if p.exists():\n"
                "    pid = int(p.read_text())\n"
                "    cmdline = Path(f'/proc/{pid}/cmdline')\n"
                f"    owned = cmdline.exists() and {str(self.remote / 'comparison_config.json')!r}.encode() in cmdline.read_bytes().split(b'\\x00')\n"
                "    if owned:\n"
                "        try: os.kill(pid, signal.SIGTERM)\n"
                "        except ProcessLookupError: pass\n"
                "    for _ in range(30):\n"
                "        try: os.kill(pid, 0)\n"
                "        except ProcessLookupError: break\n"
                "        time.sleep(1)\n"
                f"args = {args!r}\nargs[0] = sys.executable\nsubprocess.run(args, check=True)\n"))
            if self.status["stages"][self.current]["download"] != "complete":
                self.secondary("Partial stage download", lambda: self.download(self.current, partial=True))
        self.secondary("Root metadata download", lambda: self.download("root", partial=True))
        self.secondary("Remote log download", self.stream_log)

    def shutdown(self):
        sessions = self.sessions()
        endpoint = sessions.get(self.session)
        if endpoint:
            if self.endpoint and endpoint != self.endpoint:
                raise RuntimeError("Session endpoint changed; refusing to stop an unowned session")
            self.endpoint = endpoint
            self.call("stop", "--session", self.session)
        remaining = self.sessions()
        if self.session in remaining or (self.endpoint and self.endpoint in remaining.values()):
            raise RuntimeError("Owned session remains active after stop")
        self.status["shutdown"] = "verified_stopped"
        self.status["shutdown_utc"] = now()
        self.save()

    def launch(self):
        self.save()
        code = 0
        source_temp = tempfile.TemporaryDirectory(prefix="sqmc-source-")
        try:
            available = self.sessions()
            if self.session in available:
                raise RuntimeError("Session name already exists; refusing to reuse it")
            remote_ref = self.run(["git", "ls-remote", self.config["repo_url"], "refs/heads/" + self.config["repo_branch"]], timeout=60)
            if not remote_ref.split() or remote_ref.split()[0] != self.config["source_commit"]:
                raise RuntimeError("Push the exact source commit to repo_branch before provisioning")
            bundle = self.prepare_source(source_temp.name)
            self.attempted = True
            self.call("run", "--keep", "--gpu", self.config["gpu"], "--session", self.session,
                      "--timeout", self.config["setup_timeout"], SCRIPTS / "run_comparison_gpu.py",
                      "--action", "provision", "--config-json", json.dumps(self.config),
                      timeout=self.config["setup_timeout"] + 120)
            self.endpoint = self.sessions().get(self.session)
            if not self.endpoint:
                raise RuntimeError("Provisioned session missing from server session list")
            self.status["endpoint"] = self.endpoint
            self.setup_from_bundle(bundle)
            self.download("root")
            for stage in STAGES:
                self.current = stage
                self.status["stages"][stage].update(execution="running", started_utc=now())
                self.save()
                self.start_stage(stage)
                self.wait_stage(stage)
                self.download(stage)
                self.download("root")
            self.current = None
            self.status["status"] = "complete"
        except BaseException as error:
            code = 130 if isinstance(error, KeyboardInterrupt) else getattr(error, "returncode", 1)
            code = code if isinstance(code, int) and 1 <= code <= 255 else 1
            self.status.update(status="failed", error=f"{type(error).__name__}: {error}", exit_code=code)
            if self.current:
                entry = self.status["stages"][self.current]
                if entry["execution"] == "running":
                    entry["execution"] = "failed"
                entry["error"] = self.status["error"]
                if entry["download"] == "downloading":
                    entry["download"] = "failed"
            traceback.print_exc()
            self.save()
            # One Ctrl-C initiates recovery. Ignore further interrupts only
            # during bounded salvage/cleanup so GPU release is still attempted.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if self.attempted:
                self.recover()
        finally:
            if self.attempted:
                self.secondary("Shutdown", self.shutdown)
                if self.status["shutdown"] != "verified_stopped":
                    self.status["shutdown"] = "failed"
                    if code == 0:
                        code = 1
                        self.status.update(status="failed", error="Shutdown verification failed")
            else:
                self.status["shutdown"] = "not_provisioned"
            self.status.update(finished_utc=now(), exit_code=code)
            self.save()
            source_temp.cleanup()
        return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides merged into the full profile")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = resolve(args.config)
    output = REPO / "sqmc/comparison/outputs" / config["run_id"]
    if output.exists():
        raise FileExistsError(f"UTC timestamp collision: {output}; wait for the next minute")
    if args.dry_run:
        print(json.dumps(config, indent=2))
        print(f"Output: {output}")
        print("git bundle create <temporary source.bundle> refs/heads/" + config["repo_branch"])
        print("colab run --keep --gpu", config["gpu"], "--session", config["session_name"], "--timeout", config["setup_timeout"], "run_comparison_gpu.py --action provision --config-json <effective JSON plus bundle SHA-256>")
        print("colab upload <source.bundle>; colab upload <bootstrap.py>; colab exec <setup pinned checkout>")
        for stage in STAGES:
            print(shlex.join(stage_command(config, stage, Path("/content/sqmc-comparison-" + config["run_id"]) / stage)))
            print(f"colab exec --session {config['session_name']} --file <start {stage} worker>; poll; download and verify {stage}.tar.gz")
        print("colab stop --session", config["session_name"], "; colab sessions (verify shutdown)")
        return 0
    if not shutil.which("colab"):
        raise RuntimeError("colab must be available on PATH; activate the local virtual environment")
    # Numerical archive validation is required locally, before any hardware cost.
    import numpy  # noqa: F401
    output.mkdir(parents=True, exist_ok=False)
    with (output / "logs.txt").open("w", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            write_json(output / "comparison_config.json", config)
            print(f"Comparison output: {output}", flush=True)
            return Launcher(config, output).launch()


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    sys.exit(main())
