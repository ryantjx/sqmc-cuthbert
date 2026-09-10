"""Local Colab lifecycle for the SQMC–EKF comparison (stdlib only).

Mirrors ``sqmc/comparison/scripts/run_comparison_local.py`` but for a single
run. It resolves the config, provisions a Colab session, uploads a git bundle
+ config, pins the exact source commit on the VM, runs the comparison (storing
config/settings/results on the VM), downloads + verifies the archives, and
stops the session.
"""

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

from sqmc_ekf_protocol import (REMOTE_REPO, ROOT_FILES, config_digest, digest,
                               now, read_json, unpack_verified, validate_config, validate_run,
                               write_json)

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[2]
REPO_ROOT = REPO.parent  # repository root: contains the ``sqmc`` package and the frozen data


class RemoteRunFailed(RuntimeError):
    """The remote worker explicitly reported a terminal training failure."""


class ReconnectFailed(RuntimeError):
    """The saved Colab session is gone; the run cannot be reconnected."""


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

    def close(self):
        # absl.logging captures the redirected stream and calls close() on it
        # during interpreter shutdown; without this, exit raises
        # AttributeError: 'Tee' object has no attribute 'close'.
        self.log.close()


def command(argv, timeout=600, quiet=False, cwd=None, env=None):
    if not quiet:
        print("+ " + shlex.join(map(str, argv)), flush=True)
    child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, start_new_session=True)
    expired = threading.Event()

    def kill():
        expired.set()
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    # A local comparison may legitimately run for hours; timeout=None disables
    # the watchdog instead of relying on Timer's indefinite wait.
    timer = threading.Timer(timeout, kill) if timeout else None
    if timer:
        timer.start()
    lines = []
    try:
        for line in child.stdout:
            if not quiet:
                print(line, end="", flush=True)
            lines.append(line)
        code = child.wait()
        if expired.is_set():
            raise TimeoutError(f"Local command exceeded {timeout}s: {argv[0:2]}")
        if code:
            # A failing command is never quiet: surface what went wrong.
            if quiet:
                print("+ " + shlex.join(map(str, argv)), flush=True)
                sys.stdout.writelines(lines)
                sys.stdout.flush()
            raise subprocess.CalledProcessError(code, argv, output="".join(lines))
        return "".join(lines)
    except BaseException:
        kill()
        child.wait()
        raise
    finally:
        if timer:
            timer.cancel()
        child.stdout.close()


def resolve(path=None, environ=None, repo=REPO, clock=None):
    environ = os.environ if environ is None else environ
    config = read_json(SCRIPTS / "config/config_gpu.json")
    if path:
        overrides = read_json(path)
        config.update(overrides)
    for env, key in (("GPU_TYPE", "gpu"), ("COLAB_TIMEOUT", "colab_timeout"),
                     ("SESSION", "session"), ("REPO_BRANCH", "repo_branch")):
        if env in environ:
            config[key] = float(environ[env]) if key == "colab_timeout" else environ[env]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    config.setdefault("source_commit", git("rev-parse", "HEAD"))
    config.setdefault("repo_branch", git("branch", "--show-current"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", config["repo_branch"]) or ".." in config["repo_branch"]:
        raise ValueError("A valid pushed repo_branch is required")
    timestamp = (clock or datetime.now(timezone.utc)).strftime("%d%m%Y_%H%M")
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
    def __init__(self, config, output, *, run=command, sleep=time.sleep, stream_logs=True, methods=("ekf", "sqmc")):
        self.config, self.output, self.run, self.sleep = config, Path(output), run, sleep
        self.methods = tuple(methods)
        self.colab = shutil.which("colab") or "colab"
        self.session = config["session_name"]
        self.remote = Path("/content/sqmc-ekf-" + config["run_id"])
        self.attempted = False
        self.worker_dispatched = False
        self.endpoint = None
        self.log_offset = 0
        self.proxy_refresh_at = 0
        self.stream_logs = stream_logs
        self.status = {"status": "running", "run_id": config["run_id"], "source_commit": config["source_commit"],
                       "config_sha256": config_digest(config), "started_utc": now(),
                       "session": self.session, "shutdown": "pending", "secondary_errors": [],
                       "methods": list(self.methods),
                       "run": {"execution": "pending", "download": "pending"}}

    def save(self):
        self.status.update(log_offset=self.log_offset, worker_dispatched=self.worker_dispatched,
                           methods=list(self.methods))
        write_json(self.output / "status.json", self.status)

    @classmethod
    def from_output(cls, output, **kwargs):
        """Restore identity and progress without resolving a new run/config."""
        output = Path(output).resolve()
        config = read_json(output / "comparison_config.json")
        status = read_json(output / "status.json")
        validate_config(config)
        expected = {"run_id": config["run_id"], "session": config["session_name"],
                    "source_commit": config["source_commit"], "config_sha256": config_digest(config)}
        if any(status.get(key) != value for key, value in expected.items()):
            raise ValueError("Saved run identity or configuration mismatch")
        instance = cls(config, output, **kwargs)
        instance.status = status
        instance.methods = tuple(status.get("methods") or ("ekf", "sqmc"))
        instance.endpoint = status.get("endpoint")
        instance.log_offset = status.get("log_offset", 0)
        instance.attempted = instance.worker_dispatched = bool(status.get("worker_dispatched"))
        return instance

    def reconnect(self):
        # Refresh against server assignments first; a missing CLI cache entry
        # can be restored only for this saved endpoint, without provisioning.
        self.proxy_refresh_at = 0
        self.refresh_proxy()
        with tempfile.TemporaryDirectory(prefix="sqmc-reconnect-") as temp:
            target = Path(temp) / "comparison_config.json"
            self.call("download", "--session", self.session, self.remote / target.name, target)
            if read_json(target) != self.config:
                raise ValueError("Remote effective configuration mismatch")
        self.status.update(status="running", shutdown="pending", reconnected_utc=now())
        for key in ("error", "finished_utc", "detached_utc", "failure_kind"):
            self.status.pop(key, None)
        self.save()
        print(f"Reconnected to {self.session}; continuing from saved progress.", flush=True)

    def call(self, *args, timeout=None, quiet=True):
        if args[0] == "exec":
            raise ValueError("Use execute(); CLI exec can prune a live session on connection errors")
        if self.endpoint and args[0] in {"upload", "download"}:
            self.refresh_proxy()
        return self.run([self.colab, *map(str, args)], timeout=timeout or self.config["transfer_timeout"], quiet=quiet)

    def proxy_command(self):
        # Use Colab's own Python environment: the launcher needs only stdlib,
        # and the CLI may be installed in a separate uv/pipx environment.
        with Path(self.colab).open() as entrypoint:
            shebang = entrypoint.readline().strip()
        if not shebang.startswith("#!") or "python" not in shebang:
            raise RuntimeError("Expected a Python colab entrypoint for credential refresh")
        return [*shlex.split(shebang[2:]), str(SCRIPTS / "refresh_colab_proxy.py"),
                self.session, self.endpoint]

    def refresh_proxy(self):
        if time.monotonic() < self.proxy_refresh_at:
            return
        response = self.run(self.proxy_command(),
                            timeout=self.config["transfer_timeout"], quiet=True)
        lifetime = json.loads(response)["expires_in_seconds"]
        if not isinstance(lifetime, (int, float)) or not 0 < lifetime < float("inf"):
            raise ValueError("Invalid Colab proxy credential lifetime")
        self.proxy_refresh_at = time.monotonic() + min(1200, lifetime / 2)

    def sessions(self):
        return session_map(self.call("sessions"))

    def execute(self, code, timeout=None):
        with tempfile.TemporaryDirectory(prefix="sqmc-dispatch-") as temp:
            path = Path(temp) / "dispatch.py"
            path.write_text(code)
            duration = timeout or self.config["transfer_timeout"]
            # This adapter refreshes credentials and borrows the kernel without
            # invoking CLI exec's session-pruning handler. Do not retry dispatch:
            # an acknowledgement can be lost after the worker actually starts.
            return self.run([*self.proxy_command(), "--exec-file", str(path),
                             "--timeout", str(duration)], timeout=duration + 30, quiet=True)

    def prepare_source(self, directory):
        ref = "refs/heads/" + self.config["repo_branch"]
        sha = self.run(["git", "-C", str(REPO), "rev-parse", ref], timeout=60).strip()
        if sha != self.config["source_commit"]:
            raise RuntimeError("Local source branch does not match the pushed commit")
        bundle = Path(directory) / "source.bundle"
        self.run(["git", "-C", str(REPO), "bundle", "create", str(bundle), ref], timeout=120)
        self.config["source_bundle_sha256"] = digest(bundle)
        self.status["config_sha256"] = config_digest(self.config)
        write_json(self.output / "comparison_config.json", self.config)
        self.save()
        return bundle

    def setup_from_bundle(self, bundle):
        self.call("upload", "--session", self.session, bundle, self.remote / "source.bundle")
        self.call("upload", "--session", self.session, SCRIPTS / "run_sqmc_ekf_gpu.py", self.remote / "bootstrap.py")
        args = [str(self.remote / "bootstrap.py"), "--action", "setup", "--config", str(self.remote / "comparison_config.json")]
        self.execute(f"import subprocess, sys\nsubprocess.run([sys.executable, *{args!r}], check=True)\n", timeout=self.config["setup_timeout"])

    def worker_command(self, action):
        command = ["/usr/bin/python3", str(REMOTE_REPO / "rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_gpu.py"),
                    "--action", action, "--config", str(self.remote / "comparison_config.json")]
        if action == "run":
            command += ["--methods", ",".join(self.methods)]
        return command

    def start_run(self):
        args = self.worker_command("run")
        response = self.execute("import subprocess, sys\nfrom pathlib import Path\n"
                     f"args = {args!r}\nargs[0] = sys.executable\n"
                     f"with open({str(self.remote / 'run_worker.log')!r}, 'w') as log:\n"
                     "    worker = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)\n"
                     f"Path({str(self.remote / 'run.pid')!r}).write_text(str(worker.pid))\n"
                     "print('Started worker', worker.pid)\n")
        if not re.search(r"Started worker \d+", response):
            raise RuntimeError("Colab did not acknowledge starting the run worker")

    def poll_run(self):
        # Filter on the VM and transfer only new complete progress lines plus
        # status. A failed request leaves the byte offset unchanged for retry.
        scripts = REMOTE_REPO / "rbsqmc/comparison/sqmc_ekf/scripts"
        prefix = "SQMC_EKF_PROGRESS="
        response = self.execute(
            "import sys, json\n"
            f"sys.path.insert(0, {str(scripts)!r})\n"
            "from sqmc_ekf_protocol import progress_snapshot\n"
            f"print({prefix!r} + json.dumps(progress_snapshot("
            f"{str(self.remote)!r}, {self.log_offset}, {self.stream_logs!r})))\n"
        )
        packets = [line[len(prefix):] for line in response.splitlines() if line.startswith(prefix)]
        if len(packets) != 1:
            raise RuntimeError(f"Colab returned no unique progress snapshot: {response}")
        packet = json.loads(packets[0])
        state = packet["run"]
        for line in packet["lines"]:
            print(line, end="", flush=True)
        self.log_offset = packet["offset"]
        self.status["run"].update(state)
        self.save()
        return state

    def download_log(self):
        # Full diagnostics are fetched once on failure; successful runs carry
        # them in the verified root archive instead of each progress poll.
        with tempfile.TemporaryDirectory(prefix="sqmc-log-") as temp:
            target = Path(temp) / "remote_logs.txt"
            self.call("download", "--session", self.session, self.remote / "remote_logs.txt", target)
            shutil.copyfile(target, self.output / "remote_logs.txt")

    def wait_run(self):
        # The worker enforces its own colab_timeout; the monitor polls until the
        # worker reports archive_ready (or the session disappears), with a buffer
        # so a slow final archive/transfer is not mistaken for a dead run.
        deadline = time.monotonic() + self.config["colab_timeout"] + 2 * self.config["transfer_timeout"]
        failures = 0
        while time.monotonic() < deadline:
            try:
                state = self.poll_run()
            except (subprocess.CalledProcessError, TimeoutError) as error:
                failures += 1
                self.proxy_refresh_at = 0  # force credential refresh on the next call
                if failures >= 5:
                    raise
                print(f"Progress poll failed ({failures}/5); retrying in 15s: {error}", flush=True)
                self.sleep(15)
                continue
            failures = 0
            if state.get("archive_ready"):
                self.status["run"].update(state)
                self.save()
                if state["execution"] != "complete":
                    raise RemoteRunFailed(f"run: {state.get('error', state['execution'])}")
                return
            self.sleep(15)
        raise TimeoutError("Run worker did not finish and archive within its deadline")

    def download(self, kind, *, partial=False):
        # Root metadata is refreshed after results arrive. Its lifecycle must
        # not reset or reject the completed results download.
        if kind == "run":
            if self.status["run"]["download"] == "complete":
                raise RuntimeError("Refusing to overwrite a complete download")
            self.status["run"]["download"] = "downloading"
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
                    validate_run(verified, self.config, methods=self.methods)
                shutil.copytree(verified / "results", self.output / "results", dirs_exist_ok=True)
                shutil.copytree(verified / "images", self.output / "images", dirs_exist_ok=True)
                shutil.copyfile(verified / "run_manifest.json", self.output / "run_manifest.json")
                self.status["run"].update(download="partial" if partial else "complete", downloaded_utc=now())
                self.save()
        print(f"Downloaded and verified {kind}{' (partial)' if partial else ''}.", flush=True)

    def secondary(self, label, action):
        try:
            action()
            return True
        except Exception as error:
            message = f"{label}: {type(error).__name__}: {error}"
            print(message, flush=True)
            self.status["secondary_errors"].append(message)
            self.save()
            return False

    def recover(self):
        metadata = self.secondary("Root metadata download", lambda: self.download("root", partial=True))
        logs = self.secondary("Remote log download", self.download_log)
        # The worker archives the run bundle in its finally block even when
        # validation fails, so completed training results may exist on the VM
        # even though the run was marked failed. Salvage them (skipping the
        # validation gate, which is what failed) before the session stops.
        if self.status["run"].get("archive_ready"):
            self.secondary("Run artifact salvage", lambda: self.download("run", partial=True))
        return metadata and logs

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

    def provision(self, source_directory):
        print(f"[1/6] Verifying source commit {self.config['source_commit'][:12]} "
              f"is pushed to {self.config['repo_branch']}...", flush=True)
        available = self.sessions()
        if self.session in available:
            raise RuntimeError("Session name already exists; refusing to reuse it")
        remote_ref = self.run(["git", "ls-remote", self.config["repo_url"], "refs/heads/" + self.config["repo_branch"]], timeout=60)
        if not remote_ref.split() or remote_ref.split()[0] != self.config["source_commit"]:
            raise RuntimeError("Push the exact source commit to repo_branch before provisioning")
        bundle = self.prepare_source(source_directory)
        self.attempted = True
        print(f"[2/6] Provisioning {self.config['gpu']} session "
              f"'{self.session}'...", flush=True)
        self.call("run", "--keep", "--gpu", self.config["gpu"], "--session", self.session,
                  "--timeout", self.config["setup_timeout"], SCRIPTS / "run_sqmc_ekf_gpu.py",
                  "--action", "provision", "--config-json", json.dumps(self.config),
                  timeout=self.config["setup_timeout"] + 120)
        self.endpoint = self.sessions().get(self.session)
        if not self.endpoint:
            raise RuntimeError("Provisioned session missing from server session list")
        self.status["endpoint"] = self.endpoint
        print("[3/6] Uploading source bundle and pinning the checkout "
              "(installs deps, asserts GPU)...", flush=True)
        self.setup_from_bundle(bundle)
        print("[4/6] Verifying remote metadata...", flush=True)
        self.download("root")
        self.status["run"].update(execution="running", started_utc=now())
        self.save()
        print("[5/6] Running the comparison (per-epoch progress is mirrored "
              "below; use --no-stream-logs to silence it)...", flush=True)
        # Persist dispatch intent before the RPC: the worker may start even
        # if its acknowledgement is lost. Reconnection must never start it twice.
        self.worker_dispatched = True
        self.save()
        self.start_run()

    def launch(self, *, resume=False):
        self.save()
        code = 0
        stop_session = False
        try:
            if resume:
                if self.status.get("status") == "complete" and self.status["run"].get("download") == "complete":
                    # A previous monitor already collected the results; verify
                    # and reuse that completed copy without touching the session.
                    validate_run(self.output, self.config, methods=self.methods)
                    print("Run already complete and collected.", flush=True)
                    return 0
                if self.status.get("shutdown") == "verified_stopped":
                    raise ValueError("The saved session was stopped; its worker cannot be reconnected")
                if not self.worker_dispatched or not self.endpoint:
                    raise ValueError("No saved worker dispatch and endpoint to reconnect")
                self.reconnect()
            else:
                with tempfile.TemporaryDirectory(prefix="sqmc-source-") as directory:
                    self.provision(directory)
            self.wait_run()
            print("[6/6] Downloading and verifying run artifacts...", flush=True)
            if self.status["run"]["download"] == "complete":
                # A previous monitor may have finished results but disconnected
                # while fetching metadata. Verify and reuse that completed copy.
                validate_run(self.output, self.config, methods=self.methods)
            else:
                self.download("run")
            self.download("root")
            self.status["status"] = "complete"
            stop_session = True
        except BaseException as error:
            code = 130 if isinstance(error, KeyboardInterrupt) else getattr(error, "returncode", 1)
            code = code if isinstance(code, int) and 1 <= code <= 255 else 1
            self.status.update(error=f"{type(error).__name__}: {error}", exit_code=code)
            traceback.print_exc()
            if self.worker_dispatched:
                if isinstance(error, ReconnectFailed):
                    # The saved session is gone; there is nothing to preserve or
                    # reconnect to. Mark the run failed and stop.
                    self.status.update(status="failed", failure_kind="monitoring",
                                       shutdown="failed", finished_utc=now())
                    self.save()
                else:
                    # A monitoring/transport failure says nothing about whether
                    # training failed. Preserve the last remote execution state
                    # and leave the session running so it can be reconnected.
                    self.status.update(status="detached", shutdown="deferred",
                                       detached_utc=now(), failure_kind="monitoring")
                    if isinstance(error, RemoteRunFailed):
                        # The worker reports whether results were exported:
                        # a validation failure leaves completed training
                        # artifacts on the VM (salvaged by recover()), while a
                        # training crash leaves nothing worth keeping.
                        failure_kind = self.status["run"].get("failure_kind", "training")
                        self.status.update(status="failed", failure_kind=failure_kind)
                        stop_session = self.recover()
                    self.save()
                    if not stop_session:
                        script = SCRIPTS / "run_sqmc_ekf_colab.sh"
                        print("Local monitoring ended; the Colab session was left untouched.\n"
                              "Reconnect with: " + shlex.join([str(script), "--resume", str(self.output)]),
                              flush=True)
            else:
                self.status.update(status="failed", failure_kind="local")
                if self.attempted:
                    self.recover()
                stop_session = self.attempted
        finally:
            if stop_session:
                self.secondary("Shutdown", self.shutdown)
                if self.status["shutdown"] != "verified_stopped":
                    self.status["shutdown"] = "failed"
                    if code == 0:
                        code = 1
                        self.status.update(status="detached", error="Shutdown verification failed")
            elif not self.attempted:
                self.status["shutdown"] = "not_provisioned"
            # This timestamp ends the local monitoring attempt, not the worker.
            self.status.update(monitor_finished_utc=now(), exit_code=code)
            if stop_session or not self.worker_dispatched:
                self.status["finished_utc"] = now()
            self.save()
        return code


def ensure_sobol_data():
    """Build the gitignored Sobol direction numbers from the tracked table.

    The SQMC module loads this file at import time, so even the EKF stage
    (which imports the SQMC filter for scoring) requires it to exist.
    """
    data = REPO_ROOT / "sqmc" / "qmc" / "_sobol_direction_numbers.npz"
    if not data.exists():
        command([sys.executable, "sqmc/qmc/_generate_sobol_data.py", "--verify-scipy"],
                cwd=REPO_ROOT)


def local_environment():
    """Match the GPU worker's precision environment, pinned to CPU.

    ``RBSQMC_PLATFORM=cpu`` makes the platform explicit: the model modules pin
    ``jax_platforms`` from it, and leaving it unset would silently depend on
    the default rather than the documented CPU execution.
    """
    return dict(os.environ, JAX_ENABLE_X64="true", RBSQMC_PLATFORM="cpu",
                MPLBACKEND="Agg", PYTHONUNBUFFERED="1")


def run_local(config, output, *, smoke=False, methods=("ekf", "sqmc")):
    """Run the whole comparison on this machine (CPU only): EKF then SQMC then combine.

    Intended for smoke/small runs. For the real comparison use
    :func:`run_hybrid`, which runs SQMC on a Colab GPU. Each stage stores its
    artifacts under ``output/<stage>`` and its progress in ``status.json``, so
    a failed SQMC stage never discards completed EKF results; the stored
    partials can be combined again with ``run.py --combine`` without retraining.
    """
    if not smoke and set(methods) == {"ekf", "sqmc"}:
        print("Note: SQMC will run on CPU. For the real run use the default "
              "mode (EKF local, SQMC on Colab GPU) or pass --smoke.", flush=True)
    effective = dict(config)
    if smoke:
        effective.update(smoke=True, n_epochs=min(config["n_epochs"], 3),
                         n_reps=min(config["n_reps"], 2),
                         n_particles=min(config["n_particles"], 64))
    config_path = output / "comparison_config.json"
    write_json(config_path, effective)
    module = "rbsqmc.comparison.sqmc_ekf.run"
    data = REPO / "data" / "results.csv"
    status = {"mode": "local-cpu", "run_id": config["run_id"], "status": "running",
              "config_sha256": config_digest(effective), "started_utc": now(),
              "methods": list(methods), "stages": {}}
    write_json(output / "status.json", status)
    code, stage = 0, None
    try:
        ensure_sobol_data()
        environment = local_environment()
        for method in methods:
            stage = method
            status["stages"][method] = "running"
            write_json(output / "status.json", status)
            command([sys.executable, "-u", "-m", module, "--config", str(config_path),
                     "--data", str(data), "--methods", method,
                     "--output-dir", str(output / method)],
                    timeout=None, cwd=REPO_ROOT, env=environment)
            status["stages"][method] = "complete"
            write_json(output / "status.json", status)
        if set(methods) == {"ekf", "sqmc"}:
            stage = "combine"
            status["stages"]["combine"] = "running"
            write_json(output / "status.json", status)
            command([sys.executable, "-u", "-m", module, "--config", str(config_path),
                     "--data", str(data), "--combine", str(output / "ekf"),
                     str(output / "sqmc"), "--output-dir", str(output / "combined")],
                    timeout=None, cwd=REPO_ROOT, env=environment)
            status["stages"]["combine"] = "complete"
        status["status"] = "complete"
    except BaseException as error:
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        if stage:
            status["stages"][stage] = "failed"
        traceback.print_exc()
    finally:
        status["finished_utc"] = now()
        write_json(output / "status.json", status)
    return code


def run_hybrid(config, output, *, smoke=False, methods=("ekf", "sqmc"), stream_logs=True):
    """The real comparison: EKF runs locally on CPU, SQMC runs on a Colab GPU.

    Stages, in order:
      1. EKF locally (fast; needs no GPU) -> ``output/ekf``
      2. SQMC on a provisioned Colab GPU session -> ``output/sqmc``
      3. Combine the two partials locally -> ``output/combined``

    Progress (including Colab provisioning/download, which the Launcher drives)
    is recorded in ``output/status.json``. Reconnect a lost monitoring session
    with ``run_sqmc_ekf_colab.sh --resume <output>``.
    """
    if set(methods) < {"ekf", "sqmc"}:
        raise ValueError("run_hybrid requires both methods")
    return _run_hybrid(config, output, smoke=smoke, stream_logs=stream_logs)


def _run_ekf_local(config, output, config_path, data):
    """Run the EKF stage on this machine's CPU."""
    command([sys.executable, "-u", "-m", "rbsqmc.comparison.sqmc_ekf.run",
             "--config", str(config_path), "--data", str(data),
             "--methods", "ekf", "--output-dir", str(output / "ekf")],
            timeout=None, cwd=REPO_ROOT, env=local_environment())


def _run_hybrid(config, output, *, smoke=False, resume=False, stream_logs=True):
    """Drive EKF locally, SQMC on Colab GPU, then combine, with a shared status."""
    effective = dict(config)
    if smoke:
        effective.update(smoke=True, n_epochs=min(config["n_epochs"], 3),
                         n_reps=min(config["n_reps"], 2),
                         n_particles=min(config["n_particles"], 64))
    config_path = output / "comparison_config.json"
    data = REPO / "data" / "results.csv"
    module = "rbsqmc.comparison.sqmc_ekf.run"
    if resume:
        status = read_json(output / "status.json")
        if status["config_sha256"] != config_digest(effective):
            raise ValueError("Saved hybrid configuration mismatch")
        if status["stages"].get("ekf") != "complete":
            raise ValueError("No completed local EKF stage to resume from")
        status.update(status="running", reconnected_utc=now())
        status.pop("error", None)
        status.pop("finished_utc", None)
    else:
        write_json(config_path, effective)
        status = {"mode": "hybrid", "run_id": config["run_id"], "status": "running",
                  "config_sha256": config_digest(effective), "started_utc": now(),
                  "smoke": bool(smoke), "stages": {}}
    write_json(output / "status.json", status)
    code, stage = 0, None
    try:
        ensure_sobol_data()
        # Stage 1: EKF locally on CPU.
        if not resume:
            stage = "ekf"
            status["stages"]["ekf"] = "running"
            write_json(output / "status.json", status)
            _run_ekf_local(effective, output, config_path, data)
            status["stages"]["ekf"] = "complete"
            write_json(output / "status.json", status)

        # Stage 2: SQMC on a Colab GPU, driven by the single-run Launcher.
        stage = "sqmc"
        status["stages"]["sqmc"] = "running"
        write_json(output / "status.json", status)
        (output / "sqmc").mkdir(parents=True, exist_ok=True)
        # The child owns its transport metadata; preserve the parent's frozen
        # configuration and completed EKF artifacts across reconnect attempts.
        launcher = (Launcher.from_output(output / "sqmc", stream_logs=stream_logs) if resume
                    else Launcher(dict(effective), output / "sqmc", stream_logs=stream_logs,
                                  methods=("sqmc",)))
        launch_code = launcher.launch(resume=resume)
        if launch_code:
            if launcher.status["status"] == "detached":
                code = launch_code
                status.update(status="detached", error=launcher.status.get("error"))
                status["stages"]["sqmc"] = "detached"
                print("Resume the comparison with: " + shlex.join([
                    "bash", str(SCRIPTS / "run_sqmc_ekf_colab.sh"), "--resume", str(output)]), flush=True)
                return code
            raise RuntimeError(f"Colab SQMC stage returned exit code {launch_code}")
        status["stages"]["sqmc"] = "complete"
        write_json(output / "status.json", status)

        # Stage 3: combine the EKF and SQMC partials locally.
        stage = "combine"
        status["stages"]["combine"] = "running"
        write_json(output / "status.json", status)
        command([sys.executable, "-u", "-m", module, "--config", str(config_path),
                 "--data", str(data), "--combine", str(output / "ekf"),
                 str(output / "sqmc"), "--output-dir", str(output / "combined")],
                timeout=None, cwd=REPO_ROOT, env=local_environment())
        status["stages"]["combine"] = "complete"
        status["status"] = "complete"
    except BaseException as error:
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        if stage:
            status["stages"][stage] = "failed"
        traceback.print_exc()
    finally:
        status["monitor_finished_utc"] = now()
        if status["status"] != "detached":
            status["finished_utc"] = now()
        write_json(output / "status.json", status)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides merged into the full profile")
    parser.add_argument("--resume", type=Path, metavar="OUTPUT_DIR",
                        help="Reconnect to a previously detached run and collect its results")
    parser.add_argument("--local", action="store_true",
                        help="Run everything on this machine's CPU (for smoke/small runs); "
                             "the default runs EKF locally and SQMC on a Colab GPU")
    parser.add_argument("--smoke", action="store_true",
                        help="Use a small smoke configuration")
    parser.add_argument("--methods", default="ekf,sqmc",
                        help="Methods for --local mode (default: ekf,sqmc)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-stream-logs", action="store_true",
                        help="Do not mirror per-epoch progress from the remote log "
                             "(the full log is still downloaded at the end)")
    args = parser.parse_args(argv)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    if not methods or not set(methods) <= {"ekf", "sqmc"}:
        parser.error("--methods must be a non-empty subset of {ekf, sqmc}")
    if args.resume:
        if args.config or args.dry_run or args.local or args.smoke or args.methods != "ekf,sqmc":
            parser.error("--resume uses the saved configuration; only --no-stream-logs may accompany it")
        output = Path(args.resume).resolve()
        if not (output / "status.json").is_file():
            raise FileNotFoundError(f"No saved run status at {output}")
        if not shutil.which("colab"):
            raise RuntimeError("colab must be available on PATH; activate the local virtual environment")
        with (output / "logs.txt").open("a", buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                print(f"Resuming comparison output: {output}", flush=True)
                if read_json(output / "status.json").get("mode") == "hybrid":
                    config = read_json(output / "comparison_config.json")
                    validate_config(config)
                    return _run_hybrid(config, output, resume=True, stream_logs=not args.no_stream_logs)
                return Launcher.from_output(output, stream_logs=not args.no_stream_logs).launch(resume=True)
    if not args.local and methods != ("ekf", "sqmc"):
        parser.error("The hybrid comparison requires both methods; --methods is for --local")
    config = resolve(args.config)
    output = SCRIPTS.parents[0] / "outputs" / config["run_id"]
    if output.exists():
        raise FileExistsError(f"UTC timestamp collision: {output}; wait for the next minute")
    if args.dry_run:
        print(json.dumps(config, indent=2))
        print(f"Output: {output}")
        if args.local:
            print("Local CPU methods: " + ", ".join(methods) + "; combine locally when both finish")
            return 0
        print("EKF: local CPU ->", output / "ekf")
        print("RB-SQMC: Colab GPU, --methods sqmc ->", output / "sqmc")
        print("Combine and validate locally ->", output / "combined")
        print("git bundle create <temporary source.bundle> refs/heads/" + config["repo_branch"])
        print("colab run --keep --gpu", config["gpu"], "--session", config["session_name"], "--timeout", config["setup_timeout"], "run_sqmc_ekf_gpu.py --action provision --config-json <effective JSON plus bundle SHA-256>")
        print("colab upload <source.bundle>; colab upload <bootstrap.py>; Colab transport: setup pinned checkout")
        print("Colab transport (preserve session on error):", config["session_name"], "; start worker; poll; download and verify run.tar.gz + root.tar.gz")
        print("colab stop --session", config["session_name"], "; colab sessions (verify shutdown)")
        return 0
    if args.local:
        # Pure-CPU smoke/small run; the real run places SQMC on a Colab GPU.
        if args.no_stream_logs:
            parser.error("--local cannot be combined with --no-stream-logs")
        output.mkdir(parents=True, exist_ok=False)
        with (output / "logs.txt").open("w", buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                print(f"Local (CPU) comparison output: {output}", flush=True)
                return run_local(config, output, smoke=args.smoke, methods=methods)
    # Default real mode: EKF locally, SQMC on Colab GPU, then combine locally.
    if not shutil.which("colab"):
        raise RuntimeError("colab must be available on PATH; activate the local virtual environment")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "logs.txt").open("w", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f"Comparison output: {output}", flush=True)
            return run_hybrid(config, output, smoke=args.smoke, stream_logs=not args.no_stream_logs)


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGTERM, terminate)
    sys.exit(main())
