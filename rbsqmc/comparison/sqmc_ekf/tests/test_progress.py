"""Regression checks for Colab polling, diagnostics and expiring credentials."""

import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rbsqmc.comparison.sqmc_ekf.scripts import sqmc_ekf_protocol as protocol
from rbsqmc.comparison.sqmc_ekf.scripts.refresh_colab_proxy import execute as remote_execute, refresh


@pytest.fixture
def local(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(protocol.__file__).parent))
    return importlib.import_module("run_sqmc_ekf_local")


@pytest.fixture
def launcher(local, tmp_path):
    config = dict(session_name="test", run_id="test", source_commit="a" * 40,
                  transfer_timeout=10, colab_timeout=60)
    return local.Launcher(config, tmp_path, sleep=lambda seconds: None)


def test_quiet_command_preserves_original_failure_through_tee(local):
    terminal, log = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(local.Tee(terminal, log)):
        with pytest.raises(subprocess.CalledProcessError) as caught:
            local.command([sys.executable, "-c", "print('HTTP 404'); raise SystemExit(7)"], quiet=True)
    assert caught.value.returncode == 7
    assert caught.value.output == "HTTP 404\n"
    assert terminal.getvalue() == log.getvalue()
    assert "HTTP 404" in terminal.getvalue()


def proxy_state():
    session = SimpleNamespace(endpoint="owned", token="old", url="url", kernel_id="kernel",
                              session_id="connection", keep_alive_pid=123)
    cache = {"test": session}
    proxy = SimpleNamespace(token="fresh", url="url", token_expires_in_seconds=3600)
    assignments = [SimpleNamespace(endpoint="owned", runtime_proxy_info=proxy,
                                   variant=SimpleNamespace(name="GPU"),
                                   accelerator=SimpleNamespace(value="A100"))]
    state = SimpleNamespace(
        store=SimpleNamespace(get=cache.get, list=lambda: cache,
                              add=lambda value: cache.update(test=value)),
        client=SimpleNamespace(list_assignments=lambda: assignments),
        auth_provider="auth", config_path=None,
        prune_session=lambda *args: pytest.fail("Pruned the live session"))
    return state, cache, assignments


@pytest.mark.parametrize("error", [RuntimeError("HTTP 401"), RuntimeError("HTTP 404"), TimeoutError("offline")])
def test_failed_remote_execution_preserves_cache_kernel_and_keep_alive(error, capsys):
    state, cache, _ = proxy_state()
    closes = []
    fail = [True]

    class Runtime:
        def __init__(self, url, token, **kwargs):
            assert token == "fresh"
            assert kwargs["kernel_id"] == "kernel"

        def execute_code(self, code, timeout):
            if fail[0]:
                raise error
            return [{"output_type": "stream", "text": "sqmc epoch 36/50\n"}]

        def stop(self, shutdown_kernel):
            closes.append(shutdown_kernel)

    with pytest.raises(type(error), match=str(error)):
        remote_execute(state, "test", "owned", "poll", 10, Runtime)
    assert cache["test"].keep_alive_pid == 123
    assert cache["test"].kernel_id == "kernel"
    fail[0] = False
    remote_execute(state, "test", "owned", "poll", 10, Runtime)
    assert closes == [False, False]
    assert capsys.readouterr().out == "sqmc epoch 36/50\n"


def test_missing_cache_is_restored_only_for_existing_saved_endpoint():
    state, cache, assignments = proxy_state()
    cache.clear()
    starts = []

    def start(endpoint, name, **kwargs):
        assert "test" in cache
        starts.append((endpoint, name))
        return 456

    refresh(state, "test", "owned", session_factory=SimpleNamespace, keep_alive=start)
    assert cache["test"].endpoint == "owned"
    assert cache["test"].keep_alive_pid == 456
    refresh(state, "test", "owned", session_factory=SimpleNamespace, keep_alive=start)
    assert starts == [("owned", "test")]
    cache.clear()
    assignments.clear()
    with pytest.raises(RuntimeError, match="no longer assigned"):
        refresh(state, "test", "owned", session_factory=SimpleNamespace, keep_alive=start)
    assert not cache
    assert len(starts) == 1


def test_restoring_cache_refuses_endpoint_registered_under_another_name():
    state, cache, _ = proxy_state()
    cache["other"] = cache.pop("test")
    with pytest.raises(RuntimeError, match="different session name"):
        refresh(state, "test", "owned", session_factory=SimpleNamespace)
    assert set(cache) == {"other"}


def test_launcher_execute_uses_non_pruning_adapter(local, launcher, tmp_path):
    cli = tmp_path / "colab"
    cli.write_text("#!/cli/bin/python3\n")
    launcher.colab, launcher.endpoint = str(cli), "owned"
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert Path(argv[argv.index("--exec-file") + 1]).read_text() == "print('poll')"
        return "poll\n"

    launcher.run = run
    assert launcher.execute("print('poll')") == "poll\n"
    assert calls[0][0] == "/cli/bin/python3"
    assert calls[0][1].endswith("refresh_colab_proxy.py")
    assert "exec" not in calls[0]
    with pytest.raises(ValueError, match="prune"):
        launcher.call("exec")


def test_reconnect_restores_cache_before_downloading_config(launcher, monkeypatch):
    actions = []
    monkeypatch.setattr(launcher, "sessions", lambda: pytest.fail("Used the CLI cache as server truth"))
    monkeypatch.setattr(launcher, "refresh_proxy", lambda: actions.append("restore"))

    def download(*args):
        assert actions == ["restore"]
        protocol.write_json(Path(args[-1]), launcher.config)
        actions.append("download")

    monkeypatch.setattr(launcher, "call", download)
    launcher.reconnect()
    assert actions == ["restore", "download"]


def test_snapshot_filters_remotely_and_preserves_partial_utf8_lines(tmp_path):
    protocol.write_json(tmp_path / "remote_status.json", {"run": {"execution": "running"}})
    log = tmp_path / "remote_logs.txt"
    epoch = "sqmc epoch 16/50: train -15178.6966, test -1083.8652, |grad| 482.8749, 178.5s\n"
    next_epoch = "sqmc epoch 17/50: train -15283.8713, test -1012.9015, |grad| 471.2261, 178.2s\n"
    noise = b"Installing dependency\n" * 10000
    partial = "sqmc epoch 18/50: diagnostic \u03bc".encode()
    log.write_bytes(noise + epoch.encode() + next_epoch.encode() + partial[:-1])
    packet = protocol.progress_snapshot(tmp_path)
    assert packet["lines"] == [epoch, next_epoch]
    assert len(json.dumps(packet)) < 500
    assert packet["offset"] == len(noise) + len(epoch) + len(next_epoch)
    assert protocol.progress_snapshot(tmp_path, packet["offset"])["lines"] == []
    with log.open("ab") as stream:
        stream.write(partial[-1:] + b"\n")
    after = protocol.progress_snapshot(tmp_path, packet["offset"])
    assert after["lines"] == [partial.decode() + "\n"]
    log.write_bytes(b"")
    with pytest.raises(ValueError, match="shrank"):
        protocol.progress_snapshot(tmp_path, after["offset"])


def test_status_only_does_not_read_log(tmp_path):
    protocol.write_json(tmp_path / "remote_status.json", {"run": {"execution": "running"}})
    assert protocol.progress_snapshot(tmp_path, 123, False) == {
        "run": {"execution": "running"}, "offset": 123, "lines": [],
    }


def test_generated_remote_poll_code_prints_each_epoch_once(local, launcher, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(local, "REMOTE_REPO", Path(__file__).resolve().parents[4])
    launcher.remote = tmp_path
    protocol.write_json(tmp_path / "remote_status.json", {"run": {"execution": "running"}})
    epoch = "ekf epoch 1/50: train -10, test -2, |grad| 1, 0.1s\n"
    (tmp_path / "remote_logs.txt").write_text("pip setup chatter\n" + epoch)

    def remote_exec(code):
        return subprocess.check_output([sys.executable, "-c", code], text=True)

    monkeypatch.setattr(launcher, "execute", remote_exec)
    assert launcher.poll_run()["execution"] == "running"
    launcher.poll_run()
    assert capsys.readouterr().out == epoch
    offset = launcher.log_offset
    monkeypatch.setattr(launcher, "execute", lambda code: "remote exception, no packet")
    with pytest.raises(RuntimeError, match="no unique progress"):
        launcher.poll_run()
    assert launcher.log_offset == offset


@pytest.mark.parametrize("persistent", [False, True])
def test_poll_transport_failure_retries_but_never_completes_on_error(launcher, monkeypatch, persistent):
    calls = []

    def poll():
        calls.append(True)
        if persistent or len(calls) < 3:
            raise subprocess.CalledProcessError(1, ["colab", "exec"], output="connection failed")
        return {"execution": "complete", "archive_ready": True}

    launcher.proxy_refresh_at = float("inf")
    monkeypatch.setattr(launcher, "poll_run", poll)
    if persistent:
        with pytest.raises(subprocess.CalledProcessError):
            launcher.wait_run()
        assert launcher.status["run"]["execution"] != "complete"
    else:
        launcher.wait_run()
        assert launcher.status["run"]["execution"] == "complete"
    assert len(calls) == (5 if persistent else 3)
    assert launcher.proxy_refresh_at == 0


def test_remote_model_failure_is_not_retried(launcher, monkeypatch):
    def poll():
        return {"execution": "failed", "archive_ready": True, "error": "Non-finite gradient"}

    monkeypatch.setattr(launcher, "poll_run", poll)
    with pytest.raises(RuntimeError, match="Non-finite gradient"):
        launcher.wait_run()


def test_proxy_refresh_updates_credentials_only_for_owned_endpoint():
    session = SimpleNamespace(endpoint="owned", token="old", url="old-url", kernel_id="kernel")
    writes = []
    proxy = SimpleNamespace(token="new", url="new-url", token_expires_in_seconds=3600)
    assignments = [SimpleNamespace(endpoint="owned", runtime_proxy_info=proxy)]
    state = SimpleNamespace(store=SimpleNamespace(get=lambda name: session, add=writes.append),
                            client=SimpleNamespace(list_assignments=lambda: assignments))
    assert refresh(state, "test", "owned") == {"expires_in_seconds": 3600}
    assert (session.token, session.url, session.kernel_id) == ("new", "new-url", "kernel")
    assert writes == [session]
    with pytest.raises(RuntimeError, match="endpoint changed"):
        refresh(state, "test", "different")
    assignments.clear()
    with pytest.raises(RuntimeError, match="no longer assigned"):
        refresh(state, "test", "owned")
    assert len(writes) == 1


def test_launcher_refreshes_before_expiry_using_cli_interpreter(local, launcher, tmp_path, monkeypatch):
    cli = tmp_path / "colab"
    cli.write_text("#!/cli-env/bin/python3\n")
    launcher.colab, launcher.endpoint = str(cli), "owned"
    now = [100]
    monkeypatch.setattr(local.time, "monotonic", lambda: now[0])
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return '{"expires_in_seconds": 3600}' if argv[0] == "/cli-env/bin/python3" else ""

    launcher.run = run
    launcher.call("download", "first")
    launcher.call("upload", "second")
    assert len(calls) == 3
    assert calls[0][-2:] == ["test", "owned"]
    now[0] += 1201
    launcher.call("download", "third")
    assert len(calls) == 5
    assert calls[3][0] == "/cli-env/bin/python3"


@pytest.mark.parametrize("error", [TimeoutError("offline"), RuntimeError("missing snapshot"),
                                  KeyboardInterrupt(), ValueError("bad archive")])
def test_monitor_errors_leave_worker_and_remote_state_untouched(launcher, monkeypatch, error):
    def provision(directory):
        launcher.attempted = launcher.worker_dispatched = True
        launcher.endpoint = "owned"
        launcher.status["run"]["execution"] = "running"

    def wait():
        raise error

    monkeypatch.setattr(launcher, "provision", provision)
    monkeypatch.setattr(launcher, "wait_run", wait)
    monkeypatch.setattr(launcher, "shutdown", lambda: pytest.fail("Stopped a background worker"))
    monkeypatch.setattr(launcher, "recover", lambda: pytest.fail("Recovery on a disconnected monitor"))
    assert launcher.launch() != 0
    saved = protocol.read_json(launcher.output / "status.json")
    assert saved["status"] == "detached"
    assert saved["shutdown"] == "deferred"
    assert saved["run"]["execution"] == "running"
    assert "error" not in saved["run"]
    assert "finished_utc" not in saved


def test_resume_collects_without_dispatching_again(launcher, monkeypatch):
    launcher.attempted = launcher.worker_dispatched = True
    launcher.endpoint = "owned"
    actions = []
    monkeypatch.setattr(launcher, "reconnect", lambda: actions.append("reconnect"))
    monkeypatch.setattr(launcher, "provision", lambda directory: pytest.fail("Reprovisioned"))
    monkeypatch.setattr(launcher, "start_run", lambda: pytest.fail("Retrained"))
    monkeypatch.setattr(launcher, "wait_run", lambda: actions.append("wait"))
    monkeypatch.setattr(launcher, "download", lambda kind: actions.append(kind))

    def stop():
        actions.append("stop")
        launcher.status["shutdown"] = "verified_stopped"

    monkeypatch.setattr(launcher, "shutdown", stop)
    assert launcher.launch(resume=True) == 0
    assert actions == ["reconnect", "wait", "run", "root", "stop"]


def test_hybrid_detach_and_resume_reuses_local_ekf(local, tmp_path, monkeypatch):
    config = protocol.read_json(Path(protocol.__file__).parent / "config/config_gpu.json")
    config.update(run_id="test", session_name="test", source_commit="a" * 40)
    actions = []
    monkeypatch.setattr(local, "ensure_sobol_data", lambda: None)
    monkeypatch.setattr(local, "_run_ekf_local", lambda *args: actions.append("ekf"))
    monkeypatch.setattr(local, "command", lambda argv, **kwargs: actions.append("combine"))

    def launch(self, *, resume=False):
        assert self.methods == ("sqmc",)
        if resume:
            assert self.log_offset == 321
            assert self.config["source_bundle_sha256"] == "b" * 64
            actions.append("reconnect")
            self.status["status"] = "complete"
            return 0
        actions.append("sqmc")
        self.config["source_bundle_sha256"] = "b" * 64
        self.endpoint = "owned"
        self.worker_dispatched = self.attempted = True
        self.log_offset = 321
        self.status.update(status="detached", endpoint=self.endpoint,
                           config_sha256=protocol.config_digest(self.config))
        protocol.write_json(self.output / "comparison_config.json", self.config)
        self.save()
        return 1

    monkeypatch.setattr(local.Launcher, "launch", launch)
    assert local.run_hybrid(config, tmp_path) == 1
    saved = protocol.read_json(tmp_path / "status.json")
    assert saved["status"] == "detached"
    assert saved["stages"] == {"ekf": "complete", "sqmc": "detached"}
    assert "source_bundle_sha256" not in config
    assert "finished_utc" not in saved
    assert local._run_hybrid(config, tmp_path, resume=True) == 0
    assert actions == ["ekf", "sqmc", "reconnect", "combine"]


def test_resume_cli_never_resolves_new_configuration(local, tmp_path, monkeypatch):
    protocol.write_json(tmp_path / "status.json", {"mode": "hybrid"})
    protocol.write_json(tmp_path / "comparison_config.json", {"saved": True})
    monkeypatch.setattr(local, "resolve", lambda *args: pytest.fail("Resolved a new run"))
    monkeypatch.setattr(local, "validate_config", lambda cfg: None)
    monkeypatch.setattr(local.shutil, "which", lambda name: "/colab")
    monkeypatch.setattr(local, "_run_hybrid", lambda cfg, output, **kw: 0 if cfg == {"saved": True} and kw["resume"] else 1)
    assert local.main(["--resume", str(tmp_path)]) == 0


def test_ekf_subprocess_is_local_cpu(local, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("RBSQMC_PLATFORM", "cuda")
    monkeypatch.setattr(local, "command", lambda argv, **kwargs: calls.append((argv, kwargs)))
    local._run_ekf_local({}, tmp_path, tmp_path / "config.json", tmp_path / "data.csv")
    argv, kwargs = calls[0]
    assert argv[0] == sys.executable
    assert argv[argv.index("--methods") + 1] == "ekf"
    assert kwargs["env"]["RBSQMC_PLATFORM"] == "cpu"
    assert kwargs["env"]["JAX_ENABLE_X64"] == "true"


@pytest.mark.parametrize("methods,argument", [(("sqmc",), "sqmc"), (("ekf", "sqmc"), "both")])
def test_remote_worker_method_argument_and_validation(local, tmp_path, monkeypatch, methods, argument):
    gpu = importlib.import_module("run_sqmc_ekf_gpu")
    shared = importlib.import_module("sqmc_ekf_protocol")
    commands, validations = [], []
    monkeypatch.setattr(gpu, "run", lambda argv, **kwargs: commands.append(argv))
    monkeypatch.setattr(shared, "validate_run", lambda root, cfg, methods: validations.append(methods))
    monkeypatch.setattr(shared, "make_archive", lambda *args: None)
    gpu.run_comparison({"colab_timeout": 60}, tmp_path, methods)
    assert commands[0][commands[0].index("--methods") + 1] == argument
    assert validations == [methods]
