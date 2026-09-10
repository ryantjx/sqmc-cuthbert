"""Offline tests: no Colab credentials, network or hardware allocation."""
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tarfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import comparison_protocol as protocol
import run_comparison_local as local


@pytest.fixture
def config():
    return local.resolve(environ={}, clock=datetime(2026, 9, 7, 1, 2, tzinfo=timezone.utc))


def test_full_profile_and_environment_resolution(tmp_path):
    override = tmp_path / "config.json"
    override.write_text(json.dumps({"qmc": {"seed": 99, "warmups": 4}, "sqmc": {"budget_seconds": [.2]}}))
    config = local.resolve(override, environ={"GPU_TYPE": "L4", "COLAB_TIMEOUT": "81", "SESSION": "test"})
    assert (config["gpu"], config["colab_timeout"], config["session"]) == ("L4", 81, "test")
    assert config["qmc"]["seed"] == 99
    assert config["hilbert_sort"]["n_values"][-1] == 131072
    for stage in protocol.STAGES:
        assert config[stage]["dimensions"] == [2, 5, 10, 30, 60]
        command = protocol.stage_command(config, stage, "/output")
        # Parse the actual benchmark CLI to independently check every forwarded option.
        import argparse
        import importlib
        from unittest.mock import patch
        captured = {}
        original = argparse.ArgumentParser.parse_args
        def parse(parser, args=None, namespace=None):
            result = original(parser, command[4:], namespace)
            captured.update(vars(result))
            raise StopIteration
        with patch.object(argparse.ArgumentParser, "parse_args", parse), pytest.raises(StopIteration):
            importlib.import_module(f"sqmc.comparison.benchmark_{stage}").main()
        captured.pop("output_dir")
        assert captured == config[stage]


@pytest.mark.parametrize("edit", [lambda c: c["qmc"].update(dimensions=[0]), lambda c: c["sqmc"].update(budget_seconds=[float('nan')]), lambda c: c.update(gpu="typo"), lambda c: c["hilbert_sort"].update(platforms=["cpu"]), lambda c: c["qmc"].update(n_values=[3]), lambda c: c["sqmc"].update(unknown=1)])
def test_invalid_config(config, edit):
    edit(config)
    with pytest.raises(ValueError):
        protocol.validate_config(config)


def test_dry_run_and_collision(config, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(local, "REPO", tmp_path)
    monkeypatch.setattr(local, "resolve", lambda *a: config)
    monkeypatch.setattr(local.Launcher, "launch", lambda *a: pytest.fail("provisioned during dry run"))
    assert local.main(["--dry-run"]) == 0
    assert list(tmp_path.iterdir()) == []
    assert "--dimensions 2 5 10 30 60" in capsys.readouterr().out
    output = tmp_path / "sqmc/comparison/outputs" / config["run_id"]
    output.mkdir(parents=True)
    sentinel = output / "logs.txt"
    sentinel.write_text("original")
    with pytest.raises(FileExistsError):
        local.main([])
    assert sentinel.read_text() == "original"


def bundle(tmp_path, config):
    root = tmp_path / "remote"
    (root / "qmc").mkdir(parents=True)
    (root / "qmc/logs.txt").write_text("stage log")
    protocol.make_archive(root, "qmc", config)
    return root


def test_archive_checksums_and_paths(tmp_path, config):
    root = bundle(tmp_path, config)
    target = tmp_path / "local"
    protocol.unpack_verified(root / "qmc.tar.gz", root / "qmc.tar.gz.sha256", target, "qmc", config)
    assert (target / "qmc/logs.txt").read_text() == "stage log"
    changed = copy.deepcopy(config)
    changed["sqmc"]["seed"] += 1
    with pytest.raises(ValueError, match="provenance"):
        protocol.unpack_verified(root / "qmc.tar.gz", root / "qmc.tar.gz.sha256", target, "qmc", changed)
    (root / "qmc.tar.gz.sha256").write_text("bad")
    with pytest.raises(ValueError, match="checksum"):
        protocol.unpack_verified(root / "qmc.tar.gz", root / "qmc.tar.gz.sha256", target, "qmc", config)


@pytest.mark.parametrize("name,kind", [("../escape", tarfile.REGTYPE), ("/escape", tarfile.REGTYPE), ("qmc/link", tarfile.SYMTYPE), ("logs.txt", tarfile.REGTYPE), ("qmc/a/../../escape", tarfile.REGTYPE)])
def test_unsafe_archives(tmp_path, config, name, kind):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        item = tarfile.TarInfo(name)
        item.type = kind
        item.linkname = "/tmp/target"
        tar.addfile(item, io.BytesIO())
    checksum = tmp_path / "checksum"
    checksum.write_text(protocol.digest(archive))
    with pytest.raises(ValueError):
        protocol.unpack_verified(archive, checksum, tmp_path / "out", "qmc", config)
    assert not (tmp_path / "out").exists()


class MockLauncher(local.Launcher):
    def prepare_source(self, directory):
        return Path(directory) / "source.bundle"

    def setup_from_bundle(self, bundle):
        self.events.append("setup_bundle")

    def __init__(self, config, output, failure=None, existing=False, shutdown_failure=False):
        self.events, self.active, self.failure = [], existing, failure
        self.shutdown_failure = shutdown_failure
        super().__init__(config, output, run=self.fake, sleep=lambda _: None)

    def fake(self, argv, timeout):
        if argv[0] == "git":
            return self.config["source_commit"] + "\trefs/heads/main\n"
        action = argv[1]
        self.events.append(action)
        if action == "sessions":
            return f"[{self.session}] owned | Hardware: A100 | Variant: GPU\n[unrelated] other | Hardware: L4 | Variant: GPU" if self.active else "[unrelated] other | Hardware: L4 | Variant: GPU"
        if action == "run":
            self.active = True
            if self.failure == "setup":
                raise subprocess.CalledProcessError(7, argv)
        if action == "stop":
            assert argv[-1] == self.session
            if self.shutdown_failure:
                raise RuntimeError("cannot stop")
            self.active = False
        return ""

    def start_stage(self, stage):
        self.events.append("start:" + stage)
        # All prior stage downloads must have been verified before starting.
        for previous in protocol.STAGES[:protocol.STAGES.index(stage)]:
            assert self.status["stages"][previous]["download"] == "complete"

    def wait_stage(self, stage):
        if self.failure == stage:
            raise subprocess.CalledProcessError(9, [stage])
        self.status["stages"][stage]["execution"] = "complete"

    def download(self, stage, partial=False):
        self.events.append("download:" + stage + (":partial" if partial else ""))
        if self.failure == "download:" + stage and not partial:
            raise ValueError("corrupt archive")
        if stage in protocol.STAGES:
            self.status["stages"][stage]["download"] = "partial" if partial else "complete"

    def execute(self, code, timeout=None):
        self.events.append("snapshot")

    def stream_log(self):
        self.events.append("logs")


@pytest.fixture(autouse=True)
def restore_signals():
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in original.items():
        signal.signal(sig, handler)


def test_stage_order_incremental_downloads_ownership(config, tmp_path):
    launcher = MockLauncher(config, tmp_path)
    assert launcher.launch() == 0
    assert [e for e in launcher.events if e.startswith(("start:", "download:"))] == ["download:root", "start:qmc", "download:qmc", "download:root", "start:hilbert_sort", "download:hilbert_sort", "download:root", "start:sqmc", "download:sqmc", "download:root"]
    assert launcher.status["status"] == "complete"
    assert launcher.status["shutdown"] == "verified_stopped"
    assert launcher.events.count("run") == 1


@pytest.mark.parametrize("failure,code", [("setup", 7), ("qmc", 9), ("hilbert_sort", 9), ("download:qmc", 1)])
def test_partial_failure_stops_later_stages(config, tmp_path, failure, code):
    launcher = MockLauncher(config, tmp_path, failure=failure)
    assert launcher.launch() == code
    assert "start:sqmc" not in launcher.events
    assert launcher.status["status"] == "failed"
    assert launcher.status["shutdown"] == "verified_stopped"
    assert "download:root:partial" in launcher.events
    assert launcher.events.index("download:root:partial") < launcher.events.index("stop")


def test_session_collision_never_stops_existing(config, tmp_path):
    launcher = MockLauncher(config, tmp_path, existing=True)
    assert launcher.launch() == 1
    assert "run" not in launcher.events and "stop" not in launcher.events


def test_original_error_survives_cleanup_error(config, tmp_path):
    launcher = MockLauncher(config, tmp_path, failure="qmc", shutdown_failure=True)
    assert launcher.launch() == 9
    assert "CalledProcessError" in launcher.status["error"]
    assert launcher.status["shutdown"] == "failed"
    assert any("Shutdown" in e for e in launcher.status["secondary_errors"])


def test_incomplete_stage_rejected(tmp_path, config):
    with pytest.raises(ValueError, match="Missing"):
        protocol.validate_stage(tmp_path, "qmc", config)


def test_root_bundle_never_overwrites_local_log(tmp_path, config):
    remote, output = tmp_path / "remote", tmp_path / "output"
    remote.mkdir()
    output.mkdir()
    protocol.write_json(remote / "comparison_config.json", config)
    protocol.write_json(remote / "run_config.json", {"source_commit": config["source_commit"], "config_sha256": protocol.config_digest(config)})
    (remote / "remote_logs.txt").write_text("remote")
    (remote / "logs.txt").write_text("must not copy")
    protocol.make_archive(remote, "root", config)
    (output / "logs.txt").write_text("local")
    def fake(argv, timeout):
        import shutil
        shutil.copyfile(remote / Path(argv[-2]).name, argv[-1])
        return ""
    launcher = local.Launcher(config, output, run=fake)
    launcher.download("root")
    assert (output / "logs.txt").read_text() == "local"
    assert (output / "remote_logs.txt").read_text() == "remote"


def test_endpoint_change_is_not_stopped(config, tmp_path):
    launcher = MockLauncher(config, tmp_path, existing=True)
    launcher.endpoint = "original-endpoint"
    with pytest.raises(RuntimeError, match="unowned"):
        launcher.shutdown()
    assert "stop" not in launcher.events


def test_shutdown_must_disappear_from_server(config, tmp_path):
    launcher = MockLauncher(config, tmp_path, existing=True)
    launcher.endpoint = "owned"
    original = launcher.fake
    def run(argv, timeout):
        if argv[1] == "stop":
            return "pretended to stop"
        return original(argv, timeout)
    launcher.run = run
    with pytest.raises(RuntimeError, match="remains active"):
        launcher.shutdown()
    assert launcher.status["shutdown"] != "verified_stopped"


def test_interrupt_salvages_before_shutdown(config, tmp_path):
    launcher = MockLauncher(config, tmp_path)
    def interrupted(stage):
        raise KeyboardInterrupt
    launcher.wait_stage = interrupted
    assert launcher.launch() == 130
    assert "download:qmc:partial" in launcher.events
    assert "start:hilbert_sort" not in launcher.events
    assert launcher.status["shutdown"] == "verified_stopped"


def test_duplicate_and_changed_member_rejected(config, tmp_path):
    root = bundle(tmp_path, config)
    archive = root / "qmc.tar.gz"
    with tarfile.open(archive, "r:gz") as tar:
        parts = [(m, tar.extractfile(m).read()) for m in tar.getmembers()]
    for duplicate in (False, True):
        with tarfile.open(archive, "w:gz") as tar:
            for member, content in parts:
                if member.name == "qmc/logs.txt" and not duplicate:
                    content = b"changed!!"
                    member.size = len(content)
                tar.addfile(member, io.BytesIO(content))
            if duplicate:
                tar.addfile(parts[0][0], io.BytesIO(parts[0][1]))
        (root / "qmc.tar.gz.sha256").write_text(protocol.digest(archive))
        with pytest.raises(ValueError, match="Duplicate|Member checksum"):
            protocol.unpack_verified(archive, root / "qmc.tar.gz.sha256", tmp_path / "out", "qmc", config)


def test_remote_worker_timeout_kills_child(tmp_path, monkeypatch):
    import run_comparison_gpu as remote
    monkeypatch.setattr(remote, "REPO", tmp_path)
    with pytest.raises(TimeoutError):
        remote.run([sys.executable, "-c", "import time; print('starting', flush=True); time.sleep(20)"], timeout=.15)


def test_colab_bootstrap_allows_cli_prelude():
    # colab run prepends imports and sys.argv and provides no __file__.
    body = (SCRIPTS / "run_comparison_gpu.py").read_text()
    compile("import sys\nsys.argv = ['run_comparison_gpu.py']\n" + body, "colab_payload", "exec")


def test_source_bundle_contains_exact_commit_only(config, tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    def git(*args, cwd=source):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    git("init", "-b", config["repo_branch"])
    (source / "tracked.py").write_text("original")
    git("add", "tracked.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "fixture")
    config["source_commit"] = git("rev-parse", "HEAD")
    (source / "tracked.py").write_text("unstaged edit")
    (source / "untracked.txt").write_text("must not upload")
    monkeypatch.setattr(local, "REPO", source)
    output = tmp_path / "output"
    output.mkdir()
    launcher = local.Launcher(config, output)
    bundle = launcher.prepare_source(tmp_path)
    assert protocol.digest(bundle) == launcher.config["source_bundle_sha256"]
    clone = tmp_path / "clone"
    clone.mkdir()
    git("init", cwd=clone)
    git("fetch", str(bundle), "refs/heads/" + config["repo_branch"], cwd=clone)
    git("checkout", "--detach", "FETCH_HEAD", cwd=clone)
    assert git("rev-parse", "HEAD", cwd=clone) == config["source_commit"]
    assert (clone / "tracked.py").read_text() == "original"
    assert not (clone / "untracked.txt").exists()
    assert (source / "tracked.py").read_text() == "unstaged edit"


def test_fresh_qmc_config_contract(config):
    assert config['qmc']['modes'] == ['fresh']
    assert config['qmc']['scramble'] is True
    assert config['qmc']['implementations'] == ['jax', 'scipy']
    for edit in ({'modes': ['sample']}, {'scramble': False}, {'implementations': ['jax']}):
        invalid = copy.deepcopy(config)
        invalid['qmc'].update(edit)
        with pytest.raises(ValueError):
            protocol.validate_config(invalid)


def test_scipy_numerical_validation(config, tmp_path):
    from sqmc.comparison.benchmark_qmc import comparisons
    from validate_artifacts import validate_numerical
    config = copy.deepcopy(config)
    config['qmc'].update(dimensions=[2], n_values=[16], repeats=2)
    rows = []
    for sequence in ['sobol', 'halton']:
        for impl, backend, t in [('jax', 'cpu', 4.), ('jax', 'gpu', 2.), ('scipy', 'cpu', 6.)]:
            rows.append(dict(sequence=sequence, dimension=2, n=16, mode='fresh', scramble=True,
                             implementation=impl, backend=backend, median_seconds=t, samples_seconds=[t,t],
                             q25_seconds=t, q75_seconds=t, repetition_seeds=[4,5]))
    jax_pairs, scipy_pairs = comparisons(rows)
    for name, content in [('results.json', rows), ('cpu_gpu_comparison.json', jax_pairs), ('scipy_comparison.json', scipy_pairs)]:
        protocol.write_json(tmp_path/name, content)
    validate_numerical(tmp_path, 'qmc', config)
    scipy_pairs[0]['scipy_over_jax'] = 100
    protocol.write_json(tmp_path/'scipy_comparison.json', scipy_pairs)
    with pytest.raises(ValueError, match='SciPy speedup'):
        validate_numerical(tmp_path, 'qmc', config)
