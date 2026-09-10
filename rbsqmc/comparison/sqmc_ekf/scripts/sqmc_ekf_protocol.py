"""Configuration, command and verified archive contract for the SQMC–EKF run.

Standard-library only, mirroring ``sqmc/comparison/scripts/comparison_protocol.py``
but for a single comparison run (no multi-stage benchmark). The run is archived
as ``run.tar.gz`` (the ``results/`` and ``images/`` subfolders) plus a
``root.tar.gz`` (config, settings, compute setup, status, logs).
"""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from datetime import datetime, timezone

REMOTE_REPO = Path("/content/rbsqmc")
ROOT_FILES = {
    "comparison_config.json",
    "run_config.json",
    "remote_status.json",
    "remote_logs.txt",
}
# Subfolders of the run directory that are archived together as the run bundle.
RUN_DIRS = ("results", "images")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def progress_snapshot(root, offset=0, stream_logs=True):
    """Read on the VM; return status and only newly appended progress lines."""
    root = Path(root)
    state = read_json(root / "remote_status.json")["run"]
    lines = []
    if stream_logs:
        pattern = re.compile(r"^(?:ekf|sqmc)(?:: compiled in| epoch \d+/\d+:)|^Comparison complete")
        with (root / "remote_logs.txt").open("rb") as log:
            if offset < 0 or offset > log.seek(0, 2):
                raise ValueError("Remote log shrank or progress offset is invalid")
            log.seek(offset)
            # Leave an unfinished line for the next poll, including a UTF-8
            # character split across writes. Never advance past unseen bytes.
            for raw in log:
                if not raw.endswith(b"\n"):
                    break
                offset += len(raw)
                line = raw.decode("utf-8", errors="replace")
                if pattern.search(line):
                    lines.append(line)
    return {"run": state, "offset": offset, "lines": lines}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config_digest(config):
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_config(config):
    """Validate the effective run configuration (single-run protocol)."""
    allowed = {
        "training_start_date", "test_start_date", "prediction_start_date",
        "n_particles", "max_goals", "seed", "n_epochs", "learning_rate",
        "n_reps", "include_friendly", "teams", "match_scale",
        "gauss_hermite_degree", "gpu", "gpu_type", "colab_timeout",
        "setup_timeout", "transfer_timeout", "session", "repo_url",
        "repo_branch", "source_commit", "run_id", "session_name",
        "resolved_utc", "source_bundle_sha256", "source_transport", "smoke",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown configuration fields: {set(config) - allowed}")
    if "smoke" in config and type(config["smoke"]) is not bool:
        raise ValueError("smoke must be a boolean")
    if config["gpu"] not in {"A100", "H100", "T4", "L4", "G4"}:
        raise ValueError("Unsupported GPU")
    for key in ("colab_timeout", "setup_timeout", "transfer_timeout"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["session"]):
        raise ValueError("session must contain only letters, digits, underscore or hyphen")
    if not isinstance(config["repo_url"], str) or not config["repo_url"].startswith("https://"):
        raise ValueError("repo_url must be an HTTPS URL")
    if "source_commit" in config and not re.fullmatch(r"[0-9a-f]{40}", config["source_commit"]):
        raise ValueError("source_commit must be a full Git SHA")
    if "source_bundle_sha256" in config and not re.fullmatch(r"[0-9a-f]{64}", config["source_bundle_sha256"]):
        raise ValueError("Invalid source bundle checksum")
    for key in ("n_particles", "max_goals", "n_epochs", "n_reps"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(config["learning_rate"]) not in (int, float) or not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")


def make_archive(root, kind, config):
    """Archive the run (results/images) or the root metadata files.

    Only regular files are included, with an independently downloaded archive
    checksum. ``kind`` is ``"run"`` (the results/images subfolders) or
    ``"root"`` (the top-level metadata files).
    """
    root = Path(root)
    if kind == "run":
        files = sorted(
            p for d in RUN_DIRS for p in (root / d).rglob("*") if p.is_file()
        )
    else:
        files = [root / name for name in sorted(ROOT_FILES) if (root / name).exists()]
    if any(p.is_symlink() for p in files):
        raise ValueError("Symlinks are not allowed in result bundles")
    manifest = {
        "kind": kind,
        "source_commit": config["source_commit"],
        "config_sha256": config_digest(config),
        "files": {str(p.relative_to(root)): {"sha256": digest(p), "size": p.stat().st_size} for p in files},
    }
    manifest_path = root / f"{kind}_manifest.json"
    write_json(manifest_path, manifest)
    archive = root / f"{kind}.tar.gz"
    temporary = root / f"{kind}.tar.gz.tmp"
    with tarfile.open(temporary, "w:gz") as tar:
        tar.add(manifest_path, arcname=manifest_path.name, recursive=False)
        for path in files:
            tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
    temporary.replace(archive)
    (root / f"{kind}.tar.gz.sha256").write_text(digest(archive) + "\n")


def unpack_verified(archive, checksum, destination, kind, config):
    """Validate every member BEFORE extracting anything; reject links/traversal."""
    if digest(archive) != Path(checksum).read_text().strip():
        raise ValueError("Archive checksum mismatch")
    destination = Path(destination).resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive path")
        manifest_name = f"{kind}_manifest.json"
        for member in members:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts or "\\" in member.name or str(name) != member.name:
                raise ValueError("Unsafe archive member")
            if kind == "run":
                allowed = member.name == manifest_name or (
                    len(name.parts) > 1 and name.parts[0] in RUN_DIRS
                )
            else:
                allowed = member.name == manifest_name or member.name in ROOT_FILES
            if not allowed:
                raise ValueError("Unexpected archive path")
        manifest = json.load(tar.extractfile(manifest_name))
        if manifest["kind"] != kind or manifest["source_commit"] != config["source_commit"] or manifest["config_sha256"] != config_digest(config):
            raise ValueError("Archive provenance mismatch")
        if set(names) != set(manifest["files"]) | {manifest_name}:
            raise ValueError("Manifest file list mismatch")
        for name, metadata in manifest["files"].items():
            with tar.extractfile(name) as stream:
                if tar.getmember(name).size != metadata["size"] or hashlib.file_digest(stream, "sha256").hexdigest() != metadata["sha256"]:
                    raise ValueError("Member checksum mismatch")
        for member in members:
            target = destination / member.name
            if any(p.is_symlink() for p in [target, *target.parents]):
                raise ValueError("Extraction destination contains a symlink")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    return manifest


def validate_run(root, config, methods=("ekf", "sqmc")):
    """Use the same content checks before archiving and after downloading."""
    if __package__:
        from .validate_sqmc_ekf_outputs import validate_artifacts, validate_partial
    else:
        from validate_sqmc_ekf_outputs import validate_artifacts, validate_partial
    if set(methods) == {"ekf", "sqmc"}:
        validate_artifacts(root, config)
    elif len(methods) == 1 and methods[0] in {"ekf", "sqmc"}:
        validate_partial(root, config, methods[0])
    else:
        raise ValueError("Invalid methods for artifact validation")
