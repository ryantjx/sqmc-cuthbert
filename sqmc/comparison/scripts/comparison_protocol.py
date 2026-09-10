"""Configuration, command and verified archive contract; standard library only."""
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from datetime import datetime, timezone

STAGES = ("qmc", "hilbert_sort", "sqmc")
REMOTE_REPO = Path("/content/rbsqmc")
ROOT_FILES = {"comparison_config.json", "run_config.json", "remote_status.json", "remote_logs.txt"}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_config(config):
    defaults = read_json(Path(__file__).parent / "config/comparison_config.json")
    allowed = set(defaults) | {"source_commit", "repo_branch", "run_id", "session_name", "resolved_utc", "source_bundle_sha256", "source_transport"}
    if set(config) - allowed or set(defaults) - set(config):
        raise ValueError("Unknown or missing top-level configuration fields")
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
    for stage in STAGES:
        args = config[stage]
        if set(args) != set(defaults[stage]):
            raise ValueError(f"Unknown or missing {stage} parameters")
        if args["platforms"] != ["cpu", "gpu"]:
            raise ValueError("Every Colab stage must request platforms [cpu, gpu]")
        for key in ("dimensions", "particle_counts" if stage == "sqmc" else "n_values"):
            values = args[key]
            if not isinstance(values, list) or not values or any(type(v) is not int or v < 1 for v in values) or len(set(values)) != len(values):
                raise ValueError(f"Invalid {stage}.{key}")
        max_dimension = 10000 if stage == "qmc" else 62
        if max(args["dimensions"]) > max_dimension:
            raise ValueError(f"{stage} dimension exceeds {max_dimension}")
        counts = args.get("n_values", args.get("particle_counts"))
        if any(n > 2**30 or n & (n - 1) for n in counts):
            raise ValueError("Counts must be powers of two up to 2**30")
        for key in ("repeats", "warmups", "seed", "n_steps", "datasets", "selection_reps", "validation_reps", "bootstrap_reps"):
            if key in args and (type(args[key]) is not int or args[key] < (0 if key in {"warmups", "seed"} else 1)):
                raise ValueError(f"Invalid {stage}.{key}")
        for key, choices in (("sequences", {"sobol", "halton"}), ("modes", {"sample", "fresh"})):
            if key in args and (not isinstance(args[key], list) or not args[key] or any(v not in choices for v in args[key]) or len(set(args[key])) != len(args[key])):
                raise ValueError(f"Invalid {stage}.{key}")
        if stage == "qmc" and (args["scramble"] is not True or args["modes"] != ["fresh"] or args["implementations"] != ["jax", "scipy"]):
            raise ValueError("QMC requires fresh scrambling with jax and scipy")
        if stage == "hilbert_sort" and args["distribution"] not in {"normal", "uniform"}:
            raise ValueError("Invalid Hilbert distribution")
        if stage == "sqmc":
            budgets = args["budget_seconds"]
            if not isinstance(budgets, list) or not budgets or any(type(b) not in (int, float) or not math.isfinite(b) or b <= 0 for b in budgets) or len(set(budgets)) != len(budgets):
                raise ValueError("Invalid SQMC budgets")


def stage_command(config, stage, output, python="python"):
    command = [python, "-u", "-m", f"sqmc.comparison.benchmark_{stage}"]
    for name, value in config[stage].items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + name.replace("_", "-"))
        else:
            command.append(flag)
            command.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    return command + ["--output-dir", str(output)]


def make_archive(root, kind, config):
    """Only regular files, with an independently downloaded archive checksum."""
    root = Path(root)
    files = sorted(p for p in (root / kind).rglob("*") if p.is_file()) if kind in STAGES else [root / name for name in sorted(ROOT_FILES) if (root / name).exists()]
    if any(p.is_symlink() for p in files):
        raise ValueError("Symlinks are not allowed in result bundles")
    manifest = {"kind": kind, "source_commit": config["source_commit"],
                "config_sha256": config_digest(config),
                "files": {str(p.relative_to(root)): {"sha256": digest(p), "size": p.stat().st_size} for p in files}}
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
            allowed = member.name == manifest_name or (len(name.parts) > 1 and name.parts[0] == kind if kind in STAGES else member.name in ROOT_FILES)
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
        # Manual extraction avoids version-dependent tar extraction behavior.
        for member in members:
            target = destination / member.name
            if any(p.is_symlink() for p in [target, *target.parents]):
                raise ValueError("Extraction destination contains a symlink")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    return manifest


def validate_stage(root, stage, config):
    """Check complete grids and exact effective configuration, not just exit code."""
    root = Path(root)
    required = {"logs.txt", "config.json", "metadata.json", "status.json", "results.json", "results.csv", "cpu_gpu_comparison.json", "runtime.png"}
    if stage == "qmc" and "implementations" in config[stage]:
        required.add("scipy_comparison.json")
    if stage == "sqmc":
        required |= {"accuracy_records.json", "budget_summary.json", "budget_summary.csv", "accuracy_vs_runtime.png"}
    if any(not (root / name).is_file() for name in required):
        raise ValueError(f"Missing {stage} artifacts")
    if read_json(root / "status.json")["status"] != "complete":
        raise ValueError(f"{stage} benchmark did not complete")
    actual = read_json(root / "config.json")
    actual.pop("output_dir")
    if actual != config[stage]:
        raise ValueError(f"{stage} parameters differ from effective config")
    if read_json(root / "metadata.json")["git_commit"] != config["source_commit"]:
        raise ValueError("Benchmark commit mismatch")
    rows = read_json(root / "results.json")
    args = config[stage]
    fields = ["backend", "dimension", "n"]
    values = [args["platforms"], args["dimensions"], args.get("n_values", args.get("particle_counts"))]
    if stage != "sqmc":
        fields.append("sequence")
        values.append(args["sequences"])
    if stage == "qmc":
        fields.append("mode")
        values.append(args["modes"])
    expected_grid = set(itertools.product(*values))
    actual_grid = [tuple(r[k] for k in fields) for r in rows]
    if stage == "qmc" and "implementations" in args:
        expected_grid = {("jax",) + v for v in expected_grid} | {("scipy", "cpu") + v[1:] for v in expected_grid}
        actual_grid = [(r["implementation"],) + v for r, v in zip(rows, actual_grid)]
        if read_json(root / "metadata.json").get("qmc_contract_version") != 2:
            raise ValueError("Missing QMC contract version")
        if any(r.get("scramble") is not True or r["mode"] != "fresh" for r in rows):
            raise ValueError("QMC requires fresh scrambled results")
    if len(rows) != len(expected_grid) or set(actual_grid) != expected_grid:
        raise ValueError(f"Incomplete {stage} CPU/GPU result grid")
    for row in rows:
        samples = row["samples_seconds"]
        if len(samples) != args["repeats"] or any(not math.isfinite(s) or s <= 0 for s in samples):
            raise ValueError("Invalid timing samples")
        import statistics
        if not math.isclose(row["median_seconds"], statistics.median(samples), rel_tol=1e-12):
            raise ValueError("Incorrect timing median")
    if stage == "sqmc":
        summaries = read_json(root / "budget_summary.json")
        expected = set(itertools.product(args["platforms"], args["dimensions"], args["budget_seconds"]))
        if len(summaries) != len(expected) or {(s["backend"], s["dimension"], s["budget_seconds"]) for s in summaries} != expected:
            raise ValueError("Incomplete budget grid")
        for summary in summaries:
            eligible = [r for r in rows if r["backend"] == summary["backend"] and r["dimension"] == summary["dimension"] and r["median_seconds"] <= summary["budget_seconds"]]
            if not eligible:
                if summary["status"] != "no_measured_configuration_within_budget":
                    raise ValueError("Infeasible budget marked selected")
                continue
            chosen = min(eligible, key=lambda r: (r["selection_accuracy"]["rmse"], r["median_seconds"], r["n"]))
            if summary["status"] != "selected" or summary["n"] != chosen["n"] or summary["median_seconds"] != chosen["median_seconds"] or summary["validation_rmse"] != chosen["validation_accuracy"]["rmse"]:
                raise ValueError("Budget selection does not match measured candidate results")
            if not (root / "accuracy_at_budget.png").exists():
                raise ValueError("Missing budget figure")
