"""Evidence-based draft writer for the SQMC–EKF comparison."""

import json
import os
from datetime import datetime


def _fmt(v):
    return "N/A" if v is None else f"{v:.4g}"


def _metrics_table(metrics):
    rows = [
        ("mean_brier_score", "Mean Brier score"),
        ("uniform_reference_brier_score", "Uniform reference Brier (2/3)"),
        ("brier_skill_score_vs_uniform", "Brier skill score vs uniform"),
        ("mean_log_likelihood", "Mean predictive log score"),
        ("exact_score_accuracy", "Exact-score accuracy"),
        ("outcome_accuracy", "Outcome accuracy"),
        ("n_scored", "Scored matches"),
    ]
    lines = ["| Metric | Value |", "|---|---|"]
    for key, label in rows:
        lines.append(f"| {label} | {_fmt(metrics.get(key))} |")
    return "\n".join(lines)


def _headline(m):
    """World Cup-only headline metrics when present, else all-scored."""
    return m.worldcup if m.worldcup is not None else m.all


def write_report(output_dir, dataset, results):
    """Write the run report and an evidence-based draft markdown file."""
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ekf = results["ekf"]
    sqmc = results["sqmc"]

    report = []
    report.append("# SQMC–EKF Comparison Report")
    report.append("")
    report.append(f"Generated: {now}")
    report.append(f"Dataset rows: {dataset.metadata['source_rows']} "
                  f"(train {dataset.train_count}, test {dataset.test_count}, "
                  f"prediction count {dataset.metadata['prediction_count']}).")
    report.append(f"Config: {json.dumps({k: v for k, v in results['cfg'].items()})}")
    report.append("")
    report.append("## Training")
    report.append("")
    report.append("| | EKF | SQMC |")
    report.append("|---|---|---|")
    report.append(f"| Final train logZ | {_fmt(ekf['summary']['final_train_logz'])} "
                  f"| {_fmt(sqmc['summary']['final_train_logz'])} |")
    report.append(f"| Final test logZ | {_fmt(ekf['summary']['final_test_logz'])} "
                  f"| {_fmt(sqmc['summary']['final_test_logz'])} |")
    report.append(f"| Best test logZ (epoch) | {_fmt(ekf['summary']['best_test_logz'])} "
                  f"({ekf['summary']['best_test_epoch']}) | "
                  f"{_fmt(sqmc['summary']['best_test_logz'])} ({sqmc['summary']['best_test_epoch']}) |")
    report.append(f"| Compile (s) / execute (s) | {_fmt(ekf['summary']['compilation_sec'])} / "
                  f"{_fmt(ekf['summary']['execution_sec'])} | {_fmt(sqmc['summary']['compilation_sec'])} / "
                  f"{_fmt(sqmc['summary']['execution_sec'])} |")
    report.append("")
    report.append("> Note on logZ: the EKF value is a **Gaussian-approximate** "
                  "normalising constant from moment filtering; the SQMC value is a "
                  "particle likelihood estimate. They are not directly comparable "
                  "as exact marginal likelihoods, so any raw gap must not be "
                  "interpreted as a superiority verdict on its own.")
    report.append("")
    report.append("## Headline prediction metrics (World Cup 2026 eligible)")
    report.append("")
    report.append("### EKF")
    report.append(_metrics_table(_headline(ekf["metrics"])))
    report.append("")
    report.append("### SQMC")
    report.append(_metrics_table(_headline(sqmc["metrics"])))
    report.append("")
    report.append("## Verdict (evidence-based)")
    report.append("")
    report.append(_write_verdict(ekf["metrics"], sqmc["metrics"]))
    report.append("")

    with open(os.path.join(output_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(report))

    draft = report + _write_draft_sections(ekf, sqmc, dataset, results["cfg"])
    # Draft is the report plus a longer narrative; store separately.
    with open(os.path.join(output_dir, "DRAFT.md"), "w") as f:
        f.write("\n".join(draft))


def _write_verdict(ekf_metrics, sqmc_metrics):
    """Compose the evidence-based verdict, distinguishing logZ caveats."""
    e_metrics = _headline(ekf_metrics)
    s_metrics = _headline(sqmc_metrics)

    lines = []
    brier_e = e_metrics.get("mean_brier_score")
    brier_s = s_metrics.get("mean_brier_score")
    if brier_e is not None and brier_s is not None:
        better = "EKF" if brier_e < brier_s else "SQMC" if brier_s < brier_e else "tied"
        lines.append(
            f"- On the three-outcome Brier score, {better} had the lower mean "
            f"({_fmt(min(brier_e, brier_s))} vs {_fmt(max(brier_e, brier_s))})."
        )

    acc_e = e_metrics.get("outcome_accuracy")
    acc_s = s_metrics.get("outcome_accuracy")
    if acc_e is not None and acc_s is not None:
        lines.append(
            f"- Outcome accuracy was {_fmt(acc_e)} (EKF) vs {_fmt(acc_s)} (SQMC)."
        )

    lines.append(
        "- The test logZ gap alone is not treated as a correctness verdict: "
        "the EKF reports a Gaussian-approximate logZ and the SQMC a particle "
        "estimate, so differences may reflect approximation quality and "
        "estimation noise rather than only the modelled correlations."
    )
    lines.append(
        "- Any remaining differences are attributed to correlated vs factorial "
        "dynamics only after the shared-input, shared-optimizer protocol is "
        "confirmed and prediction metrics are inspected."
    )
    return "\n".join(lines)


def _write_draft_sections(ekf, sqmc, dataset, cfg):
    return [
        "",
        "## Method",
        "",
        "Both methods consume one chronological, per-match dataset "
        "(train `[1980-01-01, 2024-01-01)`, test `[2024-01-01, 2026-06-11)`, "
        "prediction from June 11 2026). Same-day repeats are kept as separate "
        "`dt=0` rows. The EKF propagates only the two playing teams per match "
        "by each team's last-appearance gap; the SQMC shared-`dt` comparator "
        "propagates the full covariance per row.",
        "",
        "Both used Adam with cosine decay, seed 0 and exactly "
        f"{cfg['n_epochs']} updates. SQMC gradients were averaged over "
        f"{cfg['n_reps']} independently randomised replicas.",
        "",
        "## Limitations",
        "",
        "- EKF logZ is a Gaussian approximation; SQMC logZ is a particle "
        "estimate with finite-sample error.",
        "- Different time-step conventions (only-playing-teams vs shared-dt) "
        "mean the two logZ scales are not exactly aligned.",
        "- Headline prediction metrics are restricted to 2026 World Cup "
        "fixtures; other prediction matches are reported for completeness.",
    ]
