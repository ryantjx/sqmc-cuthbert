"""Metric computation for the SQMC–EKF comparison.

Predictions are scored per match on the shared score grid: truncation mass
(mass outside ``0..max_goals`` after normalization), three-outcome Brier score
vs the fixed ``2/3`` uniform baseline, exact-score accuracy, outcome accuracy,
and predictive log scores. Headline metrics are reported only over eligible
(typically 2026 World Cup) fixtures; a separate report covers all scored
prediction matches.
"""

from dataclasses import dataclass

import numpy as np

from rbsqmc.src.model.rbsmc.predict import evaluate_match_predictions


def build_records(dataset, grids, logp):
    """Build the per-match prediction-dict schema from grids + frame.

    ``grids`` are the normalized score grids over the held-out prediction
    split; ``logp`` their predictive log-probabilities. Rows align with
    ``dataset.frame`` rows in the prediction-split range.
    """
    rng_start = dataset.train_count + dataset.test_count
    frame_rows = dataset.frame.iloc[rng_start:rng_start + grids.shape[0]]

    records = []
    for (_, row), grid, lp in zip(frame_rows.iterrows(), grids, logp):
        G = grid.shape[0]
        ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
        grid = np.asarray(grid)
        prob_home = float(grid[ii > jj].sum())
        prob_draw = float(grid[ii == jj].sum())
        prob_away = float(grid[ii < jj].sum())
        flat = int(np.argmax(grid))
        pred_h, pred_a = flat // G, flat % G
        score_probabilities = [
            {"home": int(i), "away": int(j), "probability": float(grid[i, j])}
            for i in range(G) for j in range(G)
        ]
        records.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "home": row["home_team"],
            "away": row["away_team"],
            "actual_home_score": int(row["home_score"]),
            "actual_away_score": int(row["away_score"]),
            "predicted_home_score": pred_h,
            "predicted_away_score": pred_a,
            "log_likelihood": float(lp),
            "prob_home_win": prob_home,
            "prob_draw": prob_draw,
            "prob_away_win": prob_away,
            "score_probabilities": score_probabilities,
            "tournament": row.get("tournament", ""),
            "worldcup_eligible": bool(
                (row.get("tournament", "") == "FIFA World Cup")
                and (getattr(row.get("date"), "year", 0) == 2026)
            ),
        })
    return records


@dataclass
class Metrics:
    all: dict       # metrics over all scored prediction matches
    worldcup: dict  # metrics over 2026 World Cup fixtures only


def truncation_mass(grids):
    """Grids are normalized on 0..max_goals, so truncation mass is 0 by
    construction; reported for completeness against the raw (pre-normalization)
    GH mass in the EKF case."""
    return 0.0


def compute_metrics(records):
    scored = [r for r in records if r["actual_home_score"] >= 0 and r["actual_away_score"] >= 0]
    all_metrics = evaluate_match_predictions([dict(r) for r in records])
    wc = [r for r in scored if r["worldcup_eligible"]]
    wc_metrics = evaluate_match_predictions([dict(r) for r in wc]) if wc else None
    return Metrics(all=all_metrics, worldcup=wc_metrics)
