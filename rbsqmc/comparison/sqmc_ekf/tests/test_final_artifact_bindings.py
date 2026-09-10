"""Final-particle consistency and scalar-table column regression checks."""

import csv
import json

import jax.numpy as jnp
import numpy as np
import pytest

from rbsqmc.comparison.sqmc_ekf import run
from rbsqmc.comparison.sqmc_ekf.scripts import validate_sqmc_ekf_outputs as validator
from rbsqmc.comparison.sqmc_ekf.tests.test_integrity import complete_run
from rbsqmc.src.model.rbsmc.predict import predict_match_score_with_mass


@pytest.mark.parametrize("field,value", [
    ("raw_grid_mass", .123), ("current_scale", 999.),
    ("posterior_means", 999.), ("predictive_means", 999.),
    ("ess_after_update", 4.),
])
def test_final_diagnostic_corruption_rejected(tmp_path, field, value):
    complete_run(tmp_path)
    p = tmp_path / "results/sqmc_prediction_diagnostics.json"
    payload = json.loads(p.read_text())
    entry = payload["forecasts"][-1]
    if field.endswith("means"):
        entry[field]["home_attack"] = value
        # Preserve the previously checked predictive total identity.
        if field == "predictive_means":
            entry["total_strength_difference"]["mean"] = (
                value + entry[field]["home_defence"]
                - entry[field]["away_attack"] - entry[field]["away_defence"])
    else:
        entry[field] = value
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="[Dd]iagnostic"):
        validator.validate_artifacts(tmp_path)


def test_friendly_final_scale_and_weighted_means(tmp_path):
    cfg = complete_run(tmp_path)
    b = tmp_path / "results"
    fitted = json.loads((b / "sqmc/fitted_params.json").read_text())
    record = json.loads((b / "sqmc_predictions.json").read_text())[-1]
    record["tournament"] = "International FRIENDLY match"
    scale = validator._fixture_scale(record, fitted, dict(cfg, match_scale=1.5))
    assert scale == fitted["constrained"]["friendly_scale"]
    # Two equally frequent predictive coordinates, unequal posterior mass.
    particles = np.zeros((8, 2, 2))
    particles[:4, 1, 0] = 1.
    particles[4:, 1, 0] = 3.
    weights = np.array([.2] * 4 + [.05] * 4)
    alpha, beta = .2, -4.
    grid, raw_mass = predict_match_score_with_mass(
        jnp.asarray(particles), jnp.zeros(8), 1, 0, alpha, beta, 1, scale)
    grid = np.asarray(grid)
    record.update(prob_home_win=float(grid[1, 0]), prob_draw=float(np.trace(grid)),
                  prob_away_win=float(grid[0, 1]), score_probabilities=[
                      dict(home=h, away=a, probability=float(grid[h, a]))
                      for h in range(2) for a in range(2)])
    p = b / "sqmc_final_fixture.npz"
    np.savez(p, particles_x_predictive=particles, log_weights_predictive=np.zeros(8),
             log_weights_posterior=np.log(weights), home_id=1, away_id=0,
             scale=scale, alpha=alpha, beta=beta, max_goals=1)
    diagnostic = dict(raw_grid_mass=float(raw_mass), current_scale=scale,
                      predictive_means=dict(home_attack=2., home_defence=0., away_attack=0., away_defence=0.),
                      posterior_means=dict(home_attack=1.4, home_defence=0., away_attack=0., away_defence=0.),
                      ess_after_update=float(1 / (weights @ weights)))
    validator._validate_final_npz(p, cfg, records=[record],
                                 team_id_to_name={"0": "A", "1": "B"},
                                 final_scale=scale, checkpoint_alpha=alpha,
                                 checkpoint_beta=beta, final_diagnostic=diagnostic)


def test_final_scale_checked_when_worldcup_scope_excludes_final(tmp_path):
    cfg = complete_run(tmp_path)
    b = tmp_path / "results"
    records = json.loads((b / "sqmc_predictions.json").read_text())
    records[-1].update(tournament="Friendly", worldcup_eligible=False)
    p = b / "sqmc_prediction_diagnostics.json"
    payload = json.loads(p.read_text())
    payload["forecasts"] = payload["forecasts"][:-1]
    p.write_text(json.dumps(payload))
    metadata = json.loads((b / "dataset_metadata.json").read_text())
    # The NPZ still contains unit scale; a friendly needs the fitted scale.
    with pytest.raises(ValueError, match="scale does not match"):
        validator._validate_sqmc_diagnostics(b, metadata, cfg, records=records)


@pytest.mark.parametrize("mutation", ["swap_models", "swap_rows", "duplicate", "extra", "decoy"])
def test_scalar_table_requires_exact_rows_and_columns(tmp_path, mutation):
    rows = [dict(parameter=p, meaning=meaning[0], ekf=e, sqmc=s)
            for (p, meaning), e, s in zip(run._SCALAR_MEANINGS.items(),
                                         [.1, -3., .002, 1.5], [.3, -5., .004, 2.5])]
    run._write_scalar_latex(tmp_path, rows, dict(checkpoint_epoch=3),
                           dict(checkpoint_epoch=3), "run_with_underscores")
    p = tmp_path / "final_scalar_params_table.tex"
    validator._validate_scalar_table(p, rows)
    original = p.read_text()
    if mutation == "swap_models":
        changed = original.replace("0.1000 & 0.3000", "0.3000 & 0.1000")
    elif mutation == "swap_rows":
        lines = original.splitlines()
        a = next(i for i, line in enumerate(lines) if line.startswith(r"$\alpha$"))
        lines[a], lines[a + 1] = lines[a + 1], lines[a]
        changed = "\n".join(lines)
    elif mutation in ("duplicate", "extra"):
        row = next(line for line in original.splitlines() if line.startswith(r"$\alpha$"))
        if mutation == "extra":
            row = row.replace(r"$\alpha$", "covariance")
        changed = original.replace(r"\bottomrule", row + "\n" + r"\bottomrule")
    else:
        # Correct numbers elsewhere must not validate the wrong data row.
        changed = original.replace("0.1000 & 0.3000", "9.0000 & 8.0000")
        changed += "\n% expected numbers: 0.1000 0.3000\n"
    p.write_text(changed)
    with pytest.raises(ValueError, match="rows/Interpretation/model columns"):
        validator._validate_scalar_table(p, rows)


def test_derived_difference_checked_even_when_csv_matches_json(tmp_path):
    complete_run(tmp_path)
    b = tmp_path / "results"
    p = b / "final_scalar_params_comparison.json"
    payload = json.loads(p.read_text())
    payload["rows"][4]["sqmc_minus_ekf"] = 999.
    p.write_text(json.dumps(payload))
    csv_path = b / "final_scalar_params_comparison.csv"
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    rows[4]["sqmc_minus_ekf"] = "999.0"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="difference"):
        validator.validate_artifacts(tmp_path)
