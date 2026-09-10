import os
import jax
import pandas as pd
import json

from rbsqmc.src.data.data import get_results, get_training_data, concat_football_results
from rbsqmc.src.model.rbsmc.optimization import (
    run_filter_unbiased,
    logmarginal_maximize,
)
from rbsqmc.src.utils.helpers import default_init_params, resolve_teams, save_params
from rbsqmc.src.utils.graphic import (
    plot_all,
    plot_gradient_norm_curve,
    plot_logmarginal_history_train_test,
)
from datetime import datetime

def main():
    date_text = datetime.now().strftime("gd_%Y%m%d_%H%M%S")
    output_dir = f"rbsqmc/outputs/train_model/{date_text}/"

    cfg = {
        "training_start_date": "1980-01-01",
        "test_start_date": "2024-01-01",
        "prediction_start_date": "2026-06-11",
        "n_particles": 500,          # N
        "max_goals": 8,               # MAX_GOALS
        "seed": 0,                    # PRNG seed
        # optimization
        "n_epochs": 100,
        "learning_rate": 0.05,
        "n_reps": 30,
        "patience": 15,
        # "gamma_0_prior_params" : {
        #     "scale" : 1.0,
        #     "dof" : 5.0,
        #     "strength" : 1.0
        # },
        # data / output
        "include_friendly": True,
        "teams": "worldcup2026",
        "output_dir": output_dir,
    }
    if not os.path.exists(cfg["output_dir"]):    
        os.makedirs(cfg["output_dir"], exist_ok=True)
        print(f"Created output directory: {cfg['output_dir']}")
    else:
        print(f"Output directory already exists: {cfg['output_dir']}")

    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote run config to {os.path.join(output_dir, 'run_config.json')}")

    key = jax.random.PRNGKey(cfg["seed"])
    teams_only = resolve_teams(cfg)
    (train_df, test_df, prediction_df), (train_model_inputs, test_model_inputs, prediction_model_inputs), team_id_to_name = get_training_data(
        train_start_date=cfg["training_start_date"],
        test_start_date=cfg["test_start_date"],
        prediction_start_date=cfg["prediction_start_date"],
        max_goals=cfg["max_goals"],
        include_friendly=cfg["include_friendly"],
        teams_only=teams_only,
    )
    print(f"Extracted training data:")
    print(f"  Training data: {len(train_df)} matches. Training data from {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"  Test data: {len(test_df)} matches. Test data from {test_df['date'].min().date()} to {test_df['date'].max().date()}")
    print(f"  Prediction data: {len(prediction_df)} matches. Prediction data from {prediction_df['date'].min().date()} to {prediction_df['date'].max().date()}")

    num_teams = len(team_id_to_name)
    params = default_init_params(num_teams=num_teams, team_id_to_name=team_id_to_name)

    ############# LOG MARGINALIZATION OPTIMIZATION ################
    key, opt_key = jax.random.split(key, 2)
    (best_params, train_logz_history, test_logz_history, grad_norm_history) = logmarginal_maximize(
        key=opt_key,
        train_model_inputs=train_model_inputs,       # train on the train split
        test_model_inputs=test_model_inputs,   # score the held-out test split each epoch
        params=params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
        n_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"],
        n_reps=cfg["n_reps"],
        gamma_0_prior_params=cfg.get("gamma_0_prior_params"),
        patience=cfg.get("patience"),
    )
    train_logz = [float(v) for v in train_logz_history]
    test_logz = [float(v) for v in test_logz_history]
    grad_norms = [float(v) for v in grad_norm_history]

    # plot the logZ history
    plot_logmarginal_history_train_test(
        train_logz_history=train_logz_history,
        train_match_count=int(train_model_inputs.match_mask.sum()),
        test_logz_history=test_logz_history,
        test_match_count=int(test_model_inputs.match_mask.sum()),
        save_path=os.path.join(cfg["output_dir"], "logmarginal_history_train_test.png"),
    )
    plot_gradient_norm_curve(
        grad_norm_history=grad_norm_history,
        save_path=os.path.join(output_dir, "gradient_norm_curve.png"),
    )
    # save best params to output_dir
    save_params(
        params=best_params,
        path=os.path.join(cfg["output_dir"], "best_params.json")
    )
    ############# FILTERING WITH OPTIMIZED PARAMETERS ################
    # Concatenate train + test so the final filtered state reflects everything
    # through the end of the test split.
    observed_inputs = concat_football_results(train_model_inputs, test_model_inputs)
    key, filter_key = jax.random.split(key, 2)
    final_states, final_model_inputs_rbpf = run_filter_unbiased(
        key=filter_key,
        model_inputs=observed_inputs,  # filter on train + test
        params=best_params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
    )

    os.makedirs(os.path.join(cfg["output_dir"], "final_filter"), exist_ok=True)
    # Real match dates for the timeseries x-axis (train then test).
    full_dates = pd.concat([train_df["date"], test_df["date"]]).to_numpy()
    plot_all(
        filtered_states=final_states,
        augmented_results=final_model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=os.path.join(cfg["output_dir"], "final_filter"),
        timestamps=full_dates,  # date x-axis for timeseries
        params=best_params,
    )

    ############### Run Sequential Prediction for Upcoming Matches ###############
    key, pred_key = jax.random.split(key, 2)
    from rbsqmc.src.model.rbsmc.predict import run_sequential_predict
    from rbsqmc.src.utils.helpers import (
        build_match_predictions,
        save_match_predictions,
    )

    pred_grids, pred_logprobs, daily_logp = run_sequential_predict(
        key=pred_key,
        observed_inputs=observed_inputs,  # filter on train + test
        prediction_inputs=prediction_model_inputs,  # predict on upcoming matches
        params=best_params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
    )

    # Build per-match prediction dicts (with win/draw/lose percentages) and save.
    predictions = build_match_predictions(
        all_grids=pred_grids,
        all_logp_actual=pred_logprobs,
        prediction_inputs=prediction_model_inputs,
        team_id_to_name=team_id_to_name,
        max_goals=cfg["max_goals"],
    )
    pred_dir = os.path.join(cfg["output_dir"], "prediction")
    # Saves per-match JSONs + predictions.json into pred_dir, and the per-match
    # outcome/score-heatmap plots into pred_dir/prediction_plots/.
    save_match_predictions(
        predictions,
        save_dir=pred_dir,
        max_goals=cfg["max_goals"],
    )

    ############### Observe team states over time (post-prediction) ###############
    # Plots selected teams' filtered attack/defense/total strengths across the
    # full train+test+prediction sequence, with vertical lines at each match in
    # the prediction window. Defaults to Spain/England/France/Argentina; override
    # with cfg["observe_teams"] = [...]. Writes into pred_dir.
    from rbsqmc.src.model.rbsmc.observe import run_observe

    run_observe(
        cfg=cfg,
        params=best_params,
        output_dir=pred_dir,
    )


if __name__ == "__main__":
    main()