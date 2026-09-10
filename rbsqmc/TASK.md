# TASK: `run_model_unbiased_colab.sh` — split GPU optimization + local filter/predict

> **Design:** Only the backward-gradient optimization (log-marginal maximization) runs on the Colab GPU. All filtering, plotting, and prediction compute runs locally. Running the filter twice (once on GPU for the baseline, once locally) is expected and fine.
>
> **Note:** `run_model_unbiased_gpu.py` currently runs the full pipeline — it needs an optimization-only mode to match this task.

## Prerequisites
- `colab` CLI installed and authenticated (needs GitHub credentials to clone the repo).
- Local `.venv` at repo root (the orchestrator activates it).
- Config: `rbpf/scripts/config/model_unbiased_gpu_config.json` — key knobs: `n_particles`, `n_epochs`, `n_reps`, `learning_rate`, `split_date`, `gpu_type`, `colab_timeout`, `output_dir`.
- Env overrides (optional): `GPU_TYPE`, `COLAB_TIMEOUT`, `SESSION`.
- Sanity check: `bash rbpf/scripts/run_model_unbiased_colab.sh --dry-run`.

## Train / test / predict split

The pipeline is split into three phases by **date** (time-series data):

- **Train** — matches on/before `split_date`. Used to fit the parameters via backward-gradient optimization.
- **Test** — matches after `split_date`. Held out during training; used to evaluate generalization (test logZ) each epoch.
- **Predict** — upcoming fixtures (from `rbpf/data/fixtures.json`), scored with the fitted params.

## Steps

1. `run_model_unbiased_colab.sh`
   1. Create folder: `rbpf/outputs/smoothing_unbiased_gpu_DDMMYYYY_HHMM`
   2. Setup Colab environment
      1. Check RAM, GPU model; print the parameters
   3. `run_model_unbiased_gpu.py` (optimization-only mode)
      1. Check required packages; install ONLY if missing
      2. Run the backward-gradient optimization (log-marginal maximization) on the **train** split
      3. Each epoch, evaluate logZ on the **test** split (forward filter only)
      4. Download results from GPU into the output folder
         1. `params_unbiased.json`
         2. `optimization_summary.json` — includes `train_logZ_history` and `test_logZ_history`
         3. `optimization_logZ_curve.png` — train + test logZ vs epoch on the same graph
         4. `run_config.json`
   4. Stop the Colab session (server no longer needed; remaining compute is local)
   5. Generate `optimization_logZ_curve.png` locally from the downloaded `optimization_summary.json` (train + test curves)
   6. Create folder: `rbpf/outputs/smoothing_unbiased_gpu_DDMMYYYY_HHMM/filtered`
   7. Run `model.py` locally — generate filtered states. Takes a params path and a run-config path.
      1. Outputs into `filtered/`
         1. `filter_states.npz`
         2. `timeseries_states.json`
      2. Generate output images
         1. `final_rankings.png`
         2. `timeseries_states.png`
         3. `top_strengths.png`
   8. Create folder: `rbpf/outputs/smoothing_unbiased_gpu_DDMMYYYY_HHMM/predict`
   9. Run `predict.py` locally — sequential prediction using the model's filtered states. Takes `predictions.json` and `filter_states.npz`.
      1. Generate predictions in `predict/`
         1. `predictions.json`
         2. `post_prediction_filter_rankings.json` — final rankings after running the filter
      2. Generate per-match images in `predict/prediction_plots/`
   10. Download logs into `rbpf/outputs/smoothing_unbiased_gpu_DDMMYYYY_HHMM`
   11. Stop the Colab session (final cleanup)

## Output

All outputs land under `rbpf/outputs/smoothing_unbiased_gpu_DDMMYYYY_HHMM/` (the run folder), with filter outputs in `filtered/` and prediction outputs in `predict/`.

### Files
- `params_unbiased.json` — fitted model parameters from the backward-gradient optimization.
- `optimization_summary.json` — optimization metrics: `baseline_logZ`, `best_logZ`, `final_filter_logZ`, plus `train_logZ_history`, `test_logZ_history`, `gradient_norm_history`, and `regularizer_history` (one entry per epoch).
- `run_config.json` — the resolved run configuration (dates, particles, epochs, split, output dir).
- `filtered/filter_states.npz` — the final filtered particle states (attack/defense per team over time).
- `filtered/timeseries_states.json` — filtered state trajectories serialized as JSON.
- `predict/predictions.json` — per-match sequential predictions with score probabilities and accuracy metrics.
- `predict/post_prediction_filter_rankings.json` — final team rankings after running the filter on the predictions.
- `run_<YYYYMMDD_HHMMSS>.log` — per-run log of the orchestrator and Colab output.

### Plots
- `optimization_logZ_curve.png` — train and test logZ vs epoch on the same axes, with train/test match counts annotated.
- `gradient_norm_curve.png` — gradient norm per epoch (convergence / instability check).
- `regularizer_curve.png` — inverse-Wishart prior contribution per epoch (prior-vs-data balance).
- `filtered/final_rankings.png` — final team rankings by attack/defense strength.
- `filtered/timeseries_states.png` — attack/defense trajectories over time for top teams.
- `filtered/top_strengths.png` — top teams by final strength.
- `predict/prediction_plots/` — one per-match image (outcome probabilities + score heatmap).

## Acceptance criteria
- All artifacts present and non-empty (see manifest above).
- `optimization_summary.json` shows `best_logZ` improved over `baseline_logZ` on the **train** split.
- `optimization_summary.json` contains both `train_logZ_history` and `test_logZ_history` (same length as `n_epochs`).
- `optimization_summary.json` contains `gradient_norm_history` and `regularizer_history` (same length as `n_epochs`).
- `optimization_logZ_curve.png` shows **two** curves (train + test) on the same axes, labeled, with the train/test match counts annotated.
- `gradient_norm_curve.png` and `regularizer_curve.png` are present and non-empty.
- Validator passes: `python rbpf/scripts/validate_model_unbiased_outputs.py <output_dir>`.

### Required config additions (`model_unbiased_gpu_config.json`)
- `split_date` — e.g. `"2025-01-01"`. Matches on/before this date are train; after are test.
- `test_n_particles` — particle count for the test logZ evaluation (can differ from `n_particles` to save GPU time).

## Failure handling
- Downloads retry 3× with 5s backoff; fail if an artifact is still empty.
- On any failure, the Colab session is stopped via the `cleanup` trap.
- Per-run log written to `<output_dir>/run_<YYYYMMDD_HHMMSS>.log`.