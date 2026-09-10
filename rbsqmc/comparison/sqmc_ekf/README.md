# SQMC vs EKF

Compare the **correlated RB-SQMC** model against the **factorial EKF** model on identical training configurations.

## Background

The factorial EKF model (from [cuthberto-carlos](https://github.com/state-space-models/cuthberto-carlos), documented in `dissertation/drafts/chapters/3_football_model_with_sqmc.tex`) assumes teams evolve **independently** (factorial state-space model). The RB-SQMC model relaxes this by allowing **correlations between the evolution of latent states** through a non-diagonal initial covariance $\Sigma_0 = \Gamma_0 \otimes B$, giving a better model of team evolution.

## Task

1. **Replicate** the factorial EKF model in `rbsqmc/src/model/ekf` — filtering via the `cuthbert` package, gradient descent via `optax`, and forward filtering of individual games.
2. **Compare** SQMC vs EKF under **identical configurations**:
   - Same train / test / predict date splits.
   - Same optimization steps (gradient descent, epochs, learning rate, seed).
3. **Deploy** to Colab, using GPU wherever available. Outputs go to `rbsqmc/comparison/sqmc_ekf/outputs/DDMMYYYY_HHMM/`.
4. **Evaluate** both models (see [Evaluation](#evaluation)).
5. **Write the draft** based on the evaluation results.

The Colab launcher polls every 15 seconds, transferring status and only newly
completed epoch/compilation lines filtered on the VM. Full logs arrive with the
verified final artifacts (or during failure recovery). `--no-stream-logs` polls
status only. Transport failures get up to five attempts; model failures still
fail the run. The launcher refreshes the existing session's expiring proxy
credentials through the installed Colab CLI's Python environment before expiry.
Remote commands use the CLI's transport directly, without its `exec` handler
that deletes session records and kills keep-alive on 401/404 errors. Credentials
are refreshed before each remote execution. If the CLI cache entry is missing,
reconnection restores it and keep-alive only after Colab confirms the saved
endpoint is still assigned; it never allocates a replacement runtime.

The default shell command runs EKF on the local CPU, RB-SQMC on the Colab GPU,
then combines and validates both outputs locally. Results are stored in
`outputs/DDMMYYYY_HHMM/{ekf,sqmc,combined}/`; each method's hardware metadata is
also retained under `combined/results/{ekf,sqmc}/run_config.json`.

After worker dispatch, a monitoring failure or Ctrl+C leaves the Colab session
untouched and records `detached`, preserving the last observed execution state.
Reconnect with the parent output directory to collect SQMC and combine it with
the completed EKF results, without retraining:

```bash
bash rbsqmc/comparison/sqmc_ekf/scripts/run_sqmc_ekf_colab.sh --resume rbsqmc/comparison/sqmc_ekf/outputs/DDMMYYYY_HHMM
```

The session is stopped after verified collection, or after a confirmed worker
failure and successful diagnostic recovery. While detached, the worker's original
timeout still applies; the launcher leaves the VM assigned until collection or an explicit
`colab stop --session <saved-session-name>`. Reconnection requires that Colab
still retains the same runtime. `--local --smoke` exercises the pipeline on CPU;
it does not verify GPU execution.

## Reference scripts

| Purpose | Script |
|---|---|
| Colab orchestrator (template) | `rbsqmc/comparison/sqmc_smc/run_model_unbiased_colab.sh` |
| Colab GPU bootstrap (template) | `rbsqmc/comparison/sqmc_smc/run_model_unbiased_gpu.py` |
| Config (template) | `rbsqmc/comparison/sqmc_smc/config/model_unbiased_gpu_config.json` |
| Output validator (template) | `rbsqmc/comparison/sqmc_smc/validate_model_unbiased_outputs.py` |
| **SMC-vs-SQMC comparison** (model for this task) | `rbsqmc/comparison/sqmc_smc/compare_smc_sqmc.py` |
| Sequential prediction (RB-SQMC) | `rbsqmc/src/model/rbsqmc/predict_rbsqmc.py` |
| Sequential prediction (SMC) | `rbsqmc/src/model/rbsmc/predict.py` |
| Filter / predict pipeline | `rbsqmc/src/model/rbsmc/train_model_gpu.py` |
| Plotting utilities | `rbsqmc/src/utils/graphic.py` |

Follow `predict_rbsqmc.py` for the sequential prediction used to evaluate model performance at the end.

## The reference model: `cuthberto-carlos`

The [cuthberto-carlos](https://github.com/state-space-models/cuthberto-carlos) repository is the **factorial EKF** baseline we replicate. It implements a moment-based (Gaussian) factorial state-space model for football scores, following Duffield, Power & Rimella (2024).

### Key packages

| Package | Role in the reference model |
|---|---|
| **`cuthbert`** | Core filtering library. `cuthbert.gaussian.moments` builds the linearised-Gaussian (EKF) filter; `cuthbert.factorial.gaussian.build_factorializer` + `cuthbert.factorial.filter` run the factorial filter that keeps per-team marginals independent. |
| **`cuthbertlib`** | Linearisation helpers. `cuthbertlib.linearize.moments` provides `MeanAndCholCovFunc` and `linearize_moments` used to linearise the OU dynamics and the bivariate-Poisson observation. |
| **`ghq`** | Gauss–Hermite quadrature. `ghq.multivariate` integrates the bivariate-Poisson likelihood over the joint skill distribution in `predict_match` to produce the score grid. |
| **`optax`** | Gradient-based parameter estimation. `train_moments.py` uses `optax.adam` with `jax.value_and_grad` on the negative log-normalising constant. |
| **`jax` / `jax.numpy`** | Differentiable array backend; all filtering, linearisation, and optimisation are JIT-compiled. |
| **`pandas` / `numpy`** | Data loading and preprocessing (`download_data`). |
| **`plotnine` / `PIL`** | Prediction graphics (`graphics.py`). |

### How it connects to our model

Both models share the **same statistical core**: an OU state-space model with a bivariate-Poisson observation likelihood, parameterised by `(alpha, beta, kappa, init_mean, init_cov, friendly_scale)`. The connection points are:

1. **Same data source.** Both read `martj42/international_results` (`results.csv`). Our `rbsqmc/src/data/data.py` (`get_results`, `get_training_data`) and the reference `cuthberto_carlos/data.py` (`download_data`, `to_jax_data`) apply the same preprocessing: days-since-origin timestamps, `friendly` flag, `-1` sentinel for future scores, and the Morocco–Senegal AFCON 2026 patch.

2. **Same likelihood.** The bivariate-Poisson `loglik` / `loglik_grid` in `cuthberto_carlos/bivariate_poisson.py` is the same function our `rbsqmc/src/data/bivariate_poisson.py` uses. This is the key shared component for a fair SQMC-vs-EKF comparison.

3. **Same dynamics.** The OU transition `x_t | x_{t-1} ~ N(mu + phi(x_{t-1} - mu), Q_t)` with `phi = exp(-kappa * dt)` and `Q_t = (1 - phi^2) * Sigma_0` appears in both. The reference uses a **diagonal** `Sigma_0` (factorial, teams independent); our RB-SQMC uses a **non-diagonal** `Sigma_0 = Gamma_0 ⊗ B` (correlated teams). This is the core modelling difference the comparison tests.

4. **The EKF is the factorial filter.** The reference's `cuthbert.gaussian.moments` filter is exactly the EKF we replicate in `rbsqmc/src/model/ekf`. It propagates each team's Gaussian marginal independently and linearises the observation around the current mean — the "factorial" approximation our RB-SQMC relaxes.

> **Reference fidelity.** The EKF in `rbsqmc/src/model/ekf` replicates the original factorial moment-based model from [`state-space-models/cuthberto-carlos`](https://github.com/state-space-models/cuthberto-carlos) (revision `f79147e`), matching the OU dynamics, the bivariate-Poisson observation moments, and the Gauss–Hermite score-grid prediction exactly. The sole deliberate deviation is that Cuthbert's explicit `init_prepare` initial state is used in place of the reference's ``add_dummy_initial_input`` prepended dummy input; this is the API's documented replacement for the outdated dummy initialisation and does not change the model.

5. **Prediction interface.** The reference's `predict_match(skills_mean, skills_cov, alpha, beta, scale, max_goals)` returns a score grid + result probabilities. Our `predict_rbsqmc.py` / `predict.py` produce the same per-match forecast structure, so both models can be scored with the same evaluation metrics (Brier, exact/outcome accuracy, log-likelihood).

6. **Parameter estimation.** The reference trains with `optax.adam` on the negative log-normalising constant (gradient descent). Our `train_model_gpu.py` / `optimization.py` do the same for RB-SQMC. To compare fairly, both must use identical `n_epochs`, `learning_rate`, `seed`, and date splits.

## Training Parameters

`n_particles` may not apply to the EKF model.

```json
{
  "training_start_date": "1980-01-01",
  "test_start_date": "2024-01-01",
  "prediction_start_date": "2026-06-11",
  "n_particles": 256,
  "max_goals": 8,
  "seed": 0,
  "n_epochs": 100,
  "learning_rate": 0.05,
  "n_reps": 25,
  "include_friendly": true,
  "teams": "worldcup2026",
  "gpu_type": "A100",
  "colab_timeout": 14400,
  "repo_url": "https://github.com/ryantjx/rbsqmc.git"
}
```

## Evaluation

Compare SQMC vs EKF on the following axes. All plots are produced by `rbsqmc/src/utils/graphic.py` unless noted.

### 1. Time comparison
- Wall-clock training time per method (recorded in the run summary, as in `compare_smc_sqmc.py`).

### 2. Train / Test marginal log-likelihood
- `plot_logmarginal_history_train_test` — train and test logZ vs epoch on the same axes, with match counts annotated.
- `plot_gradient_norm_curve` — gradient norm per epoch (convergence / instability check).
- Report final and best test logZ, and the per-match logZ difference (SQMC − EKF).

### 3. World Cup 2026 predictions
- `plot_prediction_match` / `plot_prediction_score_heatmap` / `plot_all_predictions` — per-match outcome probabilities + bivariate-Poisson score heatmap.
- Compare Brier score, exact-score accuracy, and outcome accuracy (as in `_plot_prediction_evaluation` in `compare_smc_sqmc.py`), with the uniform-reference Brier score as a baseline.

### 4. Initial correlation of the SQMC model
- `plot_initial_correlation_matrix` — heatmap of the learned prior $\Gamma_0$ (the between-team correlation structure).
- `plot_correlation_matrix` / `plot_correlation_topn_bar` — final-state team correlation matrix and strongest positive/negative pairs.

### 5. Filtered states
- `plot_final_rankings` — all teams ranked by final total strength.
- `plot_top_strengths` — top-N attack, defense, and total strengths.
- `plot_timeseries_states` — attack/defense trajectories over time for top teams.

### 6. Verdict
- Determine whether RB-SQMC performs better than the factorial EKF model, and write the draft based on these results.
