# Execute the SQMC–EKF comparison

continue with the task in rbsqmc/comparison/sqmc_ekf/PLAN.md. my match scales for [model_rbsqmc.py](/Users/ryant/Github/ryantjx/rbsqmc/rbsqmc/src/model/model_rbsqmc.py) should be 1 in the config.

## Approach

Implement the factorial EKF and a reproducible comparison pipeline. Keep new code, tests, configuration, and the results draft inside `model_ekf` and `sqmc_ekf`. Make only the shared-code corrections required for correctness.

Use your selected settings: **512 particles, 100 epochs, cosine learning-rate decay from 0.05, seed 0, 25 SQMC gradient replicas, final-epoch checkpoints, and World Cup-only headline metrics**.

## Implementation steps

1. **Prepare identical, reproducible inputs.**
   - Freeze the results dataset and record its checksum, source revision, team mapping, configuration, and dependency versions.
   - Use train `[1980-01-01, 2024-01-01)`, test `[2024-01-01, 2026-06-11)`, and prediction from June 11.
   - Preserve chronological match order, friendly flags, and per-team previous timestamps across split boundaries.
   - Correct preprocessing that drops unknown-score fixtures, incorrectly removes same-day matches, or conditionally misses the Morocco–Senegal correction. Record exclusions from the existing eight-goal cutoff explicitly.
   - **Timing semantics (decision):** keep same-day repeats as separate rows (`dt = 0` for a team's repeated appearance that day) and **do not** drop them; no per-day `_drop_duplicate_teams_per_day` aggregation. The comparison does **not** adopt the original model's shared-`dt`-per-row convention.

2. **Implement the factorial EKF in `model_ekf`.**
   - Replicate the OU dynamics, bivariate-Poisson observation moments, factorial updates, and Gauss–Hermite prediction from [cuthberto-carlos](https://github.com/state-space-models/cuthberto-carlos), recording the reference revision and attribution.
   - Use the installed `cuthbert` API’s explicit initial state; do not copy the outdated dummy-input initialization.
   - Retain attack–defence covariance within each team and discard between-team posterior covariance through factorial marginalization.
   - Expose filtering, differentiable training loss, and sequential prediction interfaces. Return actual Gaussian means, covariances, and cumulative logZ.
   - Handle zero elapsed time analytically and synchronize team states through OU propagation, without copying the reference’s internal log-normalizer shape workaround.
   - **Only playing teams move (decision):** the EKF propagates only the two teams in the current match through the OU by each team's own last-appearance gap. Non-playing teams hold their state until a later match. This is kept deliberately distinct from the original model's every-row-all-teams propagation.

3. **Make the necessary SQMC compatibility corrections.**
   - Keep friendly-scale parameters and match metadata in comparison-local adapters.
   - Add optional per-match scales to shared SQMC filtering and prediction, defaulting to `1` for existing callers.
   - Ensure unknown scores contribute no observation likelihood and forecasts precede score assimilation.
   - Reuse the existing SQMC algorithm and gradient estimator. Correct demonstrated defects directly; do not introduce monkey patches, fabricated values, or silent parameter changes.

4. **Train both methods through one comparison driver.**
   - Match initial shared parameters and per-team marginal covariance; retain SQMC’s correlated prior and EKF’s independent team blocks.
   - Apply identical Adam settings and exactly 100 updates. Average 25 independently randomized SQMC gradients; accumulate replicas sequentially to control memory without reducing the experiment.
   - Record train and test logZ at the same updated checkpoint. Compute conditional test logZ from a continuous train-plus-test pass, subtracting its training-prefix logZ.
   - Save final parameters, histories, gradient norms, and synchronized wall-clock timings, separating compilation from execution.
   - Treat non-finite losses, gradients, or states as failed runs with diagnostics. Never reuse an earlier score or silently select an earlier checkpoint.

5. **Evaluate and produce the draft.**
   - Forecast each match before updating with its known result. Process eligible intervening matches, but calculate headline metrics only for 2026 World Cup fixtures.
   - Use matching score-grid normalization and outcome ordering; report truncation mass, Brier score, the `2/3` uniform baseline, exact-score accuracy, outcome accuracy, and predictive log scores.
   - Produce the README’s convergence, prediction, correlation, ranking, and trajectory plots. Use weighted SQMC posterior moments, including residual Gaussian covariance, and native EKF moments.
   - Report final/best test logZ and per-match differences, explicitly distinguishing EKF’s Gaussian approximate logZ from SQMC’s likelihood estimate.
   - Write the evidence-based verdict and draft inside the run directory. Do not infer superiority from logZ alone or attribute every difference solely to correlations.

6. **Run and validate on Colab.**
   - Add a comparison CLI, configuration, GPU bootstrap, orchestrator, and output validator under `sqmc_ekf`.
   - Upload the exact implementation and frozen inputs; verify compatible dependencies, including `ghq`, and GPU availability before execution.
   - Run a small GPU smoke test, then the full configuration on A100 with the 14,400-second timeout.
   - Download and validate artifacts before releasing the session. Store everything under `sqmc_ekf/outputs/DDMMYYYY_HHMM/`, including failure diagnostics when applicable.

## Timing note

- The EKF uses a per-team `previous` (`(T, 2)` gap per playing team) and moves **only** the two playing teams each step. The SQMC comparator uses a shared per-row `timestamp_prev` propagated to all teams.
- Same-day repeats are kept as separate rows at `dt = 0` but are never aggregated away.
- These intentional differences are reported in the draft rather than silently aligned, so the comparison is explicit about the EKF's only-playing-teams-move approximation vs the shared-`dt` reference.

## Script checklist

### In place (done)

- [x] `rbsqmc/comparison/sqmc_ekf/scripts/config/config_gpu.json` — comparison config (add `match_scale`: `1`).
- [x] `rbsqmc/src/data/data_ekf.py` — frozen, per-row dataset for both methods; same-day repeats kept as `dt=0` rows; per-team `previous`; SQMC `FootballResults`; exclusions metadata.
- [x] `rbsqmc/comparison/sqmc_ekf/requirements-gpu.txt` — pin `cuthbert`, `cuthbertlib`, `ghq`, `optax`, `jax[cuda12]`.
- [x] `rbsqmc/comparison/sqmc_ekf/scripts/train.py` — `Methods` driver: `constrain`/`params`, `filter`, per-epoch gradient, score, checkpoint/history/summary.
- [x] `rbsqmc/src/data/data.py` — fixed date & Morocco–Senegal issue; removed `_drop_duplicate_teams_per_day` (same-day matches kept); unknown-score-safe max-goals filter.
- [x] `rbsqmc/src/model/rbsqmc/model_rbsqmc.py` — `match_scales` option (`1` default, so SQMC unchanged for existing callers); unknown-score likelihood suppression.
- [x] `rbsqmc/src/model/ekf/__init__.py` — package marker.
- [x] `rbsqmc/src/model/ekf/model.py` — factorial EKF: `MatchInputs`, `constrain`/`positive`, `build` (OU + bivariate-Poisson moments + factorializer), `run_filter`, `propagate`, `predict_match`, `sequential_predict`, `synchronized_moments`.

### To build (open)

- [x] **Prediction driver** (`scripts/predict.py`) — `predict_sqmc` via `run_sequential_predict_rbsqmc` and `predict_ekf` via OU propagation + GH quadrature; both forecast from the pre-likelihood/previous state so the current score cannot enter its own grid.
- [x] **Evaluation module** (`evaluate.py`) — score-grid records, truncation mass, Brier score, `2/3` baseline, exact/outcome accuracy, predictive log score; World Cup–eligible subset + all-scored.
- [x] **Plots module** (`plots.py`) — convergence (logmarginal + gradient-norm), prediction heatmaps, SQMC correlation matrices, rankings, strengths, trajectories; weighted SQMC posterior moments and EKF native moments via a single-particle wrapper.
- [x] **Report/draft writer** (`report.py`) — final/best test logZ, per-match logZ difference, explicit Gaussian-approx (EKF) vs particle (SQMC) distinction, verdict + `REPORT.md`/`DRAFT.md`.
- [x] **CLI entrypoint** (`run.py`) — wires config, data, training, prediction, evaluation, plots, and report into a timestamped run dir.
- [x] **GPU bootstrap** (`run_sqmc_ekf_gpu.py`) — Colab host: clone/upload, install missing deps (incl. `ghq`), assert GPU, run comparison.
- [x] **Orchestrator** (`run_sqmc_ekf_colab.sh`) — validate config/bootstrap/validator, launch on GPU, download artifacts, stop session.
- [x] **Output validator** (`validate_sqmc_ekf_outputs.py`) — assert required artifacts, finite losses/states, checkpoint/history alignment, complete metrics.
- [x] **Tests** (`rbsqmc/tests/test_sqmc_ekf.py`) — OU moments, zero-elapsed handling, init constraints, metric computation, report serialization; plus existing regression tests pass.

### Findings / caveats (run 08092026)

- The pipeline runs end-to-end on CPU (`predict → evaluate → plots → report → validate`).
- **Data limitation:** the frozen `results.csv` ends `2025-12-27` and contains **no 2026 World Cup fixtures**, so the default `prediction_start_date: 2026-06-11` yields an empty prediction split and `load_dataset` raises. For the smoke/small runs the prediction split was moved inside the data range, so World Cup–eligible headline metrics are `null` and the report falls back to all-scored metrics. The real A100 run needs a WC fixtures source (e.g. `worldcup2026.json`/a `fixtures.json` ingestion) or the frozen result set must be extended.

## Verification and comments

- Test reference EKF parity, OU moments, factorial independence, initialization, finite gradients, and quadrature predictions.
- Test split continuity, friendly scaling, repeated same-day teams, unknown scores, and forecast independence from the current or future result.
- Test checkpoint/history alignment, failure reporting, metric calculations, and complete artifact validation; run existing affected regression tests.
- Add short comments immediately before complicated operations: factorial joining/marginalization, timestamp handling, covariance propagation, parameter constraints, replica averaging, conditional logZ, and prediction-before-update. Explain the mathematics and purpose without commenting routine statements.


## Progress

- `rbsqmc/comparison/sqmc_ekf/scripts/config/config_gpu.json`
- `rbsqmc/src/data/data_ekf.py`
- `rbsqmc/comparison/sqmc_ekf/requirements-gpu.txt`
- `/Users/ryant/Github/ryantjx/rbsqmc/rbsqmc/comparison/sqmc_ekf/scripts/train.py`
- `/Users/ryant/Github/ryantjx/rbsqmc/rbsqmc/src/data/data.py`
  - fixed dates issue 
  - fixed dropping duplicate teams per day - with match level filters, it accomodates the same-day matches
- `/Users/ryant/Github/ryantjx/rbsqmc/rbsqmc/src/model/rbsqmc/model_rbsqmc.py`
  - include friendly scale (which should default to 1 for the EKF comparison)
- `/Users/ryant/Github/ryantjx/rbsqmc/rbsqmc/src/model/ekf/__init__.py`
- `/Users/ryant/Github/ryantjx/rbsqmc/rbsqmc/src/model/ekf/model.py`
- `/Users/ryant/Github/ryantjx/rbsqmc/rbsqmc/comparison/sqmc_ekf/scripts/predict.py`