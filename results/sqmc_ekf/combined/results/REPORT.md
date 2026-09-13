# SQMC–EKF Comparison Report

Generated: 2026-09-10 00:07:58
Dataset rows: 49521 (train 4702, test 305, prediction count 104).
Config: {"training_start_date": "1980-01-01", "test_start_date": "2024-01-01", "prediction_start_date": "2026-06-11", "n_particles": 512, "max_goals": 8, "seed": 0, "n_epochs": 50, "learning_rate": 0.02, "n_reps": 15, "include_friendly": true, "teams": "worldcup2026", "match_scale": 1, "gauss_hermite_degree": 32, "gpu": "A100", "colab_timeout": 14400, "setup_timeout": 900, "transfer_timeout": 600, "session": "sqmc_ekf", "repo_url": "https://github.com/ryantjx/rbsqmc.git", "repo_branch": "main", "source_commit": "336842f0e8739071141928486f54db1d08175f0c", "run_id": "09092026_1431", "session_name": "sqmc_ekf_09092026_1431_f4321a23cb", "resolved_utc": "2026-09-09T14:31:18.881201+00:00", "source_transport": "colab_git_bundle"}

## Training

| | EKF | SQMC |
|---|---|---|
| Final train logZ | -1.568e+04 | -1.465e+04 |
| Final test logZ | -1040 | -1005 |
| Best test logZ (epoch) | -1040 (50) | -960 (49) |
| Compile (s) / execute (s) | 0.8523 / 13.07 | 15.65 / 5656 |

> Note on logZ: the EKF value is a **Gaussian-approximate** normalising constant from moment filtering; the SQMC value is a particle likelihood estimate. They are not directly comparable as exact marginal likelihoods, so any raw gap must not be interpreted as a superiority verdict on its own.

## Headline prediction metrics (World Cup 2026 eligible)

### EKF
| Metric | Value |
|---|---|
| Mean Brier score | 0.5442 |
| Uniform reference Brier (2/3) | 0.6667 |
| Brier skill score vs uniform | 0.1838 |
| Mean predictive log score | -3.131 |
| Exact-score accuracy | 0.07692 |
| Outcome accuracy | 0.5288 |
| Scored matches | 104 |

### SQMC
| Metric | Value |
|---|---|
| Mean Brier score | 0.6878 |
| Uniform reference Brier (2/3) | 0.6667 |
| Brier skill score vs uniform | -0.03165 |
| Mean predictive log score | -3.664 |
| Exact-score accuracy | 0.04808 |
| Outcome accuracy | 0.4712 |
| Scored matches | 104 |

## Verdict (evidence-based)

- On the three-outcome Brier score, EKF had the lower mean (0.5442 vs 0.6878).
- Outcome accuracy was 0.5288 (EKF) vs 0.4712 (SQMC).
- The test logZ gap alone is not treated as a correctness verdict: the EKF reports a Gaussian-approximate logZ and the SQMC a particle estimate, so differences may reflect approximation quality and estimation noise rather than only the modelled correlations.
- Any remaining differences are attributed to correlated vs factorial dynamics only after the shared-input, shared-optimizer protocol is confirmed and prediction metrics are inspected.
