# CPU/GPU comparison scripts

These three standalone scripts compare the repository's shared algorithms on CPU and GPU. The filter comparison measures **SQMC on CPU versus GPU at fixed particle counts and 100 updates**, with runtime-budget selection as a secondary analysis. It does not run an SMC baseline.

Run the commands from the repository root in an environment with JAX, NumPy, SciPy, Matplotlib, Cuthbert, and Cuthbertlib installed. The shared QMC package also needs its `_sobol_direction_numbers.npz` data file; `_generate_sobol_data.py` provides the repository's generation utility if that file is absent. A GPU run requires an installed JAX GPU backend. Both CPU and GPU are requested by default; an unavailable requested device causes a logged failure, with no CPU fallback. Use `--platforms cpu` explicitly for local smoke tests.

| Script | Algorithm executed | Comparison |
|---|---|---|
| `benchmark_qmc.py` | `Halton.sample()` and `Sobol.sample()` from `sqmc/qmc/qmc.py` | Fresh-scramble JAX CPU, JAX GPU and SciPy CPU |
| `benchmark_hilbert_sort.py` | `hilbert_sort()` from `sqmc/hilbert_sort/hilbert_sort.py` | Same sorter on identical QMC-generated input arrays |
| `benchmark_sqmc.py` | `sqmc/sqmc/sqmc.py::build_filter`, shared scrambled Sobol sampling, and shared Hilbert ordering | Held-out SQMC accuracy attainable within a per-trajectory time budget on each backend |

The scripts supply models, input data, orchestration, measurement, and analysis. They do not implement alternative Sobol, Halton, Hilbert, resampling, or particle-update algorithms. The earlier files under `archive/` are outside this comparison. The Colab launcher below provisions one GPU session and runs these three entry points sequentially, downloading each stage before starting the next. The commands below also work directly on a CPU/GPU host.

## QMC generation

```bash
.venv/bin/python -m sqmc.comparison.benchmark_qmc \
  --platforms cpu gpu --sequences sobol halton \
  --dimensions 2 5 10 --n-values 128 512 2048 8192 32768 \
  --modes fresh --scramble --implementations jax scipy --repeats 7 --warmups 2
```

All three implementations use fresh scrambling, float64 outputs and index-zero blocks. Sobol uses 30 bits and power-of-two counts. Fixed-scramble sampling and `--no-scramble` are rejected.

JAX calls the shared public `sample()` inside a compiled function: runtime scrambling and generation are timed, with compilation and input transfers excluded. SciPy constructs a new RNG and scrambled engine inside every timed call, then uses `Sobol.random_base2` or `Halton.random(workers=1)` with no optimisation. Its constructor overhead is included. This compares complete implementation paths, not isolated identical scrambling kernels.

Seven distinct reproducible seeds drive the timed repetitions; JAX CPU/GPU use matching keys. Equal seeds do not imply identical SciPy/JAX scrambled points. Execution order rotates across all three implementations.

QMC contract version 2 adds `implementation` (`jax` or `scipy`) alongside `backend` (`cpu` or `gpu`). `cpu_gpu_comparison.json` retains the JAX CPU/GPU ratio; `scipy_comparison.json` adds SciPy CPU/JAX CPU and SciPy CPU/JAX GPU ratios. Earlier contract-1 results remain readable by the artifact validator but cannot supply missing SciPy timings for the new figures.

The QMC figure is 2×2: Sobol/Halton rows and runtime/speedup columns. The separate Hilbert figure is 1×2 for Sobol/Halton inputs. PDF and PNG exports are retained. Full-profile grids contain 180 QMC, 140 Hilbert and 50 SQMC timing rows.

## Hilbert sorting of generated QMC points

```bash
.venv/bin/python -m sqmc.comparison.benchmark_hilbert_sort \
  --platforms cpu gpu --sequences sobol halton \
  --distribution normal --dimensions 2 5 10 \
  --n-values 128 256 512 2048 8192 --repeats 7 --warmups 2
```

For each configuration, generate scrambled QMC points through the public generator on CPU. `--distribution normal` applies the shared finite normal-quantile transform; `--distribution uniform` retains the unit-cube points. Save this input array and its SHA-256 hash, then transfer that exact array to each requested device. QMC generation and transfer are excluded from sort timing; input-generation wall time is recorded separately.

The measured operation is the complete shared `hilbert_sort`: standardization, coordinate transformation/quantization, index computation, and sorting. Validate that every returned index vector is a permutation. When both devices run, report exact permutation agreement and the number of differing positions, in addition to CPU/GPU speedup. Floating-point reduction differences near quantization boundaries can affect ordering; a mismatch is retained in the results for investigation, rather than silently treated as equivalent. No alternate reference grid or CPU sorting algorithm is substituted.

## SQMC under a fixed runtime budget

```bash
.venv/bin/python -m sqmc.comparison.benchmark_sqmc \
  --platforms cpu gpu --dimensions 2 5 \
  --particle-counts 128 256 512 1024 2048 \
  --n-steps 100 --budget-seconds 0.01 0.05 0.1 \
  --datasets 1 --selection-reps 8 --validation-reps 16 \
  --repeats 7 --warmups 2 --bootstrap-reps 500
```

### Meaning of the budget

A budget is the allowed **median steady-state runtime of one complete filtering trajectory**, in seconds. It is not a timeout for the entire benchmark or a budget for averaging multiple filters together. Compilation, transfers, validation replications, bootstrap analysis, and plotting are outside this per-trajectory constraint. `status.json` separately records the benchmark's total wall time.

For each backend and dimension:

1. Measure complete-trajectory runtime at every requested particle count.
2. Estimate normalized filtering error using independent **selection** randomizations.
3. For each budget, retain counts whose measured median runtime is at most the budget. Select the retained count with the lowest selection error; break ties by runtime and then particle count.
4. Report that selected count's error and 95% bootstrap interval from separate **validation** randomizations. Validation error is never used to select the count.

An empty feasible set is recorded as `no_measured_configuration_within_budget`. The script does not extrapolate or assume that larger particle counts always give better observed error. Increase `--particle-counts` explicitly to widen the search. Selection is the best measured candidate on the requested grid, not proof of global optimality. A median budget is not a hard latency guarantee: the summary also reports the upper quartile and fraction of timing samples within budget.

### Model, accuracy, and randomization

Use `X_0 ~ N(0,I)`, `X_t = X_(t-1) + 0.5 Z_t`, and `Y_t = X_t + E_t`, with independent standard Gaussian noises. Process every observation `Y_1,...,Y_T`. An independent NumPy Kalman calculation supplies exact posterior means, marginal variances, and log-likelihoods.

For each replicate, record

```text
normalized_mse = mean_over_time_and_coordinates((estimated_mean - Kalman_mean)^2 / Kalman_variance)
reported_rmse = sqrt(mean_over_datasets_and_replicates(normalized_mse))
```

Confidence intervals resample datasets and replicate errors hierarchically; time steps and coordinates are not treated as independent replicates. With the default one dataset, intervals and budget comparisons are conditional on that dataset. Use `--datasets` greater than one to explore dataset variability. Selection and validation use independent filter randomizations on the same datasets; validation does not claim generalization to unseen datasets.

CPU and GPU receive identical observations and matching replicate keys to isolate backend effects. Selection, validation, and timing use separate random streams. Initialization and observation updates use separate keys, including an explicit replacement of the initialized state's consumed key before the first shared combine call. The shared filter constructs fresh LMS+shift Sobol points at index zero for each update. Particle counts must be powers of two. Initialization and propagation both use `normal_coordinates` for finite inverse-CDF values, and each log potential is scalar per particle.

The timed, fully JIT-compiled trajectory includes initialization, fresh scrambling, Hilbert sorting, resampling, propagation, weighting, posterior means at every observation, and accumulated log-likelihood. Host conversion occurs afterwards. The experiment intentionally uses scrambled Sobol: it does not exercise the shared filter's unresolved deterministic counter path or its handling of other scrambled QMC engine types. Its validated counts and finite Gaussian model also avoid the unchecked index/invalid-resampling-input cases identified in the package review. Those general-package issues still need their own fixes.

### SQMC results

- `results.json` / `results.csv`: per-count timing samples, medians/quartiles, first-call time, backend execution order, selection and validation summaries.
- `accuracy_records.json`: dataset/replicate identifiers, actual key data, normalized squared error, and exact-reference log-likelihood error.
- `reference_d*.npz`: observations, Kalman means/variances, and exact log-likelihoods.
- `<backend>_d*_n*_<phase>.npz`: posterior mean estimates in dataset-major, replicate-minor order.
- `budget_summary.json` / `budget_summary.csv`: selected counts and held-out accuracy at each budget, including unattainable-budget status.
- `cpu_gpu_comparison.json`: selected CPU/GPU particle counts and held-out error ratios at common budgets. A CPU/GPU error ratio above one favors GPU accuracy at that budget; it is not a runtime speedup ratio.
- `runtime.png`, `accuracy_vs_runtime.png`, and, when any budget is feasible, `accuracy_at_budget.png`.

## Common outputs and measurement rules

Each command creates a unique UTC timestamped directory under `sqmc/comparison/outputs/`. Use `--output-dir PATH` for an explicit fresh destination. A nonempty destination is rejected to prevent mixing runs.

Every run retains `logs.txt`, `config.json`, `metadata.json`, and `status.json`. Metadata contains device descriptions, software versions, Git commit and working-tree status, and hashes of the benchmark sources, shared implementations, and Sobol direction data. A recorded environment is not a dependency lock. Python exceptions are logged, status changes to `failed`, and completed result checkpoints remain available. Native process crashes or machine loss can prevent final status updates.

QMC and Hilbert also write `results.json`, `results.csv`, `cpu_gpu_comparison.json`, and `runtime.png`; Hilbert retains its input arrays. QMC records runner-setup wall time separately; SciPy engine construction remains inside each timed call. All scripts retain raw timed samples. First-call measurements include tracing/compilation/execution and are not labeled as isolated compile time. QMC rotates all three implementations; Hilbert and SQMC alternate CPU/GPU order across configurations. Comparisons use the first device of each requested backend, sequentially; run performance experiments without competing jobs.

## Local checks

```bash
.venv/bin/python -m pytest sqmc/comparison/tests/test_benchmarks.py -q

.venv/bin/python -m sqmc.comparison.benchmark_qmc \
  --platforms cpu --dimensions 2 --n-values 8 16 --repeats 2 --warmups 0

.venv/bin/python -m sqmc.comparison.benchmark_hilbert_sort \
  --platforms cpu --dimensions 2 --n-values 8 16 --repeats 2 --warmups 0

.venv/bin/python -m sqmc.comparison.benchmark_sqmc \
  --platforms cpu --dimensions 2 --particle-counts 8 16 --n-steps 3 \
  --selection-reps 2 --validation-reps 3 --bootstrap-reps 50 \
  --budget-seconds 0.000000001 1 --repeats 2 --warmups 0
```

Validation on 7 September 2026: all three entry points completed CPU smoke runs, and 13 focused tests passed. Tests cover public implementation delegation, keyed SQMC sampling, the independent Gaussian reference, selection/validation separation, error aggregation, invalid counts/permutations, retained failure logs, and refusal to fall back from a requested GPU. The smoke runs check execution and artifacts, not performance; they were not isolated performance experiments. Live GPU execution remains to be verified on a GPU host.


## Colab launcher

Activate the local environment containing NumPy and Pillow for artifact validation;
`colab` is discovered on the inherited `PATH` (it need not live in `.venv`).
Push the source commit before starting. Both the local and remote configured branch must point to that
exact commit; the remote checkout is detached at the SHA, so later branch changes
cannot change the experiment's source.

```bash
source .venv/bin/activate
sqmc/comparison/scripts/run_comparison_colab.sh --dry-run
sqmc/comparison/scripts/run_comparison_colab.sh
sqmc/comparison/scripts/run_comparison_colab.sh --config /path/to/overrides.json
```

The shell delegates local lifecycle management to `run_comparison_local.py`.
`run_comparison_gpu.py` bootstraps the remote checkout and runs one benchmark per
worker subprocess. Workers share the provisioned VM but have separate Python/JAX
processes. The CLI kernel remains available for cancellation and partial-output
recovery while a worker executes. The launcher polls worker status and streams
remote experiment output into the local log approximately every 15 seconds.

### Configuration

`scripts/config/comparison_config.json` defines the full chapter profile:

| Setting | QMC | Hilbert sort | SQMC |
|---|---|---|---|
| Dimensions | 2, 5, 10, 30, 60 | 2, 5, 10, 30, 60 | 2, 5, 10, 30, 60 |
| Counts | 128, 512, 2048, 8192, 32768, 131072 | 128, 256, 512, 2048, 8192, 32768, 131072 | 128, 256, 512, 1024, 2048 |
| Generator | Sobol and Halton | Sobol and Halton | Scrambled Sobol |
| Other | Fresh scrambling; JAX CPU/GPU and SciPy CPU | Normal-transformed inputs | 100 observations; 0.01, 0.05, 0.1-second budgets |
| Repetitions | 7 timed, 2 warmups | 7 timed, 2 warmups | 7 timed, 2 warmups; 8 selection, 16 validation |

All stages request both `cpu` and `gpu`. SQMC uses one dataset, 500 bootstrap
repetitions and seed 42. No SMC baseline runs. All parameter names in the three
JSON stage objects correspond to the standalone CLI flags, with underscores
replaced by hyphens. Partial configuration files merge into the full profile;
unknown fields and invalid values fail before provisioning. Example:

```json
{
  "repo_branch": "codex/colab-comparison-07092026",
  "source_commit": "<full 40-character pushed Git SHA>",
  "colab_timeout": 3600,
  "qmc": {"seed": 42}
}
```

Top-level fields are `gpu` (default A100), `colab_timeout` (execution deadline
per stage, default 3600 seconds), `setup_timeout` (900 seconds),
`transfer_timeout` (600 seconds per transfer/control command), `session`
(unique-session prefix), `repo_url`, optional `repo_branch` and `source_commit`
(default current local branch and HEAD). Existing `GPU_TYPE`, `COLAB_TIMEOUT`
and `SESSION` environment overrides take precedence over the JSON. The saved
configuration also records `source_transport=colab_git_bundle` and
`source_bundle_sha256`. It contains resolved overrides, complete experiment parameters,
UTC run identifier, unique session name and exact source commit. Runtime IDs
are regenerated even when a previous effective configuration is supplied.

`--dry-run` validates the configuration and prints effective parameters and
commands. It creates no directories, performs no network operations and allocates
no hardware. An existing UTC minute directory is rejected, including on dry run.
Wait until the next minute to start another comparison.

### Colab outputs and recovery

```text
sqmc/comparison/outputs/DDMMYYYY_HHMM/
  logs.txt
  comparison_config.json
  run_config.json
  status.json
  remote_status.json
  remote_logs.txt
  root_manifest.json
  qmc_manifest.json
  hilbert_sort_manifest.json
  sqmc_manifest.json
  qmc/
    logs.txt
    process_logs.txt
    ...standalone benchmark outputs...
  hilbert_sort/
    logs.txt
    process_logs.txt
    ...standalone benchmark outputs...
  sqmc/
    logs.txt
    process_logs.txt
    ...standalone benchmark outputs...
```

The directory timestamp uses **UTC**. `logs.txt` captures local stdout/stderr,
provisioning, commands, streamed experiments, transfers, errors and cleanup.
`process_logs.txt` also retains native subprocess stderr that Python's benchmark
logger cannot intercept. Root downloads explicitly exclude `logs.txt`.

`run_config.json` records actual CPU models/count, host memory, GPU name/memory,
NVIDIA driver/CUDA information, Python and installed package versions, JAX/JAXlib,
backend device descriptions, float64 configuration and PRNG settings. It references
the same source SHA and canonical configuration hash as the archive manifests.
The launcher creates a Git bundle from the committed branch, verifies the branch
SHA on origin, and uploads the bundle through Colab. The VM verifies its SHA-256
and Git structure, then checks out the exact commit. Uncommitted and staged edits
are excluded; remote GitHub access is not required. Setup preserves preinstalled
JAX and NVIDIA packages with pip constraints, installs
missing dependencies, generates the ignored Sobol archive from the checked-in
Joe–Kuo table, verifies it against SciPy, and requires both CPU and the requested GPU.

Each completed stage is archived remotely with a SHA-256 manifest. The launcher
downloads an independent archive checksum and verifies the archive, every member's
size/hash, safe paths, source/configuration provenance, stage completion, exact
parameter forwarding, complete CPU/GPU grids and expected artifacts. It validates
timing summaries, speedup ratios, PNG decoding, retained numerical arrays, SQMC
error calculations and budget selections before allowing the next stage to start.
Unattainable SQMC budgets remain explicit numerical outcomes, not execution failures.

On a stage error, transfer error, timeout or interrupt, subsequent stages stop.
The launcher attempts to terminate its worker, retrieve available partial outputs
and logs, then stop its uniquely named session. It verifies disappearance from the
server-backed session listing. It never stops a pre-existing session; an endpoint
change is reported instead of stopping an unowned session. Original exit/error
status is retained separately from recovery/download/shutdown errors. A cleanup
failure makes an otherwise successful comparison fail. Partial archives can be
retained for diagnosis but are never marked as successful comparisons. Complete
stage downloads are not overwritten during recovery. If setup fails before root
metadata exists, the local log still contains the setup traceback; missing remote
metadata is reported. Network or VM loss can prevent salvage or shutdown verification.

### Launcher validation

```bash
bash -n sqmc/comparison/scripts/run_comparison_colab.sh
.venv/bin/python -m pytest sqmc/tests sqmc/comparison/tests -q
```

Mocked tests cover stage ordering, all configuration forwarding through the actual
benchmark argument parsers, dry-run side effects, timestamp collisions, incremental
downloads, checksums and unsafe archives, partial failures, original error retention,
root-log protection and session ownership. Live A100 acceptance evidence is recorded
below after executing the pushed source commit; local smoke results alone do not
establish GPU performance.


### Live acceptance record — 7 September 2026

Attempt `07092026_0557` used commit
`26447649aca8e72cbcac196953c55e93848a8140` on
`codex/colab-comparison-07092026`, with the full A100 profile. A100 provisioning
succeeded, but the VM's GitHub HTTPS connection timed out during `git fetch` after
132.7 seconds. No benchmark stage started. The launcher retrieved `remote_logs.txt`,
retained the setup failure in `status.json`, stopped only its owned session
`comparison_07092026_0557_20092bab0a`, and verified no active sessions remained.
The unavailable root metadata archive was reported as a separate recovery error.
This failed attempt is not performance evidence.

That failure motivated the verified Git-bundle transport described above. Its
regression test creates a real temporary Git repository and verifies that the
uploaded bundle reproduces the pinned commit while excluding working-tree edits.
A fresh full-profile acceptance attempt follows the pushed transport fix.

The subsequent attempt `07092026_0607`, using
`308348a726dbdb67a97a445732b1cbbf2072c92f`, verified and downloaded QMC and Hilbert
sort, then lost access to the session during SQMC. Its failure status and recovery
errors are retained; session absence was verified. Its incomplete SQMC outputs
must not be combined with another run.

The completed run supplied for review, **`07092026_1002`**, uses that same source
commit and the full profile. Review of the retained configuration, manifests,
logs, numerical arrays and figures establishes:

- Hardware: A100-SXM4 40 GB; 12 exposed Xeon CPU threads at 2.20 GHz;
  JAX/JAXlib 0.11.1; float64; Threefry PRNG.
- QMC: **240** complete CPU/GPU timing rows, 120 paired comparisons; benchmark
  wall time 782.30 seconds.
- Hilbert sort: **140** timing rows, 70 paired comparisons; all paired permutations
  agree exactly and the retained input hashes agree; wall time 197.37 seconds.
- SQMC: **50** timing rows and **1,200** accuracy records, with all 30 backend,
  dimension and budget outcomes checked; wall time 331.23 seconds.
- Every manifest member's size/hash matches the retained artifact; all source
  commits and canonical configuration hashes agree. Complete grids, arguments,
  timing summaries, input arrays, SQMC estimate/error calculations, budget choices
  and PNG integrity pass the local artifact validators.
- QMC was downloaded at 10:16:53 UTC before Hilbert began at 10:16:54; Hilbert was
  downloaded at 10:21:19 before SQMC began at 10:21:20; SQMC was downloaded at
  10:27:36. The saved status/logs record verified shutdown at 10:27:41 UTC, exit 0,
  with no secondary errors. The owned session was
  `comparison_07092026_1002_92339ee396`.
- No configuration meets the 10 ms trajectory budget. Under 100 ms, GPU selects
  2,048 particles at every dimension; CPU selects 256, 512, 256, 256 and 128 at
  dimensions 2, 5, 10, 30 and 60. CPU/GPU held-out error ratios are 3.73, 1.90,
  1.95, 1.34 and 1.37. At matched counts the largest absolute RMSE difference is
  2.22e-16. Three GPU selections under 50 ms satisfy the limit in only six of
  seven timings, consistent with a median budget rather than a hard deadline.

The original QMC plot emits a `constrained_layout` warning because its legend is
crowded. Numerical and file-integrity validation succeeds, but publication use
requires a clearer plot. The dissertation review regenerates faceted figures from
this run's saved results without altering the original output folder or rerunning
any experiments. Local validation totals **197 passing package, benchmark and
launcher tests**; shell syntax and all three small CPU smoke runs also pass.

## Fresh-scramble A100 acceptance: 08092026_0048

The full comparison completed successfully from benchmark commit
`e7363f8fa53cd6e761d1abfeefacbac4423662cd` on
`codex/colab-comparison-07092026`. Results are retained in
`outputs/08092026_0048/`, with root/stage logs, configuration, hardware,
checksummed manifests and numerical artifacts. No measurements from the earlier
run are mixed into these results.

| Stage | Timing records | Benchmark wall time | Verified download (UTC) |
|---|---:|---:|---|
| QMC: JAX CPU/GPU and SciPy CPU, fresh scrambling | 180 | 658.81 s | 01:01:04 |
| Hilbert: shared sorter, Sobol/Halton inputs | 140 | 199.11 s | 01:05:44 |
| SQMC: shared CPU/GPU filter, 100 updates | 50 | 329.62 s | 01:11:37 |

Hardware was A100-SXM4 40 GB with 12 exposed Xeon CPU threads at 2.20 GHz;
JAX/JAXlib 0.11.1, SciPy 1.16.3 and float64. The configuration hash is
`c6d93d159e5ecefd9ea4ac1441129ef04ced6e31ab8f944b3a14682fb1326479`.
All stage execution/download statuses are complete. The owned session
`comparison_08092026_0048_d8c6e6d866` was stopped and its absence verified at
01:11:44 UTC, with exit code zero and no secondary failures.

At 131,072 points, JAX GPU speedups over SciPy CPU are 4.42–31.05× for Sobol
and 86.57–103.48× for Halton. The corresponding JAX CPU/GPU ratios are
5.34–45.59× and 8.82–12.29×. These use fresh scrambling and the timing boundaries
specified above; they are not fixed-scramble sampling measurements.
All 70 Hilbert input pairs give identical permutations. At 2,048 particles,
SQMC GPU speedups are 6.23–13.37×, with maximum CPU/GPU validation RMSE difference
2.22e-16 across all 25 matched configurations. These are observations on one
session and one dataset per dimension, not cross-session confidence claims.

Validation comprises 154 package tests and 50 comparison/launcher tests, shell
syntax checks, a JAX CPU/SciPy smoke run, and the full live archive/numerical
checks. Previous contract-1 QMC artifacts were also checked with the new
validator. The chapter was compiled in an isolated directory: 50 pages, no
overfull boxes or undefined references; revised figures and tables were visually
inspected. Existing global duplicate PDF destination warnings remain outside
this comparison change.

To reproduce the four publication figures from this run, use a fresh destination:

```bash
.venv/bin/python dissertation/drafts/figures/make_comparison_figures.py \
  --run-dir sqmc/comparison/outputs/08092026_0048 \
  --output-dir /tmp/sqmc-publication-figures
```

The renderer requires a run containing the new SciPy measurements and rejects
an existing output directory. It writes separate 2×2 QMC and 1×2 Hilbert PDF/PNG
figures, fixed-iteration SQMC runtime and secondary budget figures, and a
provenance JSON recording input and analysis-code hashes. The completed derived
exports and compiled dissertation are in `outputs/08092026_0048_writeup/`.
Raw experiment files remain unchanged. Later documentation/renderer commits
must not be substituted for the benchmark source commit recorded above.
