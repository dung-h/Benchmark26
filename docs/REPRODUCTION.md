# Reproduction

There are three levels, each needing more data than the previous one.

| Level | Needs | Establishes |
| --- | --- | --- |
| 1. Repository checks | This checkout | Code parses, frozen protocols are intact, unit tests pass, exported latency intervals recompute |
| 2. Upstream sources | Network access | The upstream files match their pinned SHA-256 hashes |
| 3. Study replay | The external study bundle and the upstream sources | Cohort, splits, analytical estimates and statistics can be recomputed and compared with the saved results |

## 1. Repository checks

```bash
python3 -m pip install -e ".[test]"
make check
make test
make smoke
```

`make check` parses every Python and protocol file, verifies the SHA-256 chain between the frozen files in `protocols/`, and recomputes the exported latency statistics in `results/prediction_latency/`. It succeeds without the study bundle and lists which bundle files it found. `make check-study-inputs` fails instead when they are missing.

## 2. Upstream sources

`upstream/data_sources.json` pins 249 files from the Ma–Li, Qonductor and QPack repositories by URL, revision and SHA-256. Download and verify them outside the checkout:

```bash
python3 scripts/fetch_upstream_data.py --source all --destination /path/to/upstream --list   # preview
python3 scripts/fetch_upstream_data.py --source all --destination /path/to/upstream
```

The Qonductor archive uses Git LFS. Downloading the sources does not rebuild the study tables; level 3 needs the study bundle.

## 3. Study replay

### Configure paths

All machine-specific locations are read from environment variables in `benchmark26/paths.py`. Copy `configs/paths.env.example`, edit it and `source` it.

| Variable | Default | Contents |
| --- | --- | --- |
| `B26_ARTIFACTS` | `artifacts/` | Study bundle: cohort, labels, splits, graphs, predictions, receipts |
| `B26_COHORT_PLAN` | `inputs/cohort_plan/` | Cohort plan tables (proposed observations, inventories) |
| `B26_MALI_SOURCE` | `inputs/mali/` | Ma–Li Git checkout at the pinned revision |
| `B26_MALI_QASM` | `$B26_MALI_SOURCE/data/quantum_circuits/` | Ma–Li circuits |
| `B26_QONDUCTOR_SOURCE` | `inputs/qonductor/` | Qonductor Git checkout at the pinned revision |
| `B26_QONDUCTOR_ARCHIVE` | `$B26_QONDUCTOR_SOURCE/data/database/circuits.zip` | Hydrated circuit archive |
| `B26_QONDUCTOR_DATABASE` | `$B26_QONDUCTOR_SOURCE/data/database/quantum_scheduler.db` | Qonductor database |
| `B26_WORK` | `work/` | New outputs and checkpoints |
| `B26_NEURAL_PYTHON` | current interpreter | Interpreter whose packages are recorded for neural training |

The bundle keeps the directory names of the original study (for example `postfreeze_qpu_workflow_mean_expansion_v1/`). Frozen protocols and receipts record paths in the original layout (`artifacts/...`, `benchmark_v1/protocol/...`); `benchmark26.paths.recorded()` maps them to the locations above.

### Verify against the saved results

These commands read the bundle and write only to the path given (or, for `validate_portable.py`, to `verification/` in the checkout). They can be run from any directory.

```bash
python3 scripts/check_setup.py --require-study-inputs
python3 experiments/qpu/validate_splits.py --receipt /tmp/b26/split_validation.json
python3 experiments/qpu/verify_dataset_lineage.py --output /tmp/b26/lineage
python3 experiments/qpu/replay_statistics.py --receipt /tmp/b26/statistics.json
python3 experiments/analytical/validate_portable.py
python3 experiments/qpu/train_neural.py --stage preflight --evaluation E1
```

| Command | Recomputes |
| --- | --- |
| `validate_splits.py` | All E1/E2/E3 folds, without reading runtime labels |
| `verify_dataset_lineage.py` | Source tables → cohort rows, QPack means and groups |
| `replay_statistics.py` | Every metric, paired contrast and bootstrap interval from the saved predictions; hashes of all locked inputs and graphs |
| `validate_portable.py` | Analytical metrics and intervals from the saved estimates |
| `train_neural.py --stage preflight` | Readiness of neural fitting, without fitting |

The following checks write their receipt into the bundle, as the original pipeline did. Run them on a copy of the bundle: `experiments/analytical/extract_raw.py --check-only`, `experiments/analytical/validate.py`, `validate_coverage.py`, `validate_qiskit_parity.py`, and `experiments/qpu/run_classical.py --stage preflight`.

On 3 November 2026 all eleven commands were run with this code against the original bundle, mounted read-only or as a copy-on-write overlay. The regenerated split validation, lineage tables, Qiskit parity rows and regression preflight receipt were identical to the saved files. The other receipts differed only in the hash of the script that wrote them and, for the statistics replay, in 498 additional input and graph hash checks.

### Run the pipeline

The main study runs in this order. Each step checks the outputs of the previous one.

| Step | Command |
| --- | --- |
| Cohort | `python3 experiments/qpu/prepare_cohort.py` |
| Splits | `python3 experiments/qpu/prepare_splits.py`, then `validate_splits.py` |
| Neural models | `python3 experiments/qpu/train_neural.py --stage train --evaluation E1` (also E2, E3) |
| Regression selector | `python3 experiments/qpu/run_classical.py --stage train --evaluation E1` (also E2, E3) |
| Completion check | `python3 experiments/qpu/validate_completed.py --output DIR` |
| Metrics and contrasts | `python3 experiments/qpu/aggregate_results.py --output DIR` |
| Analytical estimates | `experiments/analytical/extract_raw.py --full`, then `evaluate.py` |

Neural training needs a CUDA GPU. Fitting writes new models and predictions; it is a new experiment, not a check of the saved one.

Shared modules can also be run directly, for example `python3 -m benchmark26.mali_materialize --help`.

## Code identity

The frozen protocols and execution locks record SHA-256 hashes of the code that produced the study. The code here was adapted for portable paths and formatting, so its hashes differ from the recorded ones. Checks against the bundle therefore compare data and results, not code hashes. Files whose hashes are pinned by other frozen files (`protocols/`, including two reference programs in `protocols/reference_code/`) are kept byte-identical; `make check` verifies them.

## Not reproducible from this repository

- Individual raw timing samples and request-level latency quantiles: only aggregates are exported (see [Prediction latency](PREDICTION_LATENCY.md)).
- Historical runs of the earlier 4,515- and 8,767-row component studies. Their protocols are kept in `protocols/` because the current code verifies them, but their data and some of their runners are not part of this release. Earlier timing campaigns were removed in 0.3.0; they remain at the Git tag `archive/legacy-timing-campaigns`.
