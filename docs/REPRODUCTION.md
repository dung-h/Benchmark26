# Reproduction

The runtime of every run is public: download `runs.csv` from the [benchmark26-data folder](https://drive.google.com/drive/folders/1lXiRepBX3S7rzdscwnVyqXfis1qcp8-K) and check it with `sha256sum -c SHA256SUMS` (expected `86ee98ff24088923b96b1244814c5a93d92fa170f0b4691f4f3c6a9cd8eb959a`). The levels below go further, each needing more data than the previous one.

| Level | Needs | Establishes |
| --- | --- | --- |
| 1. Repository checks | This checkout | Code parses, frozen protocols are intact, unit tests pass, exported latency and graph-ablation statistics recompute |
| 2. Upstream sources | Network access | The upstream files match their pinned SHA-256 hashes |
| 3. Study replay | The external study bundle and the upstream sources | Dataset table, splits, analytical estimates and statistics can be recomputed and compared with the saved results |

## 1. Repository checks

```bash
python3 -m pip install -e ".[test]"
make check
make test
make smoke
```

`make check` parses every Python and protocol file, verifies the SHA-256 chain between the frozen files in `protocols/`, and recomputes the exported statistics in `results/prediction_latency/` and `results/graph_ablation/` (the latter also against the file hashes in its manifest). It succeeds without the study bundle and lists which bundle files it found. `make test` reports one skipped test (the graph-transform test) when PyTorch is not installed; install `.[research]` to run it. If a command fails with an error message, look it up in [Pitfalls](PITFALLS.md). `make check-study-inputs` fails instead when they are missing.

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
| `B26_ARTIFACTS` | `artifacts/` | Study bundle: dataset table, labels, splits, graphs, predictions, receipts |
| `B26_DATASET_INPUTS` | `inputs/dataset_inputs/` | Input tables of the dataset: the list of runs assembled from the sources (`proposed_observations.csv`, the original study's name) and its inventories |
| `B26_MALI_SOURCE` | `inputs/mali/` | Ma–Li Git checkout at the pinned revision |
| `B26_MALI_QASM` | `$B26_MALI_SOURCE/data/quantum_circuits/` | Ma–Li circuits |
| `B26_QONDUCTOR_SOURCE` | `inputs/qonductor/` | Qonductor Git checkout at the pinned revision |
| `B26_QONDUCTOR_ARCHIVE` | `$B26_QONDUCTOR_SOURCE/data/database/circuits.zip` | Hydrated circuit archive |
| `B26_QONDUCTOR_DATABASE` | `$B26_QONDUCTOR_SOURCE/data/database/quantum_scheduler.db` | Qonductor database |
| `B26_WORK` | `work/` | Scratch outputs (smoke example, verifier output) |
| `B26_RUN_DIR` | unset | Output directory of a new fitting run (see [Run the pipeline](#run-the-pipeline)) |
| `B26_NEURAL_PYTHON` | current interpreter | Interpreter whose packages are recorded for neural training |

The bundle keeps the directory names of the original study (for example `postfreeze_qpu_workflow_mean_expansion_v1/`). Frozen protocols and receipts record paths in the original layout (`artifacts/...`, `benchmark_v1/protocol/...`); `benchmark26.paths.recorded()` maps them to the locations above.

### Verify against the saved results

These commands read the bundle and write only to the path given (or, for `validate_portable.py`, to `verification/` in the checkout). They can be run from any directory.

```bash
python3 scripts/check_setup.py --require-study-inputs
python3 experiments/qpu/validate_splits.py --receipt /tmp/b26/split_validation.json
python3 experiments/qpu/verify_dataset_from_sources.py --output /tmp/b26/dataset_from_sources
python3 experiments/qpu/replay_statistics.py --receipt /tmp/b26/statistics.json
python3 experiments/analytical/validate_portable.py
python3 experiments/qpu/train_neural.py --stage preflight --evaluation E1
python3 experiments/qpu/validate_completed.py --output /tmp/b26/validation   # add --replay-all for GPU inference replay
```

| Command | Recomputes |
| --- | --- |
| `validate_splits.py` | All E1/E2/E3 folds, without reading runtime labels |
| `verify_dataset_from_sources.py` | Source tables → dataset rows, QPack means and groups |
| `replay_statistics.py` | Every metric, paired contrast and bootstrap interval from the saved predictions; hashes of all locked inputs and graphs |
| `validate_portable.py` | Analytical metrics and intervals from the saved estimates |
| `train_neural.py --stage preflight` | Dataset and split inputs, then training readiness: lock, inputs, graphs, code identity, run manifest, pinned model source, PyTorch and GPU; nothing is fitted |
| `validate_completed.py` | Structure, identities and selections of all saved fitted outputs; with `--replay-all`, every saved neural prediction from its checkpoint |

The following checks write their receipt into the bundle, as the original pipeline did. Run them on a copy of the bundle: `experiments/analytical/extract_raw.py --check-only`, `experiments/analytical/validate.py`, `validate_coverage.py`, `validate_qiskit_parity.py`, and `experiments/qpu/run_classical.py --stage preflight`.

On 3 November 2026 all eleven commands were run with this code against the original bundle, mounted read-only or as a copy-on-write overlay. The regenerated split validation, lineage tables, Qiskit parity rows and regression preflight receipt were identical to the saved files. The other receipts differed only in the hash of the script that wrote them and, for the statistics replay, in 498 additional input and graph hash checks.

### Run the pipeline

The original execution lock (`execution_lock_v2.json` in the bundle) pins the protocol, inputs, graphs and code of the saved study. The code in this repository was adapted after that lock (portable paths, formatting), so it cannot continue or overwrite the saved run. It can start a new run in a separate directory with the same protocol and inputs:

```bash
export B26_RUN_DIR=/path/to/new/run          # outside the bundle
python3 experiments/qpu/prepare_run.py       # checks locked inputs and graphs, records the current code
python3 experiments/qpu/train_neural.py --stage preflight --require-ready
```

`prepare_run.py` writes `run_manifest.json` with two inventories: `code_hashes`, the current version of every code file named in the original lock (compared with the lock to list the adapted files), and `code_dependency_hashes`, every repository file the fitting code can execute. The second is the static-import closure of the run entry points and the locked files (`core.code_dependency_inventory`), so it includes files added after the lock, such as `benchmark26/paths.py`. Fitting recomputes both and stops if any file changed, was added or was removed; use a new directory after changing code. Validating saved outputs (`validate_completed.py`) checks data identity only and is unaffected. All settings (dataset, splits, seeds, epochs, candidate sets) come from the original lock. The fitted models and predictions are a new experiment, not a check of the saved one.

| Step | Command |
| --- | --- |
| Dataset table | `python3 experiments/qpu/prepare_dataset.py` |
| Splits | `python3 experiments/qpu/prepare_splits.py`, then `validate_splits.py` |
| Neural models | `python3 experiments/qpu/train_neural.py --stage train --evaluation E1` (also E2, E3) |
| Regression selector | `python3 experiments/qpu/run_classical.py --stage train --evaluation E1` (also E2, E3) |
| Completion check | `python3 experiments/qpu/validate_completed.py` |
| Metrics and contrasts | `python3 experiments/qpu/aggregate_results.py` |
| Analytical estimates | `experiments/analytical/extract_raw.py --full`, then `evaluate.py` |

With `B26_RUN_DIR` set, fitting, completion check and aggregation read and write `training/`, `classical/`, `validation/` and `analysis_v1/` under that directory; inputs are still read from the bundle. Without it they address the saved study, where fitting is refused and validation checks the saved outputs. Neural training needs a CUDA GPU with at least 12 GiB and the Ma–Li Git checkout (`B26_MALI_SOURCE`), from which the pinned model source is loaded.

Shared modules can also be run directly, for example `python3 -m benchmark26.mali_inputs --help`.

## Code identity

Frozen protocols and execution locks record SHA-256 hashes of the code that produced the study. The code here differs from those hashes. Checks against the saved study therefore compare data and results, not code; fitting with this code requires a new run as described above. Files whose hashes are pinned by other frozen files (`protocols/`, including two reference programs in `protocols/reference_code/`) are kept byte-identical; `make check` verifies them.

### Files whose code must stay unchanged

| Set | Files | What may change |
| --- | --- | --- |
| `protocols/` | All frozen contracts, registries and reference programs | Nothing. Frozen files pin each other by SHA-256 |
| Graph-ablation run | 25 files: `benchmark26/` (all modules except `integrity.py`, `logical_inputs.py`, `mali_inputs.py`, `path_estimators.py`, `tables.py`, `codesig.py`), `experiments/graph_ablation/transforms.py`, and in `experiments/qpu/`: `core.py`, `prepare_run.py`, `prepare_dataset.py`, `prepare_splits.py`, `train_neural.py`, `run_classical.py`, `graph_ablation_runner.py`, `validate_splits.py` | Docstrings, comments and layout of the Python files. The code itself and `path_kernel.cpp` must stay identical |

For the 25 files, `make check` compares a signature of each Python file's syntax tree (docstrings removed) with `results/graph_ablation/manifest.json`, so adding a docstring or a comment passes and any other change to the code fails. Names renamed after the run are listed under `renamed_after_run` and mapped back before the comparison; the old names must no longer occur in the file. One file, `benchmark26/paths.py`, gained a mapping from an old script name and is pinned anew, with its run-time signature kept beside the new one. The signature gives the same result on Python 3.10 to 3.14. The manifest records the signatures and byte hashes at the time of the run, and lists the 48 functions of earlier studies that were removed afterwards because no live code reaches them (`removed_after_run`).

All other Python files can be changed freely. A change to the code of a pinned file means the published ablation no longer matches the code. Make such a change only together with a new run.

## Not reproducible from this repository

- Individual raw timing samples and request-level latency quantiles: only aggregates are exported (see [Prediction latency](PREDICTION_LATENCY.md)).
- Historical runs of the earlier 4,515- and 8,767-row component studies. Their protocols are kept in `protocols/` because the current code verifies them, but their data and some of their runners are not part of this release. Earlier timing campaigns were removed in 0.3.0; they remain at the Git tag `archive/legacy-timing-campaigns`.
