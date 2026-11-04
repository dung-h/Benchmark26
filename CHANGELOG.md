# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow the package version in `pyproject.toml`.

## [0.4.0] - 2026-11-04

### Added
- Graph ablation, a separate experiment on the E1/E2 splits. It has four arms with shots: MLP, Graph+MLP, Graph+MLP without edges, and Graph+MLP with node features permuted relative to the edges; 3 seeds and 120 cells. The experiment adds:
  - `experiments/graph_ablation/` (locked protocol, graph controls, analysis, verifier);
  - `experiments/qpu/graph_ablation_runner.py`;
  - aggregate results in `results/graph_ablation/`;
  - `docs/GRAPH_ABLATION.md`.

  The runner and protocol are byte-identical to the executed run. `analyze.py` is a portable port of the locked analysis script and reproduces the saved summary exactly.
- Timing of the Qiskit and QCRE duration estimators from a ready physical circuit (1,452 tasks, 3,443 observations): aggregates in `results/ready_analytical_latency/`, `experiments/latency/summarize_ready_analytical.py` and `verify_ready_analytical.py`, and a section in `docs/PREDICTION_LATENCY.md`. `make check` verifies the means and the file hashes.
- `make check` recomputes the shared graph-ablation statistics from anonymous per-group errors and checks the files against the hashes in `results/graph_ablation/manifest.json`.
- `docs/PITFALLS.md`: error messages and misreadings, each with cause and fix.
- Tests for the graph controls, the median-of-seeds estimand, the stratified paired bootstrap and the shared results.

### Removed
- Report and figure writers that nothing called and that only wrote narrative reports and plots into the private bundle: `experiments/analytical/write_report.py`, `experiments/analytical/make_figures.py` and `experiments/qpu/write_final_report.py`. Also `scripts/verify_cost.py`, which checked the aggregates of an earlier timing campaign that was removed in 0.3.0 and that no document references.

### Changed
- Code clarity: every public function now has a docstring, the longest `main` functions have step comments, and the copied CSV-reading and hashing helpers of ten scripts are replaced by `benchmark26/tables.py` and `integrity.py`. In the 47 files that no run manifest pins, imports are sorted, unused imports and dead variables are removed, and subprocess checks are explicit. In the 28 Python files that the executed graph-ablation run pinned, only docstrings, comments and layout changed. `make check` now verifies them by the signature of their syntax tree (`benchmark26/codesig.py`, the same on Python 3.10 to 3.14), so a documentation change passes and any change to the code fails. `path_kernel.cpp` stays byte-identical. `docs/REPRODUCTION.md` lists which code must stay unchanged.
- README: the first command now checks the published numbers; a table points each reader task to its document. `docs/METHODOLOGY.md` starts with definitions, and `docs/PREDICTION_LATENCY.md` states what is and is not timed.
- `experiments/qpu/graph_ablation_runner.py` joins the run entry points, so a prepared run pins 29 code files. Run manifests prepared with 0.3.2 list fewer files and are rejected; prepare a new `B26_RUN_DIR`.

## [0.3.2] - 2026-11-03

### Fixed
- A prepared new run pinned only the files named in the original lock, so a change to a file added later (for example `benchmark26/paths.py`, which maps inputs, graphs and outputs) was accepted before fitting. `run_manifest.json` now also records `code_dependency_hashes`, the static-import closure of the run entry points and locked files, and fitting stops if any of these files changed. Manifests without this inventory are rejected; prepare a new `B26_RUN_DIR`. The original lock and saved-output validation are unchanged.

### Added
- Regression tests: a prepared run rejects a change to `benchmark26/paths.py`, and a manifest without the dependency inventory.

### Documentation
- `docs/PREDICTION_LATENCY.md`: which compiled-route circuits equal the circuits scored for accuracy (Ma–Li, QPack) and which do not (all Qonductor observations), and why Qonductor-source costs cannot be paired with its accuracy.

## [0.3.1] - 2026-11-03

### Fixed
- Neural and regression fitting failed before any GPU work with `locked_code_hash_mismatch`: the original execution lock pins the code of the saved study, and this code was adapted after it. Fitting now distinguishes two modes. Without `B26_RUN_DIR` it addresses the original study outputs and still requires the original code. With `B26_RUN_DIR`, `experiments/qpu/prepare_run.py` checks the locked protocol, inputs and graphs and records the current code in a run manifest; fitting then accepts exactly that code and writes only under the new directory. The original lock is not modified.
- `train_neural.py --stage preflight` reported success after checking data and splits only. It now also reports training readiness (lock, inputs, graphs, code identity, run manifest, PyTorch, CUDA, GPU memory); `--require-ready` exits 2 when training could not start.
- `validate_completed.py` and `aggregate_results.py` validate saved outputs by data identity instead of failing on the code lock.

### Added
- `B26_RUN_DIR`, `experiments/qpu/prepare_run.py`, `tests/test_run_identity.py`.

## [0.3.0] - 2026-11-03

### Changed
- Shared code moved from `benchmark_v1/` into the `benchmark26` package with descriptive module names (for example `run_mali_full_features.py` → `mali_training.py`). `benchmark26.paths.ORIGINAL_MODULES` lists the old names.
- Protocols and registries moved to `protocols/`. Five protocols edited in 0.2.0 were restored byte for byte, which repairs the SHA-256 chain that the runners verify.
- All machine-specific locations are read in `benchmark26/paths.py`. Scripts that ignored `B26_ARTIFACTS` or required the bundle to sit inside the checkout now work with an external bundle; receipts keep the original path names.
- Python code formatted with ruff (syntax tree unchanged).
- Documentation consolidated into `README.md` and three guides in `docs/`.

### Added
- Missing pinned files: two method contracts, the compiled-representation materializer, the fold-0 validator and the Ma–Li feature materializer.
- `make check` verifies the protocol hash chain; `tests/test_protocol_integrity.py` does the same in CI.
- `--output`/`--receipt` options for verifiers that previously wrote only into the study bundle.
- Environment variables `B26_COHORT_PLAN` and `B26_WORK`.

### Removed
- `legacy/` (earlier timing campaigns, not used by the current code). Available at tag `archive/legacy-timing-campaigns`.
- `scripts/verify_reference.py` and `scripts/verify_dataset_lineage.py`, duplicates of `experiments/qpu/replay_statistics.py` and `verify_dataset_lineage.py`.
- Placeholder author email and ORCID.

## [0.2.0] - 2026-11-02

### Added
- Packaging, unit tests, Makefile, CI, license, citation metadata.
- Synthetic smoke pipeline and portable path variables.
- Prediction-latency tools and exported aggregate results.

### Changed
- Experiments separated from shared code and earlier timing campaigns.

## [0.1.0] - 2026-11-01

### Added
- Initial code release: cohort of 3,443 observations in 203 groups; neural, regression and analytical estimators; source pins for 249 upstream files.
