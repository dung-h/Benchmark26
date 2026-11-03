# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow the package version in `pyproject.toml`.

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
