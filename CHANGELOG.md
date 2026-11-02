# Changelog

All notable changes to the Benchmark26 repository are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-11-02

### Added
- Standard project packaging with `pyproject.toml` and optional dependency sets (`test`, `dev`, `analysis`, `full`).
- Unified test suite in `tests/` covering schema validation, recipe screen integrity, group-disjoint split invariants, and metric calculations.
- Declarative experiment configurations in `configs/` (`unified_benchmark.yaml` and `analytical_evaluation.yaml`).
- Makefile with streamlined targets: `make check`, `make test`, `make test-unit`, `make fetch-sources-dry`, and `make help`.
- Automated GitHub Actions CI workflow for linting, syntax validation, and unit test execution across Python 3.10 and 3.11.
- Standard project metadata: `LICENSE` (MIT), `CITATION.cff`, and this `CHANGELOG.md`.
- Canonical benchmark execution CLI: `scripts/run_benchmark_pipeline.py`.
- Step-by-step reproduction guide in `docs/REPRODUCTION.md`.

### Changed
- Refactored `scripts/verify_cost.py` to handle missing external study tables with informative error guidance instead of uncaught tracebacks.
- Replaced ambiguous sprint jargon in docstrings and JSON protocols with standard scientific terminology.
- Bridged methodology documentation and codebase naming (`logical-circuit group` <-> `structural_group`).
- Clarified third-party notices distinguishing upstream software licenses from measurement dataset permissions.

## [0.1.0] - 2026-11-01

### Added
- Initial release of the quantum program runtime estimation benchmark methodology and implementation scaffold.
- Support for 3,443 unified observations across 203 logical-circuit groups spanning Ma–Li, Qonductor, and QPack datasets.
- Implementations for neural estimators (Graph MLP, Global MLP), classical selector (Qonductor adaptation), and analytical duration routes (Qiskit Target, QCRE unitary, NoTaDS paper/code).
- Cryptographic provenance manifests in `upstream/data_sources.json` pinning 249 source files and archives.
