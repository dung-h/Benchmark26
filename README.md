# Quantum Runtime Estimation Benchmark

Predicting QPU runtime helps plan workloads and compare execution choices.
This repository contains code for preparing circuit inputs, comparing neural,
regression and analytical estimators, and measuring the cost of a prediction.
Evaluation covers new circuit groups, new observations and new backend contexts.

The study follows this workflow:

```text
source datasets → qualified circuit inputs → grouped train/test splits
                → estimator fitting → held-out runtime evaluation
```

## Repository structure

| Directory | Contents |
| --- | --- |
| `benchmark_v1/` | Circuit preparation, feature construction, and base model implementations. |
| `research/` | Cohort construction, splitting, estimator training, evaluation, analytical routes, and prediction-cost code. |
| `scripts/` | Utilities for downloading pinned inputs and checking data lineage/selection. |
| `upstream/` | Locations, revisions, and hashes for the original source inputs. |
| `docs/` | Benchmark methodology and data-access notes. |

The [implementation guide](research/README.md) maps each task to its scripts,
including the descriptive meaning of older directory and file names.

## Methodology

For evaluation on new structures, the benchmark keeps related logical circuit
variants in the same group, including variants that differ only in gate
parameters. Random-observation and backend-context evaluations use different
split rules, described in the methodology.
Feature transforms and model selection are fit within the corresponding
training partition. Metrics are reported with pooled, source-balanced, and
logical-group-balanced weighting where applicable.

Further details are available in:

- [Methodology](docs/METHODOLOGY.md)
- [Data sources and acquisition](docs/DATA_SOURCES.md)

## Environment and data access

Install the review dependencies, check the code setup, and inspect download options:

```bash
python3 -m pip install -r requirements-review.txt
python3 scripts/check_setup.py
python3 scripts/fetch_upstream_data.py --help
```

The setup check lists missing packages and representative study inputs.

`requirements-review.txt` contains the lightweight dependencies used by the
review utilities. Full feature extraction and model fitting additionally depend
on Qiskit, PyTorch, PyTorch Geometric, CUDA, and source-specific environments
referenced by the relevant scripts.

The original QPU datasets and large derived circuit caches are not redistributed
in this repository. The source pins in `upstream/data_sources.json` identify
the downloadable inputs; [DATA_SOURCES.md](docs/DATA_SOURCES.md) describes the
remaining externally supplied inputs and validation steps.

## Reproducibility scope

Full execution requires original datasets, intermediate tables, circuit caches,
and experiment configuration. See [Data access](docs/DATA_SOURCES.md) for the
inputs required by each stage. The scripts use `artifacts/` paths for study
inputs and generated results. This code release contains the implementation and
setup documentation; datasets and saved model outputs are supplied separately.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party material
and redistribution constraints.
