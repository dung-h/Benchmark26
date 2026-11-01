# Quantum Runtime Estimation Benchmark

This repository provides the implementation and experimental configuration for
benchmarking quantum-program runtime estimators. It covers circuit-input
preparation, grouped evaluation splits, learned and analytical estimators, and
prediction-cost measurement.

The benchmark pipeline is:

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

## Methodology

The benchmark keeps related logical circuit variants in the same evaluation
group, including variants that differ only in gate parameters. This avoids
placing near-duplicate logical inputs in both training and test partitions.
Feature transforms and model selection are fit within the corresponding
training partition. Metrics are reported with pooled, source-balanced, and
logical-group-balanced weighting where applicable.

Further details are available in:

- [Methodology](docs/METHODOLOGY.md)
- [Data sources and acquisition](docs/DATA_SOURCES.md)

## Environment and data access

`requirements-review.txt` contains the lightweight dependencies used by the
review utilities. Full feature extraction and model fitting additionally depend
on Qiskit, PyTorch, PyTorch Geometric, CUDA, and source-specific environments
referenced by the relevant scripts.

The original QPU datasets and large derived circuit caches are not redistributed
in this repository. The source pins in `upstream/data_sources.json` identify
the downloadable inputs; [DATA_SOURCES.md](docs/DATA_SOURCES.md) describes the
remaining externally supplied inputs and validation steps.

## Reproducibility scope

The scripts retain the study's original directory layout and therefore refer to
input/output locations under `artifacts/`. A new experiment should use a new
output directory and must not overwrite a locked result directory. This release
does not include thesis or paper sources, PDFs, trained checkpoints, saved
predictions, bootstrap draws, or third-party data archives.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party material
and redistribution constraints.
