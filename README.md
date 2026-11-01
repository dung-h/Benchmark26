# Benchmark26

Code and setup notes for a quantum-runtime-estimation benchmark.

## Start here

```text
original datasets
      ↓
qualify rows and recover usable circuit inputs
      ↓
group related logical circuits and create train/test splits
      ↓
fit estimators and score held-out runtime predictions
```

| If you want to understand... | Read... |
| --- | --- |
| where the inputs come from | [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) |
| how the benchmark is organized | [docs/METHODOLOGY.md](docs/METHODOLOGY.md) |
| logical-circuit preparation | `benchmark_v1/scripts/prepare_logical_benchmark.py` |
| neural/classical benchmark runs | `research/postfreeze_qpu_workflow_mean_expansion_v1/` |
| analytical estimators | `research/postfreeze_unified_analytical_completion_v1/` |
| prediction-cost measurement | `research/postfreeze_qpu_workflow_mean_cost_v1/` |

## Repository map

```text
benchmark_v1/   circuit preparation and base model code
research/       cohort, split, training, evaluation, and cost scripts
scripts/        source-download and consistency-check utilities
upstream/       pinned locations and hashes for original inputs
docs/           short methodology and data-source guides
```

## Before running anything

The benchmark uses original QPU datasets and large derived circuit caches that
are not included here. First read [the data-source guide](docs/DATA_SOURCES.md).
It explains what can be downloaded, what must be supplied separately, and why
the code refuses to silently substitute a newly generated circuit for a missing
historical input.

The scripts preserve the directory layout used by the study, including paths
under `artifacts/`. For a new run, configure a **new output directory**; do not
overwrite an existing result directory.

## Deliberately not included

This is the implementation/setup repository. It does not contain thesis or
paper sources, PDFs, trained checkpoints, saved predictions, bootstrap draws,
or third-party data archives. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for redistribution constraints.
