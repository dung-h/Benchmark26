# Benchmark26

Code and protocols for comparing quantum-processing-unit (QPU) runtime
estimators. The study compares Ma–Li-style MLP and graph models, a Qonductor
regression selector, and analytical timing routes on 3,443 observations from
the Ma–Li, Qonductor and QPack corpora.

This checkout contains both the reusable code and the records needed to audit
the study design. The original datasets, derived feature tables, model
checkpoints and some environment-specific inputs are not included.

## What can be run from a clean checkout?

Install the lightweight review environment:

```bash
python3 -m pip install -e ".[test]"
```

Run the local checks:

```bash
make check
make test
make verify-latency
```

Run the small synthetic smoke pipeline:

```bash
python3 scripts/run_benchmark_pipeline.py \
  --stage smoke \
  --output work/smoke
```

The smoke run exercises grouped splitting, train-only preprocessing, fitting,
held-out prediction and metric/report writing. It uses synthetic data and does
not reproduce the QPU results.

To inspect which external study inputs are present:

```bash
python3 scripts/run_benchmark_pipeline.py --status
```

## What is the main research pipeline?

The measured QPU experiment is implemented under
`research/postfreeze_qpu_workflow_mean_expansion_v1/`. Its main stages are:

1. `prepare_cohort.py` — prepare and join the qualified cohort;
2. `prepare_splits.py` — construct the E1, E2 and E3 evaluation splits;
3. `train_neural.py` — run neural preflight, one-cell checks or training;
4. `run_classical.py` — run regression preflight or fitting;
5. `validate_completed.py` — validate completed outputs;
6. `aggregate_results.py` — aggregate held-out predictions.

These scripts are the entrypoints for the measured experiment, but they are
not a single raw-download-to-report command. They require the external study
bundle described in [Data sources](docs/DATA_SOURCES.md), including prepared
cohort tables, feature ledgers, graph caches, checkpoints and execution locks.
The synthetic smoke command above is intentionally separate from this pipeline.

## Repository map

| Path | Role |
| --- | --- |
| `benchmark_v1/` | Feature construction, source adapters and frozen component protocols |
| `configs/` | Smoke-run configuration and reference specifications for measured studies |
| `research/` | Main experiment implementations and supporting historical modules |
| `scripts/` | Setup, acquisition, verification and smoke-pipeline entrypoints |
| `tests/` | Unit and invariant tests for schemas, splits, metrics and analytical routes |
| `upstream/` | Source URLs, revisions, checksums and qualification records |
| `results/` | Versioned aggregate outputs for the prediction-latency analysis |
| `docs/` | Methodology, data access, path configuration and reproduction instructions |

The `postfreeze_*` names under `research/` identify the studies that produced
the corresponding artifacts. They are not interchangeable implementations of
one generic experiment. See the [code map](docs/CODE_MAP.md) before running a
research script.

## Documentation

- [Code map](docs/CODE_MAP.md) — runnable entrypoints and historical modules
- [Reproduction guide](docs/REPRODUCTION.md) — verification levels and study-bundle workflow
- [Data sources](docs/DATA_SOURCES.md) — downloads, hashes and external inputs
- [Path configuration](docs/PATHS.md) — portable data and interpreter locations
- [Methodology](docs/METHODOLOGY.md) — labels, groups, splits and metrics
- [Prediction latency](docs/PREDICTION_LATENCY.md) — latency boundaries and exported summaries

## Important interpretation boundary

The repository's local checks verify code structure, selected invariants and
exported aggregate results. They do not prove that the full QPU experiment can
be replayed without the external study bundle and its compatible scientific
software environments.
