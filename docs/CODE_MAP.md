# Experiment steps

## Clean checkout

```bash
make check
make test
python3 scripts/run_benchmark_pipeline.py --stage smoke --output work/smoke
```

The first two commands verify code and included aggregates. The last is a complete synthetic example with generated observations, grouped splits, a fitted model, held-out predictions and a report.

## Measured QPU experiment

Use [path setup](PATHS.md) and [data access](DATA_SOURCES.md) to prepare the required external inputs. The entrypoints below are sequential dependencies, not substitutes for the prepared bundle. The runners also check their own protocols and execution configuration.

| Step | Entrypoint | Produces / checks |
| --- | --- | --- |
| 1. Prepare qualified cohort | `experiments/qpu/prepare_cohort.py` | Joined cohort and feature/target tables |
| 2. Construct splits | `experiments/qpu/prepare_splits.py` | E1/E2/E3 assignments |
| 3. Neural preflight | `experiments/qpu/train_neural.py --stage preflight --evaluation E1` | Neural readiness under the selected evaluation |
| 4. Neural fitting | `experiments/qpu/train_neural.py --stage train --evaluation E1` | Fold/seed models and held-out predictions |
| 5. Regression preflight/fitting | `experiments/qpu/run_classical.py --stage preflight` then `--stage train` | Selected fold-specific regression models and predictions |
| 6. Validate | `experiments/qpu/validate_completed.py` | Completed-result checks |
| 7. Aggregate | `experiments/qpu/aggregate_results.py` | Metrics and paired comparisons |

Select E2/E3 explicitly for their respective runs. Review a new execution configuration before launching; existing original locks describe their producing run. The repository does not currently provide a universal importer for arbitrary external bundles or a single raw-download-to-report QPU command.

## Analytical evaluation and timing

- `experiments/analytical/`: extraction, raw evaluation and historical calibration variants. The raw and calibrated variants have different interpretations.
- `experiments/latency/measure_classical.py`: resident regression inference on supplied features and a fitted model.
- `experiments/latency/compose.py`: per-input addition of matching stages, followed by summary statistics.
- `experiments/latency/verify_summary.py`: included session/source aggregates and paired intervals; also run by `make check`.
- `benchmark_v1/path_estimators/`: shared NoTaDS/HyperQ path implementation.
- `benchmark_v1/shots/`: shared shots transformation/model helpers.

The [legacy directory](../legacy/README.md) contains earlier timing campaigns. The [data boundaries](DATA_BOUNDARIES.md) distinguish source inputs, prepared bundles, result snapshots and new outputs.
