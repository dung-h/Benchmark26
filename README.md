# Benchmark26

Code for comparing QPU runtime estimators on 3,443 observations from Ma–Li, Qonductor and QPack, organized into 203 logical-circuit groups.

## Check the included results

```bash
python3 -m pip install -e ".[test]"
make check
make test
```

These commands check the code, test fixtures and exported latency statistics, including 30,000 paired bootstrap draws. They use the files in this checkout.

## Try a synthetic example

```bash
python3 scripts/run_benchmark_pipeline.py --stage smoke --output work/smoke
```

This small example runs grouped splitting, preprocessing, regression, prediction and report generation. It uses `configs/smoke.json` and synthetic data; it does not reproduce the measured QPU results.

## Reproduce the measured study

The QPU experiment uses separate entrypoints and a prepared external study bundle. See the [experiment steps](docs/CODE_MAP.md), [data access](docs/DATA_SOURCES.md) and [path setup](docs/PATHS.md). There is no single raw-download-to-QPU-report command.

| Path | Contents |
| --- | --- |
| `experiments/qpu/` | Main QPU data, split, fitting and evaluation scripts |
| `experiments/analytical/` | Analytical estimator evaluation |
| `experiments/latency/` | Portable timing, cost composition and review tools |
| `benchmark_v1/` | Shared circuit features, adapters and kernels |
| `legacy/` | Earlier timing campaigns retained for comparison |
| `configs/` | Executable smoke config, runtime path example and reference specifications |
| `results/` | Versioned aggregate result snapshots |
| `upstream/` | Source URLs, revisions and qualification records |
| `work/` | Generated local outputs; ignored by Git |

[Reproduction guide](docs/REPRODUCTION.md) · [Methodology](docs/METHODOLOGY.md) · [Latency measurement](docs/PREDICTION_LATENCY.md) · [Data boundaries](docs/DATA_BOUNDARIES.md)
