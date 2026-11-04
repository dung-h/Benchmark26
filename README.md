# Benchmark26

Code and aggregate results for a benchmark of QPU runtime estimators. The study compares neural, classical-regression and analytical estimators on 3,443 archived QPU observations from the Ma–Li, Qonductor and QPack sources, grouped into 203 logical-circuit groups.

The repository contains the experiment code, the frozen study protocols, source pins for the upstream data, and the exported prediction-latency statistics. It does not contain the runtime labels, circuit caches, fitted models or other study data; these form an external study bundle (see [Reproduction](docs/REPRODUCTION.md)).

## Quick start

```bash
python3 -m pip install -e ".[test]"
make check    # syntax, protocol hash chain, exported latency statistics (30,000 bootstrap draws)
make test     # unit tests
make smoke    # synthetic split–fit–evaluate–report example in work/smoke
```

`make smoke` uses generated data. It exercises the pipeline but does not reproduce any measured result.

## Layout

| Path | Contents |
| --- | --- |
| `experiments/qpu/` | Main study: cohort, splits, neural and regression fitting, evaluation |
| `experiments/analytical/` | Analytical duration estimators (Qiskit, QCRE, NoTaDS, HyperQ) |
| `experiments/latency/` | Prediction-latency measurement, cost composition and verification |
| `experiments/graph_ablation/` | Graph ablation (separate experiment): protocol, graph controls, analysis and verification |
| `benchmark26/` | Shared code: feature extraction, models, path estimators, path configuration |
| `protocols/` | Frozen study contracts, registries and reference code, verified by SHA-256 |
| `results/` | Exported aggregate results |
| `upstream/` | Source URLs, revisions and hashes of the upstream data |
| `scripts/` | Setup check, data download, smoke example and verification tools |
| `configs/` | Smoke configuration, study specifications, path template |

## Documentation

- [Reproduction](docs/REPRODUCTION.md): data, environment variables, pipeline and verification commands
- [Methodology](docs/METHODOLOGY.md): cohort, evaluation settings, metrics and estimands
- [Prediction latency](docs/PREDICTION_LATENCY.md): what the latency results measure and how to verify them
- [Graph ablation](docs/GRAPH_ABLATION.md): which parts of the graph input matter, results and how to verify them

## License

Code is released under the MIT license. Upstream software and runtime datasets keep their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Citation metadata is in [CITATION.cff](CITATION.cff).
