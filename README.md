# Benchmark26

Benchmark26 compares neural, classical-regression and analytical estimators that predict the runtime of a quantum circuit on a QPU. It holds the code, the pinned upstream sources and the published aggregate results of one study: 3,443 archived QPU observations from the Ma–Li, Qonductor and QPack sources, grouped into 203 logical-circuit groups.

The runtime labels, circuit caches and fitted models are not in this repository. They form an external study bundle. Without the bundle you can still recompute the published latency and graph-ablation statistics. With it you can replay the study checks.

## Check the published numbers

```bash
python3 -m pip install -e ".[test]"   # NumPy is required by the next command
make check
```

`make check` recomputes the published statistics from the files in `results/`: the prediction-latency intervals (30,000 bootstrap draws) and the graph-ablation group-balanced errors and comparisons. It also parses every Python and protocol file and verifies the SHA-256 chain of the frozen files in `protocols/`. It needs no private data and takes a few seconds.

Next steps:

- `make test` runs the unit tests.
- `make smoke` runs the pipeline end to end on generated data. It shows that the pipeline runs, not that any measured result is reproduced.
- To replay the study against the original bundle, see [Reproduction](docs/REPRODUCTION.md).

## Where to read next

| You want to | Read |
| --- | --- |
| Run the checks, set paths, replay the study or start a new run | [Reproduction](docs/REPRODUCTION.md) |
| Understand the data, the definitions and the metrics | [Methodology](docs/METHODOLOGY.md) |
| Know what the latency numbers include and exclude | [Prediction latency](docs/PREDICTION_LATENCY.md) |
| See whether graph edges help, and how that was tested | [Graph ablation](docs/GRAPH_ABLATION.md) |
| Fix an error message you ran into | [Pitfalls](docs/PITFALLS.md) |

## Layout

| Path | Contents |
| --- | --- |
| `benchmark26/` | Shared code: feature extraction, models, path estimators, path configuration |
| `experiments/qpu/` | Main study: cohort, splits, neural and regression fitting, evaluation |
| `experiments/analytical/` | Analytical duration estimators (Qiskit, QCRE, NoTaDS, HyperQ) |
| `experiments/latency/` | Prediction-latency measurement, cost composition and verification |
| `experiments/graph_ablation/` | Graph ablation: protocol, graph controls, analysis and verification |
| `results/` | Published aggregate results, each with a manifest of file hashes |
| `protocols/` | Frozen contracts of the component studies that the 3,443-observation study builds on. The code reads some of them at run time and `make check` verifies their hashes. You do not need to read them to run or review the code. |
| `upstream/` | Source URLs, revisions and hashes of the upstream data |
| `scripts/` | Setup check, data download, smoke example and verification tools |
| `configs/` | Smoke configuration, study specifications, path template |

## License

Code is released under the MIT license. Upstream software and runtime datasets keep their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Citation metadata is in [CITATION.cff](CITATION.cff).
