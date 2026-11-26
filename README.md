# Benchmark26

Benchmark26 compares neural, classical-regression and analytical estimators that predict the runtime of a quantum circuit on a QPU. It holds the code, the pinned upstream sources and the published aggregate results of one study: 3,443 archived QPU observations from the Ma–Li, Qonductor and QPack sources, grouped into 203 logical-circuit groups.

## Download the dataset

The runtimes of all 3,443 runs are published as one table, `runs.csv`, in the [benchmark26-data folder](https://drive.google.com/drive/folders/1lXiRepBX3S7rzdscwnVyqXfis1qcp8-K) on Google Drive. Each row is one run (for QPack, the mean over the timings of one workflow): source, circuit group, backend, shot count and runtime in seconds. The folder's `README.md` explains every column, and `SHA256SUMS` holds the checksum (`86ee98ff2408…`). [`scripts/export_dataset.py`](scripts/export_dataset.py) shows how the table was made from the study files.

The same folder holds the study files needed to replay the study and retrain the models (51 MB), the weights of 241 trained neural models fitted on inner folds 1–3 (2.35 GB), the finished graph-ablation run (94 MB) and the compiled circuits needed to rebuild features (376 MB), as four zip files. Its `README.md` says how to unzip them and set the paths; [Reproduction](docs/REPRODUCTION.md) lists the commands. Without the bundle you can still recompute the published latency and graph-ablation statistics. With it you can replay the study checks.

The neural errors in the paper come from models retrained on the whole outer-training partition (`experiments/qpu/retrain_neural.py`). Their seed predictions, medians and error metrics are in [`results/retrained_neural/`](results/retrained_neural/); `experiments/qpu/verify_retrained.py --runs runs.csv` recomputes the metrics.

The predictions of the Qonductor model and the training median are in `benchmark26-study-files.zip`, in `artifacts/postfreeze_qpu_workflow_mean_expansion_v1/analysis_v1/reduced_predictions.csv`. The `training/` folder next to it holds the neural models fitted on inner folds 1–3, which give the retraining its epoch counts; `experiments/qpu/replay_statistics.py` recomputes the error metrics of these study files. The analytical estimates are in `artifacts/postfreeze_unified_analytical_completion_v1/predictions.csv`, and `experiments/analytical/validate_portable.py` recomputes their metrics.

## Relation to the paper

| Paper content | Source in this repository |
| --- | --- |
| Overall, per-source and per-backend errors; correlations | Neural models: `results/retrained_neural/`; Qonductor model and training median: study bundle; analytical estimates: study bundle |
| Prediction time (table and accuracy–time figure) | `results/prediction_latency/`, `results/ready_analytical_latency/` |
| Scholten-style estimate | `results/scholten/` |
| Model family chosen by the Qonductor procedure | `results/qonductor_selection/` |
| Graph model without edges | `results/graph_ablation/` |
| Exploratory compiled-circuit inputs | `results/compiled_input/` |

The repository also contains material that the paper does not report: bootstrap intervals for paired comparisons and latencies, and the NoTaDS and HyperQ duration estimators in `experiments/analytical/`.

## Check the published numbers

```bash
python3 -m pip install -e ".[test]"   # NumPy is required by the next command
make check
```

`make check` recomputes the published statistics from the files in `results/`: the prediction-latency intervals (30,000 bootstrap draws) and the graph-ablation group-balanced errors and comparisons, and checks the retrained neural predictions. It also parses every Python and protocol file and verifies the SHA-256 chain of the frozen files in `protocols/`. It needs no private data and takes a few seconds.

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
| See how the Scholten-style throughput estimate scores, and why it overestimates | [Scholten-style estimate](docs/SCHOLTEN.md) |
| See which regression model the Qonductor procedure chose in each split | [Qonductor model choice](docs/QONDUCTOR_MODEL_CHOICE.md) |
| See whether compiled-circuit inputs help neural models (exploratory) | [Compiled-circuit inputs](docs/COMPILED_INPUT.md) |
| Fix an error message you ran into | [Pitfalls](docs/PITFALLS.md) |

## Layout

| Path | Contents |
| --- | --- |
| `benchmark26/` | Shared code: feature extraction, models, path estimators, path configuration |
| `experiments/qpu/` | Main study: dataset table, train/test splits, neural and regression fitting, evaluation |
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
