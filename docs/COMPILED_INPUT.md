# Neural models with compiled-circuit inputs

Neural runtime models given the compiled circuit had lower point group-balanced mean absolute error (GB-MAE) in this study, but the study does not show that compilation causes the difference. The new models also use different training code and shot encoding than the saved baselines. Treat every comparison here as exploratory.

The study uses the same 3,443 observations in 203 circuit-structure groups as the main benchmark, with the original E1 (held-out groups) and E2 (random observations) splits. It adapts the Ma–Li models. It is inspired by [Azizov et al.](https://arxiv.org/abs/2609.12980), who predict simulator time, and it does not reproduce that study. All models start from fresh initialization, so these are not transfer-learning results.

## Methods

The file and column names use short codes, listed here.

| Code | Name | Inputs |
| --- | --- | --- |
| `TM_structural` | Compiled-feature MLP | 51 logical + 51 compiled global features, 18 log1p compiled descriptors, log1p shots (121 inputs) |
| `TG_structural` | Compiled-DAG Graph+MLP | Full physical DAG with the Ma–Li 178-feature node encoding, 51 logical features, log1p shots |
| `TGF_structural` | Compiled-DAG + global Graph+MLP | The same DAG with the 121 global inputs |
| `M1`, `G1`, `R0` | MLP + shots, Graph+MLP + shots, Qonductor | Saved results from the main benchmark |

"Compiled" and "transpiled" mean the same thing here. The study adds no gate-error, readout-error or gate-duration node features. Compiled inputs for Ma–Li and QPack are nominal recompilations. Those for Qonductor are the submitted physical circuits. Backend snapshots are nominal. Each prediction is the median of three seed models (42, 1234, 31415) across five folds, 90 training cells in all.

## Accuracy

The table gives GB-MAE in seconds, with every group weighted equally (148 Ma–Li, 49 Qonductor, 6 QPack).

| Method | E1 | E2 |
| --- | ---: | ---: |
| MLP + shots | 0.880015 | 0.598162 |
| Graph+MLP + shots | 0.846774 | 0.584411 |
| Qonductor | 0.796531 | 0.637630 |
| Compiled-feature MLP | 0.706774 | 0.512320 |
| Compiled-DAG Graph+MLP | 0.787098 | 0.518370 |
| Compiled-DAG + global Graph+MLP | 0.696063 | 0.505408 |

Three limits apply to this table:

- The compiled-DAG + global model has the lowest GB-MAE, but its difference from the compiled-feature MLP has an interval containing zero in both settings. That leaves the value of the DAG branch unresolved; it does not show equivalence.
- Qonductor keeps the lowest pooled MAE in E1 (0.931825 s). In E1 the compiled-feature MLP has higher error than Qonductor on the Qonductor-source groups.
- `contrasts.csv` gives the seven comparisons per setting with 99.2857% paired whole-group bootstrap intervals. The intervals condition on the fitted predictions.

## Device-ready prediction time

The clock starts with normalized inputs already on the GPU and stops after the CPU median of three seed predictions. It excludes parsing, compilation, feature extraction, normalization, input transfer and model loading, so it is not the latency from a logical circuit to a prediction. It cannot be compared with the timings in [PREDICTION_LATENCY.md](PREDICTION_LATENCY.md).

Two sessions on one NVIDIA GeForce RTX 5070 Ti timed every observation for each method and setting. The group-balanced mean is about 0.34 ms for the compiled-feature MLP and about 140 ms for each graph model, dominated by large Ma–Li DAGs. See `results/compiled_input/ready_prediction_latency/`.

## Check the results

Run `make check`, or the two scripts directly:

```bash
python experiments/compiled_input/verify_summary.py        # MAE and the 14 accuracy intervals
python experiments/compiled_input/verify_ready_prediction.py  # timing summary and intervals
```

These scripts recompute the published summaries from anonymous per-group values. They do not repeat training, checkpoint replay or timing. Rerunning the training scripts needs the study files, the compiled inputs and a GPU; set `B26_ARTIFACTS` and `B26_RUN_DIR`.

## What was run and what is not included

Two scripts were renamed after the run, and imports were updated to match: `train_compiled_dag.py` was `train_physical_graph_v4.py`, and `build_structural_features.py` was `materialize_structural.py`. The folder was `transpiled_input`. Their logic is unchanged. Names such as `canonical_observation_id` and `structural_features_draft.csv` are spellings inside the saved study files. Row-level predictions, labels, circuits, graphs and checkpoints are not in this repository.

The first graph cell (E1, fold 0, compiled-DAG Graph+MLP, seed 42) continued from an epoch-11 checkpoint imported across cache and runner versions; `cells.csv` marks it. It was not rerun from scratch. A full inference replay from the saved checkpoints matched all 61,974 predictions to within 5.8e-6 s; that replay used private inputs and is not independent retraining.
