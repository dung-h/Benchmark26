# Graph ablation

This is a separate experiment from the main benchmark, run in November 2026. It asks which parts of the Graph+MLP input explain its difference from the MLP on the 3,443-observation cohort. The design was fixed in [`experiments/graph_ablation/PROTOCOL.md`](../experiments/graph_ablation/PROTOCOL.md) before fitting; the analysis script was locked before any new test metric was read.

## Design

The four arms all use the shot count. They share the seeds 42, 1234 and 31415 and the original E1/E2 outer and inner splits.

| Arm | Graph input | What a contrast with it isolates |
| --- | --- | --- |
| `M1` | none (global-feature MLP) | reference |
| `G1` | original logical-circuit DAG | reference graph model |
| `G1e` | the same nodes, no edges | `G1 − G1e`: message passing along the DAG |
| `G1p` | the same edges, node feature rows permuted | `G1 − G1p`: matching each node's features to its position in the DAG |

`G1e − M1` asks what the node set adds without edges.

Points to keep in mind when reading the arms:

- **Edges.** The DAG edges are directed and deduplicated. Nodes are the operations plus each qubit's input and output node (`benchmark26/mali_features.py`).
- **G1e is not position-blind.** Node features include the operation's gate index (column 177), so this arm still sees operation order through its features.
- **G1e has the same parameter count but not the same effective capacity.** All graph arms have 842,057 parameters. Without edges, though, each `TransformerConv` layer reduces to its skip projection, and the attention projections receive zero gradient. G1e is therefore effectively a per-node network followed by mean pooling.
- **G1p is not a relabelling of the graph.** The permutation is deterministic for each observation (SHA-256 of the observation ID plus `graph-ablation-v1`). It moves feature rows relative to fixed edges.

All arms are trained from scratch in the same way: Adam (learning rate 0.0005, weight decay 0.0001), effective batch 32, at most 500 epochs, keeping the minimum-validation checkpoint (details in `PROTOCOL.md`). `M1` and `G1` are rerun so that every contrast is computed within this run. Do not mix these numbers with the paper tables.

## Estimand and statistics

- **Metric.** For each observation, take the median of its three seed predictions. Take each group's mean absolute error, then the mean over the 203 groups (GB-MAE, seconds). This is an ensemble estimand. It is not the mean of the three per-seed GB-MAEs, which `summary.json` reports separately as `seed_group_balanced_mae_seconds`. The historical benchmark used the same estimand.
- **Contrasts.** Three paired contrasts in each of E1 and E2, six in total. Each uses 10,000 whole-group bootstrap draws (seed 42), resampled within source strata so that each source keeps its group count. Intervals are Bonferroni-adjusted two-sided 99.1667% (1 − 0.05/6).
- **What the intervals include.** They are conditional on the fitted predictions and do not include retraining variation.
- **Baseline check.** Before the ablation arms were trained, the reruns of E1 `M1` and `G1` were compared with their historical GB-MAE. This was an operational stop rule, not an equivalence test. The threshold was the width of the historical adjusted G1−M1 interval, 0.324 s. The observed differences were 0.016 s and 0.018 s.

## Results

All 120 cells completed. GB-MAE in seconds:

| Evaluation | M1 | G1 | G1e | G1p |
| --- | ---: | ---: | ---: | ---: |
| E1 (held-out groups) | 0.8964 | 0.8644 | 0.8844 | 0.8795 |
| E2 (random observations) | 0.6464 | 0.5870 | 0.5501 | 0.6923 |

Paired contrasts, with Bonferroni-adjusted 99.1667% intervals:

| Evaluation | Contrast | Δ GB-MAE (s) | Interval (s) |
| --- | --- | ---: | --- |
| E1 | G1 − G1e | −0.0199 | [−0.1080, 0.0546] |
| E1 | G1 − G1p | −0.0151 | [−0.1047, 0.0655] |
| E1 | G1e − M1 | −0.0120 | [−0.1043, 0.0883] |
| E2 | G1 − G1e | +0.0369 | [−0.0007, 0.0796] |
| E2 | G1 − G1p | −0.1053 | [−0.1672, −0.0445] |
| E2 | G1e − M1 | −0.0963 | [−0.1728, −0.0373] |

What the results support:

- **E1.** None of the three contrasts is distinguished from zero. This is not evidence that the arms are equivalent.
- **E2, node set.** The edgeless node model has lower error than the MLP.
- **E2, edges.** Removing the edges is not distinguished from keeping them; the point estimate favours removing them.
- **E2, permuted features.** Message passing over misaligned features has higher error than over the original alignment. Most of this difference comes from the Qonductor-source groups (Qonductor-source GB-MAE 1.0736 for G1p against 0.5565 for G1).
- **Taken together.** In E2, the measured advantage of the graph pipeline over the MLP is attributable to node-level information. A positive contribution of the dependency edges themselves is not detected.

Per-source values are in `results/graph_ablation/summary.json`. QPack has 6 groups, so its values are descriptive only.

Limits:

- One architecture and one node schema.
- Source-specific label boundaries.
- E1/E2 only (E3 was not run).
- Fixed fitted predictions.

The results do not establish a general property of graph neural networks for circuit runtime.

## Run it

Inputs come from outside the repository:

- the study bundle (`B26_ARTIFACTS`);
- the Ma–Li checkout pinned in `benchmark26/mali_training.py` (`B26_MALI_SOURCE`);
- a CUDA GPU with at least 12 GiB.

Outputs go to a new run directory:

```bash
export B26_RUN_DIR=/path/to/new/graph_ablation_run      # outside the bundle
python experiments/qpu/prepare_run.py                    # pins inputs and the 29-file code inventory
python experiments/qpu/graph_ablation_runner.py --stage preflight --require-ready
python experiments/qpu/graph_ablation_runner.py --stage canary
python experiments/qpu/graph_ablation_runner.py --stage train   # E1 then E2; baseline check after E1 M1/G1
python experiments/graph_ablation/analyze.py --output "$B26_RUN_DIR/analysis/graph_ablation_summary.json" \
    --export-groups "$B26_RUN_DIR/analysis/group_errors.csv"
```

`analyze.py` refuses to run unless all 120 cells are complete and the baseline check passed. Its lock check expects `analysis_lock.json` and `ablation_lock.json` in the run directory, and these are not written by a script in this repository. The original operator wrote them before fitting and before analysis. They record:

- the protocol hash;
- the run-manifest hash;
- the activation-checkpointing setting;
- the hash of the analysis script that was used.

A new run must write equivalent locks before it reads test labels.

## Code identity of the original run

The original run is `attempt_03`; it is private and is not distributed. Its run manifest pins 29 code files. In this repository:

- `experiments/qpu/graph_ablation_runner.py`, `experiments/graph_ablation/transforms.py`, `experiments/qpu/core.py` and the other pinned files are byte-identical to the executed versions. They are left unformatted on purpose.
- `PROTOCOL.md` is identical to the locked protocol.
- `analyze.py` is a portable port of the locked analysis script. That script (SHA-256 `8d48c0a4…`) read the bundle from a fixed local path. On the original run the port gives a summary identical to the saved one, field for field (`analyze.py --check`).

## Verify the shared results

`results/graph_ablation/` contains only aggregates:

- `summary.json`: the saved analysis output;
- `group_errors.csv`: anonymous per-group MAE for each arm, with groups numbered in resampling order and labelled by source;
- `cells.csv`: per-cell epochs, best epoch, best validation MSE, parameter count and fit time.

Without the bundle:

```bash
python experiments/graph_ablation/verify_summary.py
```

This recomputes the 32 GB-MAE values and the six contrasts with the same bootstrap draws, and compares them with `summary.json` (tolerance 1e-12). It checks the statistics. It does not check the predictions, labels or training; those require the private run directory and the bundle.
