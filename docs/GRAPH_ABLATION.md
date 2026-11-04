# Graph ablation

This experiment asks whether the dependency edges of a circuit help the Graph+MLP estimator predict QPU runtime. We found no sign that they do. When test and training circuits can share a group (E2), the graph model's lead over the MLP is still there after the edges are removed. When the test groups are unseen (E1), none of the comparisons separates the models.

The experiment is separate from the main benchmark. Its design was written down in [`experiments/graph_ablation/PROTOCOL.md`](../experiments/graph_ablation/PROTOCOL.md) before any model was trained. It uses the same 3,443 observations and 203 circuit groups as the main benchmark.

This page has two parts. Part 1 explains what was tested and what the results show. Part 2 shows how to rerun the analysis and check the numbers.

## Part 1. What was tested and what it shows

### The four models

All four models receive the shot count. The code names in the second column appear in the files and tables.

| Model | Code name | What it sees |
| --- | --- | --- |
| MLP | `M1` | Circuit-level features only |
| Graph | `G1` | Circuit-level features, plus one feature vector per operation, plus the dependency edges between operations |
| Graph without edges | `G1e` | The same, with every edge removed |
| Graph with shuffled nodes | `G1p` | The same edges, but each operation's feature vector is moved to a different node |

Comparing two models isolates one ingredient:

- Graph against graph without edges tests whether the edges help.
- Graph against graph with shuffled nodes tests whether it matters that each node's features match its place in the dependency graph.
- Graph without edges against MLP tests what the per-operation features add when there are no edges.

Two properties of the controls matter when you read the results:

- **Graph without edges still sees operation order.** Each node carries its position in the circuit as a feature, so removing edges does not hide the order of operations.
- **Graph without edges does not have the same effective capacity.** It has the same 842,057 parameters as the Graph model. Without edges, however, the attention weights in each layer take no part in the computation, and a check on one training batch showed zero gradient for them. The model then behaves like one network applied to each operation separately, followed by averaging.

All four models are trained from scratch in the same way. They use Adam, a batch of 32, and at most 500 epochs, and they keep the checkpoint with the lowest validation error. `PROTOCOL.md` has the details. The MLP and Graph models are retrained for this experiment, so every comparison happens inside one run. Do not compare these numbers with the tables in the paper.

### How the results are measured

- **Error.** For each circuit we take the median of the predictions from three random seeds. We average the absolute error within each of the 203 circuit groups, then average over the groups, so every group counts once. We call this the group-balanced MAE (GB-MAE), in seconds.
- **Two test settings.** In E1 the test circuits belong to groups never seen in training. In E2 test and training circuits can share a group.
- **Comparisons.** Each comparison subtracts the GB-MAE of one model from another, group by group. We resample whole groups 10,000 times, within each data source, to get an interval. The six comparisons share one error budget (Bonferroni), so each interval covers 99.17%.
- **What the intervals leave out.** They treat the trained models as fixed. They do not include the variation from retraining.
- **Safety check.** Before training the two ablation models, we retrained the MLP and Graph models for E1 and compared them with the earlier results. The largest difference was 0.018 s, below the planned stop threshold of 0.324 s. This check catches a broken pipeline. It does not prove the runs are equivalent.

### Results

GB-MAE in seconds. Lower is better.

| Setting | MLP | Graph | Graph without edges | Graph with shuffled nodes |
| --- | ---: | ---: | ---: | ---: |
| E1 (unseen groups) | 0.8964 | 0.8644 | 0.8844 | 0.8795 |
| E2 (shared groups) | 0.6464 | 0.5870 | 0.5501 | 0.6923 |

Each row below gives the difference in GB-MAE (first model minus second) and its interval. A negative difference means the first model has lower error. An interval that contains zero means the data do not separate the two models. It does not show that they are equivalent.

| Setting | Comparison | Difference (s) | Interval (s) |
| --- | --- | ---: | --- |
| E1 | Graph − Graph without edges | −0.0199 | [−0.1080, 0.0546] |
| E1 | Graph − Graph with shuffled nodes | −0.0151 | [−0.1047, 0.0655] |
| E1 | Graph without edges − MLP | −0.0120 | [−0.1043, 0.0883] |
| E2 | Graph − Graph without edges | +0.0369 | [−0.0007, 0.0796] |
| E2 | Graph − Graph with shuffled nodes | −0.1053 | [−0.1672, −0.0445] |
| E2 | Graph without edges − MLP | −0.0963 | [−0.1728, −0.0373] |

What the table shows:

1. **Unseen groups (E1).** No comparison separates the models.
2. **Shared groups (E2).** The graph model without edges has lower error than the MLP. So the gain over the MLP does not need the edges.
3. **Edges (E2).** Removing the edges changes the error by an amount the data cannot separate from zero. The point estimate even favors removing them.
4. **Shuffled nodes (E2).** Shuffling the node features raises the error, mostly on the Qonductor-source groups (1.0736 s, against 0.5565 s for the Graph model). This experiment does not test why.

Per-source values are in `results/graph_ablation/summary.json`. QPack has only 6 groups, so its values are descriptive.

### Limits

- The results cover one architecture, one node-feature schema, and the E1 and E2 settings. E3, the held-out-backend setting, was not run.
- The data sources define runtime differently, so the results apply to this dataset and design.
- The results do not show that edges are useless for graph models in general.

## Part 2. Rerun and verify

### Check the shared numbers (no private data needed)

```bash
python experiments/graph_ablation/verify_summary.py
```

This recomputes the 32 GB-MAE values and the six comparisons from `results/graph_ablation/group_errors.csv` and checks them against `summary.json` to 1e-12. `make check` runs it too. It checks the statistics only. It does not check the predictions, the labels or the training, which need the private run directory and the study bundle.

The shared files in `results/graph_ablation/` contain aggregates only:

- `summary.json`: the saved analysis output.
- `group_errors.csv`: the error of each model for each circuit group. Groups are anonymous, numbered in the order used for resampling, and labelled by source.
- `cells.csv`: epochs, best epoch, best validation error, parameter count and fit time for each of the 120 training runs.

### Rerun the training and analysis

You need three things from outside the repository: the study bundle (`B26_ARTIFACTS`), the pinned Ma–Li checkout (`B26_MALI_SOURCE`), and a CUDA GPU with at least 12 GiB. Write the outputs to a new directory.

```bash
export B26_RUN_DIR=/path/to/new/graph_ablation_run      # outside the bundle
python experiments/qpu/prepare_run.py                    # pins the inputs and the 29 code files
python experiments/qpu/graph_ablation_runner.py --stage preflight --require-ready
python experiments/qpu/graph_ablation_runner.py --stage canary
python experiments/qpu/graph_ablation_runner.py --stage train   # E1 then E2
python experiments/graph_ablation/analyze.py --output "$B26_RUN_DIR/analysis/graph_ablation_summary.json" \
    --export-groups "$B26_RUN_DIR/analysis/group_errors.csv"
```

`analyze.py` refuses to run unless all 120 training runs are complete and the safety check passed. It also expects `analysis_lock.json` and `ablation_lock.json` in the run directory. No script in this repository writes them. The original operator wrote them before training and before analysis, and they record:

- the hash of the protocol;
- the hash of the run manifest;
- the setting for activation checkpointing;
- the hash of the analysis script.

Write equivalent files before you read any test label in a new run.

### How the code in this repository relates to the original run

The original run is private and is not distributed.

- The runner (`experiments/qpu/graph_ablation_runner.py`), the graph controls (`transforms.py`), `core.py` and the other 26 pinned Python files have the same code as the executed versions on every path the run uses: their syntax trees match, and only docstrings and comments may differ. After the run, the unused trainers of earlier studies were removed from seven modules (listed in `results/graph_ablation/manifest.json`); a replay of 8 saved checkpoints with the reduced code reproduced the saved predictions (largest difference 1.9e-6 s). `path_kernel.cpp` is byte-identical. `make check` verifies this. `PROTOCOL.md` is identical to the locked protocol.
- `analyze.py` is a port of the locked analysis script (SHA-256 `8d48c0a4…`). The locked script read the bundle from a fixed local path. The port reads it through `B26_ARTIFACTS` and gives a summary identical to the saved one, field for field. Check this with `analyze.py --check <saved summary>`.
