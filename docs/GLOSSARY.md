# Glossary

Plain meanings of names that appear in the code, tables and file names. The names stay as they are because the frozen protocols and the checks in `make check` refer to them by name or hash. [Methodology](METHODOLOGY.md) gives the full definitions of the study terms.

## The data

| Name in the repository | Plain meaning |
| --- | --- |
| cohort, panel | The study dataset: the table of 3,443 observations. Each row is one archived QPU run with its circuit, shot count, backend and runtime label. Code and file names such as `prepare_cohort.py` and `logical_panel.py` build or read this table. |
| `proposed_observations.csv` | The input list of the study dataset, one row per observation. It is the starting table of the pipeline and is part of the external study files, not of this repository. The word "proposed" is historical: the list was drawn up first and then locked as the final dataset. |
| `target_seconds` | The runtime label of an observation, in seconds. Each source defines it differently; see Methodology. |
| source | Where an observation comes from: Ma–Li, Qonductor or QPack. |
| logical-circuit group (`group_id`) | Observations whose circuits differ only in gate angles. The unit of group-disjoint evaluation. |
| external study files, "bundle" | The private folder (`B26_ARTIFACTS`) with the labels, circuit caches, graphs, saved predictions and check receipts. It is not in this repository. |

## Evaluation and models

| Name | Plain meaning |
| --- | --- |
| E1, E2, E3 | The three test settings. E1 holds out whole circuit groups, E2 holds out random observations so groups can repeat, E3 holds out backends. |
| split | A division of the dataset into training, validation and test rows. |
| GB-MAE | Group-balanced mean absolute error, in seconds. Every group counts once. |
| `M1`, `G1`, `G1e`, `G1p` | Graph-ablation models: MLP, Graph, Graph without edges, Graph with shuffled node features. Columns `mae_M1`, `mae_G1`, `mae_G1e` and `mae_G1p` in `results/graph_ablation/group_errors.csv` use these codes. |

## Process words

| Name | Plain meaning |
| --- | --- |
| materialize | Compute the model input (features, graphs or circuits) from the source files and write it to disk. |
| lineage | The trace from the source tables to the dataset rows. `verify_dataset_from_sources.py` (formerly `verify_dataset_lineage.py`) recomputes that trace independently. |
| qualified | Passed the logical-input qualification checks of the original study. The criteria are in `protocols/qualified_logical_benchmark.json`. |
| frozen, lock, freeze | Fixed before the corresponding results were produced. Frozen files carry SHA-256 hashes and pin each other. Do not edit them. |
| receipt | A small JSON file that records what a check read and what it found. |
| smoke | A quick run on generated data. It shows that the pipeline runs, not that a measured result is reproduced. |
