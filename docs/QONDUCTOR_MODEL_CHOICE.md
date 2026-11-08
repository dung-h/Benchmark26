# Which model the Qonductor procedure chose

This page explains which regression model the Qonductor procedure chose in each split of the benchmark, and how to check that choice.

## Result

Tree ensembles won almost every split, and polynomial regression won only two. This differs from the Qonductor paper, where polynomial regression won on the authors' own data (execution-time R² = 0.998).

The Qonductor procedure tunes six model families and keeps the candidate with the highest mean R² in seconds over four inner folds of the training data. The table gives, for each family, the mean of its best score over the splits and how often it was chosen.

| Family | E1: mean R² | E1: chosen | E2: mean R² | E2: chosen | E3: mean R² | E3: chosen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Extra Trees | 0.73 | 5 of 5 | 0.88 | 2 of 5 | 0.78 | 8 of 10 |
| Gradient Boosting | 0.70 | 0 | 0.88 | 0 | 0.75 | 0 |
| Random Forest | 0.69 | 0 | 0.88 | 0 | 0.74 | 0 |
| Histogram Gradient Boosting | 0.68 | 0 | 0.88 | 3 of 5 | 0.74 | 0 |
| AdaBoost | 0.62 | 0 | 0.79 | 0 | 0.67 | 0 |
| Polynomial regression | 0.50 | 0 | 0.80 | 0 | 0.64 | 2 of 10 |

Polynomial regression won when the held-out backend was lagos or perth (E3). In the random split (E2), the four tree ensembles are within 0.005 of each other.

These are selection scores on the training part of each split, not test scores. Only the chosen model was refitted and tested; its test errors are in the paper and the study files.

A likely reason for the difference from the Qonductor paper: this dataset pools three sources with very different runtimes, and E1 and E3 hold out whole circuit groups or backends. This benchmark did not test that explanation.

## Definitions

- **E1, E2, E3:** held-out circuit groups (5 splits), random runs (5 splits) and one held-out backend at a time (10 splits). `docs/METHODOLOGY.md` defines them.
- **Candidates per split:** 48 for AdaBoost, 3 polynomial degrees, and 64 settings for each other family.

## Rerun or check

`make check` runs `experiments/qonductor_selection/verify_summary.py`. It needs no study files. It checks the file hashes, that each split chose its highest-scoring family, and that the summary matches the per-split table.

To rebuild the tables from the saved candidate scores, unzip `benchmark26-study-files.zip` as described in [REPRODUCTION.md](REPRODUCTION.md), then run:

```bash
export B26_ARTIFACTS=/path/to/benchmark26-study-files/artifacts
python3 experiments/qonductor_selection/summarize.py --output work/qonductor_selection
diff work/qonductor_selection/family_scores.csv results/qonductor_selection/family_scores.csv
```

The script also checks that the chosen family is the one named in the saved test predictions.
