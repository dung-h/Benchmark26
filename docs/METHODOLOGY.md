# Methodology

## Cohort

The study uses 3,443 archived QPU observations in 203 logical-circuit groups.

| Source | Observations | Groups | Runtime label |
| --- | ---: | ---: | --- |
| Ma–Li | 340 | 148 | Reported execution time |
| Qonductor | 3,058 | 49 | Job execution duration |
| QPack | 45 | 6 | Arithmetic mean over 3,854 constituent workflow timings |

The labels come from different clocks. They share a unit, not a measurement boundary, and are interpreted per source.

A logical-circuit group collects circuits that differ only in gate angles, so that closely related inputs never fall on both sides of a group-disjoint split. A group is an evaluation unit, not necessarily one QASM file, job or physical circuit. Code and tables call it `group_id` or `structural_group`.

## Evaluation settings

| Setting | Held out | Folds |
| --- | --- | --- |
| E1 | Logical-circuit groups (unseen structures) | 5 outer × 4 inner |
| E2 | Random observations, stratified by source; groups may repeat | 5 outer × 4 inner |
| E3 | Backend contexts; structures may appear on other backends | 10 outer cases |

Feature transformations are fitted on training partitions only. Neural models select their epoch on the inner validation fold. The Qonductor-style regression selector chooses one model family and candidate per outer fold on inner validation, then refits it on the outer training data; it does not choose per test row.

## Metrics

Errors are absolute errors in seconds. Three weightings answer different questions and are reported separately:

- pooled MAE weights every observation equally;
- source-balanced MAE weights every source equally;
- group-balanced MAE weights every logical-circuit group equally (primary metric).

Paired contrasts compare two pipelines on the same held-out observations with a source-stratified group bootstrap. An interval that contains zero does not establish equivalence.

## Analytical estimators

The analytical routes produce different quantities and are not interchangeable:

- raw nominal duration of one execution (Qiskit Target, QCRE);
- the same duration multiplied by the shot count (a heuristic, not a measured decomposition);
- unitless scores (NoTaDS);
- predictors calibrated on outer-training labels (historical variants in `experiments/analytical/calibrate.py`; not reported in the paper).

## Code map

| Stage | Code |
| --- | --- |
| Logical inputs and grouping | `benchmark26/logical_panel.py`, `benchmark26/circuit_representation.py` |
| Ma–Li features and graph models | `benchmark26/mali_features.py`, `mali_materialize.py`, `mali_training.py`, `mali_batching.py` |
| Qonductor features and regression | `benchmark26/qonductor_features.py`, `qonductor_materialize.py`, `qonductor_regression.py`, `qonductor_models.py` |
| Shot features | `benchmark26/shots.py` |
| Cohort, splits, fitting, evaluation | `experiments/qpu/` |
| Analytical estimators | `experiments/analytical/`, `benchmark26/path_estimators.py`, `benchmark26/path_kernel.cpp` |
| Prediction latency | `experiments/latency/` |

## Protocols

`protocols/` holds the frozen contracts of the component studies on which the unified study builds (Ma–Li features, Qonductor features, estimator execution, logical inputs). Several describe the earlier 4,515- and 8,767-row cohorts. They are kept because the shared code verifies their hashes; they are not the configuration of the 3,443-observation study, whose execution lock is part of the study bundle. `configs/study/` summarizes that study's settings for readers.
