# Methodology

## Definitions

These terms are used the same way in every document and in the code.

- **Observation**: one row of the study dataset: one archived QPU run, or for QPack the mean over the timings of one workflow. It has a circuit, a shot count, a backend and one archived runtime label.
- **Label**: the archived runtime of an observation, in seconds. Each source defines it differently (see the table below).
- **Source**: where the observation comes from. There are three: Ma–Li, Qonductor and QPack.
- **Logical circuit**: the circuit before it is compiled to a backend's native gates. The neural estimators take it as input. For Qonductor it is reconstructed from the circuit family and width, not recovered byte for byte.
- **Physical circuit**: the circuit after compilation to a backend's native gates. The analytical estimators score the saved physical circuit of each observation. For Qonductor this is the circuit that was submitted.
- **Logical-circuit group**: observations whose circuits differ only in gate angles. Details below.
- **Estimator**: anything that maps an input to a predicted runtime: a neural model, the regression selector, or an analytical duration formula.
- **Group-balanced MAE (GB-MAE)**: the mean absolute error in seconds, averaged within each group and then over groups, so every group counts once. It is the primary metric.

## Dataset

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

## Terms from the original study process

`benchmark26/circuit_representation.py` still has the preparation steps of the earlier study under the names C133 (backend snapshot registry), C134 (feature sidecar) and C135 (target preflight). They produce inputs of the 3,443-observation study or check them; they are not part of the evaluation.

The trainers, tuners and approval checks that belonged only to the earlier 8,767-row Ma-Li study and the 340-row Qonductor study were removed from the code. They produced no input of the current study.

## Code map

| Stage | Code |
| --- | --- |
| Logical inputs and grouping | `benchmark26/logical_inputs.py`, `benchmark26/circuit_representation.py` |
| Ma–Li features and graph models | `benchmark26/mali_features.py`, `mali_inputs.py`, `mali_training.py`, `mali_batching.py` |
| Qonductor features and regression | `benchmark26/qonductor_features.py`, `qonductor_inputs.py`, `qonductor_regression.py`, `qonductor_models.py` |
| Shot features | `benchmark26/shots.py` |
| Dataset table, splits, fitting, evaluation | `experiments/qpu/` |
| Analytical estimators | `experiments/analytical/`, `benchmark26/path_estimators.py`, `benchmark26/path_kernel.cpp` |
| Prediction latency | `experiments/latency/` |

## Protocols

`protocols/` holds the frozen contracts of the component studies on which the unified study builds (Ma–Li features, Qonductor features, estimator execution, logical inputs). Several describe the earlier 4,515- and 8,767-row cohorts. They are kept because the shared code verifies their hashes; they are not the configuration of the 3,443-observation study, whose execution lock is part of the study bundle.
