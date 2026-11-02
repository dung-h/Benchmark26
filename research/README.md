# Implementation guide

## Data preparation and models

The unified benchmark uses the following scripts under
`postfreeze_qpu_workflow_mean_expansion_v1/`:

| Task | Script |
| --- | --- |
| Build the cohort and join circuit features | `prepare_cohort.py` |
| Construct evaluation splits | `prepare_splits.py` |
| Train MLP and Graph+MLP, with and without shots | `train_neural.py` |
| Select and fit regression models | `run_classical.py` |
| Aggregate held-out predictions | `aggregate_results.py` |
| Calculate paired statistics | `analysis_core.py`, `replay_statistics.py` |

Base feature and model implementations are in `../benchmark_v1/scripts/`.
The shots helper is `postfreeze_mali_shots_ablation_v1/shots_core.py`.

## Analytical estimators

`postfreeze_unified_analytical_completion_v1/` contains feature extraction
(`extract_raw.py`), training-set calibration (`calibrate.py`) and evaluation
(`evaluate.py`). The NoTaDS and dependency-path kernels are in
`postfreeze_notads_hyperq_track_a_v1/`.

## Prediction latency

| Measurement task | Directory / script |
| --- | --- |
| Prediction from prepared or raw features | `postfreeze_qpu_workflow_mean_cost_v1/run.py`, `timing.py` |
| Circuit parsing and feature extraction | `postfreeze_qpu_workflow_mean_full_cost_v1/run_circuit.py` |
| Fresh-process model startup | `postfreeze_qpu_workflow_mean_full_cost_v1/run_cold.py` |
| Schedule bounded diagnostic measurements | `postfreeze_qpu_cost_completion_v1/campaign.py` |
| Circuit-to-prediction diagnostic | `postfreeze_qpu_cost_completion_v1/run_c.py`, `run_c_fixed.py` |
| Analytical/compilation CPU worker | `postfreeze_qpu_cost_completion_v1/cpu_task.py` |
| Resume outstanding measurements | `postfreeze_qpu_cost_completion_v1/execute_remaining.py` |
| Collect parallel measurement results | `postfreeze_qpu_workflow_mean_parallel_cost_v1/finish.py` |
| Aggregate accuracy and latency | `postfreeze_qpu_workflow_mean_cost_analysis_v1/` |

In older identifiers, “Clock C” means circuit-to-prediction elapsed time.
“Canary” means a small trial measurement used to check inputs and prediction
agreement before running the larger measurement set. Neither term names a
model. Cost measurements have distinct starting inputs; compare methods only
within a shared measurement boundary.

## Configuration and inputs

Directory names identify the study modules used to produce the original
experiments. They are retained because imports, execution configurations and
input hashes refer to those paths. The code also includes earlier component
studies; their populations and configurations differ from the unified study.
Use the task map above rather than treating all directories as interchangeable
implementations of one experiment.

The scripts require external study tables and caches. Their dependencies are
listed in [Data access](../docs/DATA_SOURCES.md). Source inspection and setup
checks work from the code checkout; training and result verification require
the corresponding study inputs.
