# Benchmark methodology

## Data-to-prediction flow

1. Obtain the original Ma--Li, Qonductor, and QPack inputs using the pinned
   locations in `upstream/data_sources.json`.
2. Build a canonical source table and qualify rows for which the benchmark has
   both an admissible runtime label and a usable circuit representation.
3. Construct logical-circuit groups.  Variants that differ only in gate angles
   may be placed in the same group, so that closely related logical inputs do
   not occur on opposite sides of a group-disjoint split.
   Code and tables use `group_id` or `structural_group` for these evaluation
   groups; these names do not imply a different grouping method.
4. Derive logical/global features and graph inputs from the qualified circuit
   representation.  Fit all transforms on the training partition only.
5. Create outer and inner splits, then fit the neural, classical, and
   analytical routes under their declared evaluation boundary.
6. Score held-out predictions in seconds.  Report pooled, source-balanced, and
   logical-group-balanced metrics separately; do not substitute one weighting
   for another.

## Code map

| Stage | Main implementation |
| --- | --- |
| Logical-input preparation and grouping | `benchmark_v1/scripts/prepare_logical_benchmark.py` |
| Ma--Li-style features and batched graph training | `benchmark_v1/scripts/mali_full_features.py`, `benchmark_v1/scripts/run_mali_batched.py` |
| Qonductor input materialisation | `benchmark_v1/scripts/materialize_qonductor_native_features.py` |
| Unified cohort and split construction | `research/postfreeze_qpu_workflow_mean_expansion_v1/prepare_cohort.py`, `prepare_splits.py` |
| Neural and classical fitting | `train_neural.py`, `run_classical.py` in that same directory |
| Evaluation and diagnostics | `aggregate_results.py`, `analysis_core.py`, `replay_statistics.py` |
| Analytical routes | `research/postfreeze_unified_analytical_completion_v1/` and `research/postfreeze_notads_hyperq_track_a_v1/` |
| Prediction-cost harness | `research/postfreeze_qpu_workflow_mean_cost_v1/`, `research/postfreeze_qpu_cost_completion_v1/`, and `research/postfreeze_qpu_workflow_mean_full_cost_v1/` |

## Important boundaries

- A logical-circuit group is an evaluation unit, not automatically one
  byte-identical QASM file, job, physical circuit, or historically exact source
  program.
- A runtime label is interpreted only within its recorded source-specific
  measurement boundary.  Do not merge different runtime clocks merely because
  they use the same unit.
- Directory names under `research/` identify the original experiment modules.
  The JSON files under `benchmark_v1/protocol/` describe earlier component
  studies, including a 4,515-row cohort. The unified experiment implementation
  is under `research/postfreeze_qpu_workflow_mean_expansion_v1/`; its execution
  lock and input tables are external dependencies. Earlier component protocols
  should not be used as the configuration for the unified experiment.
- The analytical routes distinguish raw nominal quantities, shot-scaled
  quantities, and train-only calibrated predictors.  These are not equivalent
  estimands.
