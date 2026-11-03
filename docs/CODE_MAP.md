# Code map

## Current reader entrypoints

| Purpose | Entrypoint | Required inputs |
| --- | --- | --- |
| Runnable smoke pipeline | `scripts/run_benchmark_pipeline.py --stage smoke --output NEW_DIR` | NumPy and `configs/smoke.json`; synthetic data generated locally |
| Inspect file availability | `scripts/run_benchmark_pipeline.py --status` | Checkout; presence checks are not execution-readiness checks |
| Download upstream sources | `scripts/run_benchmark_pipeline.py --stage acquire --inputs DESTINATION` | Network and pinned source index |
| Review published latency aggregates | `make verify-latency` | Included aggregate CSV files and NumPy |
| Compose matched stage costs | `research/prediction_latency/compose.py` | External per-observation stage CSV |
| Measure fitted regression inference | `research/prediction_latency/measure_classical.py` | External feature CSV, fitted model and compatible environment |

## Main measured QPU experiment

The implementation is retained in `research/postfreeze_qpu_workflow_mean_expansion_v1/`. Its directory name identifies the producing study; these scripts are separate from the synthetic smoke pipeline. They require the prepared study bundle described in DATA_SOURCES.md, source dependencies, compatible CPU/GPU environments and the study's input-validation sequence.

| Step | Script and relevant options |
| --- | --- |
| Prepare cohort | `prepare_cohort.py` |
| Construct E1/E2/E3 splits | `prepare_splits.py` |
| Neural preflight / one-cell / training | `train_neural.py --stage preflight`, `--stage first_cell`, `--stage train`; select `--evaluation E1`, `E2` or `E3` |
| Regression preflight / fitting | `run_classical.py --stage preflight` or `--stage train` |
| Validate completed results | `validate_completed.py` |
| Aggregate validated predictions | `aggregate_results.py` |

These are dependency-aware research entrypoints, not a complete raw-download-to-paper command. In particular, availability of upstream archives alone does not supply the derived feature ledger, model checkpoints, graph caches or original execution locks.

## Supporting and historical implementations

| Directory | Role |
| --- | --- |
| `benchmark_v1/` | Representation contracts, feature extraction, source adapters and pinned upstream implementation support |
| `research/postfreeze_unified_analytical_completion_v1/` | Saved analytical experiment implementation, including historical calibration variants; raw and calibrated variants have distinct meanings |
| `research/postfreeze_notads_hyperq_track_a_v1/` | Original analytical path kernel and supporting historical comparison |
| Other `research/postfreeze_*cost*` / `*parallel*` directories | Earlier timing campaigns, analysis and feature/replay worker orchestration |

Historical modules retain their producing paths because other modules import them. Moving them would require changing and revalidating those dependencies. They are not additional top-level workflows to run by default. The code map separates their role from the main experiment without changing scientific output identities.

## Data and outputs

- `upstream/`: source URLs, revisions and qualification decisions.
- `configs/`: the executable synthetic smoke config, plus reference specifications for measured studies. The review CLI consumes only its smoke JSON.
- `work/smoke/` (created by the example command): disposable synthetic run outputs.
- `results/prediction_latency/`: generated, versioned aggregate results for independent review.
- External `B26_ARTIFACTS`: prepared study inputs and experiment outputs. Original references and output identities remain part of each study's protocol.
