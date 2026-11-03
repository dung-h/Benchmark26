# Experiments

| Directory | Purpose | Inputs |
| --- | --- | --- |
| `qpu/` | Main measured QPU comparison | Prepared cohort, feature ledger, graphs, splits and execution configuration |
| `analytical/` | Raw/shot-scaled analytical evaluation and historical calibrated variants | Prepared circuit/backend representations and labels |
| `latency/` | Timing, stage-cost composition and exported-statistic verification | Included aggregates for verification; external features/models for new measurements |

For exact commands, see [code map](../docs/CODE_MAP.md). The synthetic smoke example is implemented separately in `scripts/smoke_pipeline.py`.

Shared features and source adapters are under `benchmark_v1/scripts/`; shots helpers are under `benchmark_v1/shots/`; NoTaDS/HyperQ path kernels are under `benchmark_v1/path_estimators/`. Earlier timing implementations are grouped under `legacy/`.

Source paths are descriptive. Original study directory identifiers remain in external output paths and protocols to preserve the link to the producing experiments.
