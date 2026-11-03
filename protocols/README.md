# Frozen protocols

These files are the contracts of the original study, copied byte for byte. Do not edit them: they pin each other by SHA-256, and the code checks those hashes before using them. `make check` verifies the chain.

| File | Contract |
| --- | --- |
| `estimator_execution.json` | Shared execution contract; pins the method contracts and registries below |
| `mali_full_features.json` | Ma–Li global and graph features (method contract) |
| `qonductor_native_features.json` | Qonductor feature materialization (method contract) |
| `simulator_estimator_ports.json`, `analytical_findings.json` | Remaining method contracts pinned by the execution contract |
| `qualified_logical_benchmark.json`, `logical_input_review.json` | Logical-input qualification |
| `circuit_representation_v3.json`, `snapshot_policy_v3.json` | Circuit representation and backend snapshot policy |
| `registry/` | Seed registry and registry guards pinned by the execution contract |
| `reference_code/` | Two programs checked by hash: the compiled-representation materializer used by Qonductor feature extraction (pinned by `qonductor_native_features.json`) and the fold-0 validator of the regression runs (pinned by a receipt in the study bundle) |

Most contracts belong to the component studies with 4,515 and 8,767 rows that preceded the 3,443-observation study. They are not that study's configuration.

The files record paths in the original repository layout (`benchmark_v1/protocol/...`, `artifacts/...`) and, in a few hint fields, the original machine paths. The code resolves recorded paths with `benchmark26.paths.recorded()` and takes machine-specific locations from environment variables, never from these hints. The `_v3` suffixes are part of the original contract identifiers.
