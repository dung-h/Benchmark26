# Earlier timing implementations

These modules retain the earlier measurement and analysis implementations. They are separate from the main experiment entrypoints in `experiments/`. Some current readers and feature adapters still import their utilities.

| Directory | Original scope |
| --- | --- |
| `feature_latency/` | Prepared-feature model loading and inference timing |
| `circuit_latency/` | Circuit extraction, process-cold and circuit-request measurements |
| `parallel_features/` | Concurrent feature extraction and subsequent prediction replay |
| `timing_completion/` | Diagnostic timing campaign and resume logic |
| `cost_analysis/` | Earlier timing aggregation, validation and plots |

A script here may require an external study bundle and its original measurement configuration. Running it is not part of the clean-checkout smoke example. The older `postfreeze_*` identifiers in output namespaces identify original studies; they are not additional code directories.

Some historical support modules and protocols are not distributed in this checkout. Full legacy execution requires these files as well as the measured data; the clean-checkout tests do not certify those runs. `B26_LEGACY_HELPERS` locates additional Python helper modules for the timing-completion implementation.
