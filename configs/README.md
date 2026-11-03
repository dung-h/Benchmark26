# Configuration

| File | Purpose |
| --- | --- |
| `smoke.json` | Settings of the synthetic example (`make smoke`) |
| `paths.env.example` | Template for the environment variables that locate the study bundle and upstream sources |
| `study/qpu.yaml` | Summary of the 3,443-observation study: cohort, splits, models, metrics |
| `study/analytical.yaml` | Summary of the analytical estimator variants |
| `study/prediction_latency.json` | Scope and resampling settings of the exported latency results |

The files in `study/` document the study for readers; the research scripts do not read them. Each script validates its inputs against the frozen protocols in `protocols/` and the execution lock in the study bundle.
