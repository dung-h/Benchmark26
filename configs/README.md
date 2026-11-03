# Configuration roles

| Path | Role | Consumer |
| --- | --- | --- |
| `smoke.json` | Executable synthetic example configuration | `scripts/run_benchmark_pipeline.py --stage smoke --config FILE` |
| `runtime/paths.env.example` | Template for locating external data and worker Python environments | Copy, edit, then source in the shell before a research run |
| `reference/unified.yaml` | Specification of the measured QPU study | Documentation; not executed by the smoke CLI |
| `reference/analytical.yaml` | Specification of analytical variants | Documentation; not executed by the smoke CLI |
| `reference/prediction_latency.json` | Measured timing scope and resampling settings | Documentation accompanying the exported results |

A reference specification does not configure the research runners automatically. Each research entrypoint exposes its own options and validates its execution inputs. See `docs/CODE_MAP.md` and `docs/PATHS.md`.
