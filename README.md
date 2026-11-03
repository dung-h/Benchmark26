# Benchmark26

Code and protocols for comparing QPU runtime estimators: Ma–Li-style MLP/Graph+MLP, a Qonductor model selector, and analytical timing methods. The measured study uses 3,443 observations from Ma–Li, Qonductor and QPack, organized into 203 logical-circuit groups.

## Run a smoke pipeline

```bash
pip install -e ".[test]"
python3 scripts/run_benchmark_pipeline.py --stage smoke --output work/smoke
```

This runs **synthetic data → group-disjoint split → train-only preprocessing → regression fit → held-out predictions → metrics and report**. It checks the software flow on a small linear example; its scores are not the published QPU benchmark. `configs/smoke.json` controls this run; pass `--config FILE.json` to change it. Use a new output directory for each run.

## Verify the included results

```bash
make check
make test
make verify-latency
```

The latency verifier reconstructs per-session/source group means and 30,000 paired bootstrap draws from the exported aggregates. Individual raw-duration replay requires external measurement records. See [prediction latency](docs/PREDICTION_LATENCY.md) for measurement boundaries and the composite processing-cost calculation.

## Run the research experiments

The QPU experiments use separate entrypoints and external inputs. The CLI above provides the synthetic smoke run, input inventory and review checks; it does not orchestrate the full QPU experiment.

- [Code map](docs/CODE_MAP.md): main experiment entrypoints and supporting historical scripts.
- [Data sources](docs/DATA_SOURCES.md): upstream downloads, qualification and external study inputs.
- [Path configuration](docs/PATHS.md): data roots and CPU/GPU Python interpreters.
- [Reproduction](docs/REPRODUCTION.md): verification scope and steps with the study bundle.
- [Methodology](docs/METHODOLOGY.md): labels, circuit groups, evaluation and weighting.

`research/postfreeze_qpu_workflow_mean_expansion_v1/` contains the main measured QPU experiment; `research/prediction_latency/` contains portable latency tools. Other versioned research directories preserve supporting experiments and are listed separately in the code map. `results/prediction_latency/` contains generated aggregate results; it is not an input dataset. `configs/unified_benchmark.yaml` documents the measured study and is not an executable CLI configuration.

Raw third-party datasets, fitted models, checkpoints and manuscript sources are external to this repository. See [third-party notices](THIRD_PARTY_NOTICES.md).
