# Prediction latency

The new timing results cover 3,443 observations and 203 logical-circuit groups under E1. The two measurements below answer different questions.

| Measurement | Start and end | Conditions |
| --- | --- | --- |
| Observed neural request latency | Resident logical circuit bytes → parsing → features → transformation → GPU prediction → host scalar | MLP/Graph variants, three serial sessions, three warmups and ten timed repetitions; three-seed median prediction |
| Composite compiled-route processing cost | Matched parse/compile time + method-specific processing + Qonductor inference, when applicable | 695 unique CPU configurations measured with eight workers; inference measured separately in three serial sessions |

Compilation is counted once per input and method. Totals are computed for each observation before calculating summary statistics. Composite costs exclude model loading, disk access, backend preparation and inter-stage communication. They are not observed isolated end-to-end latency. The constant lookup bypasses circuit processing. NoTaDS produces a unitless score, while its processing cost is measured in time units.

`results/prediction_latency/` contains source-level summaries, anonymous neural group means for each session and across sessions, and paired latency intervals. Each observation contributes the mean of its three session medians. The group-balanced mean weights groups equally; the reported median and P90 weight observations equally. The summaries therefore use different weighting.

## Verify the exported statistics

With the repository's NumPy dependency installed, run:

```bash
python3 experiments/latency/verify_summary.py
```

This recomputes five neural group-balanced means and 30,000 source-stratified paired group-bootstrap draws. Anonymous group labels retain the ordering used for resampling. Intervals describe these workloads on the measured host and sessions; they do not represent variation across machines or days.

The exported group means support interval reconstruction, but not reconstruction of individual raw durations or request-level quantiles. Those require the original external measurement records. The composite CPU stages have one sample per configuration, so their summaries have no timing confidence intervals.

## Reuse the implementation with external inputs

`experiments/latency/measure_classical.py` measures a fitted model on its five-feature inputs. Supply a CSV with `task_key,feature_0,...,feature_4`, a trusted joblib model for the corresponding fold, and a new output directory. Use the model's original sklearn environment. It records three sessions of ten timed predictions after three warmups. It performs no training or compilation. The script selects one available CPU core on Linux and records affinity, versions and model/input hashes. Use `--cpu-core` to select a particular allowed core. The optional `expected_prediction` column checks against a saved reference.

```bash
python3 experiments/latency/measure_classical.py \
  --features /path/to/fold_features.csv --model /path/to/estimator.joblib \
  --output /path/to/new_inference_samples
```

`compose.py` accepts an external per-observation CSV with `id,group_id,method,parse_compile_ns,method_stage_ns,inference_ns`. For the reference calculation, `inference_ns` is zero for analytical methods and the mean of session medians for Qonductor. Map every observation to the correct task and fold before composing. Reuse of a task measurement across observations does not create independent timing samples.

```bash
python3 experiments/latency/compose.py \
  --stages /path/to/matched_stages.csv --output /path/to/new_composites.csv
```

These portable tools expose the measurement and composition logic. Reproducing the original raw circuit measurement also requires the external circuit inputs, backend snapshots, feature extraction environment and fitted states (see [Reproduction](REPRODUCTION.md)).
