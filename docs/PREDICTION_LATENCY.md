# Prediction latency

The new timing results cover 3,443 observations and 203 logical-circuit groups under E1. The two measurements below answer different questions.

| Measurement | Start and end | Conditions |
| --- | --- | --- |
| Observed neural request latency | Resident logical circuit bytes → parsing → features → transformation → GPU prediction → host scalar | MLP/Graph variants, three serial sessions, three warmups and ten timed repetitions; three-seed median prediction |
| Composite compiled-route processing cost | Matched parse/compile time + method-specific processing + Qonductor inference, when applicable | 695 unique CPU configurations measured with eight workers; inference measured separately in three serial sessions |

Compilation is counted once per input and method. Totals are computed for each observation before calculating summary statistics. Composite costs exclude model loading, disk access, backend preparation and inter-stage communication. They are not observed isolated end-to-end latency. The constant lookup bypasses circuit processing. NoTaDS produces a unitless score, while its processing cost is measured in time units.

## What is timed

Each measurement has a stated input at the start and a stated output at the end. Everything between them counts toward the cost. Only the items in the last column are left out.

| Measurement | Input at the start | Output at the end | Not timed |
| --- | --- | --- | --- |
| Observed neural request latency | Logical circuit bytes already in memory | Prediction as a host scalar | Loading the circuit and the model; the three warm-up runs of each session |
| Composite compiled-route cost | Logical circuit | Method-specific processing finished, plus model inference for Qonductor | Model loading, disk access, backend preparation, communication between stages |
| Analytical estimators from ready physical input | Parsed physical circuit in memory, with the backend target or duration map prepared | Estimated duration | QASM parsing, compilation, model loading, the one warm-up run |

A method cannot skip a step that its input needs. Compilation counts once per input and method. The constant lookup is the one exception, because it does not process the circuit at all.

## Analytical estimators from ready physical input

This measurement times the Qiskit and QCRE duration estimators on the physical circuits that their accuracy is scored on. These are the saved native circuits of each observation: for Qonductor, the circuit that was submitted; for Ma–Li and QPack, the saved nominal compilation. Nothing is recompiled. The input is a parsed circuit in memory with the backend target or duration map already prepared.

- **Qiskit** is one call of `QuantumCircuit.estimate_duration`.
- **QCRE** is the conversion of the measurement-stripped circuit to BQSKit plus the original `estimate_runtime`. Conversion and estimator are also timed separately.
- **Design.** One timing task per unique (circuit, backend snapshot) pair: 1,452 tasks for 3,443 observations. Each task has one warm-up call and three timed calls. The value of a task is the median of its three samples, and each observation takes the value of its task.
- **Conditions.** Eight worker processes ran at the same time, each pinned to one CPU core. The numbers describe latency under that load. Do not compare them with single-worker measurements.
- **Correctness.** Before timing, the estimators were run on every task and matched the accuracy-route values exactly (largest difference 0). The check covers the timing graph and the numeric output. For the 325 Ma–Li and QPack tasks the full circuit IR after parsing differs from the saved one, so identity of the complete circuit is not claimed. All 1,127 Qonductor tasks have an identical IR.

Milliseconds. The mean weights every circuit group equally; the median weights every observation equally. The two differ because Ma–Li has 148 of the 203 groups but only 340 of the 3,443 observations.

| Source | Groups | Qiskit mean | QCRE mean | Qiskit median | QCRE median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ma–Li | 148 | 819.6 | 28,866.6 | 839.4 | 30,113.7 |
| Qonductor | 49 | 0.15 | 44.8 | 0.10 | 16.4 |
| QPack | 6 | 0.11 | 18.7 | 0.10 | 12.5 |
| All sources | 203 | 597.5 | 21,057.0 | 0.11 | 17.9 |

What the numbers show:

- The cost is set by circuit size. The Ma–Li circuits are large (median about 246,000 operations, largest about 373,000), so Qiskit takes about 0.8 s and QCRE about 29 s on them. The Qonductor circuits are small (median 173 operations, largest 7,444), and Qiskit takes about 0.1 ms on them.
- About 92% of the pooled QCRE mean is the conversion to BQSKit. The estimator itself is the smaller part.
- These numbers start from a parsed circuit. They are not comparable with the composite compiled-route table, which starts from the logical circuit and includes parsing and compilation.

`results/ready_analytical_latency/` holds the summary, the anonymous group means and a manifest. Recompute the means with `python3 experiments/latency/verify_ready_analytical.py` (`make check` runs it). Per-task samples and the measurement script need the private study bundle and are not distributed; the manifest records the hashes of the protocol, the correctness gate and the script.

## Compiled-route inputs

The compiled route compiles each logical input with `optimization_level=1`, trivial layout, SABRE routing and the translator. Ma–Li and QPack use the seed of the saved representation: for these 385 observations the compiled circuit has the same canonical IR digest as the circuit scored for accuracy. Qonductor uses seed 42, and its compiled circuit differs from the submitted physical circuit for all 3,058 observations. The submitted circuits came from Qonductor's benchmark code (level 3, fewest CX/CZ/ECR among repeated transpilations); the archive records no per-job seed, compiler version or snapshot. The new circuits never ran on a QPU, so Qonductor-source costs cannot be paired with the accuracy measured on the submitted circuits. Identity was established by IR digest against the private bundle; equal scalar outputs do not establish it (16 Qonductor observations keep the saved Qiskit duration with a different circuit).

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
