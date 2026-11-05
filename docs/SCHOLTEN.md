# Scholten-style throughput estimate

This page explains how the benchmark scores the throughput model of Scholten et al. (2023), what the results show, and how to rerun or check them.

## Result

The estimate is far too high on every source, but it ranks runs about as well as the Qiskit duration multiplied by shots. On the 2,289 runs it covers, it predicts 1,981 s on average against a mean measured runtime of 6.41 s. Its Pearson correlation with the measured runtime is 0.83 and its Spearman correlation 0.71. On the same runs, the Qiskit duration multiplied by shots has 0.81 and 0.73.

The overestimate grows with the number of shots. The table below gives the median ratio of estimate to measured runtime; the full table, with quartiles, is `results/scholten/ratio_by_shots.csv`.

| Source | Shots | Runs | Median ratio |
| --- | --- | ---: | ---: |
| Qonductor | 501–2,000 | 661 | 11 |
| Qonductor | 2,001–8,000 | 810 | 38 |
| Qonductor | more than 8,000 | 445 | 97 |
| QPack | 4,096 | 33 | 21 |
| Ma–Li | 1,024 | 340 | 541 |

## How this relates to the original paper

Scholten et al. model the runtime of a job of M circuits as M × shots × D_eff / CLOPS (their Eq. 6), with no fixed per-job cost. They tested it mainly on jobs of 100 circuits on IBM processors and report the same trend as above:

- the model is most accurate when a job resembles the jobs used to measure CLOPS, which run 100 shots;
- with few shots it underestimates (ratio 0.04–0.06 at 10 shots), since the model has no fixed per-job cost;
- with many shots it overestimates, and the ratio grows with shots (1.6–2.3 at 500 shots, 6.6–8.2 at 8,000 shots).

Every run covered here used more than 500 shots, so an overestimate was expected. Two differences from the paper probably make the overestimate larger; this benchmark did not separate their effects:

- The benchmark uses the layer depth of the compiled circuit. The paper uses an effective depth, which divides by the depth of a quantum-volume circuit of the same size. Deep circuits are therefore charged more here. This matters most for Ma–Li, whose compiled circuits have a median depth of about 22,000 layers, against about 110 for Qonductor.
- Each job here holds one circuit, while the paper's jobs hold 100.

## Definitions and choices

- **Formula:** estimate in seconds = shots × compiled layer depth / CLOPS, for one circuit per job.
- **CLOPS** (circuit layer operations per second) measures how many circuit layers a processor runs per second. IBM reports it as `clops_v` (older definition) or `clops_h` (newer definition, used by kyoto and osaka). The values are in `experiments/scholten/clops.csv`, copied from the fake-backend files of qiskit-ibm-runtime 0.49.0. belem, lima and quito report neither value, so their 1,154 runs get no estimate.
- **Nothing is fitted.** The held-out-group setting (E1) and the random-run setting (E2) therefore give the same values.
- **Scores:** group-balanced mean absolute error (the mean over circuit groups of each group's mean error), pooled mean absolute error, R² in seconds, Pearson and Spearman correlation. The same scores of five saved estimators on exactly the same runs are in `results/scholten/scores.csv`.

## Rerun or check

`make check` runs `experiments/scholten/verify_summary.py`. It needs no study files: it checks the file hashes, the CLOPS table and the run counts.

To recompute the scores, download and unzip `benchmark26-study-files.zip` as described in [REPRODUCTION.md](REPRODUCTION.md), then run:

```bash
export B26_ARTIFACTS=/path/to/benchmark26-study-files/artifacts
python3 experiments/scholten/estimate.py --output work/scholten
diff work/scholten/scores.csv results/scholten/scores.csv
```

The script refuses to write into an existing folder. `tests/test_scholten.py` compares the CLOPS table with the installed qiskit-ibm-runtime when that package is present.

Reference: T. L. Scholten et al., "A Model for Circuit Execution Runtime And Its Implications for Quantum Kernels At Practical Data Set Sizes", arXiv:2307.04980, 2023.
