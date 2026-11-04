# Pitfalls

Errors and misreadings that came up while checking this study. Each entry gives the symptom, the cause and the fix.

## Running the code

### `make check` fails because NumPy is missing

**Cause:** the check recomputes statistics with NumPy, and a bare Python install does not have it.
**Fix:** run `python3 -m pip install -e ".[test]"` first.

### `historical_code_lock_mismatch` when you start training

**Cause:** the original execution lock records the hashes of the code that produced the saved study. This repository's code was adapted after that, so it may not extend the saved run.
**Fix:** start a new run in a separate directory. Set `B26_RUN_DIR` to a path outside the study bundle, then run `python3 experiments/qpu/prepare_run.py`. See [Reproduction](REPRODUCTION.md#run-the-pipeline).

### `new_run_not_prepared`

**Cause:** `B26_RUN_DIR` is set, but `prepare_run.py` has not written `run_manifest.json` there.
**Fix:** run `prepare_run.py` with the same environment.

### `code_dependency_changed_since_run_manifest:<file>`

**Cause:** a code file changed after the run was prepared. The manifest pins every repository file the run can execute, including `benchmark26/paths.py`, and fitting stops if any of them changes.
**Fix:** use a new `B26_RUN_DIR` and prepare it again. Restoring the file also works.

### `prepare_run.py` exits with code 2: "outside the original study directory"

**Cause:** `B26_RUN_DIR` points inside the study directory of the bundle, which would mix new outputs with the saved ones.
**Fix:** choose a path outside the bundle.

### `graph_artifact_missing_or_outside_root:<id>`

**Cause:** the ledger records graph files as `artifacts/...`. They are resolved through `B26_ARTIFACTS` and must lie inside it. Usually `B26_ARTIFACTS` is unset or points at the wrong directory.
**Fix:** point `B26_ARTIFACTS` at the bundle root.

### The pinned model source is reported as missing

**Cause:** the Ma–Li model is loaded from a Git checkout at a pinned revision, and `B26_MALI_SOURCE` does not point to it.
**Fix:** set `B26_MALI_SOURCE` to the checkout. `train_neural.py --stage preflight` reports `pinned_model_source`.

### The GPU is not visible, or writes to the temporary directory fail

**Cause:** this happens when the bundle is mounted read-only inside a sandbox such as `bwrap`. The sandbox may hide the GPU and have no writable temporary directory.
**Fix:** set `TMPDIR` to a writable path. Run GPU steps (preflight, replay) outside the sandbox and send their output to a scratch directory.

## Reading the results

### Equal outputs do not mean the same circuit

Two different circuits can have the same estimated duration. When this study compared circuits by output, 401 observations looked identical. By canonical circuit IR digest, only 385 were. Compare IR digests, not scalar outputs.

### Do not mix numbers from different runs

The graph ablation retrains the MLP and Graph models inside its own run. Its numbers are not comparable with the paper tables. The same holds for any new run.

### Timings depend on the measurement conditions

The composite CPU stages were measured with eight concurrent workers. Do not compare them with single-worker timings, and do not add medians from different runs and call the result a median total.

### A confidence interval that contains zero is not evidence of equivalence

It says the data cannot separate the two models at this sample size. No equivalence margin was set in this study.
