# Reproduction guide

This repository contains the implementation and provenance needed to reproduce
the benchmark, but the original study tables, circuit caches, model outputs,
and some runtime-specific environments are intentionally distributed
separately. A clean checkout therefore has two levels of verification.

## 1. Verify the code release

```bash
python3 -m pip install -e ".[test]"
make check
make test
```

`make check` parses every Python file and protocol JSON file, then reports the
optional scientific dependencies and representative external inputs. `make
test` runs the unit tests that do not require a QPU or a training campaign.

## 2. Acquire pinned upstream inputs

List the files recorded in the source manifest without downloading them:

```bash
python3 scripts/fetch_upstream_data.py --source all \
  --destination /path/outside/Benchmark26-inputs --list
```

Download and verify the pinned source files:

```bash
python3 scripts/fetch_upstream_data.py --source all \
  --destination /path/outside/Benchmark26-inputs
```

The destination must be outside the checkout. The utility verifies file size
and SHA-256 before accepting each file; it does not reconstruct the cohort,
run circuits, or create runtime labels.

## 3. Reproduce a study result

The full benchmark additionally requires the external study bundle described
in [DATA_SOURCES.md](DATA_SOURCES.md), including the execution lock, cohort
tables, feature caches, and (for cost studies) provenance tables. Use
`research/README.md` as the task map and run stages in this order:

1. prepare the qualified cohort and feature tables;
2. construct the declared train/validation/test splits;
3. fit the selected estimators;
4. aggregate held-out predictions;
5. run validation and write the report.

Each stage should preserve its input hashes, configuration, commit SHA, random
seed, and output manifest. Do not treat files from different experiment
directories as interchangeable: the `postfreeze_*` names identify distinct
study modules and evaluation boundaries.

## Reproducibility limits

Exact bitwise equality is not promised for CUDA training. Compare the declared
metrics, split membership, tolerances, and provenance hashes instead. Source
datasets and large derived artifacts are not included in this repository and
may carry separate redistribution terms.
