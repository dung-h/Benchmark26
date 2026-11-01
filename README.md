# Benchmark26 — Quantum Runtime Estimation Benchmark Setup

This repository contains the implementation and methodological setup for the
QPU runtime-estimation benchmark.  It is a code-and-protocol release for
reviewing how the benchmark is constructed; it is not a release of thesis
sources, workshop-paper sources, trained checkpoints, saved predictions, or
headline-result reports.

## What is included

- `benchmark_v1/`: logical-circuit preparation, Ma--Li-style feature
  construction, Qonductor input materialisation, and benchmark protocols.
- `research/`: scripts used to prepare the unified cohort and splits, train the
  neural and classical routes, evaluate analytical routes, and measure
  prediction cost.
- `scripts/`: source acquisition, selection-ledger, lineage, reference,
  analytical-kernel, and cost verification utilities.
- `upstream/`: pinned source locations and metadata needed to obtain original
  inputs.
- `docs/`: a concise methodology and data-acquisition guide.

## What is intentionally excluded

The repository does not contain QPU labels, third-party archives, extracted
feature caches, trained models, prediction tables, bootstrap draws, PDFs, or
the thesis/workshop manuscripts.  Those materials either have redistribution
constraints, are large derived outputs, or belong to the separate scientific
record.  Consequently, cloning this repository alone does not reproduce a
published score.

## Reproduction boundary

The intended workflow is:

1. Read `docs/DATA_SOURCES.md` and acquire the pinned upstream
   inputs outside the repository.
2. Read `docs/METHODOLOGY.md` and run the selection/lineage scripts in
   `scripts/` against those inputs.
3. Inspect the fixed preparation, splitting, training, evaluation, and cost
   scripts in `research/` and `benchmark_v1/` before running any new study.

Several archived scripts refer to a private research-artifact layout such as
`artifacts/...`.  That is deliberate: they document the exact experimental
pipeline.  Before a fresh experiment, configure a new output namespace and
confirm that every input path and target boundary is appropriate for the new
study.  Do not overwrite an existing frozen result directory.

## Scope

This code supports methodological review of circuit grouping, data lineage,
feature construction, split construction, model fitting, analytical routes,
and prediction-cost measurement.  It does not by itself establish the
scientific validity of a result; reviewers should inspect the source data,
measurement boundary, protocol, and saved predictions for the study being
evaluated.

## Dependencies

`requirements-review.txt` is sufficient for the lightweight review utilities.
Training and full feature extraction additionally require the Qiskit, PyTorch,
PyTorch Geometric, CUDA, and source-specific environments recorded by the
corresponding scripts.  No QPU job is launched by the included verification
utilities.

## Third-party material

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).  Do not redistribute
upstream datasets unless their respective licenses and terms permit it.
