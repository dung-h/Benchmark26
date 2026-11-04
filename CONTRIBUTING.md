# Contributing

These rules keep the code, data and documents readable for someone who has never seen the project, and keep every published number checkable.

## Names

- Name files, folders, functions, variables and table columns after what they contain or do: `runs.csv`, `runtime_seconds`, `circuit_group`, `prepare_dataset.py`.
- Do not use internal or process words in names or reader-facing text, such as cohort, panel, canonical, conservative, qualified, proposed, materialize or lineage. Say "dataset", "run", "build" or "check" instead.
- Standard field terms and abbreviations are fine: MLP, QPU, DAG, MSE, CLOPS, shots, fold.
- Folder and data-package names are lowercase with hyphens and short, such as `benchmark26-data`. Do not add version numbers or words like "final" or "for-upload" to names; versions are recorded in `CHANGELOG.md`, git history and checksum files.
- Paths and column names inside the saved study files keep their original spelling, because the code reads existing data. Explain such a name where a reader meets it.

## Renaming pinned code

The files listed in `results/graph_ablation/manifest.json` must keep the code that produced the published ablation. To rename something in them:

1. Add the rename to `renamed_after_run` in that manifest (new name → old name, per file).
2. Make sure no old name remains in the file; the check refuses files that still contain one.
3. Run `make check`, `make test`, and `experiments/graph_ablation/replay_checkpoints.py` on a finished run, and record the result in the commit message.

Any other change to a pinned file needs a new run.

## Writing documents

- Give the answer or the decision in the first sentence, then the evidence.
- Write one idea per sentence, with the actor named and the verb saying what happens.
- Define every term at its first use, and keep one name per concept.
- Report only the numbers a reader needs, each with its comparison and direction. Put more than three numbers in a table.
- An interval that contains zero does not show equivalence, and a correlation is not accuracy or a cause. State an uncertainty once, at the claim it affects.
- Keep instructions, reference tables and explanations in separate sections.

## Before you commit

- Run `make check` and `make test`; run `make smoke` if the pipeline changed.
- Every number in the documents must come from a file in `results/` or from the study files, and a script must be able to recompute it.
- Describe in the commit message what changed and how it was verified.
