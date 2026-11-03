# Inputs, code and outputs

| Category | Location | How it is used |
| --- | --- | --- |
| Source code | `experiments/`, `benchmark_v1/`, `scripts/`, `legacy/` | Implementation; current and earlier experiments are separated |
| Upstream input descriptions | `upstream/` | Download URLs, revisions, hashes and qualification decisions; not raw execution data |
| External source inputs | Paths selected by `B26_MALI_*` and `B26_QONDUCTOR_*` | Original archives, source checkouts and database |
| External prepared study bundle | `B26_ARTIFACTS` | Derived cohort, labels, feature ledger, circuit graphs, execution locks and fitted states; not shipped in this code repository |
| Versioned result snapshots | `results/prediction_latency/` | Aggregate statistics and anonymous group means for review; not training inputs |
| Generated local outputs | `work/`, or the explicit new output path | Smoke data, models, predictions and reports; ignored by Git |
| Runtime path settings | `configs/runtime/paths.env` | User-local locations; ignored by Git |

The included latency aggregates reconstruct the reported group means and paired intervals. They do not reconstruct individual raw timing samples, circuit inputs or fitted models. The smoke example generates its own synthetic observations and does not consume the measured study snapshots.

Experimental output namespaces inside an external bundle retain their original `postfreeze_*` study identifiers. Code directory names have been simplified without renaming those recorded outputs or changing their values.
