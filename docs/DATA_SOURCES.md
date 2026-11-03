# Obtaining inputs

`upstream/data_sources.json` records the source URLs, revisions, sizes, and
SHA-256 pins used for the benchmark inputs.  The helper
`scripts/fetch_upstream_data.py` downloads pinned files to a directory outside
the checkout and checks their bytes.  It does not run circuits or create new
runtime labels.

The acquisition utility can display its options from this checkout:

```bash
python scripts/fetch_upstream_data.py --help
```

The remaining verification scripts require externally supplied study tables.
`verify_upstream_labels.py` compares original labels with the canonical tables;
`build_review_selection_ledger.py` needs those tables and qualification results;
`verify_dataset_lineage.py` needs the cohort and source tables. `verify_cost.py`
requires `provenance/cost_source_index.json` and its indexed cost tables.
`verify_reference.py`, `verify_analytical_kernel.py`, and `validate_portable.py`
also require saved predictions, resampling draws, or kernel fixtures.
These inputs are not included in this code release. Downloading upstream files
alone does not regenerate the intermediate tables. Scripts retain their original
relative input paths; inspect these before preparing an external study bundle.

Some source archives use Git LFS. Large derived circuit caches are also external
to this release. Missing pinned inputs must be supplied before executing the
corresponding stage. Reconstructed circuits require their own provenance records.

The source pins establish provenance of downloaded files.  They do not prove
that source datasets share a common runtime boundary; that must be assessed
from the source documentation and the benchmark protocol for the particular
study.

## Additional research execution dependencies

The prepared data bundle is necessary but does not by itself close every historical execution dependency. Some support runners reference additional component protocols, registries and helper modules that are not included here (for example `circuit_representation_v3.json`, `seed_registry.json` and `materialize_mali_full_features.py`). Resolve these components and graph references before claiming a full experiment replay. The included smoke example and aggregate-statistic verification do not require these files.
