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

The prepared data bundle is necessary but does not by itself close every historical execution dependency. The training path's helper `materialize_mali_full_features.py`, the registries pinned by `protocol/estimator_execution.json` (`registry/seed_registry.json`, `CURRENT.json`, `method_fidelity_registry_v2.json`) and `protocol/circuit_representation_v3.json` are now included; `tests/test_pinned_reference_files.py` checks the registry hashes against the pins. They are copies of the private study files with only path handling changed in the materializer (`B26_ARTIFACTS`, `B26_NEURAL_PYTHON`). Historical input manifests that pin the *private* helper hashes will not match the adapted public helpers; compare behaviour, not those pins, or rebuild the manifest with the public sources.

Still not included: `registry/data_versions/archived_qpu_v1.json` (private absolute paths and split tables; used by the historical `run_qonductor_grouped_reproduction.py`), the other registries named inside `CURRENT.json`, the registry validator (needs private artifacts) and all data/circuit caches. Resolve these before claiming a full replay of the historical Qonductor grouped runs. The smoke example and aggregate-statistic verification do not require them.

`python scripts/check_setup.py` is a code-review check: it exits 0 and only lists missing external inputs. `python scripts/check_setup.py --require-study-inputs` (also `make check-study-inputs`) exits 2 when a representative study input is absent.
