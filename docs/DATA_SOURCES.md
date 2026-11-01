# Obtaining inputs

`upstream/data_sources.json` records the source URLs, revisions, sizes, and
SHA-256 pins used for the benchmark inputs.  The helper
`scripts/fetch_upstream_data.py` downloads pinned files to a directory outside
the checkout and checks their bytes.  It does not run circuits or create new
runtime labels.

After acquisition, the following utilities document the intended checks:

```bash
python scripts/fetch_upstream_data.py --help
python scripts/verify_upstream_labels.py --help
python scripts/build_review_selection_ledger.py --help
python scripts/verify_dataset_lineage.py --help
```

Some source archives use Git LFS and some derived circuit/feature caches are
not redistributable in this release.  The code deliberately fails rather than
silently replacing a missing exact input with a newly transpiled circuit or an
unverified reconstruction.  A reviewer should record missing inputs as a
reproduction limitation, not treat a generated proxy as the historical input.

The source pins establish provenance of downloaded files.  They do not prove
that source datasets share a common runtime boundary; that must be assessed
from the source documentation and the benchmark protocol for the particular
study.
