# Reference registries

Byte-for-byte copies of the registries that `protocol/estimator_execution.json`
pins by SHA-256 (`seed_registry`, `current_guard`, `fidelity_guard`). They hold
no raw observations, machine paths or credentials.

- `CURRENT.json` points to further artifacts (`method_registry_v3.json`,
  eligibility matrices, artifact-organization authority) that belong to the
  private study workspace and are **not** exported here. It is a guard record,
  not an executable configuration.
- `data_versions/archived_qpu_v1.json` (consumed by the historical
  `run_qonductor_grouped_reproduction.py`) is not exported: it contains
  absolute private paths and references split tables outside this release.
- The registry validator and its test need private study artifacts and are not
  part of this code release.

`tests/test_pinned_reference_files.py` checks that the files here still match
the pins.
