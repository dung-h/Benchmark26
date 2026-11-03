# Path configuration

Copy `configs/runtime/paths.env.example` to `configs/runtime/paths.env`, edit the paths and source it before invoking experiment entrypoints. The local `.env` file is ignored by Git.

Executable scripts resolve code from their own location. External files and worker interpreters use environment variables, so the checkout does not need to be next to the authors' data or virtual environments.

| Variable | Default relative to repository root | Purpose |
| --- | --- | --- |
| `B26_ARTIFACTS` | `artifacts/` | External study tables, graphs, locks and experiment outputs, retaining the study's directory layout |
| `B26_MALI_SOURCE` | `inputs/mali/` | Pinned upstream Ma–Li source checkout |
| `B26_MALI_QASM` | `inputs/mali/data/quantum_circuits/` | Ma–Li input circuits |
| `B26_QONDUCTOR_SOURCE` | `inputs/qonductor/` | Pinned Qonductor checkout |
| `B26_QONDUCTOR_ARCHIVE` | `inputs/qonductor/data/database/circuits.zip` | Hydrated circuit archive |
| `B26_QONDUCTOR_DATABASE` | `inputs/qonductor/data/database/quantum_scheduler.db` | Original database |
| `B26_QONDUCTOR_TABLE` | `inputs/qonductor_observations.parquet` | Extracted table used by the original grouped regression runner |
| `B26_CPU_PYTHON` | Current Python executable | CPU feature worker environment |
| `B26_LEGACY_HELPERS` | `inputs/legacy_helpers/` | Additional helper modules required by earlier timing scripts |
| `B26_GPU_PYTHON` | Current Python executable | Neural/replay worker environment |

Use absolute paths when exporting these variables. Source checkout and circuit/archive variables are independent: setting a source root does not automatically change the archive override.

```bash
export B26_ARTIFACTS=/path/to/study-bundle/artifacts
export B26_MALI_SOURCE=/path/to/mali-checkout
export B26_MALI_QASM=/path/to/mali-checkout/data/quantum_circuits
export B26_QONDUCTOR_SOURCE=/path/to/qonductor-checkout
export B26_QONDUCTOR_ARCHIVE=/path/to/hydrated/circuits.zip
export B26_QONDUCTOR_DATABASE=/path/to/quantum_scheduler.db
export B26_CPU_PYTHON=/path/to/cpu-environment/bin/python
export B26_GPU_PYTHON=/path/to/gpu-environment/bin/python
```

Install QCRE/BQSKit dependencies in the selected CPU environment; no script adds another environment's `site-packages` to `sys.path`. Use the versions required by the relevant protocol. Selecting an interpreter is not a dependency-compatibility check.

The external study bundle must preserve content identities, split tables and graph references. An imported ledger may contain paths from its producing environment; resolve those references when preparing a relocated bundle. The code release does not include a universal importer for arbitrary historical bundles. This checkout's path changes do not replace historical execution locks or certify a full training replay. Generate and review a new execution configuration for a new run, while retaining original scientific records.

The JSON protocol fields named `local_resolution_hint` or `local_python_hint` use `${VARIABLE:-default}` as documentation. The script environment overrides above provide runtime resolution; protocol strings are not evaluated as shell code.
