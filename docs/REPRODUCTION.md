# Reproduction guide

The checkout supports local code checks, a synthetic smoke pipeline, upstream source verification and reconstruction of the exported latency intervals. Replaying the full measured QPU experiment additionally requires the external prepared study bundle and compatible model/feature environments. See [code map](CODE_MAP.md) and [path configuration](PATHS.md) before invoking research scripts.

## 1. System Requirements & Tested Environment

The following environments and requirements describe the relevant study components; the smoke pipeline and review tests do not require a GPU:

| Component | Verified Specification | Notes |
|:---|:---|:---|
| **Operating System** | Ubuntu Linux 22.04 / 24.04 LTS (x86_64) | Tested on Linux kernel 6.x |
| **Python** | Python 3.10.21 / Python 3.11 | Python >= 3.10 required |
| **C++ Compiler** | `g++` (>= 11.0) | Required only for scalar analytical kernel compilation |
| **CPU Memory** | 16 GB RAM minimum | 32 GB recommended for full DAG feature extraction |
| **GPU Hardware** | NVIDIA GPU (>= 16 GB VRAM) with CUDA 12.x | Required for neural training (`train_neural.py`); optional for testing |
| **Disk Space** | ~500 MB for code release; ~15 GB if downloading all upstream archives | Git LFS required for Qonductor archive |

---

## 2. Three Levels of Reproducibility

To ensure maximum transparency, reproducibility is divided into three distinct levels depending on data access:

```
[Level 1: Code & Logic Verification]  ──►  [Level 2: Upstream Source Audit]  ──►  [Level 3: Full Experiment Replay]
   (Clean checkout, zero network,            (Downloads 249 original files,             (Requires external study tables
      runs in a small local run)                verifies SHA256 & provenance)             & trained model checkpoints)
```

### Level 1: Code & Logic Verification (Offline, Clean Checkout)

Can be executed immediately after cloning without downloading any external datasets:

```bash
# 1. Install lightweight test dependencies
pip install -e ".[test]"

# 2. Check Python syntax, JSON protocol schemas, and study input declarations
make check

# 3. Inspect representative external-input availability
make status

# 4. Run the automated unit test suite (tests covering schemas, leakage, and metrics)
make test
```

**What this verifies:**
- Syntactic correctness of all 82+ Python files and all component protocol JSONs.
- Integrity of `upstream/data_sources.json` and `upstream/qualification_provenance.json`.
- Structural validity of `upstream/qonductor_recipe_screen.csv` (4,482 observation decisions).
- Group-disjoint E1 split logic on test fixtures; E2 intentionally allows shared groups.
- Correctness of multi-group error weighting and analytical critical-path DAG recurrence.

---

### Level 2: Upstream Source Acquisition & Byte Verification

Downloads the 249 original upstream assets from their official institutional repositories and verifies their cryptographic checksums:

```bash
# Preview the files to be downloaded (dry-run):
make fetch-sources-dry

# Download and verify all upstream assets to an external folder:
python3 scripts/fetch_upstream_data.py --source all --destination /path/outside/Benchmark26-inputs
```

**Covered upstream sources:**
- **Ma–Li Corpus:** Original QASM files and execution CSVs from `mooselab/Quantum-Execution-Time-Prediction` (commit `32c392a`).
- **Qonductor Corpus:** Hydrated Git LFS archive, SQLite database, and circuits from `manosgior/Qonductor-SC25` (commit `5d1ac8a`).
- **QPack Corpus:** JSON execution logs and configuration from `libket/qpack` (commit `9beaf65`).

---

### Level 3: Full Benchmark Reproduction (With Study Bundle)

Reproducing the full end-to-end benchmark from model training to held-out evaluation requires the external study bundle (containing `panel.csv`, `feature_ledger.csv`, and `execution_lock_v2.json`) as described in [DATA_SOURCES.md](DATA_SOURCES.md).

When the prepared bundle is placed under `artifacts/` or the configured `B26_ARTIFACTS` root (and required input references and execution configuration are valid):

| Result type | Target Command | Input Files Required |
|:---|:---|:---|
| **Main model accuracy** (MAE, MedAE, RMSE, $R^2$) | `python3 experiments/qpu/aggregate_results.py` | `cohort/targets.csv`, `training/` predictions |
| **Group vs random holdout** (E1 vs E2 comparisons) | `python3 experiments/qpu/write_final_report.py` | `analysis_v1/reduced_predictions.csv` |
| **Earlier warm-feature inference cost** | `python3 scripts/verify_cost.py` | `provenance/cost_source_index.json`, `estimator_cost/` |
| **Analytical duration routes** (Qiskit, QCRE, NoTaDS) | `python3 experiments/analytical/validate_portable.py` | `artifacts/postfreeze_unified_analytical_completion_v1/` |
| **Dataset Selection Ledger** (8,767 -> 3,443 rows) | `python3 scripts/build_review_selection_ledger.py --output verification/ledger.csv` | Canonical tables, `qonductor_recipe_screen.csv` |

---

## 3. Evaluation rules

E1 keeps logical-circuit groups disjoint between fitting and held-out evaluation. E2 uses source-stratified random observations and intentionally permits familiar groups. E3 holds out backend contexts; it may retain related structures on other backends.

Feature transformations and model selection use the permitted training partitions. Neural inner-fit preprocessing and validation selection are separated from the outer test set; the classical selector refits its selected model on outer-training data. Report pooled, source-balanced and group-balanced errors with their weighting explicitly stated. They answer different questions about the empirical mixture.

## 4. Troubleshooting & FAQ

- **Q: Why does `scripts/verify_cost.py` report missing study tables?**  
  *A:* The base repository distributes code, schemas, and verification logic. The full cost timing archive is an external study bundle. Run `make check` or `make status` to view which tables are currently detected.
- **Q: How do I test the C++ analytical recurrence?**  
  *A:* Run `pytest tests/test_analytical_path_kernel.py`. It executes self-contained recurrence checks using pure Python and NumPy without requiring `g++` or Qiskit.
