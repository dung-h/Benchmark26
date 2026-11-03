# Quantum Runtime Estimation Benchmark (Benchmark26)

[![Quality & Reproducibility CI](https://github.com/dung-h/Benchmark26/actions/workflows/quality.yml/badge.svg)](https://github.com/dung-h/Benchmark26/actions/workflows/quality.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tested on Linux](https://img.shields.io/badge/OS-Linux%20Ubuntu-orange)](https://ubuntu.com/)

Predicting physical quantum processing unit (QPU) runtime is critical for cloud job scheduling, compilation selection, and workflow cost optimization. **Benchmark26** provides an open-source, leakage-safe benchmarking suite evaluating learning-based and analytical runtime estimators on physical IBM quantum processors.

The unified study evaluates **3,443 physical QPU execution observations across 203 logical-circuit groups**, unifying three upstream corpora: Ma–Li (340 observations), Qonductor (3,058 observations), and QPack (45 workflow means from 3,854 execution records).

---

## Key Highlights

- **Leakage-Safe Grouping:** Uses group-disjoint E1 evaluation for unseen circuit groups; E2 intentionally allows familiar groups and E3 holds out backend contexts.
- **Multi-Paradigm Comparison:** Directly contrasts graph neural networks (Graph-augmented TransformerConv MLP), baseline MLPs, validation-selected regression models (Qonductor adaptation), and zero-training analytical critical-path models.
- **Fair Inference Cost Accounting:** Reports measured neural request latency and separately labelled composite costs for compiled routes.
- **Cryptographic Provenance:** Cryptographically pins 249 original upstream assets in `upstream/data_sources.json` and records recipe qualification criteria in `upstream/qualification_provenance.json`.

---

## Quickstart

From a clean checkout, verify the environment and run unit tests in under 10 seconds:

```bash
# 1. Clone repository
git clone https://github.com/dung-h/Benchmark26.git
cd Benchmark26

# 2. Install package and lightweight test dependencies
pip install -e ".[test]"

# 3. Check environment readiness and run automated test suite
make check
make test
```

To view the readiness of every benchmark pipeline stage:

```bash
make status
```

---

See [prediction latency](docs/PREDICTION_LATENCY.md) for the new measurements, composite stage costs and commands to reconstruct the paired intervals.

## Repository Structure

| Directory | Purpose | Contents |
|:---|:---|:---|
| [`configs/`](configs/) | Declarative Configurations | Canonical YAML configs for unified benchmark (`unified_benchmark.yaml`) and analytical evaluations. |
| [`tests/`](tests/) | Unit & Integration Tests | Test suite validating schema integrity, recipe screening, split leakage guards, and metric calculations. |
| [`benchmark_v1/`](benchmark_v1/) | Feature Pipeline & Adapters | Circuit materializers, AST hardware-wire restoration adapters, and frozen component protocols. |
| [`research/`](research/) | Benchmark Experiments | Unified cohort preparation, group-disjoint split generators, neural training harnesses, and cost timers. |
| [`scripts/`](scripts/) | Utilities & Entrypoints | Canonical pipeline CLI (`run_benchmark_pipeline.py`), setup checkers, and verifiers. |
| [`upstream/`](upstream/) | Provenance & Source Pins | Pinned upstream URLs, commit revisions, SHA256 checksums, and MQT 1.0.3 screening tables. |
| [`docs/`](docs/) | Documentation | Methodology guide (`METHODOLOGY.md`), data access guide (`DATA_SOURCES.md`), and reproduction manual (`REPRODUCTION.md`). |

---

## Benchmark Pipeline & Stages

```text
upstream sources ──► circuit qualification ──► group-disjoint splits ──► estimator training ──► evaluation & cost
  (Ma–Li, Qonductor,     (MQT 1.0.3 readout        (E1 group holdout,        (Graph MLP, Global,     (Pooled, source-balanced,
        QPack)             state filtering)        E2 random, E3 backend)     R0 selector, Analytical) group-balanced metrics)
```

| Stage | Command / Script | Dependencies & Input Requirements |
|:---|:---|:---|
| **1. Source Acquisition** | `python3 scripts/fetch_upstream_data.py --source all --destination <dir>` | Network access; downloads and checks 249 pinned upstream files. |
| **2. Recipe Qualification** | `pytest tests/test_qonductor_recipe_screen.py` | Local repository files; validates 4,482 screening decisions. |
| **3. Group-Disjoint Splits** | `python3 research/postfreeze_qpu_workflow_mean_expansion_v1/prepare_splits.py` | Requires cohort tables (`panel.csv`). Enforces zero-group leakage. |
| **4. Estimator Training** | `python3 research/postfreeze_qpu_workflow_mean_expansion_v1/train_neural.py` | PyTorch, PyTorch Geometric, CUDA. Trains G1, G2, M0, M1. |
| **5. Analytical Routes** | `pytest tests/test_analytical_path_kernel.py` | NumPy / g++; evaluates critical-path DAG recurrence and durations. |
| **6. Cost & Metrics Reporting** | `python3 scripts/verify_cost.py` | Validates warm inference latency (Clock B) and accuracy alignments. |

For detailed instructions on obtaining the external study bundle and reproducing specific tables, see [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

---

## Methodology Summary

- **Evaluation Unit:** A **logical-circuit group** (`group_id` / `structural_group`) groups circuits sharing topological structure. Angle-parameter variants remain grouped together to prevent data leakage.
- **Measurement Boundaries:** Runtime targets are evaluated strictly within their source-specific measurement clock (e.g., job duration vs. queue-to-completion). Clocks are never merged into an arbitrary universal unit.
- **Analytical Estimand Types:** We strictly differentiate raw nominal quantities, shot-scaled quantities, and train-only calibrated predictors.

Full details are documented in [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

---

## Citation

If you use Benchmark26 in your research, please cite:

```bibtex
@software{benchmark26_2026,
  author = {Dung Hoang},
  title = {Benchmark26: A Unified Quantum Program Runtime Estimation Benchmark},
  year = {2026},
  url = {https://github.com/dung-h/Benchmark26},
  version = {0.2.0}
}
```

See [CITATION.cff](CITATION.cff) for the complete Citation File Format metadata.

---

## License

The code and benchmark configuration scaffolding in this repository are licensed under the [MIT License](LICENSE). Third-party datasets and tools remain subject to their respective upstream licenses as detailed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
