.PHONY: help setup check test test-unit status fetch-sources-dry validate-provenance

PYTHON ?= python3

help:
	@echo "Benchmark26 — Available Commands:"
	@echo "  make setup                Install package and lightweight test dependencies"
	@echo "  make check                Validate Python syntax, protocol schemas, and study inputs"
	@echo "  make test                 Run all unit tests across tests/ and research/"
	@echo "  make test-unit            Run fast isolated unit tests in tests/"
	@echo "  make status               Display readiness status of each benchmark pipeline stage"
	@echo "  make fetch-sources-dry    List all pinned upstream source assets without downloading"
	@echo "  make validate-provenance  Verify cryptographic hash and provenance integrity"

setup:
	$(PYTHON) -m pip install -e ".[test]"

check:
	$(PYTHON) scripts/check_setup.py
	$(PYTHON) -m compileall -q benchmark_v1 experiments legacy scripts tests
	$(PYTHON) experiments/latency/verify_summary.py

test:
	$(PYTHON) -m pytest

test-unit:
	$(PYTHON) -m pytest tests/

status:
	$(PYTHON) scripts/run_benchmark_pipeline.py --status

fetch-sources-dry:
	$(PYTHON) scripts/fetch_upstream_data.py --source all --destination /tmp/benchmark26_dry_run --list

validate-provenance:
	$(PYTHON) -m pytest tests/test_provenance_schema.py tests/test_qonductor_recipe_screen.py

.PHONY: verify-latency
verify-latency:
	$(PYTHON) experiments/latency/verify_summary.py

.PHONY: smoke
smoke:
	$(PYTHON) scripts/run_benchmark_pipeline.py --stage smoke --output work/smoke
