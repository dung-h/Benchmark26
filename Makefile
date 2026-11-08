PYTHON ?= python3

.PHONY: help setup check check-study-inputs test smoke

help:
	@echo "make setup               install the package with test dependencies"
	@echo "make check               syntax, protocol hash chain and included aggregate statistics (latency, ready-input timing, graph ablation)"
	@echo "make check-study-inputs  as 'check', and fail if the external study bundle is missing"
	@echo "make test                unit tests"
	@echo "make smoke               synthetic split-fit-evaluate-report example in work/smoke"

setup:
	$(PYTHON) -m pip install -e ".[test]"

check:
	$(PYTHON) scripts/check_setup.py
	$(PYTHON) experiments/latency/verify_summary.py
	$(PYTHON) experiments/graph_ablation/verify_summary.py
	$(PYTHON) experiments/latency/verify_ready_analytical.py
	$(PYTHON) experiments/compiled_input/verify_summary.py
	$(PYTHON) experiments/compiled_input/verify_ready_prediction.py
	$(PYTHON) experiments/scholten/verify_summary.py
	$(PYTHON) experiments/qonductor_selection/verify_summary.py

check-study-inputs:
	$(PYTHON) scripts/check_setup.py --require-study-inputs

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) scripts/run_benchmark_pipeline.py --stage smoke --output work/smoke
