.PHONY: check test setup

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -e ".[test]"

check:
	$(PYTHON) scripts/check_setup.py
	$(PYTHON) -m compileall -q benchmark_v1 research scripts

test:
	$(PYTHON) -m pytest
