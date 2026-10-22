PYTHON ?= python3
RUFF ?= ruff

.PHONY: test kernels model examples check-examples lint format check synth

test:
	$(PYTHON) scripts/run_tests.py

kernels:
	$(PYTHON) scripts/run_kernel.py all

model:
	$(PYTHON) scripts/run_model.py --tokens 1

synth:
	$(PYTHON) scripts/synthesize.py --verify

examples:
	$(PYTHON) scripts/write_kernel_examples.py

check-examples:
	$(PYTHON) scripts/write_kernel_examples.py --check

lint:
	$(RUFF) check snackwell scripts tests
	$(RUFF) format --check snackwell scripts tests

format:
	$(RUFF) format snackwell scripts tests

check: lint check-examples test
