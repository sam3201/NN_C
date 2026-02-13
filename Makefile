SHELL := /bin/bash

VENV := venv
PY   := $(VENV)/bin/python
PIP  := $(PY) -m pip

# Windows fallback (if user runs Make from Git Bash)
ifeq ($(OS),Windows_NT)
	PY := $(VENV)/Scripts/python.exe
	PIP := $(PY) -m pip
endif

PORT ?= 5005

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make setup      - create venv + install deps"
	@echo "  make deps       - install python deps only"
	@echo "  make build      - build C extensions in-place"
	@echo "  make run        - run server on PORT (default 5005)"
	@echo "  make test       - run smoke tests"
	@echo "  make clean      - remove build artifacts"
	@echo "  make reset      - delete venv + build artifacts"

$(VENV):
	python -m venv $(VENV)

.PHONY: setup
setup: $(VENV) deps build

.PHONY: deps
deps: $(VENV)
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r requirements.txt

.PHONY: build
build: $(VENV)
	rm -rf build/
	$(PY) setup.py build_ext --inplace

.PHONY: run
run: $(VENV)
	PYTHONPATH=src/python:. $(PY) src/python/complete_sam_unified.py --port $(PORT)

.PHONY: test
test: $(VENV)
	$(PY) -c "import sam_sav_dual_system, sam_meta_controller_c; print('C extensions import OK')"
	$(PY) -c "from src.python.complete_sam_unified import UnifiedSAMSystem; print('System import OK')"

.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} \; || true
	find . -name "*.pyc" -delete || true

.PHONY: reset
reset: clean
	rm -rf $(VENV)

