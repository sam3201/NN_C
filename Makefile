PY ?= python3
VENV ?= venv
VPY := $(VENV)/bin/python
PIP := $(VPY) -m pip
PORT ?= 5005

ifeq ($(OS),Windows_NT)
VPY := $(VENV)/Scripts/python.exe
PIP := $(VPY) -m pip
endif

.PHONY: help setup deps build run test clean doctor

help:
	@echo "make setup | deps | build | run PORT=5005 | test | clean | doctor"

setup:
	$(PY) -m venv $(VENV)
	$(PIP) install -U pip setuptools wheel

deps:
	$(PIP) install -r requirements.txt

build:
	rm -rf build/
	$(VPY) setup.py build_ext --inplace

run: build
	PYTHONPATH=src/python:. $(VPY) src/python/complete_sam_unified.py --port $(PORT)

test:
	$(VPY) -c "import sam_sav_dual_system, sam_meta_controller_c; print('C extensions import OK')"
	$(VPY) -m compileall -q src/python/complete_sam_unified.py

clean:
	rm -rf build/ *.egg-info
	rm -f src/*.so src/*.pyd src/*.dll

doctor:
	@echo "Python:"; $(VPY) -V || true
	@echo "Executable:"; $(VPY) -c "import sys; print(sys.executable)" || true

