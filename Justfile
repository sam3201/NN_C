# Justfile for SAM / NN_C
# Usage:
#   just            # lists recipes
#   just run        # bootstrap + run (default profile=full)
#   just run experimental
#   just bootstrap
#   just build
#   just test
#   just clean

set shell := ["bash", "-cu"]

default:
  @just --list

# ---------- Platform detection helpers ----------
is_windows := if os() == "windows" { "1" } else { "0" }

# ---------- Core tasks ----------
bootstrap profile="full":
  @echo "== bootstrap (profile={{profile}}) =="
  @if [ "{{is_windows}}" = "1" ]; then \
    powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1 {{profile}} ; \
  else \
    chmod +x ./bootstrap.sh ; \
    ./bootstrap.sh {{profile}} ; \
  fi

run profile="full":
  @just bootstrap {{profile}}

build:
  @echo "== build extensions =="
  @if [ "{{is_windows}}" = "1" ]; then \
    powershell -NoProfile -ExecutionPolicy Bypass -Command "if (!(Test-Path .venv)) { py -m venv .venv }; .\.venv\Scripts\Activate.ps1; python -m pip install -U pip setuptools wheel; python setup.py build_ext --inplace" ; \
  else \
    if [ ! -d .venv ]; then python3 -m venv .venv; fi ; \
    source .venv/bin/activate ; \
    python -m pip install -U pip setuptools wheel ; \
    python setup.py build_ext --inplace ; \
  fi

deps:
  @echo "== install python deps =="
  @if [ "{{is_windows}}" = "1" ]; then \
    powershell -NoProfile -ExecutionPolicy Bypass -Command "if (!(Test-Path .venv)) { py -m venv .venv }; .\.venv\Scripts\Activate.ps1; python -m pip install -U pip setuptools wheel; if (Test-Path requirements.txt) { pip install -r requirements.txt } else { pip install requests requests-oauthlib numpy }" ; \
  else \
    if [ ! -d .venv ]; then python3 -m venv .venv; fi ; \
    source .venv/bin/activate ; \
    python -m pip install -U pip setuptools wheel ; \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; else pip install requests requests-oauthlib numpy; fi ; \
  fi

test:
  @echo "== pytest =="
  @if [ "{{is_windows}}" = "1" ]; then \
    powershell -NoProfile -ExecutionPolicy Bypass -Command ".\.venv\Scripts\Activate.ps1; pytest -q" ; \
  else \
    source .venv/bin/activate ; \
    pytest -q ; \
  fi

clean:
  @echo "== clean build artifacts =="
  @rm -rf build dist *.egg-info __pycache__ */__pycache__ .pytest_cache
  @rm -f *.so *.dylib *.pyd

# Convenience
profile-full:
  @just run full

profile-experimental:
  @just run experimental

