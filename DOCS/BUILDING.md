# Building SAM-D (ΨΔ-Core)

This guide covers how to build the SAM-D system, including its C-accelerated neural network cores.

## Prerequisites

- **Python 3.10+**
- **C Compiler**: GCC or Clang (macOS: `xcode-select --install`)
- **Python Development Headers**: Usually included with Python installers, or `python3-dev` on Linux.

## Standard Build

The easiest way to build the system is using the provided build script:

```bash
./scripts/build.sh
```

This script will:
1. Clean old build artifacts.
2. Build all C extensions in `src/c_modules/`.
3. Verify that the extensions can be imported.

## Manual Build

You can also use `setup.py` directly:

```bash
python3 setup.py build_ext --inplace
```

## Troubleshooting

### Missing Headers
If you get errors about missing `Python.h`, ensure your python development headers are installed.

### Architecture Mismatch (macOS)
If building on Apple Silicon, ensure your Python and compiler are both targeting `arm64`.

### Clean Rebuild
To force a clean rebuild:
```bash
rm -rf build/
find . -name "*.so" -delete
./scripts/build.sh
```
