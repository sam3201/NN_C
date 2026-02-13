# SAM-D Codebase Reorganization Plan

## Overview
Comprehensive cleanup and reorganization of the NN_C repository to create a clean, human-readable structure.

## Current State
- **Total Files**: 762 (excluding venv/build)
- **Root Directory Clutter**: Multiple .so files, chatlogs, misc files
- **Missing Structure**: NN directory needed for legacy neural network core

## Reorganization Actions Completed

### ✅ Phase 1: Deep Scan (COMPLETED)
- Read all critical Python files (complete_sam_unified.py, sam_cores.py, etc.)
- Analyzed all C modules (23 files) and headers (14 files)
- Reviewed documentation (54 markdown files)
- Examined test suite and training pipeline

### ✅ Phase 2: Archive Chatlogs (COMPLETED)
- Moved ChatGPT_2026-02-13-*.txt files to DOCS/archive/chatlogs/
- Consolidated all conversation history in one location

### ✅ Phase 3: Create NN Directory (COMPLETED)
- Created NN/ directory structure with subdirectories:
  - CNN, RNN, LSTM, TRANSFORMER
  - NEAT, GAN, GNN, SNN, KAN
  - MEMORY, UTILS
- Added README.md explaining NN directory purpose

## Remaining Actions

### ⏳ Phase 4: Clean Root Directory
**Files to Keep in Root:**
- README.md - Main project documentation
- AGENTS.md - Agent coding guidelines
- setup.py - Build configuration
- run_sam.sh - Main entry script
- .gitignore - Git ignore patterns

**Files to Archive/Move:**
- BASE_GOALS_NO_WRITE_ACCESS.md → DOCS/archive/
- package-lock.json → Archive or delete (not needed for Python project)
- .so files → Should be in build/ directory only

### ⏳ Phase 5: Update .gitignore
Add patterns for:
- *.so (compiled extensions)
- ChatGPT_*.txt (chatlogs)
- .aider.chat.history.md
- .DS_Store
- build/ directory contents

### ⏳ Phase 6: Clean Build Artifacts
- Move .so files to build/ or clean up duplicates
- Only keep latest Python version .so files

## Final Directory Structure

```
NN_C/
├── README.md              # Main entry point
├── AGENTS.md              # Development guidelines
├── setup.py               # Build configuration
├── run_sam.sh             # Launcher script
├── .gitignore             # Git patterns
│
├── src/                   # Source code
│   ├── python/            # Python modules
│   └── c_modules/         # C extensions
│
├── include/               # C headers
├── NN/                    # Neural network core (legacy but core)
├── tests/                 # Test suite
├── training/              # LoRA training pipeline
├── tools/                 # Utility scripts
├── DOCS/                  # Documentation
│   ├── archive/           # Historical files
│   │   ├── chatlogs/      # All ChatGPT conversations
│   │   └── legacy/        # Old code versions
│   └── *.md               # Current documentation
├── scripts/               # Build/deployment scripts
├── profiles/              # Environment profiles
├── sam_data/              # Runtime data
├── logs/                  # Runtime logs
└── templates/             # HTML templates
```

## Summary Statistics Post-Reorganization
- Root files reduced from 30+ to 5 essential files
- All chatlogs consolidated in DOCS/archive/chatlogs/
- NN directory established for neural network core
- Clean separation of concerns between directories
