# AGENTS.md - Agent Coding Guidelines for SAM-D AGI

## Project Overview

SAM-D is a hybrid Python/C recursive meta-evolutionary AGI system with a web dashboard, slash-command interface, and C-accelerated cores for meta-control and dual-system simulation. The project uses Python orchestration with C extensions for performance-critical components.

## Build Commands

### Installation & Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Build C extensions (required before running)
python setup.py build_ext --inplace

# Optional: Native optimization (requires SAM_NATIVE=1 environment variable)
SAM_NATIVE=1 python setup.py build_ext --inplace
```

### Running the Application
```bash
# Using the run script (loads profile from profiles/)
./run_sam.sh

# Or directly with Python
python3 complete_sam_unified.py

# With specific profile (full or experimental)
SAM_PROFILE=experimental ./run_sam.sh
```

### Smoke Tests
```bash
# Test C extensions import
python3 -c "import sam_sav_dual_system, sam_meta_controller_c; print('C extensions import OK')"

# Test Python system import
python3 -c "from complete_sam_unified import UnifiedSAMSystem; print('System import OK')"

# Test Python compilation
python3 -m py_compile complete_sam_unified.py
```

### Running Tests

#### Run All Tests
```bash
# Using pytest (discovers tests/ directory)
pytest -q

# Or with verbose output
pytest -v
```

#### Run a Single Test
```bash
# Run specific test file
pytest tests/test_smoke.py -v

# Run specific test function
pytest tests/test_smoke.py::test_smoke_imports -v

# Run by pattern match
pytest -k "test_smoke" -v

# Run single test file with full path
python -m pytest tests/test_orchestrator.py -v
```

#### Comprehensive & Regression Tests
```bash
# Run comprehensive system tests
SAM_TEST_MODE=1 ./venv/bin/python -c "from SAM_AGI import CompleteSAMSystem; s=CompleteSAMSystem(); s.run_comprehensive_tests()"

# Run recursive checks (includes regression suite)
./tools/run_recursive_checks.sh

# Run regression suite directly
python3 -m training.regression_suite \
  --tasks training/tasks/default_tasks.jsonl \
  --provider ollama:mistral:latest \
  --min-pass 0.7 \
  --max-examples 5
```

## Code Style Guidelines

### General Conventions

- **Python Version**: 3.10+
- **Encoding**: UTF-8, use `from __future__ import annotations` for forward references
- **Line Length**: Target under 100 characters (soft limit)
- **Indentation**: 4 spaces (no tabs)

### Naming Conventions

- **Classes**: PascalCase (e.g., `UnifiedSAMSystem`, `CircuitBreaker`)
- **Functions/Methods**: snake_case (e.g., `get_config()`, `call()`)
- **Variables**: snake_case (e.g., `failure_threshold`, `config_dict`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `COMMON_ARGS`, `DEFAULT_TIMEOUT`)
- **Private Members**: Prefix with underscore (e.g., `_on_failure()`)
- **C Extensions**: Underscore-separated lowercase (e.g., `sam_sav_dual_system`)

### Type Hints

- Use Python's `typing` module for type annotations
- Common types: `Dict`, `List`, `Any`, `Callable`, `Optional`, `Union`
- Use `type` alias for complex types
- Return types should be annotated for public methods
- Example:
  ```python
  from typing import Dict, Any, Callable, Optional, List

  def get_config(key: str = None, default: Any = None) -> Any:
      ...
  ```

### Import Organization

Order imports (separated by blank lines):
1. Standard library (`import os`, `import sys`)
2. Third-party packages (`import pytest`)
3. Local project imports (`from complete_sam_unified import ...`)

```python
import os
import sys
import time
import threading
from typing import Dict, Any, Callable, Optional
from enum import Enum

import requests

from complete_sam_unified import UnifiedSAMSystem
from src.python.circuit_breaker import CircuitBreaker
```

### Docstrings

Use docstrings for all public classes and functions:

```python
class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        ...
```

### Error Handling

- Use specific exception types rather than bare `except:`
- Include meaningful error messages
- Use try/except blocks for recoverable errors
- Example:
  ```python
  try:
      result = func(*args, **kwargs)
      self._on_success()
      return result
  except self.expected_exception as e:
      self._on_failure()
      raise e
  ```

### C Extension Integration

- C extensions are built via `setup.py` using `setuptools.Extension`
- Extension modules use underscore naming: `module_name.cpython-*.so`
- Python bindings follow C function naming conventions
- Test C extensions directly in `tests/test_*.py` files

### File Organization

```
src/python/      - Main Python source files
src/c_modules/   - C source files for extensions
tests/           - Test files (test_*.py)
tools/           - Utility scripts
DOCS/            - Documentation
include/         - C header files
profiles/        - Environment configuration
```

### Testing Patterns

Tests follow this structure:
```python
#!/usr/bin/env python3
"""Test description"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def test_something():
    """Test function description"""
    # Test code
    assert condition, "Failure message"

if __name__ == "__main__":
    test_something()
```

### Important Environment Variables

- `SAM_PROFILE` - Execution profile (full/experimental)
- `SAM_NATIVE` - Enable native C optimizations (1/0)
- `SAM_REGRESSION_ON_GROWTH` - Enable regression gate (1/0)
- `SAM_TEST_MODE` - Run in test mode (1/0)
- `SAM_POLICY_PROVIDER` - Model provider for policy decisions

### Configuration Files

- `.env.local` - Local environment overrides
- `profiles/full.env` - Full profile settings
- `profiles/experimental.env` - Experimental profile settings
- `.aider.conf.yml` - Aider AI assistant configuration

### Best Practices

1. Always build C extensions after modifying C source code
2. Run smoke tests before committing changes
3. Use the regression gate when making structural changes
4. Test C extension imports separately from Python modules
5. Follow the import order conventions (stdlib, third-party, local)
6. Add type hints for public API functions
7. Use descriptive variable and function names
8. Keep functions focused and single-purpose
