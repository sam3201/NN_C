import sys
import os

# Add the project root to sys.path for module discovery during testing
sys.path.insert(0, os.path.abspath('.'))

def test_smoke_imports():
    from complete_sam_unified import UnifiedSAMSystem

    assert UnifiedSAMSystem is not None