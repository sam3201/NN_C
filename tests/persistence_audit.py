import sys
import os
import json
import time
import importlib
from pathlib import Path

# Add src/python to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "python"))

import complete_sam_unified
print(f"DEBUG: complete_sam_unified file: {complete_sam_unified.__file__}")
import inspect
print(f"DEBUG: module source preview:\n{inspect.getsource(complete_sam_unified)[:200]}")

def test_persistence():
    print("\n--- Persistence Audit ---")
    os.environ["SAM_TEST_MODE"] = "1"
    
    # 1. Initialize and modify
    try:
        sys1 = complete_sam_unified.UnifiedSAMSystem()
    except Exception as e:
        print(f"❌ sys1 init failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print(f"DEBUG: sys1.state_path = {sys1.state_path}")
    original_val = sys1.system_metrics.get("total_conversations", 0)
    sys1.system_metrics["total_conversations"] = original_val + 42
    sys1._save_system_state()
    print(f"Saved state with total_conversations = {sys1.system_metrics['total_conversations']}")
    
    # Small delay to ensure file write is complete
    time.sleep(1)
    
    if sys1.state_path.exists():
        print(f"✅ State file exists at: {sys1.state_path}")
    else:
        print(f"❌ State file MISSING at: {sys1.state_path}")
    
    # 2. Reload in a new instance
    importlib.reload(complete_sam_unified)
    try:
        sys2 = complete_sam_unified.UnifiedSAMSystem()
    except Exception as e:
        print(f"❌ sys2 init failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print(f"DEBUG: sys2.state_path = {sys2.state_path}")
    reloaded_val = sys2.system_metrics.get("total_conversations", 0)
    print(f"Reloaded total_conversations = {reloaded_val}")
    
    if reloaded_val == original_val + 42:
        print("✅ Persistence verified.")
        return True
    else:
        print("❌ Persistence FAILED.")
        return False

if __name__ == "__main__":
    if not test_persistence():
        sys.exit(1)
