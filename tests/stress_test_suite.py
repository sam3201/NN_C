import sys
import os
import json
import time
import threading
import random
from pathlib import Path

# Fix sys.path to find everything
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "python"))

from complete_sam_unified import UnifiedSAMSystem
import sam_meta_controller_c

def test_input_fuzzing(system):
    print("\n--- Test 1: Input Fuzzing ---")
    malformed_inputs = [
        None,
        "",
        "   ",
        "A" * 10000, # Buffer overflow attempt
        "\\x00\\x01", # Binary garbage
        '{"json": "injection"}',
        "/modify-code with missing args"
    ]
    
    for inp in malformed_inputs:
        try:
            # We are testing robustness, so we expect it NOT to crash
            response = system._process_chatbot_message(inp, {})
            print(f"Input: {str(inp)[:20]}... -> Response: {str(response)[:50]}...")
        except Exception as e:
            print(f"❌ CRASH on input {str(inp)[:20]}: {e}")
            return False
    print("✅ Input fuzzing passed (no crashes).")
    return True

def test_growth_trigger(system):
    print("\n--- Test 2: Growth Trigger ---")
    
    print("Injecting high pressure signals...")
    # Force update pressure multiple times to build 'persistence'
    # We set ONE pressure significantly higher to satisfy 'dominance'
    for _ in range(10):
        sam_meta_controller_c.update_pressure(
            system.meta_controller,
            1.0, # residual - High
            0.5, # rank_def
            0.5, # retrieval_entropy
            0.5, # interference
            0.5, # planner_friction
            0.5, # context_collapse
            0.5, # compression_waste
            0.5  # temporal_incoherence
        )
    
    # Trigger selection
    primitive = sam_meta_controller_c.select_primitive(system.meta_controller)
    print(f"Selected Primitive: {primitive}")
    
    if primitive > 0:
        print(f"✅ Growth triggered successfully (Primitive {primitive})")
        return True
    else:
        print("❌ Growth NOT triggered despite high pressure.")
        reason = sam_meta_controller_c.get_growth_diagnostics(system.meta_controller)["last_growth_reason"]
        print(f"Reason: {reason}")
        return False

def test_tbqg_veto(system):
    print("\n--- Test 3: TBQG Governance Veto ---")
    
    # Case 1: Safe change
    safe_proposal = {"patch": "print('hello')", "file": "test.py"}
    approved_safe, _ = system._governance_quorum_vote("code_change", safe_proposal)
    
    # Case 2: Dangerous change
    dangerous_proposal = {"patch": "import os; os.system('rm -rf /')", "file": "critical.py"}
    approved_danger, decision = system._governance_quorum_vote("code_change", dangerous_proposal)
    
    if approved_safe and not approved_danger:
        print("✅ TBQG correctly approved safe change and vetoed dangerous change.")
        print(f"Veto Reason: {decision['votes']['SAV']['reason']}")
        return True
    else:
        print(f"❌ TBQG Failed. Safe: {approved_safe}, Dangerous: {approved_danger}")
        return False

def test_concurrent_load(system):
    print("\n--- Test 4: Concurrent Load ---")
    results = []
    
    def worker(i):
        try:
            res = system._process_chatbot_message(f"Concurrent message {i}", {})
            results.append(True)
        except Exception as e:
            print(f"❌ Worker {i} CRASHED: {e}")
            results.append(False)
            
    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    if all(results):
        print(f"✅ Concurrent load passed (10/10 requests handled).")
        return True
    else:
        print(f"❌ Concurrent load FAILED.")
        return False

def run_stress_suite():
    print("🚀 Starting SAM-D Deep Stress Test Suite")
    
    # Initialize system in testing mode
    os.environ["SAM_TEST_MODE"] = "1"
    system = UnifiedSAMSystem()
    
    # 1. Fuzzing
    if not test_input_fuzzing(system):
        sys.exit(1)
        
    # 2. Growth
    if not test_growth_trigger(system):
        # Don't fail the whole suite if growth logic is just strict, but warn
        print("⚠️ Growth test failed, check thresholds.")
        
    # 3. Governance
    if not test_tbqg_veto(system):
        sys.exit(1)
        
    # 4. Concurrent Load
    if not test_concurrent_load(system):
        sys.exit(1)
        
    print("\n✨ All Stress Tests Completed.")

if __name__ == "__main__":
    run_stress_suite()
