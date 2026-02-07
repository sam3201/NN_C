#!/usr/bin/env python3
"""Simple test to check if complete_sam_system can be imported"""

import sys
sys.path.insert(0, '.')

print("🔍 Testing complete_sam_system import...", flush=True)

try:
    print("Importing complete_sam_system...", flush=True)
    import complete_sam_system
    print("✅ Import successful", flush=True)

    print("Checking for CompleteSAMSystem class...", flush=True)
    if hasattr(complete_sam_system, 'CompleteSAMSystem'):
        print("✅ CompleteSAMSystem class found", flush=True)

        print("Trying to instantiate CompleteSAMSystem...", flush=True)
        system = complete_sam_system.CompleteSAMSystem()
        print("✅ Instantiation successful", flush=True)

        print("Checking sam_hub...", flush=True)
        if system.sam_hub is not None:
            print("✅ sam_hub exists", flush=True)
        else:
            print("❌ sam_hub is None", flush=True)

    else:
        print("❌ CompleteSAMSystem class not found", flush=True)

except Exception as e:
    print(f"❌ Error: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("Test completed", flush=True)
