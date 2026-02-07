#!/usr/bin/env python3
"""Simple test script to debug CompleteSAMSystem initialization"""

import sys
sys.path.insert(0, '.')

print("🔍 TESTING CompleteSAMSystem Initialization")
print("=" * 50)

try:
    print("1. Importing CompleteSAMSystem...")
    from complete_sam_system import CompleteSAMSystem
    print("   ✅ Import successful")

    print("2. Creating CompleteSAMSystem instance...")
    system = CompleteSAMSystem()
    print("   ✅ Instantiation successful")

    print("3. Checking sam_hub...")
    if system.sam_hub is not None:
        print("   ✅ sam_hub exists")
        print(f"   Type: {type(system.sam_hub)}")
        print(f"   Has run method: {hasattr(system.sam_hub, 'run')}")
    else:
        print("   ❌ sam_hub is None")

    print("4. Calling run()...")
    system.run()

except Exception as e:
    print(f"❌ TEST FAILED: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

print("=" * 50)
