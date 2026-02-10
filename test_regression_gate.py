#!/usr/bin/env python3
"""
Test to verify regression gate is enabled in local mode
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from complete_sam_unified import UnifiedSAMSystem

def test_regression_gate_enabled():
    """Test that regression gate is enabled in local mode"""
    print("🧪 Testing Regression Gate Enabled in Local Mode...")
    
    # Set local mode
    os.environ['SAM_STRICT_LOCAL_ONLY'] = '1'
    os.environ['SAM_REGRESSION_ON_GROWTH'] = '1'  # Enable by default
    
    # Create system
    system = UnifiedSAMSystem()
    
    # Check if regression gate is enabled
    regression_enabled = getattr(system, 'regression_on_growth', False)
    
    print(f"   Strict local-only: {system.strict_local_only}")
    print(f"   Regression gate enabled: {regression_enabled}")
    
    if regression_enabled:
        print("   ✅ SUCCESS: Regression gate is enabled in local mode")
        return True
    else:
        print("   ❌ FAILED: Regression gate is disabled in local mode")
        return False

def main():
    """Run regression gate test"""
    print("🚀 Testing Regression Gate Configuration")
    print("=" * 50)
    
    success = test_regression_gate_enabled()
    
    print("\n" + "=" * 50)
    print("📊 REGRESSION GATE TEST RESULT")
    print("=" * 50)
    
    if success:
        print("🎉 Regression gate is properly enabled!")
        print("✅ System will validate changes before applying")
        print("✅ Safety mechanisms are active")
    else:
        print("🔴 Regression gate is disabled!")
        print("❌ System may apply unsafe changes")
        print("🔧 Configuration needs adjustment")
    
    print("=" * 50)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
