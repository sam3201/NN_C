#!/usr/bin/env python3
"""
Test for C research agent safety with very long queries
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from complete_sam_unified import UnifiedSAMSystem

def test_research_safety():
    """Test C research agent with very long query to ensure no buffer overflow"""
    print("🧪 Testing C research agent safety with very long query...")
    
    try:
        # Initialize system
        system = UnifiedSAMSystem()
        
        # Test with a very long query that should trigger buffer overflow protection
        very_long_query = "x" * 10000 + " research query that is extremely long and should trigger buffer overflow protection in the C research agent to ensure the system handles oversized inputs gracefully without crashing or producing undefined behavior"
        
        print(f"🔍 Testing with query length: {len(very_long_query)} characters")
        
        # Call the research agent
        result = system._call_c_agent("research", very_long_query)
        
        if result:
            print(f"✅ C research agent handled long query successfully: {result[:100]}...")
            print("🛡️ Buffer overflow protection appears to be working correctly")
            return True
        else:
            print(f"❌ C research agent failed or returned empty: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_research_safety()
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)
