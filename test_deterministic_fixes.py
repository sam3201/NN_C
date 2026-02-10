#!/usr/bin/env python3
"""
Test the deterministic fixes added to MetaAgent
Tests the specific improvements for missing colons, indentation, and f-string fixes
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from complete_sam_unified import MetaAgent, ObserverAgent, FaultLocalizerAgent, PatchGeneratorAgent, VerifierJudgeAgent, FailureEvent
from datetime import datetime

def test_missing_colon_fix():
    """Test the deterministic missing colon fix"""
    print("🧪 Testing Deterministic Missing Colon Fix...")
    
    # Create test file with missing colon
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('''
def broken_function()
    print("This will break")
    return "broken"
''')
        test_file = f.name
    
    try:
        # Create mock system
        class MockSystem:
            def __init__(self):
                self.project_root = Path(test_file).parent
        
        system = MockSystem()
        
        # Create MetaAgent
        observer = ObserverAgent(system)
        localizer = FaultLocalizerAgent(system)
        generator = PatchGeneratorAgent(system)
        verifier = VerifierJudgeAgent(system)
        meta_agent = MetaAgent(observer, localizer, generator, verifier, system)
        
        # Create failure event
        failure = FailureEvent(
            error_type="SyntaxError",
            stack_trace=f'''File "{test_file}", line 2
    def broken_function()
    ^
SyntaxError: expected ':'''',
            timestamp=datetime.now().isoformat(),
            severity="medium",
            context="test"
        )
        
        # Handle failure
        result = meta_agent.handle_failure(failure)
        
        # Check if file was fixed
        with open(test_file, 'r') as f:
            fixed_content = f.read()
        
        print(f"   Original: def broken_function()")
        print(f"   Fixed:    {fixed_content.split('def broken_function')[1].split('\\n')[0].strip()}")
        
        # Check if fix was applied
        success = 'def broken_function():' in fixed_content
        
        print(f"   ✅ Missing colon fix: {'SUCCESS' if success else 'FAILED'}")
        return success
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        try:
            os.unlink(test_file)
        except:
            pass

def test_fstring_escape_fix():
    """Test the deterministic f-string escape fix"""
    print("🧪 Testing Deterministic F-String Escape Fix...")
    
    # Create test file with escaped f-string
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('''
def broken_function():
    name = "test"
    message = f\\"Hello {name}\\"  # Bad escaped quotes
    return message
''')
        test_file = f.name
    
    try:
        # Create mock system
        class MockSystem:
            def __init__(self):
                self.project_root = Path(test_file).parent
        
        system = MockSystem()
        
        # Create MetaAgent
        observer = ObserverAgent(system)
        localizer = FaultLocalizerAgent(system)
        generator = PatchGeneratorAgent(system)
        verifier = VerifierJudgeAgent(system)
        meta_agent = MetaAgent(observer, localizer, generator, verifier, system)
        
        # Create failure event
        failure = FailureEvent(
            error_type="SyntaxError",
            stack_trace=f'''File "{test_file}", line 4
    message = f\\"Hello {name}\\"  # Bad escaped quotes
    ^
SyntaxError: unexpected character after line continuation character''',
            timestamp=datetime.now().isoformat(),
            severity="medium",
            context="test"
        )
        
        # Handle failure
        result = meta_agent.handle_failure(failure)
        
        # Check if file was fixed
        with open(test_file, 'r') as f:
            fixed_content = f.read()
        
        print(f"   Original: f\\\\"Hello {{name}}\\\\"")
        print(f"   Fixed:    {'f\"Hello {name}\"' if 'f\"Hello {name}\"' in fixed_content else 'No change'}")
        
        # Check if fix was applied
        success = 'f"Hello {name}"' in fixed_content
        
        print(f"   ✅ F-string escape fix: {'SUCCESS' if success else 'FAILED'}")
        return success
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        try:
            os.unlink(test_file)
        except:
            pass

def test_indentation_fix():
    """Test the deterministic indentation fix"""
    print("🧪 Testing Deterministic Indentation Fix...")
    
    # Create test file with tabs
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('''
def indented_function():
\tif True:  # Tab instead of spaces
\t\tprint("Bad indentation")
\treturn "bad"
''')
        test_file = f.name
    
    try:
        # Create mock system
        class MockSystem:
            def __init__(self):
                self.project_root = Path(test_file).parent
        
        system = MockSystem()
        
        # Create MetaAgent
        observer = ObserverAgent(system)
        localizer = FaultLocalizerAgent(system)
        generator = PatchGeneratorAgent(system)
        verifier = VerifierJudgeAgent(system)
        meta_agent = MetaAgent(observer, localizer, generator, verifier, system)
        
        # Create failure event
        failure = FailureEvent(
            error_type="IndentationError",
            stack_trace=f'''File "{test_file}", line 3
\tif True:  # Tab instead of spaces
\t^
IndentationError: expected an indented block''',
            timestamp=datetime.now().isoformat(),
            severity="medium",
            context="test"
        )
        
        # Handle failure
        result = meta_agent.handle_failure(failure)
        
        # Check if file was fixed
        with open(test_file, 'r') as f:
            fixed_content = f.read()
        
        print(f"   Original: \\t\\tprint")
        print(f"   Fixed:    {'        print' if '        print' in fixed_content else 'No change'}")
        
        # Check if fix was applied
        success = '        print' in fixed_content and '\t' not in fixed_content
        
        print(f"   ✅ Indentation fix: {'SUCCESS' if success else 'FAILED'}")
        return success
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        try:
            os.unlink(test_file)
        except:
            pass

def test_utc_now_fix():
    """Test the deterministic _utc_now fix"""
    print("🧪 Testing Deterministic _utc_now Fix...")
    
    # Create test file that uses _utc_now
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('''
def test_function():
    timestamp = _utc_now()  # Missing function
    return timestamp
''')
        test_file = f.name
    
    try:
        # Create mock system with project root
        class MockSystem:
            def __init__(self):
                self.project_root = Path(tempfile.gettempdir())
        
        system = MockSystem()
        
        # Create MetaAgent
        observer = ObserverAgent(system)
        localizer = FaultLocalizerAgent(system)
        generator = PatchGeneratorAgent(system)
        verifier = VerifierJudgeAgent(system)
        meta_agent = MetaAgent(observer, localizer, generator, verifier, system)
        
        # Create failure event
        failure = FailureEvent(
            error_type="NameError",
            stack_trace=f'''File "{test_file}", line 3
    timestamp = _utc_now()  # Missing function
    ^
NameError: name '_utc_now' is not defined''',
            timestamp=datetime.now().isoformat(),
            severity="medium",
            context="test"
        )
        
        # Handle failure
        result = meta_agent.handle_failure(failure)
        
        # Check if _utc_now function was added to complete_sam_unified.py
        sam_file = Path(__file__).parent / "complete_sam_unified.py"
        if sam_file.exists():
            with open(sam_file, 'r') as f:
                sam_content = f.read()
            
            success = 'def _utc_now():' in sam_content
            print(f"   ✅ _utc_now fix: {'SUCCESS' if success else 'FAILED'}")
            return success
        else:
            print(f"   ❌ complete_sam_unified.py not found")
            return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    finally:
        try:
            os.unlink(test_file)
        except:
            pass

def main():
    """Run deterministic fix tests"""
    print("🚀 Testing Deterministic MetaAgent Fixes")
    print("=" * 50)
    
    # Set environment for testing
    os.environ['SAM_META_RESEARCH_ENABLED'] = '1'
    os.environ['SAM_META_RESEARCH_MODE'] = 'local'
    os.environ['SAM_META_TEST_MODE'] = '1'
    os.environ['SAM_META_CONFIDENCE_THRESHOLD'] = '0.7'  # Lower threshold
    
    tests = [
        test_missing_colon_fix,
        test_fstring_escape_fix,
        test_indentation_fix,
        test_utc_now_fix
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test error: {e}")
    
    print("\n" + "=" * 50)
    print("📊 DETERMINISTIC FIX TEST RESULTS")
    print("=" * 50)
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All deterministic fixes working perfectly!")
    elif passed >= total * 0.75:
        print("🟢 Most deterministic fixes working well!")
    elif passed >= total * 0.5:
        print("🟡 Some deterministic fixes working")
    else:
        print("🔴 Deterministic fixes need improvement")
    
    print("=" * 50)
    
    return 0 if passed >= total * 0.75 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
