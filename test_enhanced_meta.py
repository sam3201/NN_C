#!/usr/bin/env python3
"""
Enhanced MetaAgent Testing Suite
Tests the improved self-healing capabilities
"""

import os
import sys
import time
import tempfile
import shutil
import json
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from meta_agent_enhanced import EnhancedMetaAgent

class EnhancedMetaAgentTestSuite:
    """Test suite for Enhanced MetaAgent"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        self.test_files = {}
        
    def setup_test_environment(self):
        """Setup isolated test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="enhanced_meta_test_")
        print(f"🧪 Enhanced Test environment: {self.temp_dir}")
        
        # Create test files with various issues
        self.test_files = {
            'missing_colon.py': self._create_missing_colon_file(),
            'indentation_error.py': self._create_indentation_error_file(),
            'name_error.py': self._create_name_error_file(),
            'division_by_zero.py': self._create_division_by_zero_file(),
            'index_error.py': self._create_index_error_file(),
            'key_error.py': self._create_key_error_file(),
            'none_error.py': self._create_none_error_file(),
        }
        
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up enhanced test environment")
    
    def _create_missing_colon_file(self):
        """Create a file with missing colon"""
        content = '''
def broken_function()
    print("This will break")
    return "broken"

if True
    print("No colon here")
'''
        path = os.path.join(self.temp_dir, 'missing_colon.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_indentation_error_file(self):
        """Create a file with indentation error"""
        content = '''
def indented_function():
	if True:  # Tab instead of spaces
		print("Bad indentation")
	return "bad"
'''
        path = os.path.join(self.temp_dir, 'indentation_error.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_name_error_file(self):
        """Create a file with undefined variable"""
        content = '''
def use_undefined_var():
    result = undefined_variable + 10  # undefined_variable not defined
    return result
'''
        path = os.path.join(self.temp_dir, 'name_error.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_division_by_zero_file(self):
        """Create a file with division by zero"""
        content = '''
def divide_numbers(a, b):
    result = a / b  # Potential division by zero
    return result

# Test case
result = divide_numbers(10, 0)
'''
        path = os.path.join(self.temp_dir, 'division_by_zero.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_index_error_file(self):
        """Create a file with index error"""
        content = '''
def get_list_item(data, index):
    result = data[index]  # Potential index out of range
    return result

# Test case
my_list = [1, 2, 3]
item = get_list_item(my_list, 10)  # Will cause IndexError
'''
        path = os.path.join(self.temp_dir, 'index_error.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_key_error_file(self):
        """Create a file with key error"""
        content = '''
def get_dict_value(data, key):
    result = data[key]  # Potential key error
    return result

# Test case
my_dict = {'a': 1, 'b': 2}
value = get_dict_value(my_dict, 'c')  # Will cause KeyError
'''
        path = os.path.join(self.temp_dir, 'key_error.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_none_error_file(self):
        """Create a file with None value error"""
        content = '''
def process_value(value):
    result = value.upper()  # Will fail if value is None
    return result

# Test case
none_value = None
result = process_value(none_value)  # Will cause AttributeError
'''
        path = os.path.join(self.temp_dir, 'none_error.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def create_test_error(self, error_type, file_path):
        """Create a test error by trying to execute the file"""
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            # Try to execute to generate real error
            exec(compile(code, file_path, 'exec'))
            
        except Exception as e:
            return {
                'error_message': str(e),
                'stack_trace': traceback.format_exc(),
                'error_type': type(e).__name__
            }
        
        return {
            'error_message': 'No error generated',
            'stack_trace': '',
            'error_type': 'Unknown'
        }
    
    def test_missing_colon_fix(self):
        """Test fixing missing colon"""
        print("\n🧪 Testing Missing Colon Fix...")
        
        file_path = self.test_files['missing_colon.py']
        error = self.create_test_error('SyntaxError', file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        passed = result.get('status') == 'success'
        
        self.test_results.append({
            'test': 'missing_colon_fix',
            'passed': passed,
            'details': f"Status: {result.get('status')}, Confidence: {result.get('confidence', 0):.2f}"
        })
        
        return passed
    
    def test_indentation_fix(self):
        """Test fixing indentation error"""
        print("\n🧪 Testing Indentation Fix...")
        
        file_path = self.test_files['indentation_error.py']
        error = self.create_test_error('IndentationError', file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        passed = result.get('status') == 'success'
        
        self.test_results.append({
            'test': 'indentation_fix',
            'passed': passed,
            'details': f"Status: {result.get('status')}, Confidence: {result.get('confidence', 0):.2f}"
        })
        
        return passed
    
    def test_name_error_fix(self):
        """Test fixing name error"""
        print("\n🧪 Testing Name Error Fix...")
        
        file_path = self.test_files['name_error.py']
        error = self.create_test_error('NameError', file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        passed = result.get('status') == 'success'
        
        self.test_results.append({
            'test': 'name_error_fix',
            'passed': passed,
            'details': f"Status: {result.get('status')}, Confidence: {result.get('confidence', 0):.2f}"
        })
        
        return passed
    
    def test_division_by_zero_fix(self):
        """Test fixing division by zero"""
        print("\n🧪 Testing Division by Zero Fix...")
        
        file_path = self.test_files['division_by_zero.py']
        error = self.create_test_error('ZeroDivisionError', file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        passed = result.get('status') == 'success'
        
        self.test_results.append({
            'test': 'division_by_zero_fix',
            'passed': passed,
            'details': f"Status: {result.get('status')}, Confidence: {result.get('confidence', 0):.2f}"
        })
        
        return passed
    
    def test_index_error_fix(self):
        """Test fixing index error"""
        print("\n🧪 Testing Index Error Fix...")
        
        file_path = self.test_files['index_error.py']
        error = self.create_test_error('IndexError', file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        passed = result.get('status') == 'success'
        
        self.test_results.append({
            'test': 'index_error_fix',
            'passed': passed,
            'details': f"Status: {result.get('status')}, Confidence: {result.get('confidence', 0):.2f}"
        })
        
        return passed
    
    def test_key_error_fix(self):
        """Test fixing key error"""
        print("\n🧪 Testing Key Error Fix...")
        
        file_path = self.test_files['key_error.py']
        error = self.create_test_error('KeyError', file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        passed = result.get('status') == 'success'
        
        self.test_results.append({
            'test': 'key_error_fix',
            'passed': passed,
            'details': f"Status: {result.get('status')}, Confidence: {result.get('confidence', 0):.2f}"
        })
        
        return passed
    
    def test_none_error_fix(self):
        """Test fixing None error"""
        print("\n🧪 Testing None Error Fix...")
        
        file_path = self.test_files['none_error.py']
        error = self.create_test_error('AttributeError', file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        passed = result.get('status') == 'success'
        
        self.test_results.append({
            'test': 'none_error_fix',
            'passed': passed,
            'details': f"Status: {result.get('status')}, Confidence: {result.get('confidence', 0):.2f}"
        })
        
        return passed
    
    def test_error_detection_accuracy(self):
        """Test error type detection accuracy"""
        print("\n🧪 Testing Error Detection Accuracy...")
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        test_cases = [
            ('SyntaxError: missing colon at line 3', 'syntax_errors', 'missing_colon'),
            ('NameError: name undefined_variable is not defined', 'runtime_errors', 'name_error'),
            ('ZeroDivisionError: division by zero', 'runtime_errors', 'division_by_zero'),
            ('IndexError: list index out of range', 'runtime_errors', 'index_out_of_range'),
            ('KeyError: missing_key', 'runtime_errors', 'key_error'),
        ]
        
        correct_detections = 0
        total_tests = len(test_cases)
        
        for error_msg, expected_category, expected_type in test_cases:
            detected = enhanced_agent.detect_error_type(error_msg, '')
            
            if (detected and 
                detected.get('category') == expected_category and 
                detected.get('type') == expected_type):
                correct_detections += 1
        
        accuracy = correct_detections / total_tests
        
        passed = accuracy >= 0.8  # 80% accuracy threshold
        
        self.test_results.append({
            'test': 'error_detection_accuracy',
            'passed': passed,
            'details': f"Accuracy: {accuracy:.1%} ({correct_detections}/{total_tests})"
        })
        
        return passed
    
    def test_multiple_fix_strategies(self):
        """Test multiple fix strategies for same error"""
        print("\n🧪 Testing Multiple Fix Strategies...")
        
        file_path = self.test_files['key_error.py']
        error = self.create_test_error('KeyError', file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        # Check if multiple strategies were considered
        multiple_strategies = result.get('total_fixes_generated', 0) > 1
        
        passed = (result.get('status') == 'success') and multiple_strategies
        
        self.test_results.append({
            'test': 'multiple_fix_strategies',
            'passed': passed,
            'details': f"Fixes generated: {result.get('total_fixes_generated', 0)}, Status: {result.get('status')}"
        })
        
        return passed
    
    def test_learning_capabilities(self):
        """Test learning and improvement capabilities"""
        print("\n🧪 Testing Learning Capabilities...")
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        # Simulate multiple successful fixes
        for i in range(3):
            file_path = self.test_files['missing_colon.py']
            error = self.create_test_error('SyntaxError', file_path)
            
            result = enhanced_agent.handle_failure(
                error['error_message'], 
                error['stack_trace'], 
                file_path
            )
        
        # Check learning statistics
        stats = enhanced_agent.get_statistics()
        
        passed = (
            stats.get('successful_fixes', 0) > 0 and
            stats.get('success_rate', 0) > 0
        )
        
        self.test_results.append({
            'test': 'learning_capabilities',
            'passed': passed,
            'details': f"Success rate: {stats.get('success_rate', 0):.1%}, Fixes: {stats.get('successful_fixes', 0)}"
        })
        
        return passed
    
    def _create_mock_system(self):
        """Create a mock system for testing"""
        class MockSystem:
            def __init__(self):
                self.project_root = Path(self.temp_dir if 'self.temp_dir' in globals() else '/tmp')
        
        return MockSystem()
    
    def run_enhanced_tests(self):
        """Run all enhanced MetaAgent tests"""
        print("🚀 Starting Enhanced MetaAgent Test Suite")
        print("=" * 60)
        
        self.setup_test_environment()
        
        try:
            # Core functionality tests
            self.test_missing_colon_fix()
            self.test_indentation_fix()
            self.test_name_error_fix()
            self.test_division_by_zero_fix()
            self.test_index_error_fix()
            self.test_key_error_fix()
            self.test_none_error_fix()
            
            # Advanced capability tests
            self.test_error_detection_accuracy()
            self.test_multiple_fix_strategies()
            self.test_learning_capabilities()
            
        finally:
            self.cleanup_test_environment()
        
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 ENHANCED METAAGENT TEST SUITE SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        print("-" * 40)
        
        for result in self.test_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {result['test']}")
            print(f"    {result['details']}")
        
        print("\n🎯 Enhanced MetaAgent Assessment:")
        print("-" * 40)
        
        if failed_tests == 0:
            print("🎉 All tests passed! Enhanced MetaAgent is working excellently.")
            print("   ✅ Advanced error detection working")
            print("   ✅ Sophisticated fix strategies working")
            print("   ✅ Learning system operational")
        else:
            success_rate = (passed_tests/total_tests)*100
            if success_rate >= 80:
                print("🟢 Enhanced MetaAgent is working well!")
                print("   ✅ Significant improvements over original")
                print("   ✅ Most capabilities functional")
            elif success_rate >= 60:
                print("🟡 Enhanced MetaAgent shows improvement!")
                print("   ✅ Some capabilities working")
                print("   🔧 Further refinements needed")
            else:
                print("🔴 Enhanced MetaAgent needs more work!")
                print("   ❌ Many capabilities not working")
                print("   🔧 Major improvements required")
        
        print("=" * 60)

def main():
    """Main test runner"""
    print("🤖 Enhanced MetaAgent Testing Suite")
    print("Testing improved self-healing capabilities")
    print()
    
    # Run enhanced test suite
    test_suite = EnhancedMetaAgentTestSuite()
    test_suite.run_enhanced_tests()
    
    # Return exit code based on results
    failed_tests = sum(1 for result in test_suite.test_results if not result['passed'])
    return 0 if failed_tests == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
