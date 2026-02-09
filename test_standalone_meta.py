#!/usr/bin/env python3
"""
Standalone MetaAgent Testing Suite
Tests the enhanced MetaAgent without complex dependencies
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

class StandaloneMetaAgentTest:
    """Standalone test for Enhanced MetaAgent"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        self.test_files = {}
        
    def setup_test_environment(self):
        """Setup isolated test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="standalone_meta_test_")
        print(f"🧪 Standalone Test environment: {self.temp_dir}")
        
        # Create test files with various issues
        self.test_files = {
            'syntax_colon.py': self._create_syntax_colon_file(),
            'syntax_indent.py': self._create_syntax_indent_file(),
            'runtime_division.py': self._create_runtime_division_file(),
            'runtime_index.py': self._create_runtime_index_file(),
            'runtime_key.py': self._create_runtime_key_file(),
            'runtime_none.py': self._create_runtime_none_file(),
        }
        
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up standalone test environment")
    
    def _create_syntax_colon_file(self):
        """Create a file with missing colon"""
        content = '''
def broken_function()
    print("This will break")
    return "broken"

if True
    print("No colon here")
'''
        path = os.path.join(self.temp_dir, 'syntax_colon.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_syntax_indent_file(self):
        """Create a file with indentation error"""
        content = '''
def indented_function():
	if True:  # Tab instead of spaces
		print("Bad indentation")
	return "bad"
'''
        path = os.path.join(self.temp_dir, 'syntax_indent.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_runtime_division_file(self):
        """Create a file with division by zero"""
        content = '''
def divide_numbers(a, b):
    result = a / b  # Potential division by zero
    return result

# Test case
result = divide_numbers(10, 0)
'''
        path = os.path.join(self.temp_dir, 'runtime_division.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_runtime_index_file(self):
        """Create a file with index error"""
        content = '''
def get_list_item(data, index):
    result = data[index]  # Potential index out of range
    return result

# Test case
my_list = [1, 2, 3]
item = get_list_item(my_list, 10)  # Will cause IndexError
'''
        path = os.path.join(self.temp_dir, 'runtime_index.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_runtime_key_file(self):
        """Create a file with key error"""
        content = '''
def get_dict_value(data, key):
    result = data[key]  # Potential key error
    return result

# Test case
my_dict = {'a': 1, 'b': 2}
value = get_dict_value(my_dict, 'c')  # Will cause KeyError
'''
        path = os.path.join(self.temp_dir, 'runtime_key.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_runtime_none_file(self):
        """Create a file with None value error"""
        content = '''
def process_value(value):
    result = value.upper()  # Will fail if value is None
    return result

# Test case
none_value = None
result = process_value(none_value)  # Will cause AttributeError
'''
        path = os.path.join(self.temp_dir, 'runtime_none.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def generate_error_from_file(self, file_path):
        """Execute file to generate real error"""
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            # Try to execute to generate real error
            exec(compile(code, file_path, 'exec'))
            
        except Exception as e:
            return {
                'error_message': str(e),
                'stack_trace': traceback.format_exc(),
                'error_type': type(e).__name__,
                'file_path': file_path
            }
        
        return {
            'error_message': 'No error generated',
            'stack_trace': '',
            'error_type': 'Unknown',
            'file_path': file_path
        }
    
    def test_syntax_colon_fix(self):
        """Test fixing missing colon syntax error"""
        print("\n🧪 Testing Syntax Colon Fix...")
        
        file_path = self.test_files['syntax_colon.py']
        error = self.generate_error_from_file(file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        # Verify fix by checking if file compiles
        try:
            with open(file_path, 'r') as f:
                fixed_code = f.read()
            compile(fixed_code, file_path, 'exec')
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
        
        passed = (result.get('status') == 'success') and syntax_valid
        
        self.test_results.append({
            'test': 'syntax_colon_fix',
            'passed': passed,
            'details': f"Status: {result.get('status')}, Syntax valid: {syntax_valid}, Confidence: {result.get('confidence', 0):.2f}"
        })
        
        return passed
    
    def test_syntax_indent_fix(self):
        """Test fixing indentation syntax error"""
        print("\n🧪 Testing Syntax Indentation Fix...")
        
        file_path = self.test_files['syntax_indent.py']
        error = self.generate_error_from_file(file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        # Verify fix by checking if file compiles
        try:
            with open(file_path, 'r') as f:
                fixed_code = f.read()
            compile(fixed_code, file_path, 'exec')
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
        
        passed = (result.get('status') == 'success') and syntax_valid
        
        self.test_results.append({
            'test': 'syntax_indent_fix',
            'passed': passed,
            'details': f"Status: {result.get('status')}, Syntax valid: {syntax_valid}, Confidence: {result.get('confidence', 0):.2f}"
        })
        
        return passed
    
    def test_runtime_division_fix(self):
        """Test fixing division by zero runtime error"""
        print("\n🧪 Testing Runtime Division Fix...")
        
        file_path = self.test_files['runtime_division.py']
        error = self.generate_error_from_file(file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        # Verify fix by checking if code executes without error
        try:
            with open(file_path, 'r') as f:
                fixed_code = f.read()
            
            # Create a safe execution environment
            safe_globals = {}
            exec(fixed_code, safe_globals)
            runtime_valid = True
        except Exception:
            runtime_valid = False
        
        passed = (result.get('status') == 'success') and runtime_valid
        
        self.test_results.append({
            'test': 'runtime_division_fix',
            'passed': passed,
            'details': f"Status: {result.get('status')}, Runtime valid: {runtime_valid}, Confidence: {result.get('confidence', 0):.2f}"
        })
        
        return passed
    
    def test_runtime_index_fix(self):
        """Test fixing index error runtime error"""
        print("\n🧪 Testing Runtime Index Fix...")
        
        file_path = self.test_files['runtime_index.py']
        error = self.generate_error_from_file(file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        # Verify fix by checking if code executes without error
        try:
            with open(file_path, 'r') as f:
                fixed_code = f.read()
            
            safe_globals = {}
            exec(fixed_code, safe_globals)
            runtime_valid = True
        except Exception:
            runtime_valid = False
        
        passed = (result.get('status') == 'success') and runtime_valid
        
        self.test_results.append({
            'test': 'runtime_index_fix',
            'passed': passed,
            'details': f"Status: {result.get('status')}, Runtime valid: {runtime_valid}, Confidence: {result.get('confidence', 0):.2f}"
        })
        
        return passed
    
    def test_runtime_key_fix(self):
        """Test fixing key error runtime error"""
        print("\n🧪 Testing Runtime Key Fix...")
        
        file_path = self.test_files['runtime_key.py']
        error = self.generate_error_from_file(file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        # Verify fix by checking if code executes without error
        try:
            with open(file_path, 'r') as f:
                fixed_code = f.read()
            
            safe_globals = {}
            exec(fixed_code, safe_globals)
            runtime_valid = True
        except Exception:
            runtime_valid = False
        
        passed = (result.get('status') == 'success') and runtime_valid
        
        self.test_results.append({
            'test': 'runtime_key_fix',
            'passed': passed,
            'details': f"Status: {result.get('status')}, Runtime valid: {runtime_valid}, Confidence: {result.get('confidence', 0):.2f}"
        })
        
        return passed
    
    def test_runtime_none_fix(self):
        """Test fixing None value runtime error"""
        print("\n🧪 Testing Runtime None Fix...")
        
        file_path = self.test_files['runtime_none.py']
        error = self.generate_error_from_file(file_path)
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        result = enhanced_agent.handle_failure(
            error['error_message'], 
            error['stack_trace'], 
            file_path
        )
        
        # Verify fix by checking if code executes without error
        try:
            with open(file_path, 'r') as f:
                fixed_code = f.read()
            
            safe_globals = {}
            exec(fixed_code, safe_globals)
            runtime_valid = True
        except Exception:
            runtime_valid = False
        
        passed = (result.get('status') == 'success') and runtime_valid
        
        self.test_results.append({
            'test': 'runtime_none_fix',
            'passed': passed,
            'details': f"Status: {result.get('status')}, Runtime valid: {runtime_valid}, Confidence: {result.get('confidence', 0):.2f}"
        })
        
        return passed
    
    def test_error_detection_accuracy(self):
        """Test error type detection accuracy"""
        print("\n🧪 Testing Error Detection Accuracy...")
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        test_cases = [
            ('SyntaxError: missing colon at line 3', 'syntax_errors', 'missing_colon'),
            ('IndentationError: expected an indented block', 'syntax_errors', 'indentation_error'),
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
    
    def test_fix_strategy_generation(self):
        """Test fix strategy generation"""
        print("\n🧪 Testing Fix Strategy Generation...")
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        # Test different error types
        test_errors = [
            {'type': 'missing_colon', 'category': 'syntax_errors', 'auto_fixable': True},
            {'type': 'division_by_zero', 'category': 'runtime_errors', 'auto_fixable': True},
            {'type': 'index_out_of_range', 'category': 'runtime_errors', 'auto_fixable': True},
        ]
        
        strategies_generated = 0
        total_tests = len(test_errors)
        
        for error_info in test_errors:
            fixes = enhanced_agent.generate_fixes(error_info, '/tmp/test.py', 1)
            if len(fixes) > 0:
                strategies_generated += 1
        
        generation_rate = strategies_generated / total_tests
        
        passed = generation_rate >= 0.8  # 80% generation rate
        
        self.test_results.append({
            'test': 'fix_strategy_generation',
            'passed': passed,
            'details': f"Generation rate: {generation_rate:.1%} ({strategies_generated}/{total_tests})"
        })
        
        return passed
    
    def test_learning_system(self):
        """Test learning and improvement system"""
        print("\n🧪 Testing Learning System...")
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        # Simulate multiple successful fixes
        successful_fixes = 0
        for i in range(5):
            file_path = self.test_files['syntax_colon.py']
            error = self.generate_error_from_file(file_path)
            
            result = enhanced_agent.handle_failure(
                error['error_message'], 
                error['stack_trace'], 
                file_path
            )
            
            if result.get('status') == 'success':
                successful_fixes += 1
        
        # Check learning statistics
        stats = enhanced_agent.get_statistics()
        
        passed = (
            stats.get('successful_fixes', 0) > 0 and
            stats.get('success_rate', 0) > 0
        )
        
        self.test_results.append({
            'test': 'learning_system',
            'passed': passed,
            'details': f"Success rate: {stats.get('success_rate', 0):.1%}, Fixes: {stats.get('successful_fixes', 0)}"
        })
        
        return passed
    
    def test_comprehensive_scenarios(self):
        """Test comprehensive real-world scenarios"""
        print("\n🧪 Testing Comprehensive Scenarios...")
        
        system = self._create_mock_system()
        enhanced_agent = EnhancedMetaAgent(system)
        
        scenarios = [
            {
                'name': 'complex_syntax',
                'code': '''
def complex_function(param1, param2)
    if param1 > 0
        result = param1 / param2
    else
        result = param1 * 2
    return result
''',
                'expected_fixable': True
            },
            {
                'name': 'nested_runtime',
                'code': '''
def nested_function(data):
    results = []
    for i in range(len(data)):
        item = data[i]
        if item > 0:
            results.append(item / 0)  # Division by zero
    return results[10]  # Index error
''',
                'expected_fixable': True
            },
            {
                'name': 'mixed_errors',
                'code': '''
def mixed_errors()
    value = None
    result = value.upper()  # None error
    data = [1, 2, 3]
    return data[10]  # Index error
''',
                'expected_fixable': True
            }
        ]
        
        passed_scenarios = 0
        total_scenarios = len(scenarios)
        
        for scenario in scenarios:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(scenario['code'])
                temp_file = f.name
            
            try:
                # Generate error
                error = self.generate_error_from_file(temp_file)
                
                # Try to fix
                result = enhanced_agent.handle_failure(
                    error['error_message'], 
                    error['stack_trace'], 
                    temp_file
                )
                
                if result.get('status') == 'success':
                    passed_scenarios += 1
                    
            except Exception as e:
                print(f"   Scenario {scenario['name']} error: {e}")
            finally:
                # Cleanup
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        success_rate = passed_scenarios / total_scenarios
        
        passed = success_rate >= 0.6  # 60% success rate for complex scenarios
        
        self.test_results.append({
            'test': 'comprehensive_scenarios',
            'passed': passed,
            'details': f"Success rate: {success_rate:.1%} ({passed_scenarios}/{total_scenarios})"
        })
        
        return passed
    
    def _create_mock_system(self):
        """Create a mock system for testing"""
        class MockSystem:
            def __init__(self):
                self.project_root = Path(self.temp_dir if 'self.temp_dir' in globals() else '/tmp')
        
        return MockSystem()
    
    def run_standalone_tests(self):
        """Run all standalone tests"""
        print("🚀 Starting Standalone Enhanced MetaAgent Test Suite")
        print("=" * 60)
        
        self.setup_test_environment()
        
        try:
            # Core functionality tests
            self.test_syntax_colon_fix()
            self.test_syntax_indent_fix()
            self.test_runtime_division_fix()
            self.test_runtime_index_fix()
            self.test_runtime_key_fix()
            self.test_runtime_none_fix()
            
            # Advanced capability tests
            self.test_error_detection_accuracy()
            self.test_fix_strategy_generation()
            self.test_learning_system()
            self.test_comprehensive_scenarios()
            
        finally:
            self.cleanup_test_environment()
        
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 STANDALONE ENHANCED METAAGENT TEST SUITE SUMMARY")
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
        
        success_rate = (passed_tests/total_tests)*100
        if success_rate >= 80:
            print("🎉 EXCELLENT! Enhanced MetaAgent is working exceptionally well!")
            print("   ✅ Advanced error detection working perfectly")
            print("   ✅ Sophisticated fix strategies highly effective")
            print("   ✅ Learning system operational and improving")
            print("   ✅ Ready for production deployment")
        elif success_rate >= 60:
            print("🟢 GOOD! Enhanced MetaAgent is working well!")
            print("   ✅ Significant improvements over original")
            print("   ✅ Most capabilities functional")
            print("   ✅ Ready for deployment with monitoring")
        elif success_rate >= 40:
            print("🟡 MODERATE! Enhanced MetaAgent shows promise!")
            print("   ✅ Some capabilities working")
            print("   🔧 Further refinements needed")
            print("   📈 Good foundation for improvement")
        else:
            print("🔴 NEEDS WORK! Enhanced MetaAgent requires major improvements!")
            print("   ❌ Many capabilities not working")
            print("   🔧 Major improvements required")
            print("   🔄 Consider redesigning fix strategies")
        
        print("\n🔧 Recommendations:")
        print("-" * 40)
        
        if success_rate < 80:
            print("   - Review failed test patterns")
            print("   - Enhance error detection accuracy")
            print("   - Improve fix strategy implementation")
            print("   - Add more comprehensive test cases")
        
        if success_rate >= 60:
            print("   - Consider integrating with SAM system")
            print("   - Add real-world scenario testing")
            print("   - Implement continuous improvement")
        
        print("=" * 60)

def main():
    """Main test runner"""
    print("🤖 Standalone Enhanced MetaAgent Testing Suite")
    print("Testing improved self-healing capabilities without dependencies")
    print()
    
    # Run standalone test suite
    test_suite = StandaloneMetaAgentTest()
    test_suite.run_standalone_tests()
    
    # Return exit code based on results
    failed_tests = sum(1 for result in test_suite.test_results if not result['passed'])
    return 0 if failed_tests == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
