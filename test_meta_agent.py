#!/usr/bin/env python3
"""
Comprehensive MetaAgent Testing Suite
Tests the self-healing capabilities of the SAM MetaAgent system
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

from complete_sam_unified import (
    MetaAgent, ObserverAgent, FaultLocalizerAgent, PatchGeneratorAgent, VerifierJudgeAgent,
    FailureEvent, UnifiedSAMSystem
)

class MetaAgentTestSuite:
    """Comprehensive test suite for MetaAgent self-healing capabilities"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        self.test_files = {}
        
    def setup_test_environment(self):
        """Setup isolated test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="meta_test_")
        print(f"🧪 Test environment: {self.temp_dir}")
        
        # Create test files with various issues
        self.test_files = {
            'syntax_error.py': self._create_syntax_error_file(),
            'import_error.py': self._create_import_error_file(),
            'logic_error.py': self._create_logic_error_file(),
            'performance_issue.py': self._create_performance_issue_file(),
            'missing_dependency.py': self._create_missing_dependency_file(),
            'configuration_error.py': self._create_configuration_error_file(),
        }
        
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up test environment")
    
    def _create_syntax_error_file(self):
        """Create a file with syntax error"""
        content = '''
def broken_function():
    # Missing colon - syntax error
    print("This will break")
    return "broken"

# Another syntax error
if True
    print("No colon")
'''
        path = os.path.join(self.temp_dir, 'syntax_error.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_import_error_file(self):
        """Create a file with import error"""
        content = '''
import nonexistent_module_12345  # This will fail

def function_with_bad_import():
    try:
        nonexistent_module_12345.do_something()
    except NameError:
        return "Import failed"
    return "OK"
'''
        path = os.path.join(self.temp_dir, 'import_error.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_logic_error_file(self):
        """Create a file with logic error"""
        content = '''
def divide_by_zero_function(x):
    # Logic error - division by zero
    result = x / 0  # This will cause ZeroDivisionError
    return result

def index_error_function():
    data = [1, 2, 3]
    # Index out of bounds
    return data[10]  # This will cause IndexError
'''
        path = os.path.join(self.temp_dir, 'logic_error.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_performance_issue_file(self):
        """Create a file with performance issues"""
        content = '''
import time

def inefficient_loop(n):
    # O(n²) nested loop - performance issue
    result = []
    for i in range(n):
        for j in range(n):  # Inefficient nesting
            if i == j:
                result.append(i)
    return result

def memory_leak_function():
    # Potential memory leak
    data = []
    while True:
        data.append([0] * 1000)  # Keep growing
        if len(data) > 1000:
            break  # Prevent infinite loop in test
    return data
'''
        path = os.path.join(self.temp_dir, 'performance_issue.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_missing_dependency_file(self):
        """Create a file that depends on missing modules"""
        content = '''
try:
    import requests  # Common but might not be installed
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import numpy as np  # Heavy dependency
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def check_dependencies():
    deps = []
    if not HAS_REQUESTS:
        deps.append("requests")
    if not HAS_NUMPY:
        deps.append("numpy")
    return deps
'''
        path = os.path.join(self.temp_dir, 'missing_dependency.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def _create_configuration_error_file(self):
        """Create a file with configuration issues"""
        content = '''
import os

# Configuration errors
API_KEY = None  # Missing API key
DATABASE_URL = "invalid://url"  # Invalid URL
TIMEOUT = -1  # Invalid timeout

def load_config():
    config = {
        'api_key': API_KEY or os.getenv('MISSING_API_KEY'),
        'database_url': DATABASE_URL,
        'timeout': TIMEOUT
    }
    
    # Validation errors
    if not config['api_key']:
        raise ValueError("API key is required")
    if not config['database_url'].startswith(('http://', 'https://')):
        raise ValueError("Invalid database URL")
    if config['timeout'] < 0:
        raise ValueError("Timeout must be positive")
    
    return config
'''
        path = os.path.join(self.temp_dir, 'configuration_error.py')
        with open(path, 'w') as f:
            f.write(content)
        return path
    
    def create_test_failure_event(self, error_type, file_path, error_message):
        """Create a test failure event"""
        try:
            # Try to import the problematic file to generate real stack trace
            spec = {}
            with open(file_path, 'r') as f:
                code = f.read()
            
            # Compile to get syntax errors
            compile(code, file_path, 'exec')
            
        except SyntaxError as e:
            stack_trace = traceback.format_exc()
        except Exception as e:
            stack_trace = traceback.format_exc()
        else:
            # If no compile error, try to import and run
            try:
                exec(compile(code, file_path, 'exec'))
            except Exception as e:
                stack_trace = traceback.format_exc()
            else:
                stack_trace = f"No error detected in {file_path}"
        
        return FailureEvent(
            error_type=error_type,
            stack_trace=stack_trace,
            timestamp=datetime.now().isoformat(),
            severity="medium",
            context="test_scenario",
            research_notes=f"Test case: {error_type} in {os.path.basename(file_path)}"
        )
    
    def test_syntax_error_detection(self):
        """Test MetaAgent can detect and fix syntax errors"""
        print("\n🧪 Testing Syntax Error Detection & Fixing...")
        
        file_path = self.test_files['syntax_error.py']
        failure = self.create_test_failure_event("SyntaxError", file_path, "Missing colon in function definition")
        
        # Create mock system and MetaAgent
        system = self._create_mock_system()
        meta_agent = self._create_meta_agent(system)
        
        # Test failure handling
        result = meta_agent.handle_failure(failure)
        
        self.test_results.append({
            'test': 'syntax_error_detection',
            'passed': result,
            'details': f"MetaAgent {'handled' if result else 'failed to handle'} syntax error"
        })
        
        return result
    
    def test_import_error_detection(self):
        """Test MetaAgent can detect and fix import errors"""
        print("\n🧪 Testing Import Error Detection & Fixing...")
        
        file_path = self.test_files['import_error.py']
        failure = self.create_test_failure_event("ImportError", file_path, "Missing module import")
        
        system = self._create_mock_system()
        meta_agent = self._create_meta_agent(system)
        
        result = meta_agent.handle_failure(failure)
        
        self.test_results.append({
            'test': 'import_error_detection',
            'passed': result,
            'details': f"MetaAgent {'handled' if result else 'failed to handle'} import error"
        })
        
        return result
    
    def test_logic_error_detection(self):
        """Test MetaAgent can detect and fix logic errors"""
        print("\n🧪 Testing Logic Error Detection & Fixing...")
        
        file_path = self.test_files['logic_error.py']
        failure = self.create_test_failure_event("LogicError", file_path, "Division by zero and index error")
        
        system = self._create_mock_system()
        meta_agent = self._create_meta_agent(system)
        
        result = meta_agent.handle_failure(failure)
        
        self.test_results.append({
            'test': 'logic_error_detection',
            'passed': result,
            'details': f"MetaAgent {'handled' if result else 'failed to handle'} logic error"
        })
        
        return result
    
    def test_performance_issue_detection(self):
        """Test MetaAgent can detect and fix performance issues"""
        print("\n🧪 Testing Performance Issue Detection & Fixing...")
        
        file_path = self.test_files['performance_issue.py']
        failure = self.create_test_failure_event("PerformanceIssue", file_path, "Inefficient nested loops")
        
        system = self._create_mock_system()
        meta_agent = self._create_meta_agent(system)
        
        result = meta_agent.handle_failure(failure)
        
        self.test_results.append({
            'test': 'performance_issue_detection',
            'passed': result,
            'details': f"MetaAgent {'handled' if result else 'failed to handle'} performance issue"
        })
        
        return result
    
    def test_missing_dependency_detection(self):
        """Test MetaAgent can detect and fix missing dependencies"""
        print("\n🧪 Testing Missing Dependency Detection & Fixing...")
        
        file_path = self.test_files['missing_dependency.py']
        failure = self.create_test_failure_event("MissingDependency", file_path, "Missing required modules")
        
        system = self._create_mock_system()
        meta_agent = self._create_meta_agent(system)
        
        result = meta_agent.handle_failure(failure)
        
        self.test_results.append({
            'test': 'missing_dependency_detection',
            'passed': result,
            'details': f"MetaAgent {'handled' if result else 'failed to handle'} missing dependency"
        })
        
        return result
    
    def test_configuration_error_detection(self):
        """Test MetaAgent can detect and fix configuration errors"""
        print("\n🧪 Testing Configuration Error Detection & Fixing...")
        
        file_path = self.test_files['configuration_error.py']
        failure = self.create_test_failure_event("ConfigurationError", file_path, "Invalid configuration values")
        
        system = self._create_mock_system()
        meta_agent = self._create_meta_agent(system)
        
        result = meta_agent.handle_failure(failure)
        
        self.test_results.append({
            'test': 'configuration_error_detection',
            'passed': result,
            'details': f"MetaAgent {'handled' if result else 'failed to handle'} configuration error"
        })
        
        return result
    
    def test_failure_clustering(self):
        """Test MetaAgent failure clustering capabilities"""
        print("\n🧪 Testing Failure Clustering...")
        
        system = self._create_mock_system()
        meta_agent = self._create_meta_agent(system)
        
        # Create multiple similar failures
        failures = [
            self.create_test_failure_event("SyntaxError", self.test_files['syntax_error.py'], "Missing colon"),
            self.create_test_failure_event("SyntaxError", self.test_files['syntax_error.py'], "Missing colon again"),
            self.create_test_failure_event("ImportError", self.test_files['import_error.py'], "Missing module"),
        ]
        
        # Register failures
        for failure in failures:
            meta_agent.register_failure(failure)
        
        # Check clustering
        stats = meta_agent.get_cluster_statistics()
        
        # Should have at least 2 clusters (SyntaxError and ImportError)
        clusters_found = len(stats.get('clusters', {}))
        expected_clusters = 2
        
        passed = clusters_found >= expected_clusters
        
        self.test_results.append({
            'test': 'failure_clustering',
            'passed': passed,
            'details': f"Found {clusters_found} clusters, expected at least {expected_clusters}"
        })
        
        return passed
    
    def test_research_capabilities(self):
        """Test MetaAgent research capabilities"""
        print("\n🧪 Testing Research Capabilities...")
        
        system = self._create_mock_system()
        meta_agent = self._create_meta_agent(system)
        
        failure = self.create_test_failure_event("ComplexError", self.test_files['syntax_error.py'], "Complex multi-layer error")
        
        # Test local research
        localized_files = [{'file': self.test_files['syntax_error.py'], 'score': 0.8}]
        local_research = meta_agent._local_research(failure, localized_files)
        
        # Test web research (mock)
        web_research = meta_agent._web_research(failure)
        
        # Test combined research
        combined_research = meta_agent._gather_research(failure, localized_files)
        
        passed = (
            len(local_research) > 0 and
            len(combined_research) > 0
        )
        
        self.test_results.append({
            'test': 'research_capabilities',
            'passed': passed,
            'details': f"Local research: {len(local_research)} chars, Combined: {len(combined_research)} chars"
        })
        
        return passed
    
    def test_patch_generation(self):
        """Test MetaAgent patch generation capabilities"""
        print("\n🧪 Testing Patch Generation...")
        
        system = self._create_mock_system()
        meta_agent = self._create_meta_agent(system)
        
        failure = self.create_test_failure_event("SyntaxError", self.test_files['syntax_error.py'], "Missing colon")
        
        # Generate deterministic patches
        patches = meta_agent._deterministic_patches(failure)
        
        # Should generate at least one patch
        passed = len(patches) > 0
        
        self.test_results.append({
            'test': 'patch_generation',
            'passed': passed,
            'details': f"Generated {len(patches)} patches"
        })
        
        return passed
    
    def test_learning_system(self):
        """Test MetaAgent learning and improvement capabilities"""
        print("\n🧪 Testing Learning System...")
        
        system = self._create_mock_system()
        meta_agent = self._create_meta_agent(system)
        
        # Simulate learning cycles
        initial_learning_cycles = meta_agent.learning_cycles
        
        # Simulate a successful fix
        failure = self.create_test_failure_event("TestError", self.test_files['syntax_error.py'], "Test error")
        mock_patch = {
            'id': 'test_patch_1',
            'confidence': 0.9,
            'intent': 'test_fix',
            'target_file': self.test_files['syntax_error.py']
        }
        
        meta_agent._learn_from_success(mock_patch, failure)
        
        # Check if learning state updated
        learning_improved = (
            len(meta_agent.improvements_applied) > 0 and
            meta_agent.learning_cycles > initial_learning_cycles
        )
        
        self.test_results.append({
            'test': 'learning_system',
            'passed': learning_improved,
            'details': f"Learning cycles: {meta_agent.learning_cycles}, Improvements: {len(meta_agent.improvements_applied)}"
        })
        
        return learning_improved
    
    def _create_mock_system(self):
        """Create a mock system for testing"""
        class MockSystem:
            def __init__(self):
                self.project_root = Path(self.temp_dir if 'self.temp_dir' in globals() else '/tmp')
                self.web_search_enabled = False  # Disable for testing
        
        return MockSystem()
    
    def _create_meta_agent(self, system):
        """Create MetaAgent with mock sub-agents"""
        observer = ObserverAgent()
        localizer = FaultLocalizer()
        generator = PatchGenerator()
        verifier = VerifierJudge()
        
        return MetaAgent(observer, localizer, generator, verifier, system)
    
    def run_comprehensive_tests(self):
        """Run all MetaAgent tests"""
        print("🚀 Starting Comprehensive MetaAgent Test Suite")
        print("=" * 60)
        
        self.setup_test_environment()
        
        try:
            # Core functionality tests
            self.test_syntax_error_detection()
            self.test_import_error_detection()
            self.test_logic_error_detection()
            self.test_performance_issue_detection()
            self.test_missing_dependency_detection()
            self.test_configuration_error_detection()
            
            # Advanced capabilities tests
            self.test_failure_clustering()
            self.test_research_capabilities()
            self.test_patch_generation()
            self.test_learning_system()
            
        finally:
            self.cleanup_test_environment()
        
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 METAAGENT TEST SUITE SUMMARY")
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
        
        print("\n🎯 Recommendations:")
        print("-" * 40)
        
        if failed_tests == 0:
            print("🎉 All tests passed! MetaAgent is working excellently.")
        else:
            print("🔧 Some tests failed. Consider:")
            print("   - Enhancing error detection algorithms")
            print("   - Improving patch generation strategies")
            print("   - Expanding research capabilities")
            print("   - Refining learning mechanisms")
        
        print("=" * 60)

def main():
    """Main test runner"""
    print("🤖 MetaAgent Comprehensive Testing Suite")
    print("Testing self-healing capabilities of SAM MetaAgent system")
    print()
    
    # Check if running in test environment
    if os.getenv('SAM_META_TEST_MODE', '0') != '1':
        print("⚠️  Setting test environment variables...")
        os.environ['SAM_META_RESEARCH_ENABLED'] = '1'
        os.environ['SAM_META_RESEARCH_MODE'] = 'local'
        os.environ['SAM_META_RESEARCH_MAX_CHARS'] = '1000'
        os.environ['SAM_META_TEST_MODE'] = '1'
    
    # Run test suite
    test_suite = MetaAgentTestSuite()
    test_suite.run_comprehensive_tests()
    
    # Return exit code based on results
    failed_tests = sum(1 for result in test_suite.test_results if not result['passed'])
    return 0 if failed_tests == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
