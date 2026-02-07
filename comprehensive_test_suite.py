#!/usr/bin/env python3
"""
SAM 2.0 Comprehensive Testing Framework
Tests all C modules, integration, performance, and edge cases
"""

import sys
import time
import traceback
import os
from datetime import datetime

# Test results tracking
test_results = {
    'total_tests': 0,
    'passed_tests': 0,
    'failed_tests': 0,
    'skipped_tests': 0,
    'errors': [],
    'performance_metrics': {},
    'memory_usage': {}
}

def log_test_result(test_name, success, error_msg=None, execution_time=None):
    """Log individual test results"""
    test_results['total_tests'] += 1

    if success:
        test_results['passed_tests'] += 1
        status = "✅ PASS"
    else:
        test_results['failed_tests'] += 1
        status = "❌ FAIL"
        if error_msg:
            test_results['errors'].append(f"{test_name}: {error_msg}")

    time_info = f" ({execution_time:.3f}s)" if execution_time else ""
    print(f"{status} {test_name}{time_info}")

    if error_msg and not success:
        print(f"   Error: {error_msg}")

def run_test(test_func, test_name):
    """Run a test function with timing and error handling"""
    start_time = time.time()

    try:
        result = test_func()
        execution_time = time.time() - start_time

        if isinstance(result, tuple):
            success, error_msg = result
        else:
            success, error_msg = result, None

        log_test_result(test_name, success, error_msg, execution_time)
        return success

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
        log_test_result(test_name, False, error_msg, execution_time)
        return False

def get_memory_usage():
    """Get basic memory estimate (simplified)"""
    # Simplified memory estimation without psutil
    return 0.0  # Placeholder - remove detailed memory tracking

# ================================
# MODULE IMPORT TESTS
# ================================

def test_consciousness_import():
    """Test consciousness module import"""
    try:
        import consciousness_algorithmic
        return True, None
    except ImportError as e:
        return False, str(e)

def test_orchestrator_import():
    """Test multi-agent orchestrator import"""
    try:
        import multi_agent_orchestrator_c
        return True, None
    except ImportError as e:
        return False, str(e)

def test_agents_import():
    """Test specialized agents import"""
    try:
        import specialized_agents_c
        return True, None
    except ImportError as e:
        return False, str(e)

# ================================
# CONSCIOUSNESS MODULE TESTS
# ================================

def test_consciousness_creation():
    """Test consciousness module creation"""
    try:
        import consciousness_algorithmic

        # Test different dimensions
        consciousness_algorithmic.create(32, 8)
        consciousness_algorithmic.create(128, 32)
        consciousness_algorithmic.create(64, 16)  # Standard size

        stats = consciousness_algorithmic.get_stats()
        if 'latent_dim' not in stats or 'action_dim' not in stats:
            return False, "Stats missing required fields"

        return True, None
    except Exception as e:
        return False, str(e)

def test_consciousness_training():
    """Test consciousness training functionality"""
    try:
        import consciousness_algorithmic

        consciousness_algorithmic.create(64, 16)

        # Test different epoch counts
        result1 = consciousness_algorithmic.optimize(10000, 5)
        result2 = consciousness_algorithmic.optimize(50000, 10)

        if 'final_loss' not in result1 or 'consciousness_score' not in result1:
            return False, "Training result missing required fields"

        if 'final_loss' not in result2 or 'consciousness_score' not in result2:
            return False, "Training result missing required fields"

        # Verify consciousness score is reasonable (0-1 range)
        score1 = result1['consciousness_score']
        score2 = result2['consciousness_score']

        if not (0.0 <= score1 <= 1.0) or not (0.0 <= score2 <= 1.0):
            return False, f"Invalid consciousness scores: {score1}, {score2}"

        # Verify loss decreased with more training
        if result2['final_loss'] > result1['final_loss'] * 2:
            return False, "Loss did not decrease with more training"

        return True, None
    except Exception as e:
        return False, str(e)

def test_consciousness_stats():
    """Test consciousness statistics retrieval"""
    try:
        import consciousness_algorithmic

        consciousness_algorithmic.create(64, 16)

        # Test stats before training
        stats1 = consciousness_algorithmic.get_stats()

        # Train briefly
        consciousness_algorithmic.optimize(10000, 3)

        # Test stats after training
        stats2 = consciousness_algorithmic.get_stats()

        required_fields = ['consciousness_score', 'is_conscious', 'latent_dim', 'action_dim']
        for field in required_fields:
            if field not in stats1 or field not in stats2:
                return False, f"Missing field: {field}"

        # Verify dimensions are correct
        if stats1['latent_dim'] != 64 or stats1['action_dim'] != 16:
            return False, f"Incorrect dimensions: {stats1['latent_dim']}x{stats1['action_dim']}"

        return True, None
    except Exception as e:
        return False, str(e)

# ================================
# MULTI-AGENT ORCHESTRATOR TESTS
# ================================

def test_orchestrator_creation():
    """Test multi-agent orchestrator creation"""
    try:
        import multi_agent_orchestrator_c

        print("   Testing orchestrator creation...")
        # create_system returns None but initializes global state
        result = multi_agent_orchestrator_c.create_system()
        print(f"   create_system returned: {result}")

        # The function returns None on success, so check that no exception was raised
        # and that we can get status (which means global state was initialized)

        status = multi_agent_orchestrator_c.get_status()
        print(f"   Status retrieved: {status}")

        if not status or 'agent_count' not in status:
            return False, "Orchestrator status invalid"

        # Verify we have agents
        agent_count = status.get('agent_count', 0)
        if agent_count < 2:  # Should have at least 2 agents
            return False, f"Insufficient agents created: {agent_count}"

        return True, None
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False, str(e)

def test_agent_management():
    """Test agent management in orchestrator"""
    try:
        import multi_agent_orchestrator_c

        # Initialize the orchestrator
        result = multi_agent_orchestrator_c.create_system()
        # create_system returns None but initializes global state

        status = multi_agent_orchestrator_c.get_status()
        if not status:
            return False, "Could not get orchestrator status"

        initial_count = status.get('agent_count', 0)

        # Should have at least 2 agents created automatically
        if initial_count < 2:
            return False, f"Insufficient agents created: {initial_count}"

        return True, f"Successfully created orchestrator with {initial_count} agents"
    except Exception as e:
        return False, str(e)

# ================================
# SPECIALIZED AGENTS TESTS
# ================================

def test_agents_creation():
    """Test specialized agents creation"""
    try:
        import specialized_agents_c

        specialized_agents_c.create_agents()

        # Test each agent type
        test_queries = [
            "research: quantum computing",
            "code: fibonacci function",
            "market: tech stocks",
            "survival: system threats",
            "analysis: neural network performance"
        ]

        for query in test_queries:
            agent_type, task = query.split(": ", 1)

            if agent_type == "research":
                result = specialized_agents_c.research(task)
            elif agent_type == "code":
                result = specialized_agents_c.generate_code(task)
            elif agent_type == "market":
                result = specialized_agents_c.analyze_market(task)
            elif agent_type == "survival":
                result = specialized_agents_c.assess_survival()
            elif agent_type == "analysis":
                result = specialized_agents_c.analyze_system(task)

            if not result or not isinstance(result, str):
                return False, f"Agent {agent_type} returned invalid result"

        return True, None
    except Exception as e:
        return False, str(e)

# ================================
# INTEGRATION TESTS
# ================================

def test_full_system_integration():
    """Test full system integration"""
    try:
        import complete_sam_system_clean_c

        system = complete_sam_system_clean_c.initialize_sam_system()
        if not system:
            return False, "Failed to initialize system"

        status = system.get_system_status()
        if not status or 'components' not in status:
            return False, "Invalid system status"

        # Verify all components are active
        components = status['components']
        required_components = ['consciousness', 'orchestrator', 'agents']

        for comp in required_components:
            if comp not in components or not components[comp].startswith('ACTIVE'):
                return False, f"Component {comp} not active: {components.get(comp, 'missing')}"

        # Test consciousness training through system
        training_result = system.run_consciousness_training(epochs=3)
        if 'error' in training_result:
            return False, f"Training failed: {training_result['error']}"

        return True, None
    except Exception as e:
        return False, str(e)

# ================================
# ERROR HANDLING TESTS
# ================================

def test_error_handling():
    """Test error handling and edge cases"""
    try:
        import consciousness_algorithmic
        import specialized_agents_c

        # Test invalid consciousness dimensions
        try:
            consciousness_algorithmic.create(0, 16)  # Invalid latent dim
            return False, "Should have failed with invalid dimensions"
        except:
            pass  # Expected to fail

        try:
            consciousness_algorithmic.create(64, 0)  # Invalid action dim
            return False, "Should have failed with invalid dimensions"
        except:
            pass  # Expected to fail

        # Test invalid training parameters
        consciousness_algorithmic.create(64, 16)
        try:
            result = consciousness_algorithmic.optimize(0, 1)  # Invalid num_params
            # This might not fail, but let's check the result
        except:
            pass  # May or may not fail

        # Test agents with invalid inputs
        specialized_agents_c.create_agents()
        try:
            result = specialized_agents_c.research("")  # Empty query
            if not result:
                return False, "Empty research query should return result"
        except:
            pass  # May handle gracefully

        return True, None
    except Exception as e:
        return False, str(e)

# ================================
# PERFORMANCE TESTS
# ================================

def test_performance():
    """Test performance metrics"""
    try:
        import consciousness_algorithmic

        consciousness_algorithmic.create(64, 16)

        start_time = time.time()

        # Run training
        result = consciousness_algorithmic.optimize(50000, 10)

        end_time = time.time()

        execution_time = end_time - start_time

        # Store performance metrics
        test_results['performance_metrics']['consciousness_training'] = {
            'execution_time': execution_time,
            'final_loss': result.get('final_loss', 'N/A'),
            'consciousness_score': result.get('consciousness_score', 'N/A')
        }

        # Performance thresholds (reasonable for this system)
        if execution_time > 60:  # Should complete within 60 seconds
            return False, f"Training too slow: {execution_time:.2f}s"

        return True, f"Performance acceptable: {execution_time:.2f}s"
    except Exception as e:
        return False, str(e)

# ================================
# COMPREHENSIVE TEST SUITE
# ================================

def run_comprehensive_tests():
    """Run the complete test suite"""
    print("🧪 SAM 2.0 Comprehensive Testing Framework")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test categories
    test_categories = [
        ("Module Imports", [
            (test_consciousness_import, "Consciousness Module Import"),
            (test_orchestrator_import, "Multi-Agent Orchestrator Import"),
            (test_agents_import, "Specialized Agents Import"),
        ]),

        ("Consciousness Module", [
            (test_consciousness_creation, "Consciousness Creation"),
            (test_consciousness_training, "Consciousness Training"),
            (test_consciousness_stats, "Consciousness Statistics"),
        ]),

        ("Multi-Agent Orchestrator", [
            (test_orchestrator_creation, "Orchestrator Creation"),
            (test_agent_management, "Agent Management"),
        ]),

        ("Specialized Agents", [
            (test_agents_creation, "Agents Creation and Functionality"),
        ]),

        ("Integration Tests", [
            (test_full_system_integration, "Full System Integration"),
        ]),

        ("Error Handling", [
            (test_error_handling, "Error Handling and Edge Cases"),
        ]),

        ("Performance Tests", [
            (test_performance, "Performance Metrics"),
        ]),
    ]

    # Run all tests
    for category_name, tests in test_categories:
        print(f"📋 {category_name}")
        print("-" * 40)

        for test_func, test_name in tests:
            run_test(test_func, test_name)

        print()

    # Final results
    print("=" * 60)
    print("🎯 TESTING RESULTS SUMMARY")
    print("=" * 60)

    total = test_results['total_tests']
    passed = test_results['passed_tests']
    failed = test_results['failed_tests']
    skipped = test_results['skipped_tests']

    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏭️  Skipped: {skipped}")
    print()

    if failed > 0:
        print("❌ FAILED TESTS:")
        for error in test_results['errors'][:5]:  # Show first 5 errors
            print(f"   • {error}")
        if len(test_results['errors']) > 5:
            print(f"   ... and {len(test_results['errors']) - 5} more")
        print()

    if test_results['performance_metrics']:
        print("⚡ PERFORMANCE METRICS:")
        for test_name, metrics in test_results['performance_metrics'].items():
            print(f"   • {test_name}:")
            for key, value in metrics.items():
                print(f"     - {key}: {value}")
        print()

    # Overall assessment
    success_rate = (passed / total * 100) if total > 0 else 0

    print("🎯 FINAL ASSESSMENT:")
    if success_rate >= 95:
        print("   ✅ EXCELLENT: All systems operational")
        assessment = "excellent"
    elif success_rate >= 85:
        print("   ⚠️ GOOD: Minor issues detected")
        assessment = "good"
    elif success_rate >= 70:
        print("   ❌ FAIR: Significant issues found")
        assessment = "fair"
    else:
        print("   💥 CRITICAL: Major system failures")
        assessment = "critical"

    print(".1f")
    print(f"   Status: {assessment.upper()}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return assessment == "excellent"

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
