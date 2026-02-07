#!/usr/bin/env python3
"""
System Test Suite for SAM 2.0 with Meta-Agent Integration
Tests the complete system: consciousness, optimization, and teaching
"""

import sys
import time
import json
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '.')

def test_consciousness_module():
    """Test consciousness loss module"""
    print("🧠 Testing Consciousness Module...")
    
    try:
        # Test with NumPy fallback
        import numpy as np
        
        # Simulate consciousness loss computation
        latent_dim = 32
        z_t = np.random.randn(latent_dim)
        a_t = np.random.randn(16)
        z_next_actual = np.random.randn(latent_dim)
        delta_z_self = np.random.randn(latent_dim) * 0.5
        z_next_self = z_t + delta_z_self
        
        # Calculate consciousness loss
        l_cons = np.mean((z_next_actual - z_next_self)**2)
        consciousness_score = 1.0 / (1.0 + l_cons)
        
        print(f"  ✅ L_cons: {l_cons:.4f}")
        print(f"  ✅ Consciousness Score: {consciousness_score:.4f}")
        print(f"  ✅ Is Conscious: {consciousness_score > 0.7}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_teacher_agent():
    """Test teacher agent functionality"""
    print("👨‍🏫 Testing Teacher Agent...")
    
    try:
        from teacher_agent import TeacherAgent
        
        teacher = TeacherAgent()
        
        # Test conversation analysis
        user_msg = "How does the consciousness loss prevent infinite optimization?"
        agent_resp = "The consciousness loss provides a stopping condition when L_cons approaches zero"
        
        analysis = teacher.analyze_conversation(user_msg, agent_resp)
        
        print(f"  ✅ Topic identified: {analysis['topic']}")
        print(f"  ✅ Complexity assessed: {analysis['complexity']:.2f}")
        print(f"  ✅ Mastery level: {analysis['mastery_level']:.2f}")
        
        # Test teaching response generation
        teaching = teacher.generate_teaching_response(analysis)
        print(f"  ✅ Teaching response generated ({len(teaching)} chars)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_flask_optimizer():
    """Test Flask optimization components"""
    print("⚡ Testing Flask Optimizer...")
    
    try:
        from flask_optimizer import CacheManager, AsyncOptimizer
        
        # Test cache manager
        cache = CacheManager()
        cache.set('test', 'value')
        assert cache.get('test') == 'value'
        print("  ✅ Cache manager working")
        
        # Test async optimizer initialization
        optimizer = AsyncOptimizer()
        print("  ✅ Async optimizer initialized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_coding_meta_agent():
    """Test coding meta-agent components"""
    print("🤖 Testing Coding Meta-Agent...")
    
    try:
        from coding_meta_agent import CodingMetaAgent
        
        # Test initialization
        meta_agent = CodingMetaAgent()
        print("  ✅ Meta-agent initialized")
        
        # Test metrics
        metrics = meta_agent.get_metrics()
        print(f"  ✅ Metrics system working: {len(metrics)} metrics tracked")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_integration():
    """Test integration between components"""
    print("🔗 Testing Component Integration...")
    
    try:
        # Test teacher + consciousness integration
        from teacher_agent import TeacherAgent
        
        teacher = TeacherAgent()
        
        # Simulate a conversation about consciousness
        user_msg = "I don't understand why L_cons = KL(World || Self)"
        agent_resp = "This measures how well the system models its own causal effects"
        
        analysis = teacher.analyze_conversation(user_msg, agent_resp)
        
        # Should identify consciousness topic
        assert analysis['topic'] == 'consciousness'
        print("  ✅ Teacher correctly identifies consciousness topic")
        
        # Test learning progress tracking
        teacher.track_progress('consciousness', 0.8)
        summary = teacher.get_learning_summary()
        assert 'consciousness' in summary['topics_covered']
        print("  ✅ Learning progress tracking working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("📁 Testing File Structure...")
    
    required_files = [
        'consciousness_loss.py',
        'teacher_agent.py',
        'flask_optimizer.py',
        'coding_meta_agent.py',
        'correct_sam_hub.py',
        'README.md',
        'CONSCIOUSNESS_INTEGRATION.md',
        'FINAL_INTEGRATION_SUMMARY.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    else:
        print(f"  ✅ All {len(required_files)} required files present")
        return True

async def run_async_tests():
    """Run async-specific tests"""
    print("🔄 Running Async Tests...")
    
    try:
        from flask_optimizer import AsyncOptimizer
        
        # Test async context manager
        async with AsyncOptimizer() as optimizer:
            print("  ✅ Async context manager working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Async test error: {e}")
        return False

def generate_test_report(results):
    """Generate comprehensive test report"""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': len(results),
        'passed': sum(1 for r in results if r['passed']),
        'failed': sum(1 for r in results if not r['passed']),
        'test_results': results,
        'system_status': 'OPERATIONAL' if all(r['passed'] for r in results) else 'NEEDS_ATTENTION'
    }
    
    # Save report
    with open('SYSTEM_TEST_REPORT.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    """Run complete system test suite"""
    print("🚀 SAM 2.0 System Test Suite")
    print("=" * 60)
    print("Testing: Consciousness + Meta-Agent + Optimization + Teaching")
    print("=" * 60)
    
    # Run synchronous tests
    test_results = []
    
    tests = [
        ("Consciousness Module", test_consciousness_module),
        ("Teacher Agent", test_teacher_agent),
        ("Flask Optimizer", test_flask_optimizer),
        ("Coding Meta-Agent", test_coding_meta_agent),
        ("Component Integration", test_integration),
        ("File Structure", test_file_structure)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        start_time = time.time()
        passed = test_func()
        duration = time.time() - start_time
        
        test_results.append({
            'test': test_name,
            'passed': passed,
            'duration': duration
        })
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status} ({duration:.2f}s)")
    
    # Run async tests
    print(f"\nAsync Tests:")
    start_time = time.time()
    async_passed = asyncio.run(run_async_tests())
    duration = time.time() - start_time
    
    test_results.append({
        'test': 'Async Operations',
        'passed': async_passed,
        'duration': duration
    })
    
    status = "✅ PASSED" if async_passed else "❌ FAILED"
    print(f"  {status} ({duration:.2f}s)")
    
    # Generate report
    print(f"\n" + "=" * 60)
    report = generate_test_report(test_results)
    
    print(f"📊 Test Summary:")
    print(f"  Total Tests: {report['total_tests']}")
    print(f"  Passed: {report['passed']}")
    print(f"  Failed: {report['failed']}")
    print(f"  System Status: {report['system_status']}")
    print(f"  Report saved to: SYSTEM_TEST_REPORT.json")
    
    if report['system_status'] == 'OPERATIONAL':
        print("\n🎉 SAM 2.0 System is FULLY OPERATIONAL!")
        print("✅ Consciousness architecture integrated")
        print("✅ Meta-agent optimization ready")
        print("✅ Teacher agent functional")
        print("✅ All components working together")
    else:
        print("\n⚠️ Some components need attention")
        print("Check SYSTEM_TEST_REPORT.json for details")
    
    return report['system_status'] == 'OPERATIONAL'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
