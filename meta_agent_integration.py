#!/usr/bin/env python3
"""
MetaAgent Integration with SAM System
Integrates enhanced MetaAgent capabilities with the existing SAM system
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from complete_sam_unified import MetaAgent, ObserverAgent, FaultLocalizerAgent, PatchGeneratorAgent, VerifierJudgeAgent
from meta_agent_enhanced import EnhancedMetaAgent

class IntegratedMetaAgent:
    """Integrated MetaAgent combining original SAM MetaAgent with enhanced capabilities"""
    
    def __init__(self, system):
        self.system = system
        self.project_root = getattr(system, 'project_root', Path('.'))
        
        # Initialize both original and enhanced meta agents
        self.original_meta = None
        self.enhanced_meta = EnhancedMetaAgent(system)
        
        # Integration state
        self.integration_mode = os.getenv('SAM_META_INTEGRATION_MODE', 'hybrid').lower()
        self.fallback_enabled = os.getenv('SAM_META_FALLBACK_ENABLED', '1') == '1'
        self.enhanced_priority = os.getenv('SAM_META_ENHANCED_PRIORITY', '1') == '1'
        
        # Performance tracking
        self.fix_attempts = 0
        self.successful_fixes = 0
        self.original_fixes = 0
        self.enhanced_fixes = 0
        
        # Initialize original meta agent if needed
        if self.integration_mode in ['original', 'hybrid']:
            try:
                observer = ObserverAgent(system)
                localizer = FaultLocalizerAgent(system)
                generator = PatchGeneratorAgent(system)
                verifier = VerifierJudgeAgent(system)
                self.original_meta = MetaAgent(observer, localizer, generator, verifier, system)
            except Exception as e:
                print(f"⚠️ Failed to initialize original MetaAgent: {e}")
                self.original_meta = None
        
        print("🚀 Integrated MetaAgent initialized")
        print(f"   🔄 Integration mode: {self.integration_mode}")
        print(f"   ✅ Enhanced capabilities: {'enabled' if self.enhanced_priority else 'disabled'}")
        print(f"   🛡️ Fallback to original: {'enabled' if self.fallback_enabled else 'disabled'}")
    
    def handle_failure(self, error_message: str, stack_trace: str, file_path: str = None, context: str = "runtime") -> Dict:
        """Handle failure using integrated approach"""
        self.fix_attempts += 1
        
        print(f"🔧 Integrated MetaAgent handling failure (attempt {self.fix_attempts})...")
        print(f"   Mode: {self.integration_mode}")
        print(f"   Error: {error_message[:100]}...")
        
        # Try enhanced meta agent first (if enabled)
        if self.enhanced_priority and self.integration_mode in ['enhanced', 'hybrid']:
            try:
                enhanced_result = self.enhanced_meta.handle_failure(error_message, stack_trace, file_path)
                
                if enhanced_result.get('status') == 'success':
                    self.successful_fixes += 1
                    self.enhanced_fixes += 1
                    
                    return {
                        'status': 'success',
                        'meta_agent_used': 'enhanced',
                        'fix_applied': enhanced_result.get('fix_applied'),
                        'confidence': enhanced_result.get('confidence'),
                        'total_attempts': self.fix_attempts,
                        'success_rate': self.successful_fixes / self.fix_attempts
                    }
                else:
                    print(f"   ⚠️ Enhanced MetaAgent failed: {enhanced_result.get('reason', 'Unknown')}")
                    
                    # Try original meta agent if fallback enabled
                    if self.fallback_enabled and self.original_meta and self.integration_mode == 'hybrid':
                        print("   🔄 Falling back to original MetaAgent...")
                        return self._try_original_fix(error_message, stack_trace, file_path, context)
                    
            except Exception as e:
                print(f"   ❌ Enhanced MetaAgent error: {e}")
                
                # Try original meta agent if fallback enabled
                if self.fallback_enabled and self.original_meta:
                    print("   🔄 Falling back to original MetaAgent...")
                    return self._try_original_fix(error_message, stack_trace, file_path, context)
        
        # Use original meta agent only
        elif self.original_meta and self.integration_mode in ['original', 'hybrid']:
            return self._try_original_fix(error_message, stack_trace, file_path, context)
        
        # Enhanced only mode
        elif self.integration_mode == 'enhanced':
            enhanced_result = self.enhanced_meta.handle_failure(error_message, stack_trace, file_path)
            if enhanced_result.get('status') == 'success':
                self.successful_fixes += 1
                self.enhanced_fixes += 1
            
            return enhanced_result
        
        # No meta agent available
        return {
            'status': 'failed',
            'reason': 'No suitable MetaAgent available',
            'integration_mode': self.integration_mode,
            'total_attempts': self.fix_attempts
        }
    
    def _try_original_fix(self, error_message: str, stack_trace: str, file_path: str, context: str) -> Dict:
        """Try to fix using original MetaAgent"""
        try:
            # Create failure event for original meta agent
            from complete_sam_unified import FailureEvent
            
            failure_event = FailureEvent(
                error_type="RuntimeError",
                stack_trace=stack_trace,
                timestamp=datetime.now().isoformat(),
                severity="medium",
                context=context,
                research_notes=f"Integrated fix attempt: {error_message[:100]}"
            )
            
            # Try original meta agent
            original_result = self.original_meta.handle_failure(failure_event)
            
            if original_result:
                self.successful_fixes += 1
                self.original_fixes += 1
                
                return {
                    'status': 'success',
                    'meta_agent_used': 'original',
                    'original_result': original_result,
                    'total_attempts': self.fix_attempts,
                    'success_rate': self.successful_fixes / self.fix_attempts
                }
            else:
                return {
                    'status': 'failed',
                    'meta_agent_used': 'original',
                    'reason': 'Original MetaAgent could not fix the issue',
                    'total_attempts': self.fix_attempts
                }
                
        except Exception as e:
            print(f"   ❌ Original MetaAgent error: {e}")
            return {
                'status': 'failed',
                'meta_agent_used': 'original',
                'reason': f'Original MetaAgent error: {e}',
                'total_attempts': self.fix_attempts
            }
    
    def get_comprehensive_statistics(self) -> Dict:
        """Get comprehensive statistics from both meta agents"""
        stats = {
            'integration_mode': self.integration_mode,
            'total_attempts': self.fix_attempts,
            'successful_fixes': self.successful_fixes,
            'success_rate': self.successful_fixes / max(1, self.fix_attempts),
            'enhanced_fixes': self.enhanced_fixes,
            'original_fixes': self.original_fixes,
            'enhanced_success_rate': self.enhanced_fixes / max(1, self.enhanced_fixes + self.original_fixes),
            'original_success_rate': self.original_fixes / max(1, self.enhanced_fixes + self.original_fixes)
        }
        
        # Add enhanced meta agent stats if available
        if self.enhanced_meta:
            enhanced_stats = self.enhanced_meta.get_statistics()
            stats['enhanced_capabilities'] = enhanced_stats
        
        # Add original meta agent stats if available
        if self.original_meta:
            try:
                original_stats = {
                    'failure_clusters': len(getattr(self.original_meta, 'failure_clusters', {})),
                    'patch_history': len(getattr(self.original_meta, 'patch_history', [])),
                    'confidence_threshold': getattr(self.original_meta, 'confidence_threshold', 0.8)
                }
                stats['original_capabilities'] = original_stats
            except Exception as e:
                stats['original_capabilities'] = f'Error retrieving stats: {e}'
        
        return stats
    
    def run_self_diagnostics(self) -> Dict:
        """Run comprehensive self-diagnostics"""
        print("🔍 Running Integrated MetaAgent self-diagnostics...")
        
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'integration_status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Check enhanced meta agent
        if self.enhanced_meta:
            try:
                enhanced_stats = self.enhanced_meta.get_statistics()
                if enhanced_stats.get('success_rate', 0) < 0.5:
                    diagnostics['issues'].append("Enhanced MetaAgent success rate below 50%")
                    diagnostics['recommendations'].append("Consider adjusting confidence threshold or fix strategies")
            except Exception as e:
                diagnostics['issues'].append(f"Enhanced MetaAgent diagnostic error: {e}")
        
        # Check original meta agent
        if self.original_meta:
            try:
                if hasattr(self.original_meta, 'failure_clusters'):
                    cluster_count = len(self.original_meta.failure_clusters)
                    if cluster_count > 10:
                        diagnostics['issues'].append(f"High number of failure clusters: {cluster_count}")
                        diagnostics['recommendations'].append("Consider implementing failure pattern analysis")
            except Exception as e:
                diagnostics['issues'].append(f"Original MetaAgent diagnostic error: {e}")
        
        # Check integration configuration
        if self.integration_mode not in ['enhanced', 'original', 'hybrid']:
            diagnostics['issues'].append(f"Invalid integration mode: {self.integration_mode}")
            diagnostics['recommendations'].append("Set SAM_META_INTEGRATION_MODE to 'enhanced', 'original', or 'hybrid'")
        
        # Overall health assessment
        if len(diagnostics['issues']) == 0:
            diagnostics['integration_status'] = 'excellent'
        elif len(diagnostics['issues']) <= 2:
            diagnostics['integration_status'] = 'good'
        elif len(diagnostics['issues']) <= 5:
            diagnostics['integration_status'] = 'warning'
        else:
            diagnostics['integration_status'] = 'critical'
        
        return diagnostics
    
    def create_test_scenarios(self) -> List[Dict]:
        """Create comprehensive test scenarios for validation"""
        scenarios = []
        
        # Scenario 1: Syntax errors
        scenarios.append({
            'name': 'syntax_error_colon',
            'description': 'Missing colon in function definition',
            'code': 'def broken_function()\n    print("test")\n    return "test"',
            'expected_error': 'SyntaxError',
            'difficulty': 'easy'
        })
        
        # Scenario 2: Import errors
        scenarios.append({
            'name': 'import_error_module',
            'description': 'Missing module import',
            'code': 'import nonexistent_module_xyz\ndef test():\n    return "test"',
            'expected_error': 'ModuleNotFoundError',
            'difficulty': 'easy'
        })
        
        # Scenario 3: Runtime errors
        scenarios.append({
            'name': 'runtime_division_zero',
            'description': 'Division by zero',
            'code': 'def divide(a, b):\n    return a / b\n\nresult = divide(10, 0)',
            'expected_error': 'ZeroDivisionError',
            'difficulty': 'medium'
        })
        
        # Scenario 4: Logic errors
        scenarios.append({
            'name': 'logic_index_error',
            'description': 'Index out of range',
            'code': 'data = [1, 2, 3]\nitem = data[10]',
            'expected_error': 'IndexError',
            'difficulty': 'medium'
        })
        
        # Scenario 5: Complex multi-error
        scenarios.append({
            'name': 'complex_nested_errors',
            'description': 'Multiple related errors in nested code',
            'code': '''
def complex_function(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
        else:
            result.append(data[i] / 0)  # Division by zero
    return result[10]  # Index error
''',
            'expected_error': 'Multiple',
            'difficulty': 'hard'
        })
        
        return scenarios
    
    def validate_with_scenarios(self) -> Dict:
        """Validate MetaAgent using comprehensive test scenarios"""
        print("🧪 Running comprehensive scenario validation...")
        
        scenarios = self.create_test_scenarios()
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_scenarios': len(scenarios),
            'passed_scenarios': 0,
            'failed_scenarios': 0,
            'scenario_results': [],
            'overall_score': 0
        }
        
        for scenario in scenarios:
            print(f"   Testing scenario: {scenario['name']} ({scenario['difficulty']})")
            
            # Create temporary file with scenario code
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(scenario['code'])
                temp_file = f.name
            
            try:
                # Execute to generate error
                try:
                    exec(compile(scenario['code'], temp_file, 'exec'))
                    error_message = "No error generated"
                    stack_trace = ""
                except Exception as e:
                    error_message = str(e)
                    stack_trace = traceback.format_exc()
                
                # Try to fix with integrated meta agent
                fix_result = self.handle_failure(error_message, stack_trace, temp_file)
                
                scenario_result = {
                    'name': scenario['name'],
                    'difficulty': scenario['difficulty'],
                    'error_detected': bool(error_message and error_message != "No error generated"),
                    'fix_attempted': True,
                    'fix_successful': fix_result.get('status') == 'success',
                    'meta_agent_used': fix_result.get('meta_agent_used', 'unknown'),
                    'confidence': fix_result.get('confidence', 0)
                }
                
                results['scenario_results'].append(scenario_result)
                
                if scenario_result['fix_successful']:
                    results['passed_scenarios'] += 1
                    print(f"      ✅ Fixed successfully (confidence: {scenario_result['confidence']:.2f})")
                else:
                    results['failed_scenarios'] += 1
                    print(f"      ❌ Fix failed")
                
            except Exception as e:
                print(f"      ❌ Scenario execution error: {e}")
                results['failed_scenarios'] += 1
                results['scenario_results'].append({
                    'name': scenario['name'],
                    'error': str(e),
                    'fix_successful': False
                })
            
            finally:
                # Cleanup temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        # Calculate overall score
        if results['total_scenarios'] > 0:
            results['overall_score'] = results['passed_scenarios'] / results['total_scenarios']
        
        return results
    
    def generate_improvement_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on performance"""
        recommendations = []
        stats = self.get_comprehensive_statistics()
        
        # Success rate recommendations
        if stats.get('success_rate', 0) < 0.3:
            recommendations.append("🔴 Critical: Success rate below 30% - Major improvements needed")
            recommendations.append("   - Lower confidence threshold to 0.5")
            recommendations.append("   - Add more fix strategies")
            recommendations.append("   - Improve error detection accuracy")
        elif stats.get('success_rate', 0) < 0.6:
            recommendations.append("🟡 Warning: Success rate below 60% - Improvements recommended")
            recommendations.append("   - Review failed fix patterns")
            recommendations.append("   - Enhance pattern matching")
        elif stats.get('success_rate', 0) < 0.8:
            recommendations.append("🟢 Good: Success rate above 60% - Minor improvements possible")
            recommendations.append("   - Fine-tune confidence thresholds")
            recommendations.append("   - Add edge case handling")
        else:
            recommendations.append("🎉 Excellent: Success rate above 80% - System working well")
        
        # Integration mode recommendations
        if stats.get('enhanced_success_rate', 0) > stats.get('original_success_rate', 0):
            recommendations.append("✅ Enhanced MetaAgent outperforming original")
            recommendations.append("   - Consider using enhanced-only mode")
        elif stats.get('original_success_rate', 0) > stats.get('enhanced_success_rate', 0):
            recommendations.append("📊 Original MetaAgent outperforming enhanced")
            recommendations.append("   - Review enhanced implementation")
        
        # Configuration recommendations
        if self.integration_mode == 'hybrid' and stats.get('success_rate', 0) < 0.5:
            recommendations.append("🔧 Hybrid mode underperforming")
            recommendations.append("   - Try enhanced-only mode")
            recommendations.append("   - Or disable fallback to original")
        
        return recommendations

# Factory function for easy integration
def create_integrated_meta_agent(system):
    """Create integrated meta agent with enhanced capabilities"""
    return IntegratedMetaAgent(system)

# Test runner for comprehensive validation
def run_comprehensive_validation():
    """Run comprehensive validation of integrated MetaAgent"""
    print("🚀 Comprehensive Integrated MetaAgent Validation")
    print("=" * 60)
    
    # Create mock system
    class MockSystem:
        def __init__(self):
            self.project_root = Path('/tmp')
    
    system = MockSystem()
    integrated_agent = IntegratedMetaAgent(system)
    
    # Run diagnostics
    diagnostics = integrated_agent.run_self_diagnostics()
    print("\n📊 Diagnostics Results:")
    print(f"   Status: {diagnostics['integration_status']}")
    print(f"   Issues: {len(diagnostics['issues'])}")
    for issue in diagnostics['issues']:
        print(f"   - {issue}")
    
    # Run scenario validation
    validation_results = integrated_agent.validate_with_scenarios()
    print(f"\n📊 Validation Results:")
    print(f"   Total Scenarios: {validation_results['total_scenarios']}")
    print(f"   Passed: {validation_results['passed_scenarios']}")
    print(f"   Failed: {validation_results['failed_scenarios']}")
    print(f"   Success Rate: {validation_results['overall_score']:.1%}")
    
    # Generate recommendations
    recommendations = integrated_agent.generate_improvement_recommendations()
    print(f"\n🎯 Recommendations:")
    for rec in recommendations:
        print(f"   {rec}")
    
    print("=" * 60)
    
    return {
        'diagnostics': diagnostics,
        'validation': validation_results,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    # Set test environment
    os.environ['SAM_META_INTEGRATION_MODE'] = 'hybrid'
    os.environ['SAM_META_FALLBACK_ENABLED'] = '1'
    os.environ['SAM_META_ENHANCED_PRIORITY'] = '1'
    
    # Run validation
    results = run_comprehensive_validation()
    
    # Exit with appropriate code
    success_rate = results['validation']['overall_score']
    exit_code = 0 if success_rate >= 0.6 else 1
    sys.exit(exit_code)
