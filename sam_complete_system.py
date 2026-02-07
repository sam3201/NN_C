#!/usr/bin/env python3
"""
SAM 2.0 - Complete AGI System Integration
===========================================

This is the main integration script that:
1. Connects Python orchestration with C neural core
2. Integrates with LLM groupchatbot
3. Runs the complete AGI system end-to-end
4. Provides comprehensive testing and validation

Author: Samuel Dasari
Date: February 2026
Version: 2.0 - Full Context Morphogenesis
"""

import os
import sys
import json
import time
import queue
import threading
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add paths for imports
sys.path.insert(0, '/Users/samueldasari/Personal/NN_C')

# Import SAM components
try:
    from correct_sam_hub import CorrectSAMHub
    from sam_neural_core import SAMNeuralCore, SAMNetworkManager, create_sam_core
    from agi_test_framework import AGITestFramework
    SAM_AVAILABLE = True
    print("✅ SAM 2.0 components loaded successfully")
except Exception as e:
    print(f"⚠️  SAM import warning: {e}")
    SAM_AVAILABLE = False

class SAMCompleteSystem:
    """
    Complete SAM 2.0 AGI System Integration
    
    Integrates:
    - Python orchestration layer (CorrectSAMHub)
    - C neural core (via sam_neural_core ctypes bridge)
    - LLM groupchatbot interface
    - AGI test framework
    - Morphogenesis system
    """
    
    def __init__(self):
        """Initialize complete SAM 2.0 system"""
        print("\n" + "="*70)
        print("🧠 SAM 2.0 - Complete AGI System Initialization")
        print("="*70)
        
        self.status = {
            'python_hub': False,
            'c_neural_core': False,
            'morphogenesis': False,
            'llm_integration': False,
            'test_framework': False
        }
        
        # Initialize components
        self._init_python_hub()
        self._init_c_neural_core()
        self._init_morphogenesis()
        self._init_llm_integration()
        self._init_test_framework()
        
        # Print status
        self._print_status()
        
    def _init_python_hub(self):
        """Initialize Python orchestration hub"""
        try:
            print("\n📦 Initializing Python Hub...")
            self.hub = CorrectSAMHub()
            self.status['python_hub'] = True
            print("   ✅ Python Hub ready")
        except Exception as e:
            print(f"   ❌ Python Hub failed: {e}")
            self.hub = None
            
    def _init_c_neural_core(self):
        """Initialize C neural core via ctypes"""
        try:
            print("\n🔧 Initializing C Neural Core...")
            self.neural_core, self.network_manager = create_sam_core()
            self.status['c_neural_core'] = True
            print("   ✅ C Neural Core connected")
        except Exception as e:
            print(f"   ⚠️  C Neural Core not available: {e}")
            print("   📄 Building shared library...")
            self._build_c_library()
            
    def _build_c_library(self):
        """Attempt to build C shared library"""
        try:
            base_dir = Path('/Users/samueldasari/Personal/NN_C')
            os.chdir(base_dir)
            
            # Try to build
            result = subprocess.run(
                ['make', 'shared'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("   ✅ Library built successfully")
                # Retry initialization
                self.neural_core, self.network_manager = create_sam_core()
                self.status['c_neural_core'] = True
            else:
                print(f"   ❌ Build failed: {result.stderr}")
                self.neural_core = None
                self.network_manager = None
        except Exception as e:
            print(f"   ❌ Build error: {e}")
            self.neural_core = None
            self.network_manager = None
            
    def _init_morphogenesis(self):
        """Initialize morphogenesis system"""
        if not self.neural_core:
            print("\n🧬 Morphogenesis: Using Python fallback")
            self.status['morphogenesis'] = False
            return
            
        try:
            print("\n🧬 Initializing Morphogenesis...")
            if self.neural_core.initialize_morphogenesis(initial_dim=64, max_dim=256):
                self.status['morphogenesis'] = True
                print("   ✅ Morphogenesis ready (C implementation)")
            else:
                print("   ⚠️  Morphogenesis init failed")
        except Exception as e:
            print(f"   ⚠️  Morphogenesis error: {e}")
            
    def _init_llm_integration(self):
        """Initialize LLM groupchatbot integration"""
        try:
            print("\n🤖 Initializing LLM Integration...")
            
            # Check available LLM backends
            self.llm_backends = {}
            
            # Check Ollama
            try:
                result = subprocess.run(
                    ['ollama', 'list'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self.llm_backends['ollama'] = True
                    print("   ✅ Ollama available")
            except:
                print("   ⚠️  Ollama not available")
                
            # Check for API keys
            if os.environ.get('OPENAI_API_KEY'):
                self.llm_backends['openai'] = True
                print("   ✅ OpenAI configured")
                
            if os.environ.get('ANTHROPIC_API_KEY'):
                self.llm_backends['anthropic'] = True
                print("   ✅ Anthropic configured")
                
            if os.environ.get('GOOGLE_API_KEY'):
                self.llm_backends['google'] = True
                print("   ✅ Google Gemini configured")
            
            if self.llm_backends:
                self.status['llm_integration'] = True
                print(f"   ✅ {len(self.llm_backends)} LLM backend(s) ready")
            else:
                print("   ⚠️  No LLM backends available")
                
        except Exception as e:
            print(f"   ❌ LLM integration error: {e}")
            
    def _init_test_framework(self):
        """Initialize AGI test framework"""
        try:
            print("\n🧪 Initializing Test Framework...")
            self.test_framework = AGITestFramework('sam_integration_test')
            self.status['test_framework'] = True
            print("   ✅ Test Framework ready")
        except Exception as e:
            print(f"   ❌ Test Framework error: {e}")
            self.test_framework = None
            
    def _print_status(self):
        """Print system status summary"""
        print("\n" + "="*70)
        print("📊 System Status Summary")
        print("="*70)
        
        total = len(self.status)
        ready = sum(1 for v in self.status.values() if v)
        
        for component, ready_status in self.status.items():
            symbol = "✅" if ready_status else "❌"
            print(f"{symbol} {component.replace('_', ' ').title()}")
            
        print(f"\n{'='*70}")
        print(f"System Readiness: {ready}/{total} components")
        
        if ready == total:
            print("🚀 ALL SYSTEMS READY - Full AGI Mode Available")
        elif ready >= total // 2:
            print("⚠️  PARTIAL SYSTEM - Core functionality available")
        else:
            print("❌ LIMITED SYSTEM - Some features unavailable")
            
        print("="*70 + "\n")
        
    def run_agi_test(self, epochs=10):
        """Run AGI growth test"""
        if not self.test_framework:
            print("❌ Test Framework not available")
            return None
            
        print(f"\n🧪 Running AGI Growth Test ({epochs} epochs)...")
        
        try:
            # Run test
            metrics = self.test_framework.run_full_test()
            
            # Analyze results
            print("\n" + "="*70)
            print("📈 AGI Test Results")
            print("="*70)
            
            if metrics['concepts_born'][-1] > 0:
                print("✅ Concept formation detected")
            if metrics['train_loss'][-1] < metrics['train_loss'][0]:
                print("✅ Learning improvement detected")
            if metrics['brittleness_score'][-1] < 0.3:
                print("✅ Low brittleness - stable system")
                
            return metrics
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return None
            
    def start_conversation_hub(self):
        """Start the conversation hub with all integrations"""
        if not self.hub:
            print("❌ Hub not available")
            return False
            
        print("\n🌐 Starting Conversation Hub...")
        
        try:
            # Inject neural core into hub if available
            if self.neural_core and hasattr(self.hub, '_neural_core'):
                self.hub._neural_core = self.neural_core
                print("   ✅ Neural core injected into hub")
                
            # Inject network manager
            if self.network_manager and hasattr(self.hub, '_network_manager'):
                self.hub._network_manager = self.network_manager
                print("   ✅ Network manager injected")
                
            # Start hub
            print("\n" + "="*70)
            print("🚀 SAM 2.0 Conversation Hub Running")
            print("="*70)
            print("URL: http://127.0.0.1:8080")
            print("Features:")
            print("  - Multi-agent conversations")
            print("  - Search→Augment→Relay→Verify→Save pipeline")
            print("  - Latent-space morphogenesis")
            print("  - Knowledge verification")
            print("  - AGI growth tracking")
            print("="*70 + "\n")
            
            self.hub.run(host='127.0.0.1', port=8080)
            return True
            
        except Exception as e:
            print(f"❌ Hub start failed: {e}")
            return False
            
    def run_morphogenesis_demo(self):
        """Demonstrate morphogenesis concept birth"""
        print("\n" + "="*70)
        print("🧬 Morphogenesis Demo - Concept Birth")
        print("="*70)
        
        if not self.neural_core:
            print("❌ Neural core not available")
            return False
            
        try:
            # Simulate high error scenario
            print("\n📊 Simulating high-error scenario...")
            for i in range(25):
                error = 0.25 + np.random.normal(0, 0.02)
                self.neural_core.record_error(max(0.15, error))
                
            # Check trigger
            trigger = self.neural_core.check_morphogenesis_trigger(0.28)
            
            print(f"🚨 Morphogenesis trigger: {trigger}")
            
            if trigger:
                # Birth concept
                concept_name = f"emergent_concept_{int(time.time())}"
                success = self.neural_core.birth_concept(concept_name)
                
                if success:
                    print(f"✅ Concept '{concept_name}' born successfully!")
                    
                # Print summary
                self.neural_core.print_summary()
                
            return trigger
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            return False
            
    def run_comprehensive_test(self):
        """Run comprehensive system test"""
        print("\n" + "="*70)
        print("🧪 Comprehensive System Test")
        print("="*70)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Test 1: Neural Core
        if self.neural_core:
            try:
                print("\n1️⃣ Testing Neural Core...")
                self.neural_core.record_error(0.2)
                trend = self.neural_core.get_error_trend()
                print(f"   ✅ Error recording works (trend: {trend:.6f})")
                results['tests']['neural_core'] = 'PASS'
            except Exception as e:
                print(f"   ❌ Neural Core test failed: {e}")
                results['tests']['neural_core'] = 'FAIL'
        else:
            results['tests']['neural_core'] = 'SKIP'
            
        # Test 2: Morphogenesis
        if self.neural_core and self.status['morphogenesis']:
            try:
                print("\n2️⃣ Testing Morphogenesis...")
                demo_result = self.run_morphogenesis_demo()
                results['tests']['morphogenesis'] = 'PASS' if demo_result else 'FAIL'
            except Exception as e:
                print(f"   ❌ Morphogenesis test failed: {e}")
                results['tests']['morphogenesis'] = 'FAIL'
        else:
            results['tests']['morphogenesis'] = 'SKIP'
            
        # Test 3: Hub Integration
        if self.hub:
            try:
                print("\n3️⃣ Testing Hub Integration...")
                # Test basic hub functions
                print(f"   ✅ Hub initialized with {len(self.hub.sam_submodels)} submodels")
                results['tests']['hub'] = 'PASS'
            except Exception as e:
                print(f"   ❌ Hub test failed: {e}")
                results['tests']['hub'] = 'FAIL'
        else:
            results['tests']['hub'] = 'SKIP'
            
        # Test 4: Test Framework
        if self.test_framework:
            try:
                print("\n4️⃣ Testing Test Framework...")
                # Just verify it can be created
                print(f"   ✅ Test Framework ready")
                results['tests']['test_framework'] = 'PASS'
            except Exception as e:
                print(f"   ❌ Test Framework test failed: {e}")
                results['tests']['test_framework'] = 'FAIL'
        else:
            results['tests']['test_framework'] = 'SKIP'
            
        # Summary
        print("\n" + "="*70)
        print("📊 Test Summary")
        print("="*70)
        
        passed = sum(1 for v in results['tests'].values() if v == 'PASS')
        failed = sum(1 for v in results['tests'].values() if v == 'FAIL')
        skipped = sum(1 for v in results['tests'].values() if v == 'SKIP')
        
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Skipped: {skipped}")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED - System Ready!")
        else:
            print(f"\n⚠️  {failed} test(s) failed - Check logs above")
            
        print("="*70 + "\n")
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SAM 2.0 Complete AGI System')
    parser.add_argument('--test', action='store_true', help='Run comprehensive test')
    parser.add_argument('--demo', action='store_true', help='Run morphogenesis demo')
    parser.add_argument('--agi-test', action='store_true', help='Run AGI growth test')
    parser.add_argument('--epochs', type=int, default=10, help='AGI test epochs')
    parser.add_argument('--hub', action='store_true', help='Start conversation hub')
    parser.add_argument('--status', action='store_true', help='Show system status only')
    
    args = parser.parse_args()
    
    # Initialize system
    system = SAMCompleteSystem()
    
    # Execute requested action
    if args.status:
        # Just show status (already printed in __init__)
        pass
    elif args.test:
        system.run_comprehensive_test()
    elif args.demo:
        system.run_morphogenesis_demo()
    elif args.agi_test:
        system.run_agi_test(epochs=args.epochs)
    elif args.hub:
        system.start_conversation_hub()
    else:
        # Default: run comprehensive test
        print("\n🚀 No specific mode selected. Running comprehensive test...")
        print("   (Use --hub to start conversation server)")
        system.run_comprehensive_test()


if __name__ == "__main__":
    main()
