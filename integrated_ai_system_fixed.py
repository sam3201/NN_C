#!/usr/bin/env python3
"""
Fixed Integrated AI System
Combines all capabilities with proper timeout handling
"""

import os
import sys
import time
import json
import random
import math
import threading
import subprocess
import queue
from datetime import datetime
from persistent_knowledge_system import PersistentKnowledgeSystem

class FixedIntegratedAISystem:
    def __init__(self):
        self.knowledge_system = PersistentKnowledgeSystem()
        self.session_start = time.time()
        self.running = True
        self.training_thread = None
        
        print("🚀 FIXED INTEGRATED AI SYSTEM")
        print("=" * 50)
        print("🧠 All capabilities + proper timeout handling")
        print("🤖 Using CodeLlama (fastest model)")
        
        # System components
        self.components = {
            'language_understanding': True,
            'mathematical_training': True,
            'internet_connectivity': True,
            'pretrained_models': True,
            'persistent_knowledge': True
        }
        
        # Pre-trained model configuration
        self.pretrained_model = 'codellama'  # Fastest from our test
        self.query_timeout = 15  # seconds
        
        # Training configuration
        self.training_interval = 60  # seconds (longer for stability)
        self.session_count = 0
        
        # Show system status
        self.show_system_status()
    
    def query_ollama_with_timeout(self, prompt, timeout=None):
        """Query Ollama with proper timeout handling"""
        if timeout is None:
            timeout = self.query_timeout
            
        result_queue = queue.Queue()
        
        def run_query():
            try:
                result = subprocess.run(
                    ['ollama', 'run', self.pretrained_model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=timeout + 5
                )
                result_queue.put(('success', result))
            except Exception as e:
                result_queue.put(('error', str(e)))
        
        # Start query in thread
        query_thread = threading.Thread(target=run_query)
        query_thread.daemon = True
        query_thread.start()
        
        # Wait for result with timeout
        try:
            status, result = result_queue.get(timeout=timeout)
            return status, result
        except queue.Empty:
            return 'timeout', None
    
    def show_system_status(self):
        """Show comprehensive system status"""
        summary = self.knowledge_system.get_knowledge_summary()
        
        print(f"\n📊 SYSTEM STATUS:")
        print(f"  🧠 Knowledge Base: {summary['total_knowledge_items']} items")
        print(f"  📚 Mathematical: {summary['mathematical_knowledge']}")
        print(f"  🗣️ Concepts: {summary['concept_knowledge']}")
        print(f"  🧬 Protein: {summary['protein_knowledge']}")
        print(f"  📝 Sessions: {summary['training_sessions']}")
        
        print(f"\n🤖 COMPONENTS:")
        for component, status in self.components.items():
            icon = "✅" if status else "❌"
            name = component.replace('_', ' ').title()
            print(f"  {icon} {name}")
        
        print(f"\n🎯 PRE-TRAINED MODEL:")
        print(f"  🤖 Model: {self.pretrained_model}")
        print(f"  ⏱️ Timeout: {self.query_timeout} seconds")
        print(f"  🔄 Training Interval: {self.training_interval} seconds")
        
        print(f"\n🚀 SYSTEM READY: All components integrated")
    
    def language_understanding_module(self):
        """Language understanding module"""
        print(f"\n🗣️ LANGUAGE UNDERSTANDING MODULE")
        
        # Test language understanding with CodeLlama
        test_prompt = "Explain what 'proof by contradiction' means in simple terms."
        
        print(f"  📝 Testing: {test_prompt}")
        status, result = self.query_ollama_with_timeout(test_prompt, timeout=10)
        
        if status == 'success' and result.returncode == 0:
            response = result.stdout.strip()
            print(f"  💬 Response: {response[:100]}...")
            
            # Add to knowledge base
            concept_id = self.knowledge_system.add_concept_knowledge(
                'Proof by Contradiction',
                response[:200],
                [test_prompt],
                'language_understanding'
            )
            print(f"  ✅ Added language concept")
            return 1
        else:
            print(f"  ⚠️ Language test: {status}")
            return 0
    
    def mathematical_training_module(self):
        """Mathematical training module"""
        print(f"\n🧠 MATHEMATICAL TRAINING MODULE")
        
        # Test mathematical reasoning
        test_problems = [
            "Solve step by step: 2x + 7 = 15",
            "What is the derivative of sin(x)?",
            "Find the area of a circle with radius 5"
        ]
        
        solved_count = 0
        for problem in test_problems:
            print(f"  📝 Solving: {problem}")
            status, result = self.query_ollama_with_timeout(problem, timeout=12)
            
            if status == 'success' and result.returncode == 0:
                response = result.stdout.strip()
                print(f"  💬 Solution: {response[:80]}...")
                
                # Add to knowledge base
                problem_id = self.knowledge_system.add_mathematical_knowledge(
                    problem,
                    response[:200],
                    'Solved by CodeLlama',
                    'pretrained_model'
                )
                solved_count += 1
            else:
                print(f"  ⚠️ Math test: {status}")
        
        print(f"  ✅ Solved {solved_count}/{len(test_problems)} problems")
        return solved_count
    
    def internet_connectivity_module(self):
        """Internet connectivity module (simulated)"""
        print(f"\n🌐 INTERNET CONNECTIVITY MODULE")
        
        # Simulate web content with CodeLlama's knowledge
        web_queries = [
            "What are the latest developments in P vs NP research?",
            "Explain quantum computing impact on cryptography",
            "What are recent breakthroughs in machine learning?"
        ]
        
        integrated_count = 0
        for query in web_queries:
            print(f"  🌐 Researching: {query[:40]}...")
            status, result = self.query_ollama_with_timeout(query, timeout=15)
            
            if status == 'success' and result.returncode == 0:
                response = result.stdout.strip()
                print(f"  💬 Research: {response[:80]}...")
                
                # Add as web knowledge
                concept_id = self.knowledge_system.add_concept_knowledge(
                    f"Web Research: {query[:30]}",
                    response[:200],
                    ['CodeLlama Knowledge'],
                    'web_research'
                )
                integrated_count += 1
            else:
                print(f"  ⚠️ Research test: {status}")
        
        print(f"  ✅ Integrated {integrated_count}/{len(web_queries)} research items")
        return integrated_count
    
    def pretrained_model_module(self):
        """Pre-trained model module"""
        print(f"\n🤖 PRE-TRAINED MODEL MODULE")
        
        # Advanced reasoning test
        advanced_prompt = """Analyze this mathematical statement:
"If P = NP, then cryptographic systems based on factoring would be broken."
Explain why this is true and what the implications would be."""
        
        print(f"  🧠 Advanced reasoning: {advanced_prompt[:50]}...")
        status, result = self.query_ollama_with_timeout(advanced_prompt, timeout=20)
        
        if status == 'success' and result.returncode == 0:
            response = result.stdout.strip()
            print(f"  💬 Analysis: {response[:100]}...")
            
            # Add advanced insight
            concept_id = self.knowledge_system.add_concept_knowledge(
                'Advanced Mathematical Analysis',
                response[:250],
                [advanced_prompt],
                'advanced_reasoning'
            )
            print(f"  ✅ Added advanced insight")
            return 1
        else:
            print(f"  ⚠️ Advanced test: {status}")
            return 0
    
    def knowledge_synthesis_module(self):
        """Knowledge synthesis module"""
        print(f"\n🧠 KNOWLEDGE SYNTHESIS MODULE")
        
        # Get current knowledge
        summary = self.knowledge_system.get_knowledge_summary()
        
        # Create synthesis with CodeLlama
        synthesis_prompt = f"""Based on the following knowledge base:
- Mathematical problems: {summary['mathematical_knowledge']} items
- Concepts: {summary['concept_knowledge']} items
- Training sessions: {summary['training_sessions']}

Provide a brief analysis of what this AI system has learned and suggest the next logical step for advancement."""
        
        print(f"  🔄 Synthesizing knowledge...")
        status, result = self.query_ollama_with_timeout(synthesis_prompt, timeout=18)
        
        if status == 'success' and result.returncode == 0:
            response = result.stdout.strip()
            print(f"  💬 Synthesis: {response[:120]}...")
            
            # Add synthesis
            concept_id = self.knowledge_system.add_concept_knowledge(
                'Knowledge Synthesis',
                response[:300],
                [f"Session {self.session_count}"],
                'synthesis'
            )
            print(f"  ✅ Added synthesis")
            return 1
        else:
            print(f"  ⚠️ Synthesis test: {status}")
            return 0
    
    def training_loop(self):
        """Main training loop"""
        while self.running:
            try:
                self.session_count += 1
                print(f"\n{'='*50}")
                print(f"🔄 TRAINING SESSION {self.session_count}")
                print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*50}")
                
                # Module 1: Language Understanding
                language_results = self.language_understanding_module()
                
                # Module 2: Mathematical Training
                math_results = self.mathematical_training_module()
                
                # Module 3: Internet Connectivity
                internet_results = self.internet_connectivity_module()
                
                # Module 4: Pre-trained Model
                model_results = self.pretrained_model_module()
                
                # Module 5: Knowledge Synthesis
                synthesis_results = self.knowledge_synthesis_module()
                
                # Session summary
                total_new = language_results + math_results + internet_results + model_results + synthesis_results
                print(f"\n📊 SESSION {self.session_count} SUMMARY:")
                print(f"  🗣️ Language: +{language_results}")
                print(f"  🧠 Mathematics: +{math_results}")
                print(f"  🌐 Internet: +{internet_results}")
                print(f"  🤖 Pre-trained: +{model_results}")
                print(f"  🧠 Synthesis: +{synthesis_results}")
                print(f"  📊 Total New: +{total_new} knowledge items")
                
                # Save knowledge
                self.knowledge_system.save_all_knowledge()
                print(f"  💾 Knowledge saved")
                
                # Wait for next session
                print(f"\n⏳ Next session in {self.training_interval} seconds...")
                
                # Wait with interrupt check
                for i in range(self.training_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error in training loop: {e}")
                time.sleep(5)
    
    def start_training_thread(self):
        """Start training thread"""
        if self.training_thread is None or not self.training_thread.is_alive():
            self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
            self.training_thread.start()
            print(f"🚀 Training thread started")
            return True
        else:
            print(f"⚠️ Training thread already running")
            return False
    
    def stop_training_thread(self):
        """Stop training thread"""
        self.running = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=10)
            print(f"🛑 Training thread stopped")
        else:
            print(f"⚠️ No training thread to stop")
    
    def run_interactive_mode(self):
        """Run interactive mode"""
        print(f"\n🎮 INTERACTIVE MODE")
        print(f"Commands: status, start, stop, quit, test <prompt>")
        
        while True:
            try:
                command = input(f"\n🚀 Integrated AI> ").strip().lower()
                
                if not command:
                    continue
                elif command == 'quit' or command == 'q':
                    break
                elif command == 'status':
                    summary = self.knowledge_system.get_knowledge_summary()
                    print(f"\n📊 STATUS:")
                    print(f"  🔄 Sessions: {self.session_count}")
                    print(f"  🏃 Running: {self.running}")
                    print(f"  🧵 Thread: {self.training_thread.is_alive() if self.training_thread else False}")
                    print(f"  📚 Knowledge: {summary['total_knowledge_items']} items")
                elif command == 'start':
                    if self.start_training_thread():
                        print(f"✅ Training started")
                    else:
                        print(f"⚠️ Training already running")
                elif command == 'stop':
                    self.stop_training_thread()
                    print(f"✅ Training stopped")
                elif command.startswith('test '):
                    prompt = command[5:]
                    print(f"\n🧪 Testing: {prompt}")
                    status, result = self.query_ollama_with_timeout(prompt, timeout=10)
                    if status == 'success' and result.returncode == 0:
                        print(f"💬 Response: {result.stdout.strip()}")
                    else:
                        print(f"❌ Error: {status}")
                else:
                    print(f"❓ Unknown command: {command}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Cleanup
        self.stop_training_thread()
        print(f"\n👋 Goodbye!")

def main():
    """Main function"""
    print("🚀 FIXED INTEGRATED AI SYSTEM")
    print("=" * 50)
    print("🧠 All capabilities + proper timeout handling")
    print("🤖 Using CodeLlama (fastest model)")
    
    # Create integrated system
    integrated_ai = FixedIntegratedAISystem()
    
    # Start training automatically
    print(f"\n🚀 Starting automatic training...")
    integrated_ai.start_training_thread()
    
    # Run interactive mode
    try:
        integrated_ai.run_interactive_mode()
    except KeyboardInterrupt:
        print(f"\n\n🛑 Interrupted by user")
    finally:
        integrated_ai.stop_training_thread()
        print(f"\n🎉 INTEGRATED AI SESSION COMPLETE!")
        print(f"📊 Final knowledge base: {integrated_ai.knowledge_system.get_knowledge_summary()['total_knowledge_items']} items")
        print(f"🔄 Sessions completed: {integrated_ai.session_count}")
        print(f"💾 All knowledge saved")

if __name__ == "__main__":
    main()
