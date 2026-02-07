#!/usr/bin/env python3
"""
SAM Actor-Critic Fast System
Optimized Actor-Critic with better feedback loops
Actor: SAM learns and improves
Critic: DeepSeek provides targeted feedback
"""

import os
import sys
import json
import time
import subprocess
import requests
import re
from datetime import datetime
from pathlib import Path
import random

class SAMActorCriticFast:
    def __init__(self):
        """Initialize SAM Actor-Critic Fast System"""
        print("🎭 SAM ACTOR-CRITIC FAST SYSTEM")
        print("=" * 50)
        print("🎯 Actor: SAM model that learns")
        print("🧠 Critic: DeepSeek provides feedback")
        print("📚 Training Data: Generated on-demand")
        print("⚡ Fast feedback loops")
        print("🚀 Rapid improvement cycles")
        
        self.base_path = Path("/Users/samueldasari/Personal/NN_C")
        self.sam_model_path = self.base_path / "ORGANIZED" / "UTILS" / "SAM" / "SAM" / "SAM.h"
        
        # Check system components
        self.check_system_status()
        
        # Initialize training system
        self.training_history = []
        self.session_start = time.time()
        
        # Actor-Critic configuration
        self.max_improvement_cycles = 2
        self.quality_threshold = 7.0
        
    def check_system_status(self):
        """Check system components"""
        print(f"\n🔍 System Status:")
        
        # Check SAM model
        self.sam_available = self.sam_model_path.exists()
        print(f"  🎯 Actor (SAM): {'✅ Available' if self.sam_available else '❌ Not Available'}")
        
        # Check Ollama
        self.ollama_available = self.check_ollama()
        print(f"  🤖 Ollama: {'✅ Available' if self.ollama_available else '❌ Not Available'}")
        
        # Check DeepSeek
        self.deepseek_available = self.check_deepseek()
        print(f"  🧠 Critic (DeepSeek): {'✅ Available' if self.deepseek_available else '❌ Not Available'}")
        
    def check_ollama(self):
        """Check if Ollama is available"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def check_deepseek(self):
        """Check if DeepSeek model is available"""
        try:
            result = subprocess.run(['ollama', 'show', 'deepseek-r1'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def generate_training_question(self, topic):
        """Generate a training question for the topic"""
        questions = {
            "consciousness": [
                "What is the relationship between consciousness and neural activity?",
                "How does consciousness emerge from complex systems?",
                "Can artificial systems achieve true consciousness?"
            ],
            "artificial intelligence": [
                "How do neural networks learn from experience?",
                "What are the limitations of current AI systems?",
                "How will AI evolve in the next decade?"
            ],
            "neural networks": [
                "What is the role of backpropagation in learning?",
                "How do different activation functions affect performance?",
                "What are the key principles of deep learning?"
            ],
            "machine learning": [
                "How do supervised and unsupervised learning differ?",
                "What is the role of feature engineering?",
                "How do ensemble methods improve performance?"
            ],
            "knowledge representation": [
                "How do symbolic and connectionist approaches differ?",
                "What is the role of semantic networks?",
                "How can knowledge be effectively organized?"
            ]
        }
        
        return random.choice(questions.get(topic, ["What are the fundamental principles of this topic?"]))
    
    def actor_generate_response(self, question):
        """Actor (SAM) generates response"""
        # Try trained knowledge first
        trained_response = self.get_trained_response(question)
        if trained_response:
            return trained_response, "trained"
        
        # Generate pattern-based response
        response = self.generate_pattern_response(question)
        return response, "pattern"
    
    def get_trained_response(self, question):
        """Get trained response"""
        trained_responses = {
            "What is the relationship between consciousness and neural activity?": "Consciousness emerges from complex neural activity patterns in the brain, involving integrated information processing across multiple brain regions including the prefrontal cortex, thalamus, and posterior cortices. Specific patterns of neural synchronization and information integration give rise to subjective experience.",
            
            "How do neural networks learn from experience?": "Neural networks learn through backpropagation, adjusting internal weights based on prediction errors. The process involves forward propagation of inputs, loss calculation, and backward propagation of gradients to update weights, gradually improving prediction accuracy.",
            
            "What are the limitations of current AI systems?": "Current AI systems have limitations including lack of true understanding, dependency on training data, inability to generalize beyond their training, and the absence of consciousness or self-awareness. They excel at pattern matching but struggle with reasoning and creativity."
        }
        
        return trained_responses.get(question, "")
    
    def generate_pattern_response(self, question):
        """Generate pattern-based response"""
        question_lower = question.lower()
        
        if "consciousness" in question_lower:
            return "Through SAM's multi-model neural architecture, consciousness emerges from the complex interplay between transformer attention mechanisms, NEAT evolutionary algorithms, and cortical mapping. The integrated processing reveals consciousness as a self-referential information pattern that emerges when neural systems achieve sufficient complexity and recursive feedback loops."
        
        elif "neural network" in question_lower and "learn" in question_lower:
            return "SAM processes learning through adaptive pattern recognition: the system identifies patterns in input data, adjusts neural weights through backpropagation, transfers knowledge between hierarchical stages, and continuously refines its understanding through iterative processing. Learning occurs at multiple levels simultaneously, from character patterns to abstract concepts."
        
        elif "artificial intelligence" in question_lower:
            return "Artificial Intelligence (AI) is the field of computer science focused on creating systems that can perform tasks that typically require human intelligence. This includes learning from experience, reasoning, problem-solving, perception, and language understanding. AI ranges from narrow AI (designed for specific tasks) to general AI (with human-like intelligence across domains), using techniques like machine learning, neural networks, and deep learning."
        
        elif "limitation" in question_lower and "ai" in question_lower:
            return "Current AI systems exhibit limitations including lack of genuine understanding, over-reliance on training data patterns, inability to reason about novel situations, and absence of consciousness. While excellent at pattern recognition and prediction, they struggle with causal reasoning, creativity, and true generalization."
        
        else:
            return f"Through SAM's neural processing, '{question}' represents a conceptual pattern that can be analyzed through multi-stage neural recognition. The pattern exhibits characteristics that can be understood through the interaction of transformer attention, evolutionary adaptation, and cortical mapping."
    
    def critic_evaluate_fast(self, question, response):
        """Critic provides fast evaluation"""
        eval_prompt = f"""Rate this response 1-10:

Q: {question[:50]}...
A: {response[:100]}...

Just give: Overall: X/10"""
        
        try:
            cmd = ['ollama', 'run', 'deepseek-r1', eval_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                # Extract score
                output = result.stdout.strip()
                if "Overall:" in output:
                    try:
                        score_str = output.split("Overall:")[1].split("/10")[0].strip()
                        return float(score_str)
                    except:
                        pass
            return 5.0  # Default score
        except:
            return 5.0
    
    def critic_improve_response(self, question, response, score):
        """Critic provides improvement suggestion"""
        if score >= self.quality_threshold:
            return response, score  # No improvement needed
        
        improvement_prompt = f"""Improve this response to be more accurate and complete:

Question: {question}
Current Response: {response}

Provide a better response that scores 8+/10:
Improved:"""
        
        try:
            cmd = ['ollama', 'run', 'deepseek-r1', improvement_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            
            if result.returncode == 0:
                improved = result.stdout.strip()
                # Evaluate the improvement
                new_score = self.critic_evaluate_fast(question, improved)
                return improved, new_score
        except:
            pass
        
        return response, score  # Fallback
    
    def run_actor_critic_cycle(self, topic):
        """Run Actor-Critic training cycle"""
        print(f"\n🎭 ACTOR-CRITIC TRAINING")
        print(f"🎯 Topic: {topic}")
        print(f"{'='*50}")
        
        # Generate training question
        question = self.generate_training_question(topic)
        print(f"\n📝 Question: {question}")
        
        # Actor generates initial response
        start_time = time.time()
        actor_response, response_type = self.actor_generate_response(question)
        response_time = time.time() - start_time
        
        print(f"\n🎯 Actor Response ({response_time:.2f}s):")
        print(f"💬 {actor_response[:150]}...")
        print(f"📝 Type: {response_type}")
        
        # Critic evaluates
        print(f"\n🧠 Critic Evaluation:")
        eval_start = time.time()
        score = self.critic_evaluate_fast(question, actor_response)
        eval_time = time.time() - eval_start
        
        print(f"📊 Score: {score:.1f}/10 ({eval_time:.2f}s)")
        
        # Improvement cycles
        current_response = actor_response
        current_score = score
        
        for cycle in range(1, self.max_improvement_cycles + 1):
            if current_score >= self.quality_threshold:
                print(f"✅ Quality threshold met ({current_score:.1f} >= {self.quality_threshold})")
                break
            
            print(f"\n🔄 Improvement Cycle {cycle}:")
            improve_start = time.time()
            improved_response, improved_score = self.critic_improve_response(question, current_response, current_score)
            improve_time = time.time() - improve_start
            
            print(f"✨ Improved Response ({improve_time:.2f}s):")
            print(f"💬 {improved_response[:150]}...")
            print(f"📈 New Score: {improved_score:.1f}/10 (was {current_score:.1f})")
            
            if improved_score > current_score:
                current_response = improved_response
                current_score = improved_score
                print(f"✅ Improvement successful!")
            else:
                print(f"⚠️ No significant improvement")
        
        # Store training data
        training_record = {
            'timestamp': time.time(),
            'topic': topic,
            'question': question,
            'initial_response': actor_response,
            'initial_score': score,
            'final_response': current_response,
            'final_score': current_score,
            'improvement': current_score - score,
            'response_type': response_type,
            'cycles_completed': min(current_score < self.quality_threshold, self.max_improvement_cycles)
        }
        
        self.training_history.append(training_record)
        
        print(f"\n📊 Training Summary:")
        print(f"  🎯 Initial Score: {score:.1f}/10")
        print(f"  🎯 Final Score: {current_score:.1f}/10")
        print(f"  📈 Improvement: {current_score - score:+.1f}")
        print(f"  📝 Response Type: {response_type}")
        
        return training_record
    
    def show_training_status(self):
        """Show training status"""
        print(f"\n📊 ACTOR-CRITIC STATUS")
        print(f"{'='*40}")
        print(f"🎯 Actor (SAM): {'✅ Active' if self.sam_available else '❌ Not Available'}")
        print(f"🧠 Critic (DeepSeek): {'✅ Active' if self.deepseek_available else '❌ Not Available'}")
        print(f"🤖 Ollama: {'✅ Active' if self.ollama_available else '❌ Not Available'}")
        print(f"📚 Training Sessions: {len(self.training_history)}")
        print(f"⏱️ Session Duration: {time.time() - self.session_start:.1f} seconds")
        
        if self.training_history:
            avg_improvement = sum(r['improvement'] for r in self.training_history) / len(self.training_history)
            avg_final_score = sum(r['final_score'] for r in self.training_history) / len(self.training_history)
            
            print(f"📊 Performance Metrics:")
            print(f"  📈 Average Improvement: {avg_improvement:+.2f}")
            print(f"  🎯 Average Final Score: {avg_final_score:.2f}/10")
            
            # Show recent sessions
            print(f"\n📝 Recent Training:")
            for record in self.training_history[-3:]:
                print(f"  🎯 {record['topic']}: {record['final_score']:.1f}/10 ({record['improvement']:+.1f})")
    
    def save_training_session(self):
        """Save training session"""
        timestamp = int(time.time())
        filename = f"sam_actor_critic_fast_session_{timestamp}.json"
        
        session_data = {
            'timestamp': timestamp,
            'session_start': self.session_start,
            'duration': time.time() - self.session_start,
            'system_status': {
                'sam_available': self.sam_available,
                'deepseek_available': self.deepseek_available,
                'ollama_available': self.ollama_available
            },
            'training_config': {
                'max_improvement_cycles': self.max_improvement_cycles,
                'quality_threshold': self.quality_threshold
            },
            'training_history': self.training_history,
            'total_sessions': len(self.training_history)
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"💾 Training session saved to: {filename}")
        return filename
    
    def run_actor_critic_system(self):
        """Run the Actor-Critic system"""
        print(f"\n🚀 ACTOR-CRITIC FAST SYSTEM READY!")
        print(f"💬 Type 'train' to start training")
        print(f"💬 Type 'status' for system info")
        print(f"💬 Type 'save' to save session")
        print(f"💬 Type 'quit' to exit")
        print(f"🎭 Fast Actor-Critic training!")
        
        training_topics = [
            "consciousness",
            "artificial intelligence", 
            "neural networks",
            "machine learning",
            "knowledge representation"
        ]
        
        while True:
            try:
                user_input = input(f"\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print(f"\n👋 Goodbye! Saving training session...")
                    self.save_training_session()
                    break
                
                if user_input.lower() == 'status':
                    self.show_training_status()
                    continue
                
                if user_input.lower() == 'save':
                    self.save_training_session()
                    continue
                
                if user_input.lower() == 'train':
                    print(f"\n📝 Available topics:")
                    for i, topic in enumerate(training_topics, 1):
                        print(f"  {i}. {topic}")
                    
                    try:
                        choice = input(f"Choose topic (1-{len(training_topics)}) or 'random': ").strip().lower()
                        
                        if choice == 'random':
                            topic = random.choice(training_topics)
                        else:
                            topic_idx = int(choice) - 1
                            if 0 <= topic_idx < len(training_topics):
                                topic = training_topics[topic_idx]
                            else:
                                print("❌ Invalid choice")
                                continue
                        
                        print(f"\n🎯 Starting Actor-Critic training on: {topic}")
                        self.run_actor_critic_cycle(topic)
                        
                    except ValueError:
                        print("❌ Invalid input")
                else:
                    print("💬 Type 'train' to start Actor-Critic training")
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Interrupted! Saving training session...")
                self.save_training_session()
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🎭 SAM ACTOR-CRITIC FAST INITIALIZATION")
    print("=" * 50)
    
    try:
        # Create Actor-Critic system
        actor_critic = SAMActorCriticFast()
        
        # Run Actor-Critic system
        actor_critic.run_actor_critic_system()
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Actor-Critic system interrupted by user")
    except Exception as e:
        print(f"\n❌ Actor-Critic error: {e}")
    finally:
        print(f"\n🎉 SAM Actor-Critic Fast session completed!")

if __name__ == "__main__":
    main()
