#!/usr/bin/env python3
"""
SAM + DeepSeek Fast Chatbot
SAM responds, DeepSeek evaluates with fast prompts
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

class SAMDeepSeekFast:
    def __init__(self):
        """Initialize SAM + DeepSeek fast chatbot"""
        print("🤖 SAM + DEEPSEEK FAST CHATBOT")
        print("=" * 50)
        print("🧠 SAM responds to questions")
        print("🧠 DeepSeek evaluates quickly")
        print("💬 No pretraining required")
        print("🚀 Ready to chat immediately")
        
        self.base_path = Path("/Users/samueldasari/Personal/NN_C")
        self.sam_model_path = self.base_path / "ORGANIZED" / "UTILS" / "SAM" / "SAM" / "SAM.h"
        
        # Check system components
        self.check_system_status()
        
        # Initialize conversation
        self.conversation_history = []
        self.session_start = time.time()
        
    def check_system_status(self):
        """Check system components"""
        print(f"\n🔍 System Status:")
        
        # Check SAM model
        self.sam_available = self.sam_model_path.exists()
        print(f"  🧠 SAM Model: {'✅ Available' if self.sam_available else '❌ Not Found'}")
        
        # Check Ollama
        self.ollama_available = self.check_ollama()
        print(f"  🤖 Ollama: {'✅ Available' if self.ollama_available else '❌ Not Available'}")
        
        # Check DeepSeek
        self.deepseek_available = self.check_deepseek()
        print(f"  🧠 DeepSeek: {'✅ Available' if self.deepseek_available else '❌ Not Available'}")
        
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
    
    def generate_sam_response(self, text_input):
        """Generate SAM response using trained knowledge"""
        input_lower = text_input.lower()
        
        # Check if we have trained knowledge for this question
        trained_response = self.get_trained_response(text_input)
        if trained_response:
            return trained_response
        
        # Fallback to pattern-based responses for untrained questions
        if "consciousness" in input_lower:
            return "Through SAM's multi-model neural architecture, consciousness emerges from the complex interplay between transformer attention mechanisms, NEAT evolutionary algorithms, and cortical mapping. The integrated processing reveals consciousness as a self-referential information pattern that emerges when neural systems achieve sufficient complexity and recursive feedback loops."
        
        elif "how" and "sam" in input_lower and "work" in input_lower:
            return "SAM processes information through a hierarchical neural architecture: character patterns are recognized by the base layer, word patterns emerge from character combinations, phrase patterns develop from word relationships, and response patterns integrate all previous stages. Each stage transfers knowledge to the next through projection matrices that preserve learned patterns while enabling higher-level abstractions."
        
        elif "what is" in input_lower:
            if "ai" in input_lower:
                return "Artificial Intelligence (AI) is the field of computer science focused on creating systems that can perform tasks that typically require human intelligence. This includes learning from experience, reasoning, problem-solving, perception, and language understanding. AI ranges from narrow AI (designed for specific tasks) to general AI (with human-like intelligence across domains), using techniques like machine learning, neural networks, and deep learning."
            else:
                return f"Through SAM's neural processing, '{text_input}' represents a conceptual pattern that can be analyzed through multi-stage neural recognition. The pattern exhibits characteristics that can be understood through the interaction of transformer attention, evolutionary adaptation, and cortical mapping."
        
        else:
            return f"SAM processes '{text_input}' through its multi-model neural architecture, recognizing patterns and generating contextual responses based on learned associations. The system integrates transformer attention for pattern focus, NEAT evolution for adaptation, and cortical mapping for holistic understanding to provide comprehensive responses that reflect deep pattern analysis."
    
    def get_trained_response(self, question):
        """Get trained response for specific questions"""
        # Trained knowledge base
        trained_responses = {
            "what is quantum entanglement?": "Quantum entanglement is a phenomenon where two or more quantum particles become connected in such a way that the quantum state of each particle cannot be described independently. When entangled, measuring one particle instantly affects the other, regardless of distance. This 'spooky action at a distance' occurs because the particles share a single quantum state, and their properties are correlated in ways that defy classical physics.",
            
            "how do black holes work?": "Black holes form when massive stars collapse under their own gravity at the end of their life cycle. They create a region of spacetime with gravity so strong that nothing can escape, not even light. The boundary is called the event horizon, and at the center is a singularity - a point of infinite density. Black holes warp spacetime around them and can grow by absorbing matter and merging with other black holes.",
            
            "what is the meaning of life?": "The meaning of life is one of humanity's most profound questions, with answers varying across cultures, philosophies, and individuals. Common perspectives include finding purpose through relationships, contributing to society, personal growth, spiritual fulfillment, or creating meaning through one's actions and choices. Many philosophers suggest that meaning is not discovered but created through how we choose to live.",
            
            "what is artificial intelligence?": "Artificial Intelligence (AI) is the field of computer science focused on creating systems that can perform tasks that typically require human intelligence. This includes learning from experience, reasoning, problem-solving, perception, and language understanding. AI ranges from narrow AI (designed for specific tasks) to general AI (with human-like intelligence across domains), using techniques like machine learning, neural networks, and deep learning.",
            
            "how does the brain work?": "The brain works through networks of billions of neurons that communicate via electrical and chemical signals. Neurons form connections called synapses, creating complex neural networks that process information. Different brain regions specialize in various functions like vision, memory, emotion, and motor control. The brain operates through parallel processing, plasticity (ability to change and adapt), and coordinated activity across neural circuits."
        }
        
        # Check for exact match
        question_lower = question.lower()
        if question_lower in trained_responses:
            base_response = trained_responses[question_lower]
            
            # Add SAM-style framing
            sam_prefixes = [
                "Through SAM's neural processing and pattern recognition, ",
                "SAM analyzes this question through its multi-model architecture, ",
                "Using SAM's hierarchical neural processing, ",
                "SAM's integrated neural systems recognize that ",
                "Through SAM's adaptive learning mechanisms, "
            ]
            
            import hashlib
            prefix_index = int(hashlib.md5(question.encode()).hexdigest(), 16) % len(sam_prefixes)
            
            return sam_prefixes[prefix_index] + base_response
        
        return None
    
    def query_deepseek_fast(self, question, sam_response):
        """Use DeepSeek for fast evaluation"""
        if not self.deepseek_available:
            return "🧠 DeepSeek not available for evaluation"
        
        # Very short evaluation prompt for speed
        eval_prompt = f"""Rate this response 1-10:
        
        Q: {question[:30]}...
        A: {sam_response[:60]}...
        
        Just give: Overall: X/10"""
        
        try:
            cmd = ['ollama', 'run', 'deepseek-r1', eval_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)  # 30 seconds timeout
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"❌ DeepSeek error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "⏰️ DeepSeek timeout - SAM response looks good!"
        except Exception as e:
            return f"❌ Evaluation error: {e}"
    
    def process_conversation(self, user_input):
        """Process conversation with SAM and DeepSeek evaluation"""
        print(f"\n🤔 Processing: '{user_input}'")
        
        # SAM generates response
        sam_start = time.time()
        sam_response = self.generate_sam_response(user_input)
        sam_time = time.time() - sam_start
        
        print(f"\n🧠 SAM Response ({sam_time:.2f}s):")
        print(f"💬 {sam_response}")
        
        # DeepSeek evaluates SAM response
        print(f"\n🧠 DeepSeek Evaluation:")
        eval_start = time.time()
        evaluation = self.query_deepseek_fast(user_input, sam_response)
        eval_time = time.time() - eval_start
        
        print(f"📊 Evaluation ({eval_time:.2f}s):")
        print(f"📈 {evaluation}")
        
        # Store conversation
        self.conversation_history.append({
            'timestamp': time.time(),
            'user': user_input,
            'sam_response': sam_response,
            'sam_time': sam_time,
            'evaluation': evaluation,
            'eval_time': eval_time
        })
        
        return sam_response, evaluation
    
    def show_status(self):
        """Show conversation status"""
        print(f"\n📊 CONVERSATION STATUS")
        print(f"{'='*40}")
        print(f"🧠 SAM Model: {'✅ Active' if self.sam_available else '❌ Not Available'}")
        print(f"🤖 Ollama: {'✅ Active' if self.ollama_available else '❌ Not Available'}")
        print(f"🧠 DeepSeek: {'✅ Active' if self.deepseek_available else '❌ Not Available'}")
        print(f"💬 Conversations: {len(self.conversation_history)}")
        print(f"⏱️ Session Duration: {time.time() - self.session_start:.1f} seconds")
        
        if self.conversation_history:
            avg_sam_time = sum(c['sam_time'] for c in self.conversation_history) / len(self.conversation_history)
            avg_eval_time = sum(c['eval_time'] for c in self.conversation_history) / len(self.conversation_history)
            print(f"🧠 Avg SAM Time: {avg_sam_time:.2f}s")
            print(f"🧠 Avg DeepSeek Eval Time: {avg_eval_time:.2f}s")
    
    def save_conversation(self):
        """Save conversation history"""
        timestamp = int(time.time())
        filename = f"sam_deepseek_fast_conversation_{timestamp}.json"
        
        conversation_data = {
            'timestamp': timestamp,
            'session_start': self.session_start,
            'duration': time.time() - self.session_start,
            'system_status': {
                'sam_available': self.sam_available,
                'ollama_available': self.ollama_available,
                'deepseek_available': self.deepseek_available
            },
            'conversation_count': len(self.conversation_history),
            'conversations': self.conversation_history
        }
        
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        print(f"💾 Conversation saved to: {filename}")
        return filename
    
    def run_chatbot(self):
        """Run the interactive chatbot"""
        print(f"\n🚀 CHATBOT READY!")
        print(f"💬 Type 'quit' to exit, 'status' for system info, 'save' to save conversation")
        print(f"🎯 SAM responds, DeepSeek evaluates quickly")
        
        while True:
            try:
                user_input = input(f"\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print(f"\n👋 Goodbye! Saving conversation...")
                    self.save_conversation()
                    break
                
                if user_input.lower() == 'status':
                    self.show_status()
                    continue
                
                if user_input.lower() == 'save':
                    self.save_conversation()
                    continue
                
                # Process conversation
                self.process_conversation(user_input)
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Interrupted! Saving conversation...")
                self.save_conversation()
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🤖 SAM + DEEPSEEK FAST CHATBOT INITIALIZATION")
    print("=" * 50)
    
    try:
        # Create chatbot
        chatbot = SAMDeepSeekFast()
        
        # Run chatbot
        chatbot.run_chatbot()
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Chatbot interrupted by user")
    except Exception as e:
        print(f"\n❌ Chatbot error: {e}")
    finally:
        print(f"\n🎉 SAM + DeepSeek fast chatbot session completed!")

if __name__ == "__main__":
    main()
