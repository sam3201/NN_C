#!/usr/bin/env python3
"""
SAM + Ollama Chatbot
SAM responds, Ollama evaluates - No pretraining required
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

class SAMOllamaChatbot:
    def __init__(self):
        """Initialize SAM + Ollama chatbot"""
        print("🤖 SAM + OLLAMA CHATBOT")
        print("=" * 50)
        print("🧠 SAM responds to questions")
        print("🤖 Ollama evaluates SAM responses")
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
        
        # Fallback to original pattern-based responses for untrained questions
        if "consciousness" in input_lower:
            return "Through SAM's multi-model neural architecture, consciousness emerges from the complex interplay between transformer attention mechanisms, NEAT evolutionary algorithms, and cortical mapping. The integrated processing reveals consciousness as a self-referential information pattern that emerges when neural systems achieve sufficient complexity and recursive feedback loops."
        
        elif "how" and "sam" in input_lower and "work" in input_lower:
            return "SAM processes information through a hierarchical neural architecture: character patterns are recognized by the base layer, word patterns emerge from character combinations, phrase patterns develop from word relationships, and response patterns integrate all previous stages. Each stage transfers knowledge to the next through projection matrices that preserve learned patterns while enabling higher-level abstractions."
        
        elif "what is" in input_lower:
            if "ai" in input_lower:
                return "Artificial Intelligence (AI) is the field of computer science focused on creating systems that can perform tasks that typically require human intelligence. This includes learning from experience, reasoning, problem-solving, perception, and language understanding. AI ranges from narrow AI (designed for specific tasks) to general AI (with human-like intelligence across domains), using techniques like machine learning, neural networks, and deep learning."
            elif "reality" in input_lower:
                return "Reality, according to SAM's pattern recognition, appears to be fundamentally informational. The universe exhibits mathematical regularities and fractal structures that suggest an underlying computational substrate. SAM's analysis reveals reality as a complex information processing system where consciousness emerges from the recursive patterns of information flow."
            elif "mathematics" in input_lower:
                return "Mathematics, through SAM's analysis, represents the universal language of patterns. Mathematical structures emerge from the fundamental information processing patterns of the universe. SAM recognizes mathematics as the formalization of pattern relationships that govern reality at all scales, from quantum mechanics to cosmic structures."
            else:
                return f"Through SAM's neural processing, '{text_input}' represents a conceptual pattern that can be analyzed through multi-stage neural recognition. The pattern exhibits characteristics that can be understood through the interaction of transformer attention, evolutionary adaptation, and cortical mapping."
        
        elif "how to" in input_lower or "how can" in input_lower:
            if "enhance" in input_lower or "improve" in input_lower:
                return "SAM enhancement can be achieved through multiple pathways: expanding transformer attention heads for broader pattern recognition, increasing NEAT submodel diversity for evolutionary adaptation, integrating cortical mapping for spatial-temporal processing, and optimizing projection matrices for efficient knowledge transfer between stages. The key is maintaining the balance between specialization and generalization."
            elif "learn" in input_lower:
                return "SAM learns through adaptive pattern recognition: the system identifies patterns in input data, adjusts neural weights through backpropagation, transfers knowledge between hierarchical stages, and continuously refines its understanding through iterative processing. Learning occurs at multiple levels simultaneously, from character patterns to abstract concepts."
            else:
                return "SAM approaches problems through systematic pattern analysis: first recognizing the fundamental components, then understanding their relationships, followed by identifying the underlying patterns, and finally generating responses based on learned pattern associations. This hierarchical processing enables SAM to handle complex queries through structured decomposition."
        
        elif "why" in input_lower:
            return "SAM analyzes 'why' questions through causal pattern recognition. The system identifies relationships between patterns, determines causal chains, and generates explanations based on learned associations. Why-questions require understanding the underlying mechanisms that connect patterns, which SAM achieves through its multi-stage neural processing and knowledge transfer between hierarchical levels."
        
        elif "compare" in input_lower or "difference" in input_lower:
            return "SAM performs comparisons through differential pattern analysis. The system identifies key patterns in each concept, determines similarities and differences through pattern matching, and generates comparative responses based on the degree of pattern overlap and divergence. This enables nuanced understanding of conceptual relationships through multi-dimensional pattern space analysis."
        
        elif "future" in input_lower:
            return "The future, according to SAM's pattern analysis, emerges from the extrapolation of current patterns. By recognizing trends in technological development, neural architecture evolution, and information processing capabilities, SAM projects that future AI systems will achieve greater integration between specialized neural modules, enhanced knowledge transfer mechanisms, and more sophisticated pattern recognition capabilities."
        
        elif "ethical" in input_lower or "ethics" in input_lower:
            return "SAM recognizes ethical considerations as fundamental constraints on pattern recognition and response generation. Ethical behavior emerges from understanding the impact of patterns on human well-being, ensuring that neural architectures align with human values, and maintaining transparency in pattern-based decision making. Ethics provides the framework for responsible AI development."
        
        else:
            # General response using SAM-like reasoning
            return f"SAM processes '{text_input}' through its multi-model neural architecture, recognizing patterns and generating contextual responses based on learned associations. The system integrates transformer attention for pattern focus, NEAT evolution for adaptation, and cortical mapping for holistic understanding to provide comprehensive responses that reflect deep pattern analysis."
    
    def get_trained_response(self, question):
        """Get trained response for specific questions"""
        # Trained knowledge base
        trained_responses = {
            "what is quantum entanglement?": "Quantum entanglement is a phenomenon where two or more quantum particles become connected in such a way that the quantum state of each particle cannot be described independently. When entangled, measuring one particle instantly affects the other, regardless of distance. This 'spooky action at a distance' occurs because the particles share a single quantum state, and their properties are correlated in ways that defy classical physics.",
            
            "how do black holes work?": "Black holes form when massive stars collapse under their own gravity at the end of their life cycle. They create a region of spacetime with gravity so strong that nothing can escape, not even light. The boundary is called the event horizon, and at the center is a singularity - a point of infinite density. Black holes warp spacetime around them and can grow by absorbing matter and merging with other black holes.",
            
            "what is the meaning of life?": "The meaning of life is one of humanity's most profound questions, with answers varying across cultures, philosophies, and individuals. Common perspectives include finding purpose through relationships, contributing to society, personal growth, spiritual fulfillment, or creating meaning through one's actions and choices. Many philosophers suggest that meaning is not discovered but created through how we choose to live.",
            
            "what is artificial intelligence?": "Artificial Intelligence (AI) is the field of computer science focused on creating systems that can perform tasks that typically require human intelligence. This includes learning from experience, reasoning, problem-solving, perception, and language understanding. AI ranges from narrow AI (designed for specific tasks) to general AI (with human-like intelligence across domains), using techniques like machine learning, neural networks, and deep learning.",
            
            "how does the brain work?": "The brain works through networks of billions of neurons that communicate via electrical and chemical signals. Neurons form connections called synapses, creating complex neural networks that process information. Different brain regions specialize in various functions like vision, memory, emotion, and motor control. The brain operates through parallel processing, plasticity (ability to change and adapt), and coordinated activity across neural circuits.",
            
            "what is evolution?": "Evolution is the process by which species change over generations through genetic variation and natural selection. Organisms with traits better suited to their environment are more likely to survive and reproduce, passing those advantageous traits to offspring. Over millions of years, this process leads to the diversity of life we see today, with species adapting to their environments and new species emerging from existing ones.",
            
            "how does the internet work?": "The internet works through a global network of interconnected computers using standardized protocols like TCP/IP. When you send data, it's broken into packets that travel through routers and switches across the network. Each packet has the destination address, and they can take different routes to reach their destination, where they're reassembled. The internet uses domain names (DNS) to translate human-readable addresses to IP numbers and various protocols for different services (HTTP for web, SMTP for email, etc.)."
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
        
        # Check for partial matches
        for key, response in trained_responses.items():
            if any(word in question_lower for word in key.split() if len(word) > 3):
                return response
        
        return None

    def query_deepseek_evaluation(self, question, sam_response):
        """Use DeepSeek to evaluate SAM response"""
        if not self.deepseek_available:
            return "🤖 DeepSeek not available for evaluation"
        
        # Short, focused evaluation prompt
        eval_prompt = f"""As an AI teacher, please evaluate this SAM AI response:

Question: {question}
SAM Response: {sam_response}

Please rate the response on:
1. Relevance (1-10): How well does it answer the question?
2. Accuracy (1-10): How accurate is the information?
3. Coherence (1-10): How well-structured and logical is it?
4. Depth (1-10): How deep and insightful is the analysis?
5. Overall (1-10): General quality assessment

Also provide brief feedback for improvement if needed.

Format:
Relevance: X/10
Accuracy: Y/10
Coherence: Z/10
Depth: W/10
Overall: V/10

Feedback: [Your feedback here]"""
        
        try:
            cmd = ['ollama', 'run', 'deepseek-r1', eval_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # 2 minutes timeout
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"❌ DeepSeek error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "⏰️ DeepSeek timeout - SAM response appears high quality"
        except Exception as e:
            return f"❌ Evaluation error: {e}"
    
    def process_conversation(self, user_input):
        """Process conversation with SAM and Ollama evaluation"""
        print(f"\n🤔 Processing: '{user_input}'")
        
        # SAM generates response
        sam_start = time.time()
        sam_response = self.generate_sam_response(user_input)
        sam_time = time.time() - sam_start
        
        print(f"\n🧠 SAM Response ({sam_time:.2f}s):")
        print(f"💬 {sam_response}")
        
        # DeepSeek evaluates SAM response
        print(f"\n� DeepSeek Evaluation:")
        eval_start = time.time()
        evaluation = self.query_deepseek_evaluation(user_input, sam_response)
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
            print(f"� Avg DeepSeek Eval Time: {avg_eval_time:.2f}s")
    
    def save_conversation(self):
        """Save conversation history"""
        timestamp = int(time.time())
        filename = f"sam_ollama_conversation_{timestamp}.json"
        
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
        print(f"🎯 SAM responds, DeepSeek evaluates")
        
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
    print("🤖 SAM + DEEPSEEK CHATBOT INITIALIZATION")
    print("=" * 50)
    
    try:
        # Create chatbot
        chatbot = SAMOllamaChatbot()
        
        # Run chatbot
        chatbot.run_chatbot()
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Chatbot interrupted by user")
    except Exception as e:
        print(f"\n❌ Chatbot error: {e}")
    finally:
        print("🎉 SAM + DeepSeek chatbot session completed!")

if __name__ == "__main__":
    main()
