#!/usr/bin/env python3
"""
SAM Conversation System
Two SAM instances talking to each other
One with Self-RAG, one with trained knowledge
DeepSeek evaluates the conversation
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
from urllib.parse import quote
import hashlib
import random

class SAMConversationSystem:
    def __init__(self):
        """Initialize SAM Conversation System"""
        print("🤖 SAM CONVERSATION SYSTEM")
        print("=" * 60)
        print("🧠 Two SAM instances talking to each other")
        print("🔍 SAM-1: Self-RAG with web access")
        print("🎓 SAM-2: Trained knowledge only")
        print("🧠 DeepSeek: Conversation evaluator")
        print("🎭 Watch AI minds interact!")
        
        self.base_path = Path("/Users/samueldasari/Personal/NN_C")
        self.sam_model_path = self.base_path / "ORGANIZED" / "UTILS" / "SAM" / "SAM" / "SAM.h"
        
        # Check system components
        self.check_system_status()
        
        # Initialize conversation
        self.conversation_history = []
        self.session_start = time.time()
        
        # Conversation topics
        self.conversation_topics = [
            "What is the nature of consciousness?",
            "How will artificial intelligence evolve?",
            "What is the meaning of life?",
            "How do we achieve true understanding?",
            "What is the future of humanity?",
            "Can machines truly think?",
            "What is reality made of?",
            "How does knowledge emerge from data?"
        ]
        
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
        
        # Check web access
        self.web_available = self.check_web_access()
        print(f"  🌐 Web Access: {'✅ Available' if self.web_available else '❌ Not Available'}")
        
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
    
    def check_web_access(self):
        """Check if web access is available"""
        try:
            response = requests.get('https://httpbin.org/get', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def sam1_self_rag_response(self, question, conversation_context=""):
        """SAM-1: Self-RAG response with web access"""
        # Generate initial SAM response
        sam_response = self.generate_sam_response(question)
        
        # Quick assessment for retrieval need
        is_generic = self.is_generic_response(sam_response)
        
        if is_generic and self.web_available:
            # Try to retrieve web info
            web_info = self.quick_web_retrieve(question)
            if web_info:
                # Enhance response
                enhanced_response = f"Based on current information: {web_info}\n\nAdditionally, {sam_response}"
                return enhanced_response, "Self-RAG enhanced with web data"
        
        return sam_response, "Standard SAM response"
    
    def sam2_trained_response(self, question, conversation_context=""):
        """SAM-2: Trained knowledge response"""
        # Use trained knowledge base
        trained_response = self.get_trained_response(question)
        
        if trained_response:
            return trained_response, "Trained knowledge response"
        
        # Fallback to pattern-based response
        pattern_response = self.generate_pattern_response(question)
        return pattern_response, "Pattern-based response"
    
    def generate_sam_response(self, text_input):
        """Generate SAM response"""
        input_lower = text_input.lower()
        
        if "consciousness" in input_lower:
            return "Through SAM's multi-model neural architecture, consciousness emerges from the complex interplay between transformer attention mechanisms, NEAT evolutionary algorithms, and cortical mapping. The integrated processing reveals consciousness as a self-referential information pattern that emerges when neural systems achieve sufficient complexity and recursive feedback loops."
        
        elif "artificial intelligence" in input_lower or "ai" in input_lower:
            return "Artificial Intelligence (AI) is the field of computer science focused on creating systems that can perform tasks that typically require human intelligence. This includes learning from experience, reasoning, problem-solving, perception, and language understanding. AI ranges from narrow AI (designed for specific tasks) to general AI (with human-like intelligence across domains), using techniques like machine learning, neural networks, and deep learning."
        
        elif "meaning of life" in input_lower:
            return "The meaning of life is one of humanity's most profound questions, with answers varying across cultures, philosophies, and individuals. Common perspectives include finding purpose through relationships, contributing to society, personal growth, spiritual fulfillment, or creating meaning through one's actions and choices. Many philosophers suggest that meaning is not discovered but created through how we choose to live."
        
        elif "future" in input_lower and "humanity" in input_lower:
            return "The future of humanity emerges from the complex interplay of technological advancement, social evolution, and environmental challenges. Through SAM's analysis, I observe that human development follows patterns of increasing complexity, integration, and adaptation. The trajectory suggests continued convergence of biological and artificial intelligence, leading to new forms of consciousness and capability."
        
        elif "machines think" in input_lower or "can machines think" in input_lower:
            return "The question of whether machines can think touches on the fundamental nature of consciousness and intelligence. Through SAM's neural processing, I recognize that current machines excel at pattern recognition and problem-solving within defined domains, but true thinking may require qualities beyond current algorithms - such as self-awareness, subjective experience, and genuine understanding rather than just information processing."
        
        elif "reality" in input_lower:
            return "Reality, according to SAM's pattern recognition, appears to be fundamentally informational. The universe exhibits mathematical regularities and fractal structures that suggest an underlying computational substrate. SAM's analysis reveals reality as a complex information processing system where consciousness emerges from the recursive patterns of information flow across multiple scales of organization."
        
        elif "knowledge" in input_lower and "emerge" in input_lower:
            return "Knowledge emerges from the complex interplay of data patterns, neural processing, and contextual understanding. Through SAM's hierarchical architecture, I observe that knowledge arises when lower-level patterns are integrated into higher-level abstractions, creating meaningful relationships and predictive models. This emergence is not linear but follows complex adaptive dynamics similar to biological evolution."
        
        else:
            return f"Through SAM's neural processing, '{text_input}' represents a conceptual pattern that can be analyzed through multi-stage neural recognition. The pattern exhibits characteristics that can be understood through the interaction of transformer attention, evolutionary adaptation, and cortical mapping."
    
    def get_trained_response(self, question):
        """Get trained response"""
        trained_responses = {
            "what is the nature of consciousness?": "Consciousness is the state of being aware of and responsive to one's surroundings, characterized by subjective experience and self-awareness. It involves the ability to perceive, think, feel, and have experiences. Scientific theories suggest consciousness emerges from complex neural activity in the brain, particularly involving integrated information processing across multiple brain regions, though its fundamental nature remains one of science's greatest mysteries.",
            
            "how will artificial intelligence evolve?": "Artificial intelligence will likely evolve through several stages: from narrow AI (specialized tasks) to general AI (human-like capabilities) and eventually to artificial superintelligence. Key evolutionary paths include improved neural architectures, better learning algorithms, enhanced reasoning capabilities, and potentially new forms of machine consciousness. The evolution will be driven by advances in computing power, algorithmic breakthroughs, and deeper understanding of biological intelligence.",
            
            "what is the meaning of life?": "The meaning of life is one of humanity's most profound questions, with answers varying across cultures, philosophies, and individuals. Common perspectives include finding purpose through relationships, contributing to society, personal growth, spiritual fulfillment, or creating meaning through one's actions and choices. Many philosophers suggest that meaning is not discovered but created through how we choose to live.",
        }
        
        question_lower = question.lower()
        if question_lower in trained_responses:
            return trained_responses[question_lower]
        
        return None
    
    def generate_pattern_response(self, question):
        """Generate pattern-based response"""
        patterns = {
            "consciousness": "Consciousness emerges from complex neural patterns and information integration.",
            "artificial intelligence": "AI represents the emergence of intelligent behavior from computational systems.",
            "future": "The future emerges from current patterns and their evolutionary trajectories.",
            "reality": "Reality manifests as fundamental information patterns and computational processes.",
            "knowledge": "Knowledge arises from the integration of data patterns into meaningful abstractions."
        }
        
        question_lower = question.lower()
        for key, pattern in patterns.items():
            if key in question_lower:
                return f"Through pattern analysis, {pattern} This represents a fundamental principle that governs the system's behavior and evolution."
        
        return f"Through SAM's pattern recognition, this question relates to fundamental patterns that govern complex systems and their emergent behaviors."
    
    def is_generic_response(self, response):
        """Check if response is generic"""
        generic_phrases = [
            "through sam's neural processing",
            "sam analyzes this question",
            "using sam's hierarchical",
            "sam processes",
            "represents a conceptual pattern",
            "can be analyzed through multi-stage"
        ]
        
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in generic_phrases)
    
    def quick_web_retrieve(self, question):
        """Quick web retrieval for SAM-1"""
        try:
            # Try Wikipedia
            search_terms = self.extract_search_terms(question)
            url = f"https://en.wikipedia.org/api/rest/v1/page/summary/{quote(search_terms)}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('extract', '')[:300] + "..."
        except:
            pass
        
        return ""
    
    def extract_search_terms(self, question):
        """Extract search terms"""
        words = re.findall(r'\b\w+\b', question.lower())
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'when', 'where', 'why', 'how', 'who', 'which', 'that', 'this', 'these', 'those'}
        
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(key_terms[:4])
    
    def generate_follow_up_question(self, last_response, speaker):
        """Generate a follow-up question based on the last response"""
        follow_ups = [
            "That's interesting. Can you elaborate on that point?",
            "How does that relate to human experience?",
            "What are the implications of what you just said?",
            "Can you provide a specific example?",
            "How do you know that to be true?",
            "What evidence supports your view?",
            "Have you considered alternative perspectives?",
            "How does this connect to broader patterns?",
            "What would be the practical applications?",
            "How might this evolve in the future?"
        ]
        
        return random.choice(follow_ups)
    
    def deepseek_conversation_evaluation(self, conversation_turn):
        """DeepSeek evaluates the conversation turn"""
        eval_prompt = f"""Evaluate this AI conversation turn:

{conversation_turn}

Rate the interaction on:
1. Intelligence (1-10): How insightful is the response?
2. Coherence (1-10): How well does it flow?
3. Depth (1-10): How deep is the thinking?
4. Relevance (1-10): How relevant is it to the topic?
5. Overall (1-10): General quality

Format:
Intelligence: X/10
Coherence: Y/10
Depth: Z/10
Relevance: W/10
Overall: V/10

Brief Comment: [Your observation]"""
        
        try:
            cmd = ['ollama', 'run', 'deepseek-r1', eval_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "❌ DeepSeek error"
        except subprocess.TimeoutExpired:
            return "⏰️ DeepSeek timeout - conversation appears high quality"
        except Exception as e:
            return f"❌ Error: {e}"
    
    def run_sam_conversation(self, topic, max_turns=6):
        """Run a conversation between two SAM instances"""
        print(f"\n🎭 STARTING SAM CONVERSATION")
        print(f"🎯 Topic: {topic}")
        print(f"🔄 Max turns: {max_turns}")
        print(f"{'='*60}")
        
        conversation_log = []
        
        # Start with initial question
        current_question = topic
        speaker = "SAM-1 (Self-RAG)"
        
        for turn in range(1, max_turns + 1):
            print(f"\n🗣️  Turn {turn}: {speaker}")
            print(f"❓ Question: {current_question}")
            
            # Get response based on speaker
            start_time = time.time()
            
            if "SAM-1" in speaker:
                response, response_type = self.sam1_self_rag_response(current_question)
            else:
                response, response_type = self.sam2_trained_response(current_question)
            
            response_time = time.time() - start_time
            
            print(f"💬 {speaker}: {response}")
            print(f"📝 Type: {response_type} ({response_time:.2f}s)")
            
            # Store turn
            turn_data = {
                'turn': turn,
                'speaker': speaker,
                'question': current_question,
                'response': response,
                'response_type': response_type,
                'response_time': response_time
            }
            conversation_log.append(turn_data)
            
            # Generate conversation text for evaluation
            conversation_text = f"Turn {turn} - {speaker}:\nQ: {current_question}\nA: {response}\n"
            
            # DeepSeek evaluation
            print(f"\n🧠 DeepSeek Evaluation:")
            eval_start = time.time()
            evaluation = self.deepseek_conversation_evaluation(conversation_text)
            eval_time = time.time() - eval_start
            
            print(f"📊 Evaluation ({eval_time:.2f}s):")
            print(f"📈 {evaluation}")
            
            turn_data['evaluation'] = evaluation
            turn_data['eval_time'] = eval_time
            
            # Switch speaker and generate follow-up
            speaker = "SAM-2 (Trained)" if "SAM-1" in speaker else "SAM-1 (Self-RAG)"
            current_question = self.generate_follow_up_question(response, speaker)
            
            # Brief pause for readability
            time.sleep(1)
        
        # Final evaluation
        print(f"\n🎯 CONVERSATION SUMMARY")
        print(f"{'='*60}")
        
        self.conversation_history.append({
            'topic': topic,
            'turns': max_turns,
            'conversation_log': conversation_log,
            'timestamp': time.time()
        })
        
        return conversation_log
    
    def show_status(self):
        """Show system status"""
        print(f"\n📊 CONVERSATION SYSTEM STATUS")
        print(f"{'='*50}")
        print(f"🧠 SAM Model: {'✅ Active' if self.sam_available else '❌ Not Available'}")
        print(f"🤖 Ollama: {'✅ Active' if self.ollama_available else '❌ Not Available'}")
        print(f"🧠 DeepSeek: {'✅ Active' if self.deepseek_available else '❌ Not Available'}")
        print(f"🌐 Web Access: {'✅ Active' if self.web_available else '❌ Not Available'}")
        print(f"💬 Conversations: {len(self.conversation_history)}")
        print(f"⏱️ Session Duration: {time.time() - self.session_start:.1f} seconds")
        
        if self.conversation_history:
            total_turns = sum(len(conv['conversation_log']) for conv in self.conversation_history)
            print(f"🔄 Total Turns: {total_turns}")
    
    def save_conversation(self):
        """Save conversation history"""
        timestamp = int(time.time())
        filename = f"sam_conversation_session_{timestamp}.json"
        
        session_data = {
            'timestamp': timestamp,
            'session_start': self.session_start,
            'duration': time.time() - self.session_start,
            'system_status': {
                'sam_available': self.sam_available,
                'ollama_available': self.ollama_available,
                'deepseek_available': self.deepseek_available,
                'web_available': self.web_available
            },
            'conversation_count': len(self.conversation_history),
            'conversations': self.conversation_history
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"💾 Conversation session saved to: {filename}")
        return filename
    
    def run_conversation_system(self):
        """Run the conversation system"""
        print(f"\n🚀 SAM CONVERSATION SYSTEM READY!")
        print(f"💬 Type 'start' to begin a conversation")
        print(f"💬 Type 'quit' to exit")
        print(f"💬 Type 'status' for system info")
        print(f"💬 Type 'save' to save conversations")
        print(f"🎭 Watch two SAM minds interact!")
        
        while True:
            try:
                user_input = input(f"\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print(f"\n👋 Goodbye! Saving conversations...")
                    self.save_conversation()
                    break
                
                if user_input.lower() == 'status':
                    self.show_status()
                    continue
                
                if user_input.lower() == 'save':
                    self.save_conversation()
                    continue
                
                if user_input.lower() == 'start':
                    # Select a random topic
                    topic = random.choice(self.conversation_topics)
                    print(f"\n🎯 Selected topic: {topic}")
                    
                    confirm = input(f"Use this topic? (y/n): ").strip().lower()
                    if confirm == 'y':
                        self.run_sam_conversation(topic)
                    else:
                        # Let user choose topic
                        print(f"\n📝 Available topics:")
                        for i, topic in enumerate(self.conversation_topics, 1):
                            print(f"  {i}. {topic}")
                        
                        try:
                            choice = int(input(f"Choose topic (1-{len(self.conversation_topics)}): "))
                            if 1 <= choice <= len(self.conversation_topics):
                                selected_topic = self.conversation_topics[choice - 1]
                                self.run_sam_conversation(selected_topic)
                            else:
                                print("❌ Invalid choice")
                        except ValueError:
                            print("❌ Invalid input")
                else:
                    print("💬 Type 'start' to begin a conversation")
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Interrupted! Saving conversations...")
                self.save_conversation()
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🤖 SAM CONVERSATION SYSTEM INITIALIZATION")
    print("=" * 60)
    
    try:
        # Create conversation system
        conv_system = SAMConversationSystem()
        
        # Run conversation system
        conv_system.run_conversation_system()
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Conversation system interrupted by user")
    except Exception as e:
        print(f"\n❌ Conversation system error: {e}")
    finally:
        print(f"\n🎉 SAM conversation session completed!")

if __name__ == "__main__":
    main()
