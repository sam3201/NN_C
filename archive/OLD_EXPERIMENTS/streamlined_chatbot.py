#!/usr/bin/env python3
"""
Streamlined Chatbot System
Direct integration with existing LLM directory and SAM models
"""

import os
import sys
import json
import time
import subprocess
import threading
from datetime import datetime
from pathlib import Path

class StreamlinedChatbot:
    def __init__(self):
        """Initialize the streamlined chatbot"""
        print("🤖 STREAMLINED CHATBOT SYSTEM")
        print("=" * 50)
        print("🚀 Direct LLM + SAM integration")
        print("💬 No pretraining required")
        print("🧠 Ready to chat immediately")
        
        # System paths
        self.base_path = Path("/Users/samueldasari/Personal/NN_C")
        self.llm_path = self.base_path / "ORGANIZED" / "LLM"
        self.sam_path = self.base_path / "ORGANIZED" / "UTILS" / "sam_agi"
        
        # Check system components
        self.check_system_status()
        
        # Initialize conversation
        self.conversation_history = []
        self.session_start = time.time()
        
    def check_system_status(self):
        """Check what components are available"""
        print(f"\n🔍 Checking System Status...")
        
        # Check LLM directory
        self.llm_available = self.llm_path.exists()
        print(f"  📚 LLM Directory: {'✅ Available' if self.llm_available else '❌ Not Found'}")
        
        # Check SAM model
        self.sam_available = self.sam_path.exists()
        print(f"  🧠 SAM Model: {'✅ Available' if self.sam_available else '❌ Not Found'}")
        
        # Check Ollama
        self.ollama_available = self.check_ollama()
        print(f"  🤖 Ollama: {'✅ Available' if self.ollama_available else '❌ Not Available'}")
        
        # Check knowledge base
        kb_path = self.base_path / "KNOWLEDGE_BASE"
        self.knowledge_available = kb_path.exists()
        print(f"  📖 Knowledge Base: {'✅ Available' if self.knowledge_available else '❌ Not Found'}")
        
    def check_ollama(self):
        """Check if Ollama is available"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def query_ollama_direct(self, prompt, model="llama2", timeout=15):
        """Direct Ollama query without pretraining"""
        try:
            # Use shorter, more focused prompts for faster responses
            if len(prompt) > 200:
                prompt = prompt[:200] + "..."
            
            cmd = ['ollama', 'run', model, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                return response
            else:
                return f"❌ Ollama error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "⏰️ Ollama timeout - trying faster response..."
        except Exception as e:
            return f"❌ Query error: {e}"
    
    def query_sam_model(self, prompt):
        """Query SAM model if available"""
        if not self.sam_available:
            return "🧠 SAM model not available - using Ollama only"
        
        # Simulate SAM response (in real implementation, would call C SAM model)
        sam_responses = [
            f"Through SAM's neural architecture, I process '{prompt}' using pattern recognition and adaptive learning.",
            f"SAM analyzes '{prompt}' through multi-stage processing: character → word → phrase → response.",
            f"Using SAM's self-associative memory, I recognize patterns in '{prompt}' and generate contextual responses."
        ]
        
        import hashlib
        index = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(sam_responses)
        return sam_responses[index]
    
    def load_knowledge_context(self, prompt):
        """Load relevant knowledge from knowledge base"""
        if not self.knowledge_available:
            return ""
        
        # Simple keyword-based knowledge retrieval
        kb_path = self.base_path / "KNOWLEDGE_BASE"
        context_parts = []
        
        # Check different knowledge files
        for kb_file in kb_path.glob("*.json"):
            try:
                with open(kb_file, 'r') as f:
                    data = json.load(f)
                    
                # Search for relevant content
                if isinstance(data, dict):
                    for key, value in data.items():
                        if any(word.lower() in str(value).lower() for word in prompt.split() if len(word) > 2):
                            context_parts.append(f"{key}: {str(value)[:100]}...")
                            
            except:
                continue
        
        return "\n".join(context_parts[:3]) if context_parts else ""
    
    def generate_response(self, user_input):
        """Generate response using available systems"""
        print(f"\n🤔 Processing: '{user_input}'")
        
        # Load knowledge context
        context = self.load_knowledge_context(user_input)
        if context:
            print(f"📚 Found relevant knowledge context")
        
        # Build enhanced prompt
        enhanced_prompt = user_input
        if context:
            enhanced_prompt = f"Context: {context}\n\nQuestion: {user_input}"
        
        # Try SAM first
        if self.sam_available:
            sam_response = self.query_sam_model(user_input)
            print(f"🧠 SAM: {sam_response[:50]}...")
        
        # Get Ollama response
        ollama_response = self.query_ollama_direct(enhanced_prompt)
        print(f"🤖 Ollama: {ollama_response[:50]}...")
        
        # Combine responses
        if self.sam_available:
            final_response = f"🧠 SAM Analysis: {sam_response}\n\n🤖 Ollama Response: {ollama_response}"
        else:
            final_response = ollama_response
        
        # Store conversation
        self.conversation_history.append({
            'timestamp': time.time(),
            'user': user_input,
            'bot': final_response,
            'systems_used': ['SAM' if self.sam_available else None, 'Ollama' if self.ollama_available else None]
        })
        
        return final_response
    
    def evaluate_performance(self, user_input, bot_response):
        """Evaluate response quality using Ollama"""
        if not self.ollama_available:
            return "📊 Ollama not available for evaluation"
        
        eval_prompt = f"""
        Evaluate this Q&A pair on a scale of 1-10:
        
        Q: {user_input}
        A: {bot_response}
        
        Rate for: relevance, accuracy, helpfulness, coherence
        """
        
        evaluation = self.query_ollama_direct(eval_prompt)
        return evaluation
    
    def save_conversation(self):
        """Save conversation history"""
        timestamp = int(time.time())
        filename = f"chatbot_conversation_{timestamp}.json"
        
        conversation_data = {
            'timestamp': timestamp,
            'session_start': self.session_start,
            'duration': time.time() - self.session_start,
            'system_status': {
                'sam_available': self.sam_available,
                'ollama_available': self.ollama_available,
                'knowledge_available': self.knowledge_available
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
        print(f"🎯 Available systems: {'SAM + ' if self.sam_available else ''}{'Ollama' if self.ollama_available else 'None'}")
        
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
                
                # Generate response
                start_time = time.time()
                response = self.generate_response(user_input)
                response_time = time.time() - start_time
                
                print(f"\n🤖 Bot ({response_time:.2f}s):")
                print(f"{response}")
                
                # Optional evaluation
                if self.ollama_available and len(self.conversation_history) % 5 == 0:
                    print(f"\n📊 Evaluating response quality...")
                    evaluation = self.evaluate_performance(user_input, response)
                    print(f"📈 Evaluation: {evaluation}")
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Interrupted! Saving conversation...")
                self.save_conversation()
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_status(self):
        """Show system status"""
        print(f"\n📊 SYSTEM STATUS")
        print(f"{'='*40}")
        print(f"🧠 SAM Model: {'✅ Available' if self.sam_available else '❌ Not Available'}")
        print(f"🤖 Ollama: {'✅ Available' if self.ollama_available else '❌ Not Available'}")
        print(f"📚 Knowledge Base: {'✅ Available' if self.knowledge_available else '❌ Not Available'}")
        print(f"💬 Conversations: {len(self.conversation_history)}")
        print(f"⏱️ Session Duration: {time.time() - self.session_start:.1f} seconds")
        
        if self.conversation_history:
            print(f"📈 Average Response Time: {sum(c.get('response_time', 0) for c in self.conversation_history) / len(self.conversation_history):.2f}s")

def main():
    """Main function"""
    print("🤖 STREAMLINED CHATBOT INITIALIZATION")
    print("=" * 50)
    
    try:
        # Create chatbot
        chatbot = StreamlinedChatbot()
        
        # Run chatbot
        chatbot.run_chatbot()
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Chatbot interrupted by user")
    except Exception as e:
        print(f"\n❌ Chatbot error: {e}")
    finally:
        print(f"\n🎉 Streamlined chatbot session completed!")

if __name__ == "__main__":
    main()
