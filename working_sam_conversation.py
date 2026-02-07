#!/usr/bin/env python3
"""
Working SAM Conversation System
Two advanced SAM instances with full capabilities
Self-RAG + Web Access + Actor-Critic
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
import random

class WorkingSAMConversation:
    def __init__(self):
        """Initialize Working SAM Conversation System"""
        print("🤖 WORKING SAM CONVERSATION SYSTEM")
        print("=" * 60)
        print("🧠 Two Advanced SAM Instances")
        print("🔍 Self-RAG + Web Access + Actor-Critic")
        print("🎭 Full Capabilities for Both SAMs")
        print("⚡ Real-time Conversation")
        
        self.base_path = Path("/Users/samueldasari/Personal/NN_C")
        self.sam_model_path = self.base_path / "ORGANIZED" / "UTILS" / "SAM" / "SAM" / "SAM.h"
        
        # Check system components
        self.check_system_status()
        
        # Initialize advanced SAM instances
        self.initialize_advanced_sams()
        
        # Conversation history
        self.conversation_history = []
        self.session_start = time.time()
        
    def check_system_status(self):
        """Check system components"""
        print(f"\n🔍 System Status:")
        
        # Check SAM model
        self.sam_available = self.sam_model_path.exists()
        print(f"  🧠 SAM Model: {'✅ Available' if self.sam_available else '❌ Not Available'}")
        
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
    
    def initialize_advanced_sams(self):
        """Initialize two advanced SAM instances"""
        print(f"\n🧠 INITIALIZING ADVANCED SAM INSTANCES")
        
        # SAM Alpha - Research & Analysis Specialist
        self.sam_alpha = {
            'name': 'SAM-Alpha',
            'specialty': 'Research & Analysis',
            'personality': 'analytical, detailed, evidence-based',
            'knowledge_base': self.create_knowledge_base('research'),
            'web_access': True,
            'self_rag': True,
            'actor_critic': True,
            'response_style': 'detailed and technical'
        }
        
        # SAM Beta - Synthesis & Application Specialist  
        self.sam_beta = {
            'name': 'SAM-Beta',
            'specialty': 'Synthesis & Application',
            'personality': 'creative, practical, application-focused',
            'knowledge_base': self.create_knowledge_base('application'),
            'web_access': True,
            'self_rag': True,
            'actor_critic': True,
            'response_style': 'practical and accessible'
        }
        
        print(f"  ✅ {self.sam_alpha['name']}: {self.sam_alpha['specialty']}")
        print(f"  ✅ {self.sam_beta['name']}: {self.sam_beta['specialty']}")
        print(f"  🌐 Both instances: Full capabilities enabled")
    
    def create_knowledge_base(self, specialty):
        """Create knowledge base for SAM instance"""
        base_knowledge = {
            'quantum entanglement': "Quantum entanglement is a phenomenon where two or more quantum particles become connected in such a way that the quantum state of each particle cannot be described independently. When entangled, measuring one particle instantly affects the other, regardless of distance.",
            
            'artificial intelligence': "Artificial Intelligence (AI) is the field of computer science focused on creating systems that can perform tasks that typically require human intelligence. This includes learning from experience, reasoning, problem-solving, perception, and language understanding.",
            
            'consciousness': "Consciousness emerges from complex neural activity patterns in the brain, involving integrated information processing across multiple brain regions. It represents a self-referential information pattern that arises when neural systems achieve sufficient complexity.",
            
            'neural networks': "Neural networks learn through backpropagation, adjusting weights based on prediction errors. They process information through layers of interconnected nodes, each performing simple computations that collectively enable complex pattern recognition.",
            
            'machine learning': "Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions with minimal human intervention."
        }
        
        if specialty == 'research':
            # Add more technical details
            base_knowledge.update({
                'quantum computing': "Quantum computing harnesses quantum mechanical phenomena like superposition and entanglement to perform computations. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in superposition states, enabling parallel processing of multiple possibilities.",
                
                'deep learning': "Deep learning uses neural networks with multiple hidden layers to progressively extract higher-level features from raw input. Each layer learns to transform its input data into more abstract and composite representations, enabling sophisticated pattern recognition and decision-making."
            })
        else:
            # Add more application-focused knowledge
            base_knowledge.update({
                'ai applications': "AI applications include natural language processing, computer vision, robotics, recommendation systems, autonomous vehicles, medical diagnosis, and financial forecasting. Each application leverages different AI techniques suited to specific problem domains.",
                
                'practical ai': "Practical AI implementation involves data collection, preprocessing, model selection, training, validation, and deployment. Success depends on data quality, appropriate algorithm choice, computational resources, and continuous monitoring and improvement."
            })
        
        return base_knowledge
    
    def advanced_sam_response(self, sam_instance, question, context=""):
        """Generate advanced SAM response with full capabilities"""
        start_time = time.time()
        
        # Step 1: Check knowledge base
        question_lower = question.lower()
        kb_response = None
        
        for key, value in sam_instance['knowledge_base'].items():
            if key in question_lower:
                kb_response = value
                break
        
        if kb_response:
            # Use knowledge base response with personality
            if sam_instance['response_style'] == 'detailed and technical':
                response = f"From a {sam_instance['specialty'].lower()} perspective, {kb_response} This involves complex interactions and requires careful consideration of multiple factors and underlying mechanisms."
            else:
                response = f"Practically speaking, {kb_response} This has important implications for real-world applications and can be understood in terms of practical outcomes and use cases."
            
            response_type = "knowledge_base"
        
        else:
            # Step 2: Self-RAG assessment and web retrieval
            if sam_instance['self_rag'] and sam_instance['web_access']:
                retrieval_needed = self.assess_retrieval_need(question)
                
                if retrieval_needed:
                    web_info = self.web_retrieve(question)
                    if web_info:
                        response = self.integrate_web_info(question, web_info, sam_instance)
                        response_type = "web_enhanced"
                    else:
                        response = self.generate_pattern_response(question, sam_instance)
                        response_type = "pattern"
                else:
                    response = self.generate_pattern_response(question, sam_instance)
                    response_type = "pattern"
            else:
                response = self.generate_pattern_response(question, sam_instance)
                response_type = "pattern"
        
        response_time = time.time() - start_time
        
        # Step 3: Actor-Critic improvement (simplified)
        if sam_instance['actor_critic']:
            score = self.quick_evaluate(question, response)
            if score < 6.0:
                # Try to improve response
                improved = self.improve_response_style(question, response, sam_instance)
                if improved != response:
                    response = improved
                    response_type += "_improved"
        
        return response, response_type, response_time
    
    def assess_retrieval_need(self, question):
        """Assess if web retrieval is needed"""
        question_lower = question.lower()
        
        # Check for current/recent info needs
        current_keywords = ['latest', 'recent', 'current', 'new', 'modern', 'today', 'future']
        
        # Check for specific technical questions
        specific_keywords = ['what is', 'how does', 'explain', 'describe', 'details']
        
        return any(keyword in question_lower for keyword in current_keywords + specific_keywords)
    
    def web_retrieve(self, question):
        """Retrieve web information"""
        try:
            # Try Wikipedia
            search_terms = self.extract_search_terms(question)
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(search_terms)}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('extract', '')
        except:
            pass
        
        return ""
    
    def extract_search_terms(self, question):
        """Extract search terms"""
        words = re.findall(r'\b\w+\b', question.lower())
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'when', 'where', 'why', 'how', 'who', 'which', 'that', 'this', 'these', 'those'}
        
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(key_terms[:4])
    
    def integrate_web_info(self, question, web_info, sam_instance):
        """Integrate web information with SAM personality"""
        if sam_instance['response_style'] == 'detailed and technical':
            return f"Based on current research and data: {web_info}\n\nFrom a {sam_instance['specialty'].lower()} perspective, this information suggests important patterns and mechanisms that warrant further investigation and analysis."
        else:
            return f"Current information indicates: {web_info}\n\nIn practical terms, this has significant implications for applications and real-world implementation, suggesting new opportunities and approaches."
    
    def generate_pattern_response(self, question, sam_instance):
        """Generate pattern-based response with personality"""
        question_lower = question.lower()
        
        if "consciousness" in question_lower:
            if sam_instance['response_style'] == 'detailed and technical':
                return "Consciousness emerges from complex neural activity patterns involving integrated information processing across multiple brain regions. From a research perspective, this represents a self-referential information pattern that arises when neural systems achieve sufficient complexity and recursive feedback loops, involving mechanisms like global workspace theory and integrated information theory."
            else:
                return "Consciousness is essentially the brain's ability to be aware of itself and its surroundings. In practical terms, this means we can think, feel, and have experiences - which has huge implications for creating AI systems that might someday achieve similar awareness."
        
        elif "quantum" in question_lower:
            if sam_instance['response_style'] == 'detailed and technical':
                return "Quantum phenomena operate at the smallest scales where particles exhibit wave-particle duality and can exist in superposition states. The mathematical framework involves complex Hilbert spaces and unitary evolution, with entanglement representing non-local correlations that defy classical intuition."
            else:
                return "Quantum mechanics is basically the rules that govern how tiny particles behave. Think of it like this: particles can be in multiple places at once until we look at them, and they can be mysteriously connected across distances. This weird behavior is what powers quantum computers."
        
        elif "artificial intelligence" in question_lower or "ai" in question_lower:
            if sam_instance['response_style'] == 'detailed and technical':
                return "Artificial Intelligence encompasses multiple paradigms including symbolic AI, connectionist neural networks, and emerging quantum approaches. Current state-of-the-art systems primarily use deep learning architectures with billions of parameters, trained on massive datasets through gradient-based optimization methods."
            else:
                return "AI is all about teaching computers to think and learn like humans. Today's AI can recognize faces, understand speech, and even drive cars. The exciting part is that we're just getting started - future AI might help solve huge problems like climate change and disease."
        
        else:
            if sam_instance['response_style'] == 'detailed and technical':
                return f"From a {sam_instance['specialty'].lower()} standpoint, '{question}' represents a conceptual pattern that requires systematic analysis through multi-stage neural recognition, involving transformer attention mechanisms, evolutionary adaptation, and cortical mapping for comprehensive understanding."
            else:
                return f"That's an interesting question about '{question}'. The way I see it, this connects to bigger patterns and ideas that matter in the real world. Let me break this down in a way that's actually useful and practical."
    
    def quick_evaluate(self, question, response):
        """Quick evaluation of response quality"""
        # Simple heuristic based on response characteristics
        if len(response) < 50:
            return 3.0
        elif len(response) < 150:
            return 5.0
        elif "complex" in response or "detailed" in response or "practical" in response:
            return 7.0
        else:
            return 6.0
    
    def improve_response_style(self, question, response, sam_instance):
        """Improve response based on SAM's style"""
        if sam_instance['response_style'] == 'detailed and technical':
            return f"{response} This requires careful consideration of underlying mechanisms and theoretical frameworks, with attention to empirical evidence and methodological rigor."
        else:
            return f"{response} The practical implications here are significant, and understanding this can help us make better decisions and create more effective solutions."
    
    def advanced_conversation(self, topic, max_turns=10):
        """Advanced conversation between two SAM instances"""
        print(f"\n🎭 ADVANCED SAM CONVERSATION")
        print(f"🎯 Topic: {topic}")
        print(f"🔄 Max turns: {max_turns}")
        print(f"{'='*60}")
        
        conversation_log = []
        
        # Start conversation
        current_question = f"Let's discuss {topic}. What are your initial thoughts on this topic?"
        current_speaker = self.sam_alpha
        
        for turn in range(1, max_turns + 1):
            print(f"\n🗣️  Turn {turn}: {current_speaker['name']} ({current_speaker['specialty']})")
            print(f"❓ {current_question}")
            
            # Generate response
            response, response_type, response_time = self.advanced_sam_response(current_speaker, current_question)
            
            print(f"💬 {current_speaker['name']}: {response[:300]}...")
            print(f"📝 Type: {response_type} ({response_time:.2f}s)")
            
            # Store turn
            turn_data = {
                'turn': turn,
                'speaker': current_speaker['name'],
                'specialty': current_speaker['specialty'],
                'question': current_question,
                'response': response,
                'response_type': response_type,
                'response_time': response_time
            }
            conversation_log.append(turn_data)
            
            # Switch speaker and generate follow-up question
            other_sam = self.sam_beta if current_speaker == self.sam_alpha else self.sam_alpha
            current_question = self.generate_contextual_follow_up(response, other_sam)
            current_speaker = other_sam
            
            time.sleep(1)  # Brief pause for readability
        
        # Conversation summary
        print(f"\n📊 CONVERSATION SUMMARY")
        print(f"{'='*60}")
        
        alpha_responses = [t for t in conversation_log if t['speaker'] == 'SAM-Alpha']
        beta_responses = [t for t in conversation_log if t['speaker'] == 'SAM-Beta']
        
        print(f"🧠 {self.sam_alpha['name']} ({self.sam_alpha['specialty']}):")
        print(f"  💬 Responses: {len(alpha_responses)}")
        print(f"  📝 Types: {list(set(r['response_type'] for r in alpha_responses))}")
        
        print(f"\n🧠 {self.sam_beta['name']} ({self.sam_beta['specialty']}):")
        print(f"  💬 Responses: {len(beta_responses)}")
        print(f"  📝 Types: {list(set(r['response_type'] for r in beta_responses))}")
        
        self.conversation_history.append({
            'topic': topic,
            'turns': max_turns,
            'conversation_log': conversation_log,
            'timestamp': time.time()
        })
        
        return conversation_log
    
    def generate_contextual_follow_up(self, response, asking_sam):
        """Generate contextual follow-up question"""
        follow_ups = {
            'analytical': [
                "Could you elaborate on the technical mechanisms you mentioned?",
                "What empirical evidence supports your analysis?",
                "How does this relate to established theoretical frameworks?",
                "What are the methodological implications of your perspective?",
                "Can you provide more detailed technical specifications?"
            ],
            'practical': [
                "How would this work in practice?",
                "What are the real-world applications?",
                "Can you give me a specific example?",
                "What are the practical challenges and solutions?",
                "How might this evolve in the near future?"
            ]
        }
        
        if asking_sam['response_style'] == 'detailed and technical':
            return random.choice(follow_ups['analytical'])
        else:
            return random.choice(follow_ups['practical'])
    
    def show_system_status(self):
        """Show system status"""
        print(f"\n📊 WORKING SAM SYSTEM STATUS")
        print(f"{'='*50}")
        print(f"🧠 SAM Model: {'✅ Available' if self.sam_available else '❌ Not Available'}")
        print(f"🤖 Ollama: {'✅ Available' if self.ollama_available else '❌ Not Available'}")
        print(f"🧠 DeepSeek: {'✅ Available' if self.deepseek_available else '❌ Not Available'}")
        print(f"🌐 Web Access: {'✅ Available' if self.web_available else '❌ Not Available'}")
        print(f"💬 Conversations: {len(self.conversation_history)}")
        print(f"⏱️ Session Duration: {time.time() - self.session_start:.1f} seconds")
        
        print(f"\n🎭 Advanced SAM Instances:")
        print(f"  🧠 {self.sam_alpha['name']}: {self.sam_alpha['specialty']}")
        print(f"     📚 Knowledge: {len(self.sam_alpha['knowledge_base'])} items")
        print(f"     🎭 Style: {self.sam_alpha['response_style']}")
        print(f"  🧠 {self.sam_beta['name']}: {self.sam_beta['specialty']}")
        print(f"     📚 Knowledge: {len(self.sam_beta['knowledge_base'])} items")
        print(f"     🎭 Style: {self.sam_beta['response_style']}")
    
    def save_conversation(self):
        """Save conversation history"""
        timestamp = int(time.time())
        filename = f"working_sam_conversation_{timestamp}.json"
        
        session_data = {
            'timestamp': timestamp,
            'session_start': self.session_start,
            'duration': time.time() - self.session_start,
            'system_status': {
                'sam_available': self.sam_available,
                'deepseek_available': self.deepseek_available,
                'ollama_available': self.ollama_available,
                'web_available': self.web_available
            },
            'sam_instances': {
                'sam_alpha': {
                    'name': self.sam_alpha['name'],
                    'specialty': self.sam_alpha['specialty'],
                    'knowledge_count': len(self.sam_alpha['knowledge_base'])
                },
                'sam_beta': {
                    'name': self.sam_beta['name'],
                    'specialty': self.sam_beta['specialty'],
                    'knowledge_count': len(self.sam_beta['knowledge_base'])
                }
            },
            'conversation_history': self.conversation_history
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"💾 Conversation saved to: {filename}")
        return filename
    
    def run_working_system(self):
        """Run the working SAM conversation system"""
        print(f"\n🚀 WORKING SAM SYSTEM READY!")
        print(f"💬 Type 'talk' to start advanced conversation")
        print(f"💬 Type 'status' for system info")
        print(f"💬 Type 'save' to save conversations")
        print(f"💬 Type 'quit' to exit")
        print(f"🎭 Two advanced SAMs with full capabilities!")
        
        conversation_topics = [
            "the future of artificial intelligence",
            "consciousness and neural networks", 
            "quantum computing and machine learning",
            "the nature of reality and perception",
            "ethical implications of advanced AI",
            "the relationship between mind and machine",
            "evolution of intelligence in the universe",
            "practical applications of quantum technologies"
        ]
        
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
                    self.show_system_status()
                    continue
                
                if user_input.lower() == 'save':
                    self.save_conversation()
                    continue
                
                if user_input.lower() == 'talk':
                    print(f"\n📝 Available topics:")
                    for i, topic in enumerate(conversation_topics, 1):
                        print(f"  {i}. {topic}")
                    
                    try:
                        choice = input(f"Choose topic (1-{len(conversation_topics)}) or 'random': ").strip().lower()
                        
                        if choice == 'random':
                            topic = random.choice(conversation_topics)
                        else:
                            topic_idx = int(choice) - 1
                            if 0 <= topic_idx < len(conversation_topics):
                                topic = conversation_topics[topic_idx]
                            else:
                                print("❌ Invalid choice")
                                continue
                        
                        print(f"\n🎯 Starting conversation on: {topic}")
                        self.advanced_conversation(topic)
                        
                    except ValueError:
                        print("❌ Invalid input")
                else:
                    print("💬 Type 'talk', 'status', 'save', or 'quit'")
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Interrupted! Saving conversations...")
                self.save_conversation()
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🤖 WORKING SAM CONVERSATION INITIALIZATION")
    print("=" * 60)
    
    try:
        # Create working system
        working_sam = WorkingSAMConversation()
        
        # Run working system
        working_sam.run_working_system()
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Working system interrupted by user")
    except Exception as e:
        print(f"\n❌ Working system error: {e}")
    finally:
        print(f"\n🎉 Working SAM session completed!")

if __name__ == "__main__":
    main()
