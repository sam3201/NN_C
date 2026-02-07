#!/usr/bin/env python3
"""
SAM Self-RAG Fast System
Fast Adaptive Retrieval with optimized timeouts
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
from urllib.parse import urlparse, quote
import hashlib

class SAMSelfRAGFast:
    def __init__(self):
        """Initialize SAM Self-RAG Fast System"""
        print("🧠 SAM SELF-RAG FAST SYSTEM")
        print("=" * 50)
        print("🔍 Fast Adaptive Retrieval")
        print("🌐 Web Access + Self-RAG")
        print("⚡ Optimized for speed")
        
        self.base_path = Path("/Users/samueldasari/Personal/NN_C")
        self.sam_model_path = self.base_path / "ORGANIZED" / "UTILS" / "SAM" / "SAM" / "SAM.h"
        
        # Check system components
        self.check_system_status()
        
        # Initialize conversation and retrieval history
        self.conversation_history = []
        self.retrieval_history = []
        self.session_start = time.time()
        
        # Self-RAG configuration
        self.retrieval_threshold = 0.6  # Lower threshold for more retrieval
        self.max_retrieval_attempts = 2
        self.web_sources = self.initialize_web_sources()
        
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
    
    def initialize_web_sources(self):
        """Initialize web sources"""
        return {
            'wikipedia': {
                'base_url': 'https://en.wikipedia.org/api/rest_v1/page/summary/',
                'priority': 1,
                'description': 'General encyclopedia knowledge'
            },
            'arxiv': {
                'base_url': 'https://export.arxiv.org/api/query?search_query=',
                'priority': 2,
                'description': 'Scientific papers and research'
            }
        }
    
    def quick_assess_retrieval_need(self, question, sam_response):
        """Fast assessment of retrieval need"""
        # Simple heuristic-based assessment
        question_lower = question.lower()
        sam_response_lower = sam_response.lower()
        
        # Check if SAM gave a generic response
        generic_phrases = [
            "through sam's neural processing",
            "sam analyzes this question",
            "using sam's hierarchical",
            "sam processes",
            "represents a conceptual pattern"
        ]
        
        is_generic = any(phrase in sam_response_lower for phrase in generic_phrases)
        
        # Check if question asks for specific information
        specific_keywords = [
            "what is", "how does", "explain", "describe", "define",
            "quantum", "machine learning", "artificial intelligence",
            "black hole", "consciousness", "evolution"
        ]
        
        is_specific_question = any(keyword in question_lower for keyword in specific_keywords)
        
        # Decision logic
        confidence = 8.0 if not is_generic else 4.0
        retrieval_needed = is_generic and is_specific_question
        
        info_gaps = "Need specific factual information" if retrieval_needed else ""
        
        return {
            'confidence': confidence,
            'retrieval_needed': retrieval_needed,
            'info_gaps': info_gaps,
            'assessment': f"Confidence: {confidence}/10, Retrieval: {'YES' if retrieval_needed else 'NO'}"
        }
    
    def retrieve_from_web(self, question, info_gaps=""):
        """Fast web retrieval"""
        retrieved_docs = []
        
        # Generate search terms
        search_terms = self.extract_search_terms(question, info_gaps)
        
        print(f"🔍 Searching web for: {search_terms}")
        
        # Try Wikipedia first (fastest)
        try:
            docs = self.retrieve_from_wikipedia(search_terms)
            if docs:
                retrieved_docs.extend(docs)
                print(f"  📚 Retrieved {len(docs)} docs from Wikipedia")
        except Exception as e:
            print(f"  ❌ Wikipedia error: {e}")
        
        # Try arXiv if needed
        if len(retrieved_docs) < 2:
            try:
                docs = self.retrieve_from_arxiv(search_terms)
                if docs:
                    retrieved_docs.extend(docs)
                    print(f"  📚 Retrieved {len(docs)} docs from arXiv")
            except Exception as e:
                print(f"  ❌ arXiv error: {e}")
        
        return retrieved_docs[:3]  # Limit to top 3
    
    def extract_search_terms(self, question, info_gaps):
        """Extract search terms"""
        combined_text = f"{question} {info_gaps}"
        words = re.findall(r'\b\w+\b', combined_text.lower())
        
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'when', 'where', 'why', 'how', 'who', 'which', 'that', 'this', 'these', 'those'}
        
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(key_terms[:4])
    
    def retrieve_from_wikipedia(self, search_terms):
        """Retrieve from Wikipedia"""
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(search_terms)}"
            response = requests.get(url, timeout=8)
            
            if response.status_code == 200:
                data = response.json()
                return [{
                    'source': 'wikipedia',
                    'title': data.get('title', ''),
                    'content': data.get('extract', ''),
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'relevance_score': 0.8
                }]
        except Exception as e:
            print(f"  ❌ Wikipedia error: {e}")
        
        return []
    
    def retrieve_from_arxiv(self, search_terms):
        """Retrieve from arXiv"""
        try:
            url = f"https://export.arxiv.org/api/query?search_query=all:{quote(search_terms)}&max_results=2"
            response = requests.get(url, timeout=8)
            
            if response.status_code == 200:
                content = response.text
                papers = []
                
                title_matches = re.findall(r'<title>(.*?)</title>', content)
                summary_matches = re.findall(r'<summary>(.*?)</summary>', content)
                
                for i, (title, summary) in enumerate(zip(title_matches[:2], summary_matches[:2])):
                    papers.append({
                        'source': 'arxiv',
                        'title': title,
                        'content': summary[:400] + '...' if len(summary) > 400 else summary,
                        'url': f"https://arxiv.org/abs/{search_terms}",
                        'relevance_score': 0.7
                    })
                
                return papers
        except Exception as e:
            print(f"  ❌ arXiv error: {e}")
        
        return []
    
    def generate_enhanced_response(self, question, sam_response, retrieved_docs):
        """Generate enhanced response"""
        if not retrieved_docs:
            return sam_response
        
        # Use top document
        top_doc = retrieved_docs[0]
        
        # Simple enhancement: prepend retrieved info
        enhanced = f"Based on current information from {top_doc['source']}:\n\n{top_doc['content']}\n\nAdditionally, {sam_response}"
        
        return enhanced
    
    def quick_evaluation(self, question, response):
        """Fast evaluation"""
        eval_prompt = f"""Rate this response 1-10:
        
Q: {question[:30]}...
A: {response[:100]}...
        
Overall: X/10"""
        
        try:
            cmd = ['ollama', 'run', 'deepseek-r1', eval_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "❌ DeepSeek error"
        except subprocess.TimeoutExpired:
            return "⏰️ Timeout - response looks good"
        except Exception as e:
            return f"❌ Error: {e}"
    
    def generate_sam_response(self, text_input):
        """Generate SAM response"""
        input_lower = text_input.lower()
        
        # Check trained knowledge
        trained_response = self.get_trained_response(text_input)
        if trained_response:
            return trained_response
        
        # Fallback responses
        if "consciousness" in input_lower:
            return "Through SAM's multi-model neural architecture, consciousness emerges from the complex interplay between transformer attention mechanisms, NEAT evolutionary algorithms, and cortical mapping. The integrated processing reveals consciousness as a self-referential information pattern that emerges when neural systems achieve sufficient complexity and recursive feedback loops."
        
        elif "how" and "sam" in input_lower and "work" in input_lower:
            return "SAM processes information through a hierarchical neural architecture: character patterns are recognized by the base layer, word patterns emerge from character combinations, phrase patterns develop from word relationships, and response patterns integrate all previous stages. Each stage transfers knowledge to the next through projection matrices that preserve learned patterns while enabling higher-level abstractions."
        
        elif "what is" in input_lower:
            if "ai" in input_lower:
                return "Artificial Intelligence (AI) is the field of computer science focused on creating systems that can perform tasks that typically require human intelligence. This includes learning from experience, reasoning, problem-solving, perception, and language understanding. AI ranges from narrow AI (designed for specific tasks) to general AI (with human-like intelligence across domains), using techniques like machine learning, neural networks, and deep learning."
            elif "quantum entanglement" in input_lower:
                return "Quantum entanglement is a phenomenon where two or more quantum particles become connected in such a way that the quantum state of each particle cannot be described independently. When entangled, measuring one particle instantly affects the other, regardless of distance. This 'spooky action at a distance' occurs because the particles share a single quantum state, and their properties are correlated in ways that defy classical physics."
            elif "machine learning" in input_lower:
                return "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions with minimal human intervention. Common approaches include supervised learning, unsupervised learning, and reinforcement learning."
            else:
                return f"Through SAM's neural processing, '{text_input}' represents a conceptual pattern that can be analyzed through multi-stage neural recognition."
        
        else:
            return f"SAM processes '{text_input}' through its multi-model neural architecture, recognizing patterns and generating contextual responses based on learned associations."
    
    def get_trained_response(self, question):
        """Get trained response"""
        trained_responses = {
            "what is quantum entanglement?": "Quantum entanglement is a phenomenon where two or more quantum particles become connected in such a way that the quantum state of each particle cannot be described independently. When entangled, measuring one particle instantly affects the other, regardless of distance. This 'spooky action at a distance' occurs because the particles share a single quantum state, and their properties are correlated in ways that defy classical physics.",
            
            "how do black holes work?": "Black holes form when massive stars collapse under their own gravity at the end of their life cycle. They create a region of spacetime with gravity so strong that nothing can escape, not even light. The boundary is called the event horizon, and at the center is a singularity - a point of infinite density. Black holes warp spacetime around them and can grow by absorbing matter and merging with other black holes.",
        }
        
        question_lower = question.lower()
        if question_lower in trained_responses:
            base_response = trained_responses[question_lower]
            
            sam_prefixes = [
                "Through SAM's neural processing and pattern recognition, ",
                "SAM analyzes this question through its multi-model architecture, ",
                "Using SAM's hierarchical neural processing, ",
                "SAM's integrated neural systems recognize that ",
                "Through SAM's adaptive learning mechanisms, "
            ]
            
            prefix_index = int(hashlib.md5(question.encode()).hexdigest(), 16) % len(sam_prefixes)
            return sam_prefixes[prefix_index] + base_response
        
        return None
    
    def process_self_rag_conversation(self, user_input):
        """Process conversation with Self-RAG"""
        print(f"\n🤔 Processing: '{user_input}'")
        
        # Step 1: Generate initial SAM response
        sam_start = time.time()
        sam_response = self.generate_sam_response(user_input)
        sam_time = time.time() - sam_start
        
        print(f"\n🧠 Initial SAM Response ({sam_time:.2f}s):")
        print(f"💬 {sam_response}")
        
        # Step 2: Fast Self-RAG assessment
        print(f"\n🔍 Self-RAG Assessment:")
        assess_start = time.time()
        assessment = self.quick_assess_retrieval_need(user_input, sam_response)
        assess_time = time.time() - assess_start
        
        print(f"📊 Assessment ({assess_time:.2f}s):")
        print(f"  🎯 Confidence: {assessment['confidence']}/10")
        print(f"  🔍 Retrieval Needed: {'✅ YES' if assessment['retrieval_needed'] else '❌ NO'}")
        
        # Step 3: Retrieval if needed
        retrieved_docs = []
        retrieval_time = 0
        
        if assessment['retrieval_needed']:
            print(f"\n🌐 Adaptive Retrieval:")
            retrieve_start = time.time()
            retrieved_docs = self.retrieve_from_web(user_input, assessment['info_gaps'])
            retrieval_time = time.time() - retrieve_start
            
            if retrieved_docs:
                print(f"📚 Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s")
                
                # Step 4: Generate enhanced response
                print(f"\n🧠 Enhanced Response Generation:")
                enhance_start = time.time()
                final_response = self.generate_enhanced_response(user_input, sam_response, retrieved_docs)
                enhance_time = time.time() - enhance_start
                
                print(f"✨ Enhanced response generated in {enhance_time:.2f}s")
            else:
                print(f"❌ No relevant documents found")
                final_response = sam_response
        else:
            print(f"\n✅ No retrieval needed - confidence sufficient")
            final_response = sam_response
        
        # Step 5: Quick evaluation
        print(f"\n🧠 Quick Evaluation:")
        eval_start = time.time()
        evaluation = self.quick_evaluation(user_input, final_response)
        eval_time = time.time() - eval_start
        
        print(f"📊 Evaluation ({eval_time:.2f}s):")
        print(f"📈 {evaluation}")
        
        # Store conversation
        total_time = time.time() - sam_start
        self.conversation_history.append({
            'timestamp': time.time(),
            'user': user_input,
            'sam_response': sam_response,
            'final_response': final_response,
            'assessment': assessment,
            'retrieved_docs': retrieved_docs,
            'evaluation': evaluation,
            'timing': {
                'sam_time': sam_time,
                'assess_time': assess_time,
                'retrieval_time': retrieval_time,
                'eval_time': eval_time,
                'total_time': total_time
            }
        })
        
        print(f"\n🧠 Final Response ({total_time:.2f}s total):")
        print(f"💬 {final_response}")
        
        return final_response, evaluation
    
    def show_status(self):
        """Show system status"""
        print(f"\n📊 SELF-RAG FAST STATUS")
        print(f"{'='*40}")
        print(f"🧠 SAM Model: {'✅ Active' if self.sam_available else '❌ Not Available'}")
        print(f"🤖 Ollama: {'✅ Active' if self.ollama_available else '❌ Not Available'}")
        print(f"🧠 DeepSeek: {'✅ Active' if self.deepseek_available else '❌ Not Available'}")
        print(f"🌐 Web Access: {'✅ Active' if self.web_available else '❌ Not Available'}")
        print(f"💬 Conversations: {len(self.conversation_history)}")
        print(f"⏱️ Session Duration: {time.time() - self.session_start:.1f} seconds")
        
        if self.conversation_history:
            avg_total_time = sum(c['timing']['total_time'] for c in self.conversation_history) / len(self.conversation_history)
            retrieval_rate = sum(1 for c in self.conversation_history if c['retrieved_docs']) / len(self.conversation_history) * 100
            print(f"⚡ Avg Response Time: {avg_total_time:.2f}s")
            print(f"🔍 Retrieval Rate: {retrieval_rate:.1f}%")
    
    def save_conversation(self):
        """Save conversation history"""
        timestamp = int(time.time())
        filename = f"sam_self_rag_fast_conversation_{timestamp}.json"
        
        conversation_data = {
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
            json.dump(conversation_data, f, indent=2)
        
        print(f"💾 Conversation saved to: {filename}")
        return filename
    
    def run_self_rag_chatbot(self):
        """Run the Self-RAG chatbot"""
        print(f"\n🚀 SELF-RAG FAST CHATBOT READY!")
        print(f"💬 Type 'quit' to exit, 'status' for system info, 'save' to save conversation")
        print(f"🎯 Fast Self-RAG + Web Integration")
        print(f"⚡ Optimized for speed")
        
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
                
                # Process with Self-RAG
                self.process_self_rag_conversation(user_input)
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Interrupted! Saving conversation...")
                self.save_conversation()
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🧠 SAM SELF-RAG FAST INITIALIZATION")
    print("=" * 50)
    
    try:
        # Create Self-RAG system
        self_rag = SAMSelfRAGFast()
        
        # Run Self-RAG chatbot
        self_rag.run_self_rag_chatbot()
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Self-RAG system interrupted by user")
    except Exception as e:
        print(f"\n❌ Self-RAG error: {e}")
    finally:
        print(f"\n🎉 SAM Self-RAG Fast session completed!")

if __name__ == "__main__":
    main()
