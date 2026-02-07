#!/usr/bin/env python3
"""
Intelligent SAM System
Real Self-RAG, Web Scraping, Data Augmentation
Stage-based progression with actual learning
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
from urllib.parse import quote, urlparse
import random
import threading
import signal
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("⚠️ BeautifulSoup not available - web scraping disabled")
import hashlib

class IntelligentSAMSystem:
    def __init__(self):
        """Initialize Intelligent SAM System"""
        print("🧠 INTELLIGENT SAM SYSTEM")
        print("=" * 60)
        print("🔍 Real Self-RAG + Web Scraping + Data Augmentation")
        print("📈 Stage-based progression with actual learning")
        print("🎭 True conversation with knowledge integration")
        print("🛑 Type 'shutdown' to stop")
        
        self.base_path = Path("/Users/samueldasari/Personal/NN_C")
        self.sam_model_path = self.base_path / "ORGANIZED" / "UTILS" / "SAM" / "SAM" / "SAM.h"
        
        # System state
        self.running = True
        self.current_stage = 0
        self.stage_complete = False
        
        # Initialize system
        self.check_system_status()
        self.initialize_stages()
        self.initialize_sam_instances()
        
        # Knowledge and learning
        self.knowledge_graph = {}
        self.web_cache = {}
        self.conversation_context = []
        self.session_start = time.time()
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Shutdown signal received. Saving session...")
        self.running = False
        self.save_session()
        print(f"👋 Session saved. Exiting gracefully.")
        sys.exit(0)
    
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
        
        # Check BeautifulSoup for web scraping
        try:
            import bs4
            print(f"  🕷️ BeautifulSoup: ✅ Available")
        except ImportError:
            print(f"  🕷️ BeautifulSoup: ❌ Not Available (pip install beautifulsoup4)")
            self.web_available = False
    
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
    
    def initialize_stages(self):
        """Initialize learning stages with success criteria"""
        self.stages = [
            {
                'name': 'Stage 1: Basic Knowledge Acquisition',
                'description': 'Learn fundamental concepts through web scraping',
                'success_criteria': {
                    'knowledge_items': 20,
                    'web_sources': 3,
                    'conversation_quality': 6.0
                },
                'max_attempts': 10
            },
            {
                'name': 'Stage 2: Contextual Understanding',
                'description': 'Develop contextual awareness and conversation skills',
                'success_criteria': {
                    'contextual_responses': 15,
                    'conversation_coherence': 7.0,
                    'knowledge_integration': 10
                },
                'max_attempts': 15
            },
            {
                'name': 'Stage 3: Advanced Reasoning',
                'description': 'Achieve complex reasoning and synthesis',
                'success_criteria': {
                    'reasoning_depth': 8.0,
                    'synthesis_quality': 8.0,
                    'conversation_turns': 20
                },
                'max_attempts': 20
            }
        ]
    
    def initialize_sam_instances(self):
        """Initialize two SAM instances with learning capabilities"""
        print(f"\n🧠 INITIALIZING INTELLIGENT SAM INSTANCES")
        
        self.sam_alpha = {
            'name': 'SAM-Alpha',
            'specialty': 'Research & Analysis',
            'personality': 'analytical, methodical, evidence-based',
            'knowledge_base': {},
            'learning_progress': 0,
            'conversation_history': [],
            'web_queries': 0,
            'reasoning_depth': 0,
            'response_style': 'detailed and technical'
        }
        
        self.sam_beta = {
            'name': 'SAM-Beta',
            'specialty': 'Synthesis & Application',
            'personality': 'creative, practical, application-focused',
            'knowledge_base': {},
            'learning_progress': 0,
            'conversation_history': [],
            'web_queries': 0,
            'reasoning_depth': 0,
            'response_style': 'practical and accessible'
        }
        
        print(f"  ✅ {self.sam_alpha['name']}: {self.sam_alpha['specialty']}")
        print(f"  ✅ {self.sam_beta['name']}: {self.sam_beta['specialty']}")
    
    def scrape_web_content(self, query, sam_instance):
        """Real web scraping with multiple sources"""
        if not self.web_available:
            return None
        
        # Check cache first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.web_cache:
            return self.web_cache[cache_key]
        
        scraped_content = []
        
        # Try Wikipedia
        wiki_content = self.scrape_wikipedia(query)
        if wiki_content:
            scraped_content.append({
                'source': 'wikipedia',
                'content': wiki_content,
                'relevance': 0.8
            })
        
        # Try arXiv for technical topics
        if any(word in query.lower() for word in ['quantum', 'neural', 'algorithm', 'machine learning', 'ai']):
            arxiv_content = self.scrape_arxiv(query)
            if arxiv_content:
                scraped_content.append({
                    'source': 'arxiv',
                    'content': arxiv_content,
                    'relevance': 0.9
                })
        
        # Try news sources for current topics
        if any(word in query.lower() for word in ['latest', 'recent', 'current', 'news', 'today']):
            news_content = self.scrape_news(query)
            if news_content:
                scraped_content.append({
                    'source': 'news',
                    'content': news_content,
                    'relevance': 0.7
                })
        
        # Cache results
        if scraped_content:
            self.web_cache[cache_key] = scraped_content
            sam_instance['web_queries'] += 1
            print(f"  🌐 Scraped {len(scraped_content)} sources for: {query[:30]}...")
        
        return scraped_content if scraped_content else None
    
    def scrape_wikipedia(self, query):
        """Scrape Wikipedia content"""
        try:
            # Search terms
            search_terms = self.extract_search_terms(query)
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(search_terms)}"
            response = requests.get(url, timeout=8)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('extract', '')
        except Exception as e:
            print(f"  ❌ Wikipedia scrape error: {e}")
        
        return None
    
    def scrape_arxiv(self, query):
        """Scrape arXiv papers"""
        try:
            search_terms = self.extract_search_terms(query)
            url = f"https://export.arxiv.org/api/query?search_query=all:{quote(search_terms)}&max_results=2"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                content = response.text
                papers = []
                
                # Parse XML response
                title_matches = re.findall(r'<title>(.*?)</title>', content)
                summary_matches = re.findall(r'<summary>(.*?)</summary>', content)
                
                for i, (title, summary) in enumerate(zip(title_matches[:2], summary_matches[:2])):
                    papers.append(f"Paper: {title}\nAbstract: {summary[:300]}...")
                
                return '\n\n'.join(papers) if papers else None
        except Exception as e:
            print(f"  ❌ arXiv scrape error: {e}")
        
        return None
    
    def scrape_news(self, query):
        """Scrape news sources"""
        try:
            # Use a news API or RSS feed (simplified for demo)
            search_terms = self.extract_search_terms(query)
            url = f"https://news.google.com/rss/search?q={quote(search_terms)}"
            response = requests.get(url, timeout=8)
            
            if response.status_code == 200:
                # Parse RSS feed (simplified)
                content = response.text
                titles = re.findall(r'<title>(.*?)</title>', content)
                
                if titles:
                    return f"Latest news about {search_terms}:\n" + '\n'.join(titles[:3])
        except Exception as e:
            print(f"  ❌ News scrape error: {e}")
        
        return None
    
    def extract_search_terms(self, query):
        """Extract search terms from query"""
        words = re.findall(r'\b\w+\b', query.lower())
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'when', 'where', 'why', 'how', 'who', 'which', 'that', 'this', 'these', 'those'}
        
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(key_terms[:4])
    
    def augment_knowledge(self, query, scraped_content, sam_instance):
        """Augment knowledge base with scraped content"""
        if not scraped_content:
            return False
        
        augmented = False
        for source_data in scraped_content:
            content = source_data['content']
            source = source_data['source']
            relevance = source_data['relevance']
            
            # Store in knowledge base
            sam_instance['knowledge_base'][query] = {
                'content': content,
                'source': source,
                'relevance': relevance,
                'timestamp': time.time()
            }
            
            # Extract concepts and add to knowledge graph
            concepts = self.extract_concepts(content)
            for concept in concepts:
                if concept not in self.knowledge_graph:
                    self.knowledge_graph[concept] = []
                self.knowledge_graph[concept].append({
                    'query': query,
                    'source': source,
                    'sam_instance': sam_instance['name'],
                    'timestamp': time.time()
                })
            
            augmented = True
        
        return augmented
    
    def extract_concepts(self, content):
        """Extract key concepts from content"""
        words = re.findall(r'\b\w+\b', content.lower())
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'when', 'where', 'why', 'how', 'who', 'which', 'that', 'this', 'these', 'those', 'also', 'from', 'with', 'without', 'through', 'between', 'among', 'under', 'over', 'above', 'below', 'after', 'before', 'during', 'since', 'until', 'while', 'when', 'where', 'why', 'how'}
        
        concepts = [word for word in words if word not in stop_words and len(word) > 3]
        return list(set(concepts))[:10]
    
    def self_rag_assessment(self, query, sam_instance):
        """Self-RAG: Assess if retrieval is needed"""
        # Check if query is in knowledge base
        if query in sam_instance['knowledge_base']:
            return False, "knowledge_base"
        
        # Check for related concepts in knowledge graph
        query_concepts = self.extract_concepts(query)
        related_knowledge = 0
        
        for concept in query_concepts:
            if concept in self.knowledge_graph:
                related_knowledge += len(self.knowledge_graph[concept])
        
        if related_knowledge > 3:
            return False, "knowledge_graph"
        
        # Check if query asks for current/recent information
        current_keywords = ['latest', 'recent', 'current', 'new', 'modern', 'today', 'news']
        if any(keyword in query.lower() for keyword in current_keywords):
            return True, "current_info"
        
        # Check if query is specific and technical
        specific_keywords = ['what is', 'how does', 'explain', 'describe', 'details', 'technical']
        if any(keyword in query.lower() for keyword in specific_keywords):
            return True, "specific_query"
        
        return False, "no_retrieval"
    
    def generate_intelligent_response(self, sam_instance, query):
        """Generate intelligent response with Self-RAG"""
        print(f"  🧠 {sam_instance['name']} processing: {query[:50]}...")
        
        # Step 1: Self-RAG assessment
        retrieval_needed, reason = self.self_rag_assessment(query, sam_instance)
        print(f"    🔍 Self-RAG: {'Retrieval needed' if retrieval_needed else f'Using {reason}'}")
        
        response_source = "unknown"
        response_content = ""
        
        # Step 2: Knowledge base lookup
        if not retrieval_needed and reason == "knowledge_base":
            kb_data = sam_instance['knowledge_base'][query]
            response_content = kb_data['content']
            response_source = f"knowledge_base ({kb_data['source']})"
        
        # Step 3: Knowledge graph lookup
        elif not retrieval_needed and reason == "knowledge_graph":
            related_content = self.get_related_knowledge(query, sam_instance)
            if related_content:
                response_content = related_content
                response_source = "knowledge_graph"
            else:
                # Fall back to web retrieval
                retrieval_needed = True
        
        # Step 4: Web scraping and augmentation
        if retrieval_needed:
            scraped_content = self.scrape_web_content(query, sam_instance)
            if scraped_content:
                self.augment_knowledge(query, scraped_content, sam_instance)
                
                # Use highest relevance content
                best_content = max(scraped_content, key=lambda x: x['relevance'])
                response_content = best_content['content']
                response_source = f"web_scraped ({best_content['source']})"
            else:
                # Fall back to reasoning
                response_content = self.generate_reasoned_response(query, sam_instance)
                response_source = "reasoned"
        
        # Step 5: Personalize response
        if response_content:
            final_response = self.personalize_response(response_content, sam_instance)
            
            # Update learning progress
            sam_instance['learning_progress'] += 0.1
            
            return final_response, response_source
        else:
            return self.generate_fallback_response(query, sam_instance), "fallback"
    
    def get_related_knowledge(self, query, sam_instance):
        """Get related knowledge from knowledge graph"""
        query_concepts = self.extract_concepts(query)
        related_content = []
        
        for concept in query_concepts:
            if concept in self.knowledge_graph:
                for item in self.knowledge_graph[concept]:
                    if item['sam_instance'] == sam_instance['name']:
                        related_content.append(item['query'])
        
        if related_content:
            # Return content from most related query
            related_query = related_content[0]
            if related_query in sam_instance['knowledge_base']:
                return sam_instance['knowledge_base'][related_query]['content']
        
        return None
    
    def generate_reasoned_response(self, query, sam_instance):
        """Generate reasoned response when no direct knowledge available"""
        reasoning_prompt = f"""As a {sam_instance['specialty'].lower()} specialist, provide a thoughtful response to this question:

Question: {query}

Provide a response that:
1. Acknowledges the complexity of the question
2. Draws on general knowledge and reasoning
3. Identifies key concepts and relationships
4. Suggests areas for further investigation

Response:"""
        
        try:
            cmd = ['ollama', 'run', 'deepseek-r1', reasoning_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                sam_instance['reasoning_depth'] += 1
                return result.stdout.strip()
        except:
            pass
        
        return self.generate_fallback_response(query, sam_instance)
    
    def personalize_response(self, content, sam_instance):
        """Personalize response based on SAM instance personality"""
        if sam_instance['response_style'] == 'detailed and technical':
            return f"From a {sam_instance['specialty'].lower()} perspective: {content}\n\nThis analysis requires careful consideration of underlying mechanisms and empirical evidence."
        else:
            return f"Practically speaking: {content}\n\nThis has important implications for real-world applications and implementation."
    
    def generate_fallback_response(self, query, sam_instance):
        """Generate fallback response"""
        if sam_instance['response_style'] == 'detailed and technical':
            return f"From a research perspective, '{query}' represents a complex topic that requires systematic analysis. While I don't have specific information on this, I can suggest that it involves intricate patterns and mechanisms that warrant further investigation through empirical study and theoretical frameworks."
        else:
            return f"That's an interesting question about '{query}'. While I don't have specific details, I can see this connects to important real-world patterns. The best approach would be to gather more information and consider practical applications and implications."
    
    def evaluate_response_quality(self, query, response, sam_instance):
        """Evaluate response quality"""
        quality_prompt = f"""Rate this response 1-10:

Question: {query[:50]}...
Response: {response[:100]}...

Consider:
1. Relevance to the question
2. Depth of understanding
3. Clarity and coherence
4. Use of evidence/reasoning

Overall: X/10"""
        
        try:
            cmd = ['ollama', 'run', 'deepseek-r1', quality_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if "Overall:" in output:
                    try:
                        score_str = output.split("Overall:")[1].split("/10")[0].strip()
                        return float(score_str)
                    except:
                        pass
        except:
            pass
        
        # Heuristic evaluation
        if len(response) < 50:
            return 3.0
        elif len(response) < 150:
            return 5.0
        elif "complex" in response or "detailed" in response or "practical" in response:
            return 7.0
        else:
            return 6.0
    
    def check_stage_completion(self):
        """Check if current stage is complete"""
        if self.current_stage >= len(self.stages):
            return True
        
        stage = self.stages[self.current_stage]
        criteria = stage['success_criteria']
        
        print(f"\n📊 Checking {stage['name']} completion...")
        
        # Check knowledge items
        total_knowledge = len(self.sam_alpha['knowledge_base']) + len(self.sam_beta['knowledge_base'])
        knowledge_complete = total_knowledge >= criteria['knowledge_items']
        print(f"  📚 Knowledge: {total_knowledge}/{criteria['knowledge_items']} {'✅' if knowledge_complete else '❌'}")
        
        # Check web sources
        web_sources = set()
        for kb in self.sam_alpha['knowledge_base'].values():
            if 'source' in kb:
                web_sources.add(kb['source'])
        for kb in self.sam_beta['knowledge_base'].values():
            if 'source' in kb:
                web_sources.add(kb['source'])
        
        web_complete = len(web_sources) >= criteria['web_sources']
        print(f"  🌐 Web Sources: {len(web_sources)}/{criteria['web_sources']} {'✅' if web_complete else '❌'}")
        
        # Check conversation quality (average of recent conversations)
        if len(self.conversation_context) > 0:
            recent_quality = sum(c.get('quality_score', 0) for c in self.conversation_context[-10:]) / min(10, len(self.conversation_context))
            quality_complete = recent_quality >= criteria['conversation_quality']
            print(f"  💬 Quality: {recent_quality:.1f}/{criteria['conversation_quality']} {'✅' if quality_complete else '❌'}")
        else:
            quality_complete = False
            print(f"  💬 Quality: No conversations yet {'❌'}")
        
        # Overall completion
        stage_complete = knowledge_complete and web_complete and quality_complete
        
        if stage_complete:
            print(f"  ✅ {stage['name']} COMPLETED!")
            self.current_stage += 1
            return True
        else:
            print(f"  ⏳ {stage['name']} in progress...")
            return False
    
    def run_intelligent_conversation(self):
        """Run intelligent conversation between SAM instances"""
        print(f"\n🎭 STARTING INTELLIGENT CONVERSATION")
        print(f"🎯 Current Stage: {self.stages[self.current_stage]['name'] if self.current_stage < len(self.stages) else 'All Stages Complete'}")
        print(f"{'='*70}")
        
        conversation_turn = 0
        current_question = f"Hello {self.sam_beta['name']}, I'm {self.sam_alpha['name']}. Let's explore some interesting topics together. What should we discuss first?"
        current_speaker = self.sam_alpha
        
        while self.running:
            conversation_turn += 1
            
            print(f"\n🗣️  Turn {conversation_turn}: {current_speaker['name']}")
            print(f"❓ {current_question}")
            
            # Generate intelligent response
            start_time = time.time()
            response, response_source = self.generate_intelligent_response(current_speaker, current_question)
            response_time = time.time() - start_time
            
            # Evaluate quality
            quality_score = self.evaluate_response_quality(current_question, response, current_speaker)
            
            print(f"💬 {current_speaker['name']}: {response[:400]}...")
            print(f"📝 Source: {response_source} ({response_time:.2f}s)")
            print(f"📊 Quality: {quality_score:.1f}/10")
            
            # Store conversation
            turn_data = {
                'turn': conversation_turn,
                'speaker': current_speaker['name'],
                'question': current_question,
                'response': response,
                'response_source': response_source,
                'quality_score': quality_score,
                'timestamp': time.time()
            }
            
            self.conversation_context.append(turn_data)
            current_speaker['conversation_history'].append(turn_data)
            
            # Check stage completion every 5 turns
            if conversation_turn % 5 == 0:
                if self.check_stage_completion():
                    if self.current_stage >= len(self.stages):
                        print(f"\n🎉 ALL STAGES COMPLETED!")
                        break
                    else:
                        print(f"\n🚀 ADVANCING TO NEXT STAGE!")
            
            # Generate contextual follow-up
            other_sam = self.sam_beta if current_speaker == self.sam_alpha else self.sam_alpha
            current_question = self.generate_intelligent_follow_up(response, other_sam, conversation_turn)
            current_speaker = other_sam
            
            # Brief pause
            time.sleep(2)
            
            # Status update every 10 turns
            if conversation_turn % 10 == 0:
                print(f"\n📊 Conversation Status:")
                print(f"  🗣️ Turns: {conversation_turn}")
                print(f"  🧠 Alpha Knowledge: {len(self.sam_alpha['knowledge_base'])} items")
                print(f"  🧠 Beta Knowledge: {len(self.sam_beta['knowledge_base'])} items")
                print(f"  🌐 Web Cache: {len(self.web_cache)} queries")
                print(f"  📊 Knowledge Graph: {len(self.knowledge_graph)} concepts")
                print(f"  🛑 Type 'shutdown' to stop")
    
    def generate_intelligent_follow_up(self, response, asking_sam, turn_number):
        """Generate intelligent follow-up based on conversation context"""
        
        # Analyze the response to generate relevant follow-ups
        response_lower = response.lower()
        
        # Technical follow-ups
        if any(word in response_lower for word in ['quantum', 'neural', 'algorithm', 'technical', 'mechanism']):
            technical_follow_ups = [
                "Can you explain the technical mechanisms behind that in more detail?",
                "What are the underlying principles that govern this phenomenon?",
                "How does this compare to alternative approaches or theories?",
                "What empirical evidence supports this understanding?"
            ]
            return random.choice(technical_follow_ups)
        
        # Application follow-ups
        elif any(word in response_lower for word in ['practical', 'application', 'implement', 'real-world', 'benefit']):
            application_follow_ups = [
                "How would this work in practice?",
                "What are the most promising applications?",
                "What challenges would need to be overcome?",
                "How could this benefit people in their daily lives?"
            ]
            return random.choice(application_follow_ups)
        
        # Research follow-ups
        elif any(word in response_lower for word in ['research', 'study', 'investigate', 'analyze', 'evidence']):
            research_follow_ups = [
                "What research methods would be most effective here?",
                "How could we design experiments to test these ideas?",
                "What are the current gaps in our understanding?",
                "How might this field evolve in the coming years?"
            ]
            return random.choice(research_follow_ups)
        
        # General intelligent follow-ups
        else:
            general_follow_ups = [
                "What are the broader implications of what you've described?",
                "How does this connect to other areas of knowledge?",
                "What excites you most about this topic?",
                "Where do you think the most important discoveries will be made?",
                "How might this change our understanding of the world?",
                "What questions should we be asking next?"
            ]
            return random.choice(general_follow_ups)
    
    def save_session(self):
        """Save session data"""
        timestamp = int(time.time())
        filename = f"intelligent_sam_session_{timestamp}.json"
        
        session_data = {
            'timestamp': timestamp,
            'session_start': self.session_start,
            'duration': time.time() - self.session_start,
            'current_stage': self.current_stage,
            'stages_completed': self.current_stage,
            'knowledge_graph': self.knowledge_graph,
            'web_cache': self.web_cache,
            'sam_alpha': {
                'name': self.sam_alpha['name'],
                'specialty': self.sam_alpha['specialty'],
                'knowledge_base': self.sam_alpha['knowledge_base'],
                'learning_progress': self.sam_alpha['learning_progress'],
                'web_queries': self.sam_alpha['web_queries'],
                'reasoning_depth': self.sam_alpha['reasoning_depth']
            },
            'sam_beta': {
                'name': self.sam_beta['name'],
                'specialty': self.sam_beta['specialty'],
                'knowledge_base': self.sam_beta['knowledge_base'],
                'learning_progress': self.sam_beta['learning_progress'],
                'web_queries': self.sam_beta['web_queries'],
                'reasoning_depth': self.sam_beta['reasoning_depth']
            },
            'conversation_context': self.conversation_context
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"💾 Session saved to: {filename}")
        return filename
    
    def run_intelligent_system(self):
        """Run the intelligent SAM system"""
        print(f"\n🚀 INTELLIGENT SAM SYSTEM READY!")
        print(f"🎯 Stage-based learning with real Self-RAG")
        print(f"🌐 Web scraping + Knowledge augmentation")
        print(f"🎭 Intelligent conversation until stages complete")
        print(f"🛑 Type 'shutdown' to stop anytime")
        print(f"{'='*70}")
        
        try:
            self.run_intelligent_conversation()
        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ System error: {e}")
        finally:
            print(f"\n👋 Saving session...")
            self.save_session()
            print(f"🎉 Intelligent SAM session completed!")

def main():
    """Main function"""
    print("🧠 INTELLIGENT SAM SYSTEM INITIALIZATION")
    print("=" * 60)
    
    try:
        # Create and run intelligent system
        intelligent_sam = IntelligentSAMSystem()
        intelligent_sam.run_intelligent_system()
        
    except KeyboardInterrupt:
        print(f"\n\n👋 System interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    finally:
        print(f"\n🎉 Intelligent SAM session completed!")

if __name__ == "__main__":
    main()
