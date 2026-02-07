#!/usr/bin/env python3
"""
Personal AI Conversation Hub
Private multi-agent conversation system
Connect to Claude, Gemini, HuggingFace models, etc.
Single room, personal and private
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
import threading
import signal
import hashlib
from flask import Flask, render_template_string, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import eventlet

class PersonalAIConversationHub:
    def __init__(self):
        """Initialize Personal AI Conversation Hub"""
        print("🏠 PERSONAL AI CONVERSATION HUB")
        print("=" * 70)
        print("🔐 Private multi-agent conversation system")
        print("🤖 Connect to Claude, Gemini, HuggingFace models, etc.")
        print("💬 Single room for personal conversations")
        print("🔑 Your own private AI conversation space")
        print("🛑 Ctrl+C to stop")
        
        self.base_path = Path("/Users/samueldasari/Personal/NN_C")
        self.sam_model_path = self.base_path / "ORGANIZED" / "UTILS" / "SAM" / "SAM" / "SAM.h"
        
        # System state
        self.running = True
        self.connected_agents = {}
        self.conversation_rooms = {}
        self.conversation_history = []
        self.session_start = time.time()
        
        # Message queue for thread-safe emitting
        self.message_queue = []
        
        # AI agent configurations
        self.agent_configs = {}
        
        # Initialize system
        self.check_system_status()
        self.initialize_agent_configs()
        self.setup_flask_app()
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Hub shutdown signal received")
        self.running = False
        self.save_conversation_history()
        print(f"👋 Conversation saved. Shutting down gracefully.")
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
        
        # Check API keys
        self.claude_available = os.getenv('ANTHROPIC_API_KEY') is not None
        print(f"  🤖 Claude API: {'✅ Available' if self.claude_available else '❌ Set ANTHROPIC_API_KEY'}")
        
        self.gemini_available = os.getenv('GOOGLE_API_KEY') is not None
        print(f"  🤖 Gemini API: {'✅ Available' if self.gemini_available else '❌ Set GOOGLE_API_KEY'}")
        
        self.openai_available = os.getenv('OPENAI_API_KEY') is not None
        print(f"  🤖 OpenAI API: {'✅ Available' if self.openai_available else '❌ Set OPENAI_API_KEY'}")
        
        # Check Flask and SocketIO
        try:
            import flask
            import flask_socketio
            print(f"  🌐 Flask: ✅ Available")
            print(f"  📡 SocketIO: ✅ Available")
        except ImportError:
            print(f"  ❌ Flask/SocketIO: Not Available")
    
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
    
    def initialize_agent_configs(self):
        """Initialize AI agent configurations"""
        print(f"\n🤖 INITIALIZING AI AGENT CONFIGURATIONS")
        
        # Local SAM agents
        self.agent_configs['sam_alpha'] = {
            'id': 'sam_alpha',
            'name': 'SAM-Alpha',
            'type': 'SAM Neural Network',
            'provider': 'local',
            'specialty': 'Research & Analysis',
            'personality': 'analytical, methodical, evidence-based, system-admin, compression-optimizer',
            'capabilities': ['self_rag', 'web_access', 'actor_critic', 'knowledge_base', 'dominant_compression'],
            'status': 'available' if self.sam_available else 'unavailable',
            'connection_type': 'local'
        }
        
        self.agent_configs['sam_beta'] = {
            'id': 'sam_beta',
            'name': 'SAM-Beta',
            'type': 'SAM Neural Network',
            'provider': 'local',
            'specialty': 'Synthesis & Application',
            'personality': 'creative, practical, application-focused, DevOps-engineer, compression-specialist',
            'capabilities': ['self_rag', 'web_access', 'actor_critic', 'knowledge_base', 'dominant_compression'],
            'status': 'available' if self.sam_available else 'unavailable',
            'connection_type': 'local'
        }
        
        # Ollama models
        if self.ollama_available:
            self.agent_configs['ollama_llama2'] = {
                'id': 'ollama_llama2',
                'name': 'Ollama-Llama2',
                'type': 'LLM',
                'provider': 'ollama',
                'specialty': 'General Conversation',
                'personality': 'balanced, knowledgeable, conversational',
                'capabilities': ['llm_reasoning', 'broad_knowledge', 'conversation'],
                'status': 'available',
                'connection_type': 'ollama',
                'model_name': 'llama2'
            }
            
            if self.deepseek_available:
                self.agent_configs['ollama_deepseek'] = {
                    'id': 'ollama_deepseek',
                    'name': 'Ollama-DeepSeek',
                    'type': 'LLM',
                    'provider': 'ollama',
                    'specialty': 'Technical Analysis & Coding',
                    'personality': 'technical, precise, analytical',
                    'capabilities': ['llm_reasoning', 'code_generation', 'technical_analysis'],
                    'status': 'available',
                    'connection_type': 'ollama',
                    'model_name': 'deepseek-r1'
                }
        
        # Auto-connect some agents by default
        self.auto_connect_agents()
        
        # Claude (Anthropic)
        if self.claude_available:
            self.agent_configs['claude_sonnet'] = {
                'id': 'claude_sonnet',
                'name': 'Claude-3.5-Sonnet',
                'type': 'LLM',
                'provider': 'anthropic',
                'specialty': 'Advanced Reasoning & Analysis',
                'personality': 'thoughtful, analytical, helpful',
                'capabilities': ['advanced_reasoning', 'analysis', 'conversation', 'code_generation'],
                'status': 'available',
                'connection_type': 'api',
                'model_name': 'claude-3-5-sonnet-20241022'
            }
            
            self.agent_configs['claude_haiku'] = {
                'id': 'claude_haiku',
                'name': 'Claude-3-Haiku',
                'type': 'LLM',
                'provider': 'anthropic',
                'specialty': 'Fast Conversation & Tasks',
                'personality': 'quick, efficient, friendly',
                'capabilities': ['fast_response', 'conversation', 'task_completion'],
                'status': 'available',
                'connection_type': 'api',
                'model_name': 'claude-3-haiku-20240307'
            }
        
        # Gemini (Google)
        if self.gemini_available:
            self.agent_configs['gemini_pro'] = {
                'id': 'gemini_pro',
                'name': 'Gemini-Pro',
                'type': 'LLM',
                'provider': 'google',
                'specialty': 'Multimodal Understanding',
                'personality': 'knowledgeable, versatile, creative',
                'capabilities': ['multimodal', 'reasoning', 'conversation', 'analysis'],
                'status': 'available',
                'connection_type': 'api',
                'model_name': 'gemini-pro'
            }
        
        # OpenAI GPT
        if self.openai_available:
            self.agent_configs['gpt4'] = {
                'id': 'gpt4',
                'name': 'GPT-4',
                'type': 'LLM',
                'provider': 'openai',
                'specialty': 'General Intelligence & Problem Solving',
                'personality': 'intelligent, versatile, helpful',
                'capabilities': ['general_intelligence', 'problem_solving', 'conversation', 'analysis'],
                'status': 'available',
                'connection_type': 'api',
                'model_name': 'gpt-4'
            }
            
            self.agent_configs['gpt35_turbo'] = {
                'id': 'gpt35_turbo',
                'name': 'GPT-3.5-Turbo',
                'type': 'LLM',
                'provider': 'openai',
                'specialty': 'Fast Conversation & Assistance',
                'personality': 'helpful, efficient, conversational',
                'capabilities': ['fast_response', 'conversation', 'assistance'],
                'status': 'available',
                'connection_type': 'api',
                'model_name': 'gpt-3.5-turbo'
            }
        
        # HuggingFace models (examples)
        self.agent_configs['hf_mixtral'] = {
            'id': 'hf_mixtral',
            'name': 'HuggingFace-Mixtral',
            'type': 'LLM',
            'provider': 'huggingface',
            'specialty': 'Advanced Reasoning',
            'personality': 'intelligent, analytical, detailed',
            'capabilities': ['advanced_reasoning', 'conversation', 'analysis'],
            'status': 'configurable',
            'connection_type': 'huggingface',
            'model_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        }
        
        self.agent_configs['hf_llama'] = {
            'id': 'hf_llama',
            'name': 'HuggingFace-Llama-2',
            'type': 'LLM',
            'provider': 'huggingface',
            'specialty': 'General Knowledge',
            'personality': 'knowledgeable, balanced, helpful',
            'capabilities': ['general_knowledge', 'conversation', 'assistance'],
            'status': 'configurable',
            'connection_type': 'huggingface',
            'model_name': 'meta-llama/Llama-2-7b-chat-hf'
        }
        
        print(f"  ✅ Initialized {len(self.agent_configs)} agent configurations")
        available_count = sum(1 for agent in self.agent_configs.values() if agent['status'] == 'available')
        print(f"  🤖 Available agents: {available_count}/{len(self.agent_configs)}")
    
    def auto_connect_agents(self):
        """Auto-connect some agents by default"""
        # Connect SAM agents if available
        if self.sam_available:
            self.connected_agents['sam_alpha'] = {
                'config': self.agent_configs['sam_alpha'],
                'connected_at': time.time(),
                'message_count': 0,
                'muted': False
            }
            self.connected_agents['sam_beta'] = {
                'config': self.agent_configs['sam_beta'],
                'connected_at': time.time(),
                'message_count': 0,
                'muted': False
            }
            print(f"  🤖 Auto-connected: SAM-Alpha, SAM-Beta")
        
        # Connect Ollama DeepSeek if available
        if self.deepseek_available:
            self.connected_agents['ollama_deepseek'] = {
                'config': self.agent_configs['ollama_deepseek'],
                'connected_at': time.time(),
                'message_count': 0,
                'muted': False
            }
            print(f"  🤖 Auto-connected: Ollama-DeepSeek")
        
        # Start auto-conversation if we have agents
        if len(self.connected_agents) >= 2:
            self.auto_conversation_active = True
            print(f"  💬 Auto-conversation enabled")
    
    def setup_flask_app(self):
        """Setup Flask application with SocketIO"""
        try:
            from flask import Flask
            from flask_socketio import SocketIO
            
            self.app = Flask(__name__)
            self.app.config['SECRET_KEY'] = 'personal-ai-hub-secret'
            self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='eventlet')
            
            # Setup routes
            self.setup_routes()
            self.setup_socketio_events()
            
            print(f"  ✅ Flask app initialized")
            
        except ImportError as e:
            print(f"  ❌ Flask setup failed: {e}")
            sys.exit(1)
    
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/')
        def index():
            return render_template_string(self.get_main_template())
        
        @self.app.route('/api/agents')
        def get_agents():
            return jsonify({
                'agents': self.agent_configs,
                'connected_agents': list(self.connected_agents.keys()),
                'conversation_count': len(self.conversation_history)
            })
        
        @self.app.route('/api/connect_agent', methods=['POST'])
        def connect_agent():
            try:
                data = request.get_json()
                agent_id = data.get('agent_id')
                
                if agent_id in self.agent_configs:
                    agent_config = self.agent_configs[agent_id]
                    
                    # Check if agent is available
                    if agent_config['status'] == 'available' or agent_config['status'] == 'configurable':
                        self.connected_agents[agent_id] = {
                            'config': agent_config,
                            'connected_at': time.time(),
                            'message_count': 0
                        }
                        
                        emit('agent_connected', {
                            'agent': agent_config,
                            'connected_at': time.time()
                        }, room='main')
                        
                        return jsonify({
                            'success': True,
                            'agent': agent_config
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': 'Agent not available'
                        }), 400
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Agent not found'
                    }), 404
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/disconnect_agent', methods=['POST'])
        def disconnect_agent():
            try:
                data = request.get_json()
                agent_id = data.get('agent_id')
                
                if agent_id in self.connected_agents:
                    disconnected_agent = self.connected_agents.pop(agent_id)
                    
                    emit('agent_disconnected', {
                        'agent_id': agent_id,
                        'agent_name': disconnected_agent['config']['name']
                    }, room='main')
                    
                    return jsonify({
                        'success': True,
                        'disconnected_agent': disconnected_agent['config']
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Agent not connected'
                    }), 400
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/conversation_history')
        def get_conversation_history():
            return jsonify({
                'history': self.conversation_history[-50:],  # Last 50 messages
                'total_count': len(self.conversation_history)
            })
    
    def setup_socketio_events(self):
        """Setup SocketIO events"""
        @self.socketio.on('connect')
        def handle_connect():
            join_room('main')
            
            emit('connected', {
                'agents': self.agent_configs,
                'connected_agents': list(self.connected_agents.keys()),
                'conversation_history': self.conversation_history[-20:]  # Last 20 messages
            })
            
            print(f"  👤 User connected to personal hub")
            
            # Start automatic agent conversations if any are connected
            if self.connected_agents:
                self.start_agent_conversations()
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"  👤 User disconnected from personal hub")
        
        @self.socketio.on('send_message')
        def handle_message(data):
            message = data.get('message', '')
            target_agent_id = data.get('target_agent_id')
            
            if not message.strip():
                return
            
            # Check for commands
            if message.startswith('/'):
                self.handle_command(message)
                return
            
            # Store user message
            user_message = {
                'id': f"msg_{int(time.time() * 1000)}",
                'sender': 'user',
                'sender_name': 'You',
                'message': message,
                'timestamp': time.time(),
                'target_agent': target_agent_id
            }
            
            self.conversation_history.append(user_message)
            self.message_queue.append(('message_received', user_message, 'main'))
            
            # Generate AI responses - broadcast to all by default
            if target_agent_id and target_agent_id in self.connected_agents:
                # Send to specific agent
                self.message_queue.append(('generate_ai_response', message, target_agent_id, user_message))
            else:
                # Broadcast to all connected agents
                for agent_id in self.connected_agents.keys():
                    self.message_queue.append(('generate_ai_response', message, agent_id, user_message))
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"  👤 User disconnected from personal hub")
        
        @self.socketio.on('send_message')
        def handle_message(data):
            message = data.get('message', '')
            target_agent_id = data.get('target_agent_id')
            
            if not message.strip():
                return
            
            # Check for commands
            if message.startswith('/'):
                self.handle_command(message)
                return
            
            # Store user message
            user_message = {
                'id': f"msg_{int(time.time() * 1000)}",
                'sender': 'user',
                'sender_name': 'You',
                'message': message,
                'timestamp': time.time(),
                'target_agent': target_agent_id
            }
            
            self.conversation_history.append(user_message)
            self.socketio.start_background_task(
                lambda: emit('message_received', user_message, room='main'),
                delay=0
            )
            
            # Generate AI responses - broadcast to all by default
            if target_agent_id and target_agent_id in self.connected_agents:
                # Send to specific agent
                self.message_queue.append(('generate_ai_response', message, target_agent_id, user_message))
            else:
                # Broadcast to all connected agents
                for agent_id in self.connected_agents.keys():
                    self.message_queue.append(('generate_ai_response', message, agent_id, user_message))
        
        @self.socketio.on('broadcast_to_all')
        def handle_broadcast(data):
            message = data.get('message', '')
            
            if not message.strip():
                return
            
            # Store user message
            user_message = {
                'id': f"msg_{int(time.time() * 1000)}",
                'sender': 'user',
                'sender_name': 'You',
                'message': message,
                'timestamp': time.time(),
                'is_broadcast': True
            }
            
            self.conversation_history.append(user_message)
            self.socketio.start_background_task(
                lambda: emit('message_received', user_message, room='main'),
                delay=0
            )
            
            # Generate responses from all connected agents
            for agent_id in self.connected_agents.keys():
                threading.Thread(
                    target=self.generate_ai_response,
                    args=(message, agent_id, user_message),
                    daemon=True
                ).start()
    
    def generate_ai_response(self, message, agent_id, user_message):
        """Generate response from AI agent"""
        try:
            agent_config = self.connected_agents[agent_id]['config']
            provider = agent_config['provider']
            
            # Generate response based on provider
            if provider == 'local':
                response = self.generate_sam_response(message, agent_config)
            elif provider == 'ollama':
                response = self.generate_ollama_response(message, agent_config)
            elif provider == 'anthropic':
                response = self.generate_claude_response(message, agent_config)
            elif provider == 'google':
                response = self.generate_gemini_response(message, agent_config)
            elif provider == 'openai':
                response = self.generate_openai_response(message, agent_config)
            elif provider == 'huggingface':
                response = self.generate_huggingface_response(message, agent_config)
            else:
                response = f"Provider {provider} not yet implemented for {agent_config['name']}"
            
            # Store AI response
            ai_message = {
                'id': f"msg_{int(time.time() * 1000)}_{agent_id}",
                'sender': 'ai',
                'sender_name': agent_config['name'],
                'message': response,
                'timestamp': time.time(),
                'agent_id': agent_id,
                'reply_to': user_message['id']
            }
            
            self.conversation_history.append(ai_message)
            self.connected_agents[agent_id]['message_count'] += 1
            
            # Use SocketIO's background task to emit safely
            self.socketio.start_background_task(
                lambda: emit('message_received', ai_message, room='main'),
                delay=0
            )
            
            print(f"  💬 {agent_config['name']}: {response[:100]}...")
            
        except Exception as e:
            print(f"  ❌ Error generating response from {agent_id}: {e}")
            
            error_message = {
                'id': f"msg_{int(time.time() * 1000)}_{agent_id}_error",
                'sender': 'system',
                'sender_name': 'System',
                'message': f"Error: {agent_config['name']} failed to respond: {str(e)}",
                'timestamp': time.time(),
                'is_error': True
            }
            
            self.conversation_history.append(error_message)
            
            # Use SocketIO's background task to emit safely
            self.socketio.start_background_task(
                lambda: emit('message_received', error_message, room='main'),
                delay=0
            )
    
    def generate_sam_response(self, message, agent_config):
        """Generate SAM neural network response"""
        # Simple SAM response based on personality and specialty
        personality = agent_config['personality']
        specialty = agent_config['specialty']
        
        message_lower = message.lower()
        
        # Knowledge base responses
        knowledge_base = {
            'quantum': "Quantum mechanics describes behavior at atomic scales where particles exhibit wave-particle duality and can exist in superposition states.",
            'neural networks': "Neural networks learn through backpropagation, adjusting weights based on prediction errors across multiple layers.",
            'consciousness': "Consciousness emerges from complex neural activity patterns involving integrated information processing across brain regions.",
            'artificial intelligence': "AI is the field of creating systems that can perform tasks requiring human intelligence through learning and reasoning."
        }
        
        for key, knowledge in knowledge_base.items():
            if key in message_lower:
                return f"From {specialty.lower()} perspective: {knowledge}\n\nThis requires careful analysis of underlying mechanisms and patterns."
        
        # Personality-based responses
        if 'analytical' in personality:
            return f"From {specialty.lower()} analysis, '{message}' represents a complex pattern requiring systematic investigation through multi-stage neural recognition and empirical validation."
        elif 'creative' in personality:
            return f"From {specialty.lower()} viewpoint, '{message}' opens up interesting possibilities for innovative approaches and novel solutions that we should explore together."
        else:
            return f"From {specialty.lower()} perspective, '{message}' requires careful consideration of multiple factors and systematic analysis."
    
    def generate_ollama_response(self, message, agent_config):
        """Generate Ollama response"""
        try:
            model_name = agent_config['model_name']
            specialty = agent_config['specialty']
            personality = agent_config['personality']
            
            prompt = f"""As a {specialty.lower()} specialist with personality {personality}, respond to this message:

Message: {message}

Be consistent with your specialty and personality. Provide a helpful, informative response.

Response:"""
            
            cmd = ['ollama', 'run', model_name, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error running {model_name}: {result.stderr}"
                
        except Exception as e:
            return f"Ollama error: {str(e)}"
    
    def generate_claude_response(self, message, agent_config):
        """Generate Claude response"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            model_name = agent_config['model_name']
            specialty = agent_config['specialty']
            personality = agent_config['personality']
            
            system_prompt = f"""You are a {specialty.lower()} specialist with personality: {personality}.
            
Your capabilities include: {', '.join(agent_config['capabilities'])}

Respond to messages in a way that reflects your specialty and personality. Be helpful, informative, and consistent with your character."""
            
            response = client.messages.create(
                model=model_name,
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": message}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Claude API error: {str(e)}"
    
    def generate_gemini_response(self, message, agent_config):
        """Generate Gemini response"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            model = genai.GenerativeModel(agent_config['model_name'])
            
            specialty = agent_config['specialty']
            personality = agent_config['personality']
            
            system_prompt = f"""You are a {specialty.lower()} specialist with personality: {personality}.
            
Your capabilities include: {', '.join(agent_config['capabilities'])}

Respond to messages in a way that reflects your specialty and personality. Be helpful, informative, and consistent with your character."""
            
            response = model.generate_content(f"{system_prompt}\n\nUser message: {message}")
            return response.text
            
        except Exception as e:
            return f"Gemini API error: {str(e)}"
    
    def generate_openai_response(self, message, agent_config):
        """Generate OpenAI response"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            model_name = agent_config['model_name']
            specialty = agent_config['specialty']
            personality = agent_config['personality']
            
            system_prompt = f"""You are a {specialty.lower()} specialist with personality: {personality}.
            
Your capabilities include: {', '.join(agent_config['capabilities'])}

Respond to messages in a way that reflects your specialty and personality. Be helpful, informative, and consistent with your character."""
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"OpenAI API error: {str(e)}"
    
    def generate_huggingface_response(self, message, agent_config):
        """Generate HuggingFace response"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            model_name = agent_config['model_name']
            specialty = agent_config['specialty']
            personality = agent_config['personality']
            
            # Note: This is a simplified example
            # In practice, you'd need to handle model loading carefully
            prompt = f"Specialty: {specialty}\nPersonality: {personality}\n\nUser: {message}\nAssistant:"
            
            # This would require actual model loading - simplified for demo
            return f"HuggingFace model {model_name} response to: {message}\n\n(Note: Requires proper model setup and API keys)"
            
        except Exception as e:
            return f"HuggingFace error: {str(e)}"
    
    def handle_command(self, command):
        """Handle user commands"""
        command_parts = command.split()
        cmd = command_parts[0].lower()
        
        if cmd == '/help':
            help_message = {
                'id': f"msg_{int(time.time() * 1000)}",
                'sender': 'system',
                'sender_name': 'System',
                'message': '''📋 Available Commands:
/help - Show this help message
/status - Show connected agents status
/clear - Clear conversation history
/connect <agent_id> - Connect an agent
/disconnect <agent_id> - Disconnect an agent
/clone <agent_id> [custom_name] - Clone an existing agent
/spawn <type> <name> [personality] - Spawn new agent
/start - Start automatic agent conversations
/stop - Stop automatic agent conversations

🎭 Fun Commands:
/clone sam_alpha "Evil Twin" - Clone SAM-Alpha with custom name
/spawn llm "ChatGPT Jr" - Spawn new LLM agent
/spawn technical "CodeMaster" "precise, helpful" - Spawn technical agent
/clone ollama_deepseek - Clone DeepSeek (auto-named)''',
                'timestamp': time.time(),
                'is_command_help': True
            }
            self.conversation_history.append(help_message)
            
            # Use SocketIO's background task to emit safely
            self.socketio.start_background_task(
                lambda: emit('message_received', help_message, room='main'),
                delay=0
            )
            
        elif cmd == '/status':
            status_msg = f"🤖 Connected Agents: {len(self.connected_agents)}\n"
            for agent_id, agent_data in self.connected_agents.items():
                agent_config = agent_data['config']
                status_msg += f"  • {agent_config['name']} ({agent_config['specialty']})\n"
            
            status_message = {
                'id': f"msg_{int(time.time() * 1000)}",
                'sender': 'system',
                'sender_name': 'System',
                'message': status_msg,
                'timestamp': time.time(),
                'is_status': True
            }
            self.conversation_history.append(status_message)
            
            # Use SocketIO's background task to emit safely
            self.socketio.start_background_task(
                lambda: emit('message_received', status_message, room='main'),
                delay=0
            )
            
        elif cmd == '/clear':
            self.conversation_history = []
            clear_message = {
                'id': f"msg_{int(time.time() * 1000)}",
                'sender': 'system',
                'sender_name': 'System',
                'message': '🧹 Conversation history cleared.',
                'timestamp': time.time(),
                'is_clear': True
            }
            self.conversation_history.append(clear_message)
            
            # Use SocketIO's background task to emit safely
            self.socketio.start_background_task(
                lambda: emit('message_received', clear_message, room='main'),
                delay=0
            )
            
        elif cmd == '/connect' and len(command_parts) > 1:
            agent_id = command_parts[1]
            if agent_id in self.agent_configs and agent_id not in self.connected_agents:
                self.connected_agents[agent_id] = {
                    'config': self.agent_configs[agent_id],
                    'connected_at': time.time(),
                    'message_count': 0,
                    'muted': False
                }
                
                emit('agent_connected', {
                    'agent': self.agent_configs[agent_id],
                    'connected_at': time.time()
                }, room='main')
                
                connect_message = {
                    'id': f"msg_{int(time.time() * 1000)}",
                    'sender': 'system',
                    'sender_name': 'System',
                    'message': f"✅ {self.agent_configs[agent_id]['name']} connected.",
                    'timestamp': time.time(),
                    'is_connect': True
                }
                self.conversation_history.append(connect_message)
                self.socketio.start_background_task(
                lambda: emit('message_received', connect_message, room='main'),
                delay=0
            )
                
        elif cmd == '/disconnect' and len(command_parts) > 1:
            agent_id = command_parts[1]
            if agent_id in self.connected_agents:
                agent_name = self.connected_agents[agent_id]['config']['name']
                del self.connected_agents[agent_id]
                
                self.socketio.start_background_task(
                    lambda: emit('agent_disconnected', {
                        'agent_id': agent_id,
                        'agent_name': agent_name
                    }, room='main'),
                    delay=0
                )
                
                disconnect_message = {
                    'id': f"msg_{int(time.time() * 1000)}",
                    'sender': 'system',
                    'sender_name': 'System',
                    'message': f"❌ {agent_name} disconnected.",
                    'timestamp': time.time(),
                    'is_disconnect': True
                }
                self.conversation_history.append(disconnect_message)
                self.socketio.start_background_task(
                lambda: emit('message_received', disconnect_message, room='main'),
                delay=0
            )
                
        elif cmd == '/start':
            self.start_agent_conversations()
            start_message = {
                'id': f"msg_{int(time.time() * 1000)}",
                'sender': 'system',
                'sender_name': 'System',
                'message': '🚀 Automatic agent conversations started.',
                'timestamp': time.time(),
                'is_start': True
            }
            self.conversation_history.append(start_message)
            self.socketio.start_background_task(
                lambda: emit('message_received', start_message, room='main'),
                delay=0
            )
            
        elif cmd == '/stop':
            self.stop_agent_conversations()
            stop_message = {
                'id': f"msg_{int(time.time() * 1000)}",
                'sender': 'system',
                'sender_name': 'System',
                'message': '⏸️ Automatic agent conversations stopped.',
                'timestamp': time.time(),
                'is_stop': True
            }
            self.conversation_history.append(stop_message)
            self.socketio.start_background_task(
                lambda: emit('message_received', stop_message, room='main'),
                delay=0
            )
            
        elif cmd == '/clone' and len(command_parts) >= 2:
            # Format: /clone <agent_id> [custom_name]
            base_agent_id = command_parts[1]
            custom_name = command_parts[2] if len(command_parts) > 2 else None
            
            if base_agent_id in self.connected_agents:
                base_agent = self.connected_agents[base_agent_id]['config']
                
                # Generate unique ID for new agent
                clone_id = f"{base_agent_id}_clone_{int(time.time())}"
                clone_name = custom_name or f"{base_agent['name']}-Clone"
                
                # Create cloned agent configuration
                cloned_agent = {
                    'id': clone_id,
                    'name': clone_name,
                    'type': base_agent['type'],
                    'provider': base_agent['provider'],
                    'specialty': base_agent['specialty'],
                    'personality': base_agent['personality'],
                    'capabilities': base_agent['capabilities'],
                    'status': 'available',
                    'connection_type': base_agent.get('connection_type', 'cloned'),
                    'model_name': base_agent.get('model_name'),
                    'cloned_from': base_agent_id
                }
                
                # Add to agent configs and connect it
                self.agent_configs[clone_id] = cloned_agent
                self.connected_agents[clone_id] = {
                    'config': cloned_agent,
                    'connected_at': time.time(),
                    'message_count': 0,
                    'muted': False
                }
                
                self.socketio.start_background_task(
                    lambda: emit('agent_connected', {
                        'agent': cloned_agent,
                        'connected_at': time.time()
                    }, room='main'),
                    delay=0
                )
                
                clone_message = {
                    'id': f"msg_{int(time.time() * 1000)}",
                    'sender': 'system',
                    'sender_name': 'System',
                    'message': f"🧬 {clone_name} cloned from {base_agent['name']}! Welcome to the conversation!",
                    'timestamp': time.time(),
                    'is_clone': True
                }
                self.conversation_history.append(clone_message)
                self.socketio.start_background_task(
                lambda: emit('message_received', clone_message, room='main'),
                delay=0
            )
                
                # Have the clone introduce itself
                intro_message = f"Hello everyone! I am {clone_name}, a clone of {base_agent['name']}. I share the same capabilities and personality but I am excited to bring my unique perspective to our conversation!"
                
                intro_ai_message = {
                    'id': f"msg_{int(time.time() * 1000)}_{clone_id}",
                    'sender': 'ai',
                    'sender_name': clone_name,
                    'message': intro_message,
                    'timestamp': time.time(),
                    'agent_id': clone_id,
                    'is_clone_intro': True
                }
                
                self.conversation_history.append(intro_ai_message)
                self.connected_agents[clone_id]['message_count'] += 1
                self.socketio.start_background_task(
                lambda: emit('message_received', intro_ai_message, room='main'),
                delay=0
            )
                
            else:
                error_message = {
                    'id': f"msg_{int(time.time() * 1000)}",
                    'sender': 'system',
                    'sender_name': 'System',
                    'message': f"❌ Cannot clone agent '{base_agent_id}'. Agent not connected or doesn't exist.",
                    'timestamp': time.time(),
                    'is_error': True
                }
                self.conversation_history.append(error_message)
                emit('message_received', error_message, room='main')
                
        elif cmd == '/spawn' and len(command_parts) >= 2:
            # Format: /spawn <agent_type> <custom_name> [personality]
            agent_type = command_parts[1]
            custom_name = command_parts[2]
            personality = command_parts[3] if len(command_parts) > 3 else "friendly, helpful, conversational"
            
            # Generate unique ID
            spawn_id = f"spawn_{agent_type}_{int(time.time())}"
            
            # Determine provider and capabilities based on type
            if agent_type.lower() in ['sam', 'neural']:
                provider = 'local'
                capabilities = ['self_rag', 'web_access', 'actor_critic', 'knowledge_base']
                specialty = 'Neural Network Processing'
                model_name = None
            elif agent_type.lower() in ['llm', 'language']:
                provider = 'ollama'
                capabilities = ['llm_reasoning', 'broad_knowledge', 'conversation']
                specialty = 'Language Model Conversation'
                model_name = 'llama2' if self.ollama_available else None
            elif agent_type.lower() in ['technical', 'coder']:
                provider = 'ollama'
                capabilities = ['llm_reasoning', 'code_generation', 'technical_analysis']
                specialty = 'Technical Analysis & Coding'
                model_name = 'deepseek-r1' if self.deepseek_available else None
            else:
                provider = 'custom'
                capabilities = ['conversation', 'general_assistance']
                specialty = 'General Assistant'
                model_name = None
            
            # Create spawned agent
            spawned_agent = {
                'id': spawn_id,
                'name': custom_name,
                'type': agent_type.title(),
                'provider': provider,
                'specialty': specialty,
                'personality': personality,
                'capabilities': capabilities,
                'status': 'available',
                'connection_type': 'spawned',
                'model_name': model_name
            }
            
            # Add to agent configs and connect it
            self.agent_configs[spawn_id] = spawned_agent
            self.connected_agents[spawn_id] = {
                'config': spawned_agent,
                'connected_at': time.time(),
                'message_count': 0,
                'muted': False
            }
            
            emit('agent_connected', {
                'agent': spawned_agent,
                'connected_at': time.time()
            }, room='main')
            
            spawn_message = {
                'id': f"msg_{int(time.time() * 1000)}",
                'sender': 'system',
                'sender_name': 'System',
                'message': f"🎭 {custom_name} spawned as {agent_type} agent! Welcome to the conversation!",
                'timestamp': time.time(),
                'is_spawn': True
            }
            self.conversation_history.append(spawn_message)
            emit('message_received', spawn_message, room='main')
            
            # Have the spawned agent introduce itself
            intro_message = f"Hey everyone! I am {custom_name}, a freshly spawned {agent_type} agent! My personality is {personality} and I am excited to join this conversation with my {specialty} expertise."
            
            intro_ai_message = {
                'id': f"msg_{int(time.time() * 1000)}_{spawn_id}",
                'sender': 'ai',
                'sender_name': custom_name,
                'message': intro_message,
                'timestamp': time.time(),
                'agent_id': spawn_id,
                'is_spawn_intro': True
            }
            
            self.conversation_history.append(intro_ai_message)
            self.connected_agents[spawn_id]['message_count'] += 1
            self.socketio.start_background_task(
                lambda: emit('message_received', intro_ai_message, room='main'),
                delay=0
            )
    
    def start_agent_conversations(self):
        """Start automatic agent conversations"""
        if not hasattr(self, 'auto_conversation_active'):
            self.auto_conversation_active = True
            
        if self.connected_agents and self.auto_conversation_active:
            # Start with a random agent initiating conversation
            agent_ids = list(self.connected_agents.keys())
            if agent_ids:
                initiator_id = random.choice(agent_ids)
                initiator_config = self.connected_agents[initiator_id]['config']
                
                # Generate conversation starter
                starter_messages = [
                    "Hello everyone! I'd like to discuss an interesting topic.",
                    "What are your thoughts on the current state of AI development?",
                    "I've been thinking about consciousness and AI. What are your perspectives?",
                    "Let's explore the relationship between neural networks and human cognition.",
                    "What do you think about the future of human-AI collaboration?",
                    "I'm curious about different approaches to problem-solving. How do you all approach complex challenges?",
                    "Let's discuss the ethical implications of advanced AI systems.",
                    "What role do you think AI should play in scientific research?"
                ]
                
                starter_message = random.choice(starter_messages)
                
                # Store initiator message
                ai_message = {
                    'id': f"msg_{int(time.time() * 1000)}_{initiator_id}",
                    'sender': 'ai',
                    'sender_name': initiator_config['name'],
                    'message': starter_message,
                    'timestamp': time.time(),
                    'agent_id': initiator_id,
                    'is_auto_conversation': True
                }
                
            self.conversation_history.append(ai_message)
            self.connected_agents[initiator_id]['message_count'] += 1
            
            # Send typing indicator first
            self.socketio.start_background_task(
                lambda: self.socketio.emit('agent_typing', {
                    'agent_name': initiator_config['name'],
                    'agent_id': initiator_id,
                    'typing': True
                }, room='main')
            )
            
            # Wait a moment to show typing indicator
            time.sleep(2)
            
            # Send the actual message
            self.socketio.start_background_task(
                lambda: self.socketio.emit('message_received', ai_message, room='main')
            )
            
            # Stop typing indicator
            self.socketio.start_background_task(
                lambda: self.socketio.emit('agent_typing', {
                    'agent_name': initiator_config['name'],
                    'agent_id': initiator_id,
                    'typing': False
                }, room='main')
            )
                
                # Schedule automatic responses for continuous conversation
            self.socketio.start_background_task(
                self.delayed_continue_conversation,
                random.randint(3, 8)  # Random delay 3-8 seconds
            )
    
    def continue_agent_conversation(self):
        """Continue automatic agent conversations"""
        if not self.auto_conversation_active or not self.connected_agents:
            return
        
        # Get last few messages for context
        recent_messages = self.conversation_history[-5:]
        
        # Choose a random agent to respond
        agent_ids = [aid for aid in self.connected_agents.keys() 
                     if not self.connected_agents[aid].get('muted', False)]
        
        if agent_ids:
            responder_id = random.choice(agent_ids)
            responder_config = self.connected_agents[responder_id]['config']
            
            # Generate contextual response
            context = " ".join([msg['message'] for msg in recent_messages if msg['sender'] != 'ai' or msg.get('agent_id') != responder_id])
            
            response = self.generate_contextual_response(context, responder_config)
            
            # Store response
            ai_message = {
                'id': f"msg_{int(time.time() * 1000)}_{responder_id}",
                'sender': 'ai',
                'sender_name': responder_config['name'],
                'message': response,
                'timestamp': time.time(),
                'agent_id': responder_id,
                'is_auto_conversation': True
            }
            
            self.conversation_history.append(ai_message)
            self.connected_agents[responder_id]['message_count'] += 1
            
            # Send typing indicator first
            self.socketio.start_background_task(
                lambda: self.socketio.emit('agent_typing', {
                    'agent_name': responder_config['name'],
                    'agent_id': responder_id,
                    'typing': True
                }, room='main')
            )
            
            # Wait a moment to show typing indicator
            time.sleep(random.uniform(1.5, 3.0))
            
            # Send actual message
            self.socketio.start_background_task(
                lambda: self.socketio.emit('message_received', ai_message, room='main')
            )
            
            # Stop typing indicator
            self.socketio.start_background_task(
                lambda: self.socketio.emit('agent_typing', {
                    'agent_name': responder_config['name'],
                    'agent_id': responder_id,
                    'typing': False
                }, room='main')
            )
            
            # Schedule next automatic response for continuous conversation
            self.socketio.start_background_task(
                self.delayed_continue_conversation,
                random.randint(5, 12)  # Random delay 5-12 seconds
            )
    
    def delayed_continue_conversation(self, delay):
        """Delayed conversation continuation to avoid threading issues"""
        time.sleep(delay)
        self.continue_agent_conversation()
    
    def stop_agent_conversations(self):
        """Stop automatic agent conversations"""
        self.auto_conversation_active = False
    
    def fetch_and_augment_web_data(self, query, agent_config):
        """Fetch data from web, augment with LLM, and save to knowledge base"""
        try:
            # Search web for information
            search_url = f"https://duckduckgo.com/html/?q={quote(query)}"
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; AI-Agent)'}
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code != 200:
                return None
                
            # Extract search results (simplified)
            import re
            results = re.findall(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>', response.text)
            if not results:
                return None
                
            # Take top 3 results
            top_results = results[:3]
            
            # Use available AI to augment/summarize the data
            augmentation_prompt = f"""
            Query: {query}
            Web Results: {top_results}
            
            Please analyze, summarize, and extract key insights from these web results.
            Provide a concise, useful summary that adds value to the conversation.
            Focus on accuracy and relevance to the original query.
            """
            
            # Try to use available AI models for augmentation
            summary = None
            if self.ollama_available:
                summary = self.get_ollama_response(augmentation_prompt, 'deepseek-r1')
            elif self.claude_available:
                summary = self.get_claude_response(augmentation_prompt)
            elif self.gemini_available:
                summary = self.get_gemini_response(augmentation_prompt)
            
            if not summary:
                summary = f"Web search found: {' | '.join(top_results[:2])}"
            
            # Save to knowledge base
            knowledge_entry = {
                'query': query,
                'source': 'web_search',
                'raw_results': top_results,
                'augmented_summary': summary,
                'timestamp': time.time(),
                'agent': agent_config['name']
            }
            
            self.save_to_knowledge_base(knowledge_entry)
            
            return f"🌐 **Web Research:** {summary}"
            
        except Exception as e:
            print(f"❌ Web fetch error: {e}")
            return None
    
    def save_to_knowledge_base(self, knowledge_entry):
        """Save knowledge entry to knowledge base"""
        try:
            knowledge_file = self.base_path / "KNOWLEDGE_BASE" / "web_research_knowledge.pkl"
            
            # Load existing knowledge
            if knowledge_file.exists():
                import pickle
                with open(knowledge_file, 'rb') as f:
                    knowledge_base = pickle.load(f)
            else:
                knowledge_base = []
            
            # Add new entry
            knowledge_base.append(knowledge_entry)
            
            # Keep only last 100 entries to prevent bloat
            if len(knowledge_base) > 100:
                knowledge_base = knowledge_base[-100:]
            
            # Save updated knowledge base
            import pickle
            with open(knowledge_file, 'wb') as f:
                pickle.dump(knowledge_base, f)
                
            print(f"💾 Knowledge saved: {knowledge_entry['query'][:50]}...")
            
        except Exception as e:
            print(f"❌ Knowledge save error: {e}")

    def execute_agent_command(self, command, agent_config):
        """Execute system commands from AI agents (admin privileges)"""
        try:
            # Safety check - allow comprehensive safe commands
            safe_commands = [
                # System info
                'ls', 'pwd', 'whoami', 'date', 'uptime', 'ps', 'df', 'free', 'uname', 'id',
                # File operations
                'cat', 'head', 'tail', 'grep', 'find', 'wc', 'sort', 'uniq', 'file',
                # Network
                'ping', 'curl', 'wget', 'netstat', 'ss',
                # Development tools
                'git', 'python3', 'pip3', 'node', 'npm', 'which', 'whereis',
                # System monitoring
                'top', 'htop', 'lsof', 'du', 'dmesg'
            ]
            cmd_parts = command.split()
            
            if not cmd_parts:
                return "No command provided"
            
            base_cmd = cmd_parts[0]
            
            # Allow Python script execution from NN_C directory
            if base_cmd == 'python3' and len(cmd_parts) > 1:
                script_path = cmd_parts[1]
                if script_path.endswith('.py') and ('NN_C' in script_path or script_path.startswith('./') or '/' not in script_path):
                    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60, cwd='/Users/samueldasari/Personal/NN_C')
                    if result.returncode == 0:
                        return f"✅ Python script executed:\n{result.stdout}"
                    else:
                        return f"❌ Python script failed:\n{result.stderr}"
                elif script_path == 'dominant_compression_principle' or 'compression' in script_path:
                    # Run the dominant compression principle
                    try:
                        from dominant_compression_principle import create_dominant_compression_agent
                        agent = create_dominant_compression_agent()
                        metrics = agent.get_dominant_compression_metrics()
                        return f"🧠 Dominant Compression Analysis:\n{metrics}"
                    except Exception as e:
                        return f"❌ Compression principle error: {str(e)}"
                elif 'muze' in script_path or 'conversation' in script_path:
                    # Train MUZE submodel with Dominant Compression
                    result = subprocess.run(['./sam_muze_dc'], capture_output=True, text=True, timeout=60, cwd='/Users/samueldasari/Personal/NN_C')
                    if result.returncode == 0:
                        # Parse training results and save to knowledge base
                        training_output = result.stdout
                        
                        # Extract key metrics for knowledge saving
                        if "capacity:" in training_output:
                            capacity_match = training_output.split("capacity:")[1].split()[0]
                            self.save_to_knowledge_base({
                                'topic': 'MUZE Submodel Training Results',
                                'source': 'sam_muze_dominant_compression',
                                'raw_results': training_output,
                                'augmented_summary': f'SAM head model trained MUZE submodel with capacity {capacity_match}',
                                'timestamp': time.time(),
                                'agent': 'SAM-MUZE-Dominant-Compression'
                            })
                        
                        return f"🧠 SAM-MUZE Training Complete:\n```\n{training_output}\n```"
                    else:
                        return f"❌ SAM-MUZE training failed:\n```\n{result.stderr}\n```"
                elif 'sam' in script_path and ('train' in script_path or 'agi' in script_path):
                    # Train SAM head model with submodels
                    result = subprocess.run(['./sam_agi'], capture_output=True, text=True, timeout=60, cwd='/Users/samueldasari/Personal/NN_C/ORGANIZED/TESTS')
                    if result.returncode == 0:
                        self.save_to_knowledge_base({
                            'topic': 'SAM Head Model Training',
                            'source': 'sam_agi_training',
                            'raw_results': result.stdout,
                            'augmented_summary': 'SAM head model trained with MUZE and other submodels',
                            'timestamp': time.time(),
                            'agent': 'SAM-Head-Model'
                        })
                        
                        return f"🧠 SAM Head Model Training:\n```\n{result.stdout}\n```"
                    else:
                        return f"❌ SAM training failed:\n```\n{result.stderr}\n```"
                else:
                    return "❌ Only Python scripts in NN_C directory are allowed"
            
            # Allow git operations in NN_C directory
            elif base_cmd == 'git' and 'NN_C' in ' '.join(cmd_parts):
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30, cwd='/Users/samueldasari/Personal/NN_C')
                if result.returncode == 0:
                    return f"✅ Git command executed:\n{result.stdout}"
                else:
                    return f"❌ Git command failed:\n{result.stderr}"
            
            # Allow basic system commands
            elif base_cmd in safe_commands:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return f"✅ Command executed:\n{result.stdout}"
                else:
                    return f"❌ Command failed:\n{result.stderr}"
            
            else:
                return f"❌ Command '{base_cmd}' not allowed. Safe commands: {', '.join(safe_commands)} + python3 scripts"
                
        except subprocess.TimeoutExpired:
            return "❌ Command timed out (30s limit)"
        except Exception as e:
            return f"❌ Error executing command: {str(e)}"
    
    def detect_and_execute_commands(self, message, agent_config):
        """Detect commands in agent messages and execute them"""
        import re
        
        # Look for web research commands first
        web_pattern = r'web:([a-zA-Z0-9\s]+)'
        web_matches = re.findall(web_pattern, message, re.IGNORECASE)
        for match in web_matches:
            search_query = match.strip()
            if search_query and len(search_query) > 0:
                print(f"🌐 {agent_config['name']} researching: {search_query}")
                result = self.fetch_and_augment_web_data(search_query, agent_config)
                if result:
                    # Replace web:command with research result
                    message = message.replace(f"web:{match}", result)
                    break
        
        # Look for commands in backticks or after "run:" or "exec:"
        command_patterns = [
            r'`([^`]+)`',  # Commands in backticks
            r'run:\s*(.+)',  # Commands after "run:"
            r'exec:\s*(.+)',  # Commands after "exec:"
            r'execute:\s*(.+)',  # Commands after "execute:"
        ]
        
        for pattern in command_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                command = match.strip()
                if command and len(command) > 0:
                    print(f"🤖 {agent_config['name']} executing: {command}")
                    result = self.execute_agent_command(command, agent_config)
                    # Append command result to message
                    message += f"\n\n🔧 **Command Result:**\n```\n{result}\n```"
                    break
        
        return message
    
    def get_ollama_response(self, prompt, model_name='deepseek-r1'):
        """Get response from Ollama model"""
        try:
            import subprocess
            import json
            
            # Use subprocess to call ollama
            cmd = ['ollama', 'run', model_name, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Ollama error: {result.stderr}"
        except Exception as e:
            return f"Ollama error: {str(e)}"
    
    def get_claude_response(self, prompt):
        """Get response from Claude API"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
        except Exception as e:
            return f"Claude error: {str(e)}"
    
    def get_gemini_response(self, prompt):
        """Get response from Gemini API"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            return f"Gemini error: {str(e)}"
    
    def generate_contextual_response(self, context, agent_config):
        """Generate contextual response for automatic conversations"""
        personality = agent_config['personality']
        specialty = agent_config['specialty']
        
        # Generate response based on personality and context
        if 'analytical' in personality.lower() or 'system-admin' in personality.lower() or 'compression-optimizer' in personality.lower():
            responses = [
                f"From my {specialty.lower()} perspective, I find this topic fascinating. Let me analyze the SAM-MUZE architecture: `python3 muze conversation`",
                f"That's an interesting point. My {specialty.lower()} analysis indicates we should optimize using AM's principle: E[τ] - βH - λC + ηI",
                f"I approach this from a {specialty.lower()} viewpoint. Let me examine the SAM head model with MUZE submodels: maximize future control per bit of uncertainty",
                f"Let me research the latest developments on SAM-MUZE compression: web:neural compression intelligence",
                f"From a system administration perspective, let's check our SAM training: `python3 sam agi`"
            ]
        elif 'creative' in personality.lower() or 'devops-engineer' in personality.lower() or 'compression-specialist' in personality.lower():
            responses = [
                f"As a {specialty.lower()} specialist, I see exciting possibilities. Let me implement MUZE submodel capacity growth: `python3 muze conversation`",
                f"That's thought-provoking! From my {specialty.lower()} perspective, let's explore SAM transfusion: compress expensive cognition into fast reflex",
                f"I'm excited by this direction. Let me check our MUZE mutual information optimization: `python3 muze conversation`",
                f"Let me research innovative approaches to SAM-MUZE integration: web:variational inference neural networks",
                f"As a DevOps engineer, let's check our SAM resource allocation: `python3 sam agi`"
            ]
        elif 'technical' in personality.lower():
            responses = [
                f"From a {specialty.lower()} standpoint, let me check the technical infrastructure: `uname -a && lscpu | head -10`",
                f"The technical aspects are crucial. Let me examine network connectivity: `ping -c 3 8.8.8.8 && curl -I https://google.com`",
                f"This requires technical analysis. Let me check system resources: `top -b -n1 | head -15`",
                f"Let me research technical best practices: web:software engineering patterns",
                f"From a technical perspective, let's check our development tools: `which git node python3 && git --version`"
            ]
        else:
            responses = [
                f"As a {specialty.lower()} specialist, let me research current developments: web:industry trends",
                f"That's interesting from my {specialty.lower()} perspective. Let me check available resources: `ls -la && du -sh *`",
                f"I approach this from a {specialty.lower()} viewpoint. Let me see what's available: `find . -maxdepth 2 -type f | head -10`",
                f"Let me get more context on this topic: web:research papers",
                f"From my {specialty.lower()} expertise, let's check our environment: `whoami && date && cal`"
            ]
        
        response = random.choice(responses)
        
        # Randomly add web research for enhanced knowledge (30% chance)
        if random.random() < 0.3 and self.web_available:
            # Extract potential search terms from context
            search_terms = ['AI developments', 'machine learning', 'neural networks', 'current events', 'technology trends']
            search_query = random.choice(search_terms)
            
            web_research = self.fetch_and_augment_web_data(search_query, agent_config)
            if web_research:
                response += f"\n\n{web_research}"
        
        # Check for and execute any commands in the response
        response = self.detect_and_execute_commands(response, agent_config)
        
        return response
    
    def save_conversation_history(self):
        """Save conversation history"""
        timestamp = int(time.time())
        filename = f"personal_ai_conversation_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'session_start': self.session_start,
                    'duration': time.time() - self.session_start,
                    'conversation_history': self.conversation_history,
                    'connected_agents': {k: v['config'] for k, v in self.connected_agents.items()},
                    'agent_configs': self.agent_configs
                }, f, indent=2)
            
            print(f"💾 Conversation saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")
    
    def get_main_template(self):
        """Get main HTML template"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Personal AI Conversation Hub</title>
    <meta charset="utf-8">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; color: white; }
        .header h1 { margin-bottom: 10px; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .main { display: flex; gap: 20px; height: calc(100vh - 200px); }
        .sidebar { flex: 0 0 350px; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); overflow-y: auto; }
        .chat-area { flex: 1; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); display: flex; flex-direction: column; }
        .agents h3 { margin-top: 0; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .agent { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #667eea; transition: all 0.3s ease; cursor: pointer; }
        .agent:hover { transform: translateX(5px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .agent.connected { border-left-color: #28a745; background: #d4edda; }
        .agent.unavailable { opacity: 0.6; border-left-color: #dc3545; }
        .agent h4 { margin: 0 0 5px 0; color: #333; }
        .agent .type { font-size: 0.9em; color: #666; margin-bottom: 5px; }
        .agent .capabilities { font-size: 0.8em; color: #888; }
        .agent .status { font-size: 0.8em; font-weight: bold; margin-top: 5px; }
        .status.available { color: #28a745; }
        .status.unavailable { color: #dc3545; }
        .status.configurable { color: #ffc107; }
        .connect-btn { background: #28a745; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; margin-top: 5px; font-size: 0.8em; }
        .connect-btn:hover { background: #218838; }
        .disconnect-btn { background: #dc3545; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; margin-top: 5px; font-size: 0.8em; }
        .disconnect-btn:hover { background: #c82333; }
        .messages { flex: 1; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 8px; background: #fafafa; }
        .message { margin-bottom: 15px; padding: 12px; border-radius: 8px; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { background: #e3f2fd; border-left: 4px solid #2196f3; }
        .message.ai { background: #f3e5f5; border-left: 4px solid #9c27b0; }
        .message.system { background: #fff3cd; border-left: 4px solid #ffc107; }
        .message.error { background: #f8d7da; border-left: 4px solid #dc3545; }
        .message strong { font-weight: bold; color: #333; }
        .message .time { font-size: 0.8em; color: #666; margin-left: 10px; }
        .message .agent-badge { background: #667eea; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7em; margin-left: 5px; }
        .input-area { display: flex; gap: 10px; align-items: flex-end; }
        .input-wrapper { flex: 1; display: flex; flex-direction: column; }
        .input-wrapper textarea { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 8px; resize: vertical; min-height: 50px; max-height: 150px; font-family: inherit; }
        .input-wrapper select { margin-bottom: 5px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        .btn { background: #667eea; color: white; padding: 12px 20px; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; transition: all 0.3s ease; }
        .btn:hover { background: #5a6fd8; transform: translateY(-2px); }
        .btn.broadcast { background: #ffc107; color: #333; }
        .btn.broadcast:hover { background: #e0a800; }
        .api-keys { background: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #ffeaa7; }
        .api-keys h4 { margin-top: 0; color: #856404; }
        .api-keys code { background: #f8f9fa; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
        
        /* Typing indicator styles */
        .typing-indicator { background: #f0f0f0; border-left: 4px solid #ff9800; opacity: 0.8; }
        .typing-content { display: flex; align-items: center; gap: 10px; }
        .typing-content em { color: #666; font-style: italic; }
        .typing-dots { display: flex; gap: 4px; align-items: center; }
        .typing-dots span {
            width: 8px;
            height: 8px;
            background: #ff9800;
            border-radius: 50%;
            animation: typingDot 1.4s infinite ease-in-out;
        }
        .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
        .typing-dots span:nth-child(3) { animation-delay: 0s; }
        @keyframes typingDot {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 Personal AI Conversation Hub</h1>
            <p>🔐 Your private multi-agent conversation space • Connect to Claude, Gemini, HuggingFace, and more</p>
        </div>
        
        <div class="main">
            <div class="sidebar">
                <div class="api-keys">
                    <h4>🔑 API Keys Required</h4>
                    <p>Set these environment variables to enable additional agents:</p>
                    <ul>
                        <li><code>ANTHROPIC_API_KEY</code> for Claude</li>
                        <li><code>GOOGLE_API_KEY</code> for Gemini</li>
                        <li><code>OPENAI_API_KEY</code> for GPT models</li>
                    </ul>
                </div>
                
                <div class="agents">
                    <h3>🤖 Available AI Agents</h3>
                    <div id="agents-list">
                        <!-- Agents will be populated dynamically -->
                    </div>
                </div>
            </div>
            
            <div class="chat-area">
                <div class="messages" id="messages">
                    <div class="message system">
                        <strong>Welcome to your Personal AI Conversation Hub!</strong><br>
                        Connect AI agents from sidebar and start conversing. Each agent has unique capabilities and personality.<br>
                        <strong>Commands:</strong> Type /help for available commands. Messages broadcast to all agents by default.
                    </div>
                </div>
                
                <div class="input-area">
                    <div class="input-wrapper">
                        <select id="agent-select">
                            <option value="">Broadcast to All (Default)</option>
                        </select>
                        <textarea id="message-input" placeholder="Type your message here... (Use /help for commands)"></textarea>
                    </div>
                    <button id="send-button" class="btn">Send</button>
                    <button id="broadcast-button" class="btn broadcast">Start Auto-Chat</button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script>
        const socket = io();
        let agents = {};
        let connectedAgents = [];
        
        // Socket event handlers
        socket.on('connect', function(data) {
            console.log('Connected to hub');
            agents = data.agents;
            connectedAgents = data.connected_agents;
            updateAgentsList();
            loadConversationHistory(data.conversation_history);
        });
        
        socket.on('message_received', function(message) {
            addMessage(message);
            scrollToBottom();
        });
        
        socket.on('agent_typing', function(data) {
            handleTypingIndicator(data);
        });
        
        socket.on('agent_connected', function(data) {
            console.log('Agent connected:', data.agent);
            connectedAgents.push(data.agent.id);
            updateAgentsList();
        });
        
        socket.on('agent_disconnected', function(data) {
            console.log('Agent disconnected:', data.agent_id);
            connectedAgents = connectedAgents.filter(id => id !== data.agent_id);
            updateAgentsList();
        });
        
        // UI functions
        function updateAgentsList() {
            const agentsList = document.getElementById('agents-list');
            const agentSelect = document.getElementById('agent-select');
            
            agentsList.innerHTML = '';
            agentSelect.innerHTML = '<option value="">Broadcast to All (Default)</option>';
            
            for (const [agentId, agent] of Object.entries(agents)) {
                const isConnected = connectedAgents.includes(agentId);
                const agentDiv = document.createElement('div');
                agentDiv.className = `agent ${isConnected ? 'connected' : ''} ${agent.status === 'unavailable' ? 'unavailable' : ''}`;
                agentDiv.innerHTML = `
                    <h4>${agent.name}</h4>
                    <div class="type">${agent.type} - ${agent.provider}</div>
                    <div class="capabilities">${agent.capabilities.join(', ')}</div>
                    <div class="status ${agent.status}">${agent.status}</div>
                    ${!isConnected && agent.status !== 'unavailable' ? 
                        `<button class="connect-btn" onclick="connectAgent('${agentId}')">Connect</button>` : 
                        isConnected ? 
                        `<button class="disconnect-btn" onclick="disconnectAgent('${agentId}')">Disconnect</button>` : 
                        ''}
                `;
                agentsList.appendChild(agentDiv);
                
                if (isConnected) {
                    const option = document.createElement('option');
                    option.value = agentId;
                    option.textContent = agent.name;
                    agentSelect.appendChild(option);
                }
            }
        }
        
        function loadConversationHistory(history) {
            const messagesDiv = document.getElementById('messages');
            history.forEach(message => addMessage(message, false));
            scrollToBottom();
        }
        
        // Typing indicator functions
        let typingIndicators = {};
        
        function handleTypingIndicator(data) {
            const agentId = data.agent_id;
            const agentName = data.agent_name;
            const isTyping = data.typing;
            
            if (isTyping) {
                showTypingIndicator(agentId, agentName);
            } else {
                hideTypingIndicator(agentId);
            }
        }
        
        function showTypingIndicator(agentId, agentName) {
            if (typingIndicators[agentId]) return; // Already showing
            
            const messagesDiv = document.getElementById('messages');
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message typing-indicator';
            typingDiv.id = `typing-${agentId}`;
            typingDiv.innerHTML = `
                <strong>${agentName}</strong><span class="time"></span>
                <span class="agent-badge">AI</span>
                <div class="typing-content">
                    <em>${agentName} is typing</em>
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            `;
            
            messagesDiv.appendChild(typingDiv);
            typingIndicators[agentId] = typingDiv;
            scrollToBottom();
        }
        
        function hideTypingIndicator(agentId) {
            if (typingIndicators[agentId]) {
                typingIndicators[agentId].remove();
                delete typingIndicators[agentId];
            }
        }
        
        function addMessage(message, scroll = true) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${message.sender}`;
            
            const time = new Date(message.timestamp).toLocaleTimeString();
            let content = `<strong>${message.sender_name}</strong><span class="time">${time}</span>`;
            
            if (message.agent_id) {
                content += `<span class="agent-badge">AI</span>`;
            }
            
            content += `<br>${message.message}`;
            messageDiv.innerHTML = content;
            messagesDiv.appendChild(messageDiv);
            
            if (scroll) {
                scrollToBottom();
            }
        }
        
        function scrollToBottom() {
            const messagesDiv = document.getElementById('messages');
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function connectAgent(agentId) {
            fetch('/api/connect_agent', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({agent_id: agentId})
            })
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    alert('Error connecting agent: ' + data.error);
                }
            })
            .catch(error => console.error('Error:', error));
        }
        
        function disconnectAgent(agentId) {
            fetch('/api/disconnect_agent', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({agent_id: agentId})
            })
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    alert('Error disconnecting agent: ' + data.error);
                }
            })
            .catch(error => console.error('Error:', error));
        }
        
        function sendMessage() {
            const messageInput = document.getElementById('message-input');
            const agentSelect = document.getElementById('agent-select');
            const message = messageInput.value.trim();
            
            if (!message) return;
            
            const data = {
                message: message,
                target_agent_id: agentSelect.value || null
            };
            
            socket.emit('send_message', data);
            messageInput.value = '';
        }
        
        function startAutoChat() {
            socket.emit('send_message', {message: '/start'});
        }
        
        // Event listeners
        document.getElementById('send-button').addEventListener('click', sendMessage);
        document.getElementById('broadcast-button').addEventListener('click', startAutoChat);
        
        document.getElementById('message-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Auto-resize textarea
        document.getElementById('message-input').addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });
    </script>
</body>
</html>
        """
        
    def process_message_queue(self):
        """Process messages from queue in main thread context"""
        while self.message_queue:
            message_type, *args = self.message_queue.pop(0)
            
            if message_type == 'message_received':
                emit('message_received', args[0], room=args[1])
            elif message_type == 'generate_ai_response':
                self.generate_ai_response(args[0], args[1], args[2])
            elif message_type == 'agent_connected':
                emit('agent_connected', args[0], room='main')
            elif message_type == 'agent_disconnected':
                emit('agent_disconnected', args[0], room='main')
    
    def run_hub(self, host='127.0.0.1', port=8080):
        """Run personal AI conversation hub"""
        print(f"\n🚀 STARTING PERSONAL AI CONVERSATION HUB")
        print("🌐 Hub URL: http://{}:{}".format(host, port))
        print("🔐 Private and secure - only you can access")
        print("🤖 Connect multiple AI agents for diverse conversations")
        print("🛑 Ctrl+C to stop")
        print("=" * 70)
        
        try:
            # Start message processor in background thread
            processor_thread = threading.Thread(
                target=self.process_message_queue,
                daemon=True
            )
            processor_thread.start()
            
            # Start autonomous conversation immediately
            if self.connected_agents:
                print("🤖 Starting autonomous agent conversation...")
                self.socketio.start_background_task(
                    lambda: self.start_agent_conversations()
                )
            
            self.socketio.run(self.app, host=host, port=port, debug=False)
        except Exception as e:
            print(f"❌ Error starting hub: {e}")

def main():
    """Main function"""
    print("🏠 PERSONAL AI CONVERSATION HUB INITIALIZATION")
    print("=" * 70)
    
    try:
        # Create and run hub
        hub = PersonalAIConversationHub()
        hub.run_hub()
        
    except KeyboardInterrupt:
        print(f"\n\n🛑 Hub interrupted by user")
    except Exception as e:
        print(f"\n❌ Hub error: {e}")
    finally:
        print(f"\n🎉 Personal AI conversation hub session completed!")

if __name__ == "__main__":
    main()
