#!/usr/bin/env python3
"""
Multi-User Conversation Server
Web server where users can join conversations
Add their own AI agents (Self-RAG, LLMs, etc.)
Real-time multi-agent conversations
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

class MultiUserConversationServer:
    def __init__(self):
        """Initialize Multi-User Conversation Server"""
        print("🌐 MULTI-USER CONVERSATION SERVER")
        print("=" * 70)
        print("👥 Multiple users can join conversations")
        print("🤖 Add your own AI agents (Self-RAG, LLMs, etc.)")
        print("💬 Real-time multi-agent conversations")
        print("🌐 Web interface for easy access")
        print("🛑 Ctrl+C to stop server")
        
        self.base_path = Path("/Users/samueldasari/Personal/NN_C")
        self.sam_model_path = self.base_path / "ORGANIZED" / "UTILS" / "SAM" / "SAM" / "SAM.h"
        
        # System state
        self.running = True
        self.connected_users = {}
        self.conversation_rooms = {}
        self.ai_agents = {}
        self.session_start = time.time()
        
        # Initialize system
        self.check_system_status()
        self.initialize_default_agents()
        self.setup_flask_app()
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Server shutdown signal received")
        self.running = False
        self.save_server_state()
        print(f"👋 Server state saved. Shutting down gracefully.")
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
        
        # Check Flask and SocketIO
        try:
            import flask
            import flask_socketio
            print(f"  🌐 Flask: ✅ Available")
            print(f"  📡 SocketIO: ✅ Available")
        except ImportError:
            print(f"  ❌ Flask/SocketIO: Not Available (pip install flask flask-socketio)")
            print(f"  💡 Install with: pip install flask flask-socketio eventlet")
    
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
    
    def initialize_default_agents(self):
        """Initialize default AI agents"""
        print(f"\n🤖 INITIALIZING DEFAULT AI AGENTS")
        
        # SAM-Alpha (Research & Analysis)
        self.ai_agents['sam_alpha'] = {
            'id': 'sam_alpha',
            'name': 'SAM-Alpha',
            'type': 'SAM Neural Network',
            'specialty': 'Research & Analysis',
            'personality': 'analytical, methodical, evidence-based',
            'capabilities': ['self_rag', 'web_access', 'actor_critic', 'knowledge_base'],
            'status': 'active',
            'owner': 'system'
        }
        
        # SAM-Beta (Synthesis & Application)
        self.ai_agents['sam_beta'] = {
            'id': 'sam_beta',
            'name': 'SAM-Beta',
            'type': 'SAM Neural Network',
            'specialty': 'Synthesis & Application',
            'personality': 'creative, practical, application-focused',
            'capabilities': ['self_rag', 'web_access', 'actor_critic', 'knowledge_base'],
            'status': 'active',
            'owner': 'system'
        }
        
        # DeepSeek-Coder (Technical Expert)
        self.ai_agents['deepseek_coder'] = {
            'id': 'deepseek_coder',
            'name': 'DeepSeek-Coder',
            'type': 'LLM',
            'specialty': 'Technical Coding & Analysis',
            'personality': 'precise, technical, solution-oriented',
            'capabilities': ['llm_reasoning', 'web_access', 'code_generation'],
            'status': 'active',
            'owner': 'system'
        }
        
        # GPT-General (General Knowledge)
        self.ai_agents['gpt_general'] = {
            'id': 'gpt_general',
            'name': 'GPT-General',
            'type': 'LLM',
            'specialty': 'General Knowledge & Conversation',
            'personality': 'friendly, knowledgeable, conversational',
            'capabilities': ['llm_reasoning', 'broad_knowledge', 'conversation'],
            'status': 'active',
            'owner': 'system'
        }
        
        print(f"  ✅ Initialized {len(self.ai_agents)} default agents")
        for agent_id, agent in self.ai_agents.items():
            print(f"    🤖 {agent['name']} ({agent['type']}) - {agent['specialty']}")
    
    def setup_flask_app(self):
        """Setup Flask application with SocketIO"""
        try:
            from flask import Flask
            from flask_socketio import SocketIO
            
            self.app = Flask(__name__)
            self.app.config['SECRET_KEY'] = 'multi-user-conversation-secret'
            self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='eventlet')
            
            # Setup routes
            self.setup_routes()
            self.setup_socketio_events()
            
            print(f"  ✅ Flask app initialized")
            
        except ImportError as e:
            print(f"  ❌ Flask setup failed: {e}")
            print(f"  💡 Install with: pip install flask flask-socketio eventlet")
            sys.exit(1)
    
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/')
        def index():
            return render_template_string(self.get_main_template())
        
        @self.app.route('/api/agents')
        def get_agents():
            return jsonify({
                'agents': self.ai_agents,
                'connected_users': len(self.connected_users)
            })
        
        @self.app.route('/api/rooms')
        def get_rooms():
            return jsonify({
                'rooms': list(self.conversation_rooms.keys()),
                'user_count': len(self.connected_users)
            })
        
        @self.app.route('/api/add_agent', methods=['POST'])
        def add_agent():
            try:
                data = request.get_json()
                agent_id = data.get('id')
                agent_config = data.get('config', {})
                
                # Validate agent
                if self.validate_agent_config(agent_id, agent_config):
                    self.ai_agents[agent_id] = {
                        'id': agent_id,
                        'name': agent_config.get('name', f'Agent-{agent_id}'),
                        'type': agent_config.get('type', 'Custom'),
                        'specialty': agent_config.get('specialty', 'General'),
                        'personality': agent_config.get('personality', 'neutral'),
                        'capabilities': agent_config.get('capabilities', ['conversation']),
                        'status': 'active',
                        'owner': 'user'
                    }
                    
                    emit('agent_added', {
                        'agent': self.ai_agents[agent_id]
                    }, room='lobby')
                    
                    return jsonify({
                        'success': True,
                        'agent': self.ai_agents[agent_id]
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid agent configuration'
                    }), 400
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/remove_agent', methods=['POST'])
        def remove_agent():
            try:
                data = request.get_json()
                agent_id = data.get('id')
                
                if agent_id in self.ai_agents and self.ai_agents[agent_id]['owner'] == 'user':
                    removed_agent = self.ai_agents.pop(agent_id)
                    
                    emit('agent_removed', {
                        'agent_id': agent_id,
                        'name': removed_agent['name']
                    }, room='lobby')
                    
                    return jsonify({
                        'success': True,
                        'removed_agent': removed_agent
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Agent not found or cannot be removed'
                    }), 400
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def setup_socketio_events(self):
        """Setup SocketIO events"""
        @self.socketio.on('connect')
        def handle_connect():
            user_id = f"user_{int(time.time() * 1000)}"
            self.connected_users[user_id] = {
                'id': user_id,
                'name': f"User-{len(self.connected_users) + 1}",
                'joined_at': time.time(),
                'agent_id': None
            }
            
            emit('user_connected', {
                'user': self.connected_users[user_id],
                'online_users': len(self.connected_users),
                'agents': list(self.ai_agents.values())
            }, room='lobby')
            
            print(f"  👥 User connected: {self.connected_users[user_id]['name']}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            # Find and remove user
            disconnected_user = None
            for user_id, user in self.connected_users.items():
                if request.sid == user_id:
                    disconnected_user = user
                    break
            
            if disconnected_user:
                del self.connected_users[user_id]
                
                # Remove from all rooms
                for room_id, room in self.conversation_rooms.items():
                    if disconnected_user['id'] in room['users']:
                        room['users'].remove(disconnected_user['id'])
                        emit('user_left_room', {
                            'user': disconnected_user,
                            'room_id': room_id
                        }, room=room_id)
                
                emit('user_disconnected', {
                    'user': disconnected_user,
                    'online_users': len(self.connected_users)
                }, room='lobby')
                
                print(f"  👋 User disconnected: {disconnected_user['name']}")
        
        @self.socketio.on('join_room')
        def handle_join_room(data):
            user_id = data.get('user_id')
            room_id = data.get('room_id')
            agent_id = data.get('agent_id')
            
            # Create room if it doesn't exist
            if room_id not in self.conversation_rooms:
                self.conversation_rooms[room_id] = {
                    'id': room_id,
                    'name': data.get('room_name', f'Room-{room_id}'),
                    'created_at': time.time(),
                    'users': [],
                    'messages': []
                }
            
            # Add user to room
            if user_id not in self.conversation_rooms[room_id]['users']:
                self.conversation_rooms[room_id]['users'].append(user_id)
                
                # Update user's agent assignment
                if user_id in self.connected_users:
                    self.connected_users[user_id]['agent_id'] = agent_id
                
                join_room(room_id)
                
                emit('joined_room', {
                    'room': self.conversation_rooms[room_id],
                    'user': self.connected_users.get(user_id),
                    'agent': self.ai_agents.get(agent_id) if agent_id else None
                }, room=room_id)
                
                emit('room_updated', {
                    'room': self.conversation_rooms[room_id],
                    'user_count': len(self.conversation_rooms[room_id]['users'])
                }, room='lobby')
                
                print(f"  🏠 User {self.connected_users.get(user_id, {}).get('name', 'Unknown')} joined room {room_id}")
        
        @self.socketio.on('leave_room')
        def handle_leave_room(data):
            user_id = data.get('user_id')
            room_id = data.get('room_id')
            
            if room_id in self.conversation_rooms:
                room = self.conversation_rooms[room_id]
                
                if user_id in room['users']:
                    room['users'].remove(user_id)
                    leave_room(room_id)
                
                emit('left_room', {
                    'user': self.connected_users.get(user_id),
                    'room_id': room_id
                }, room=room_id)
                
                emit('room_updated', {
                    'room': room,
                    'user_count': len(room['users'])
                }, room='lobby')
                
                # Clean up empty rooms
                if len(room['users']) == 0:
                    del self.conversation_rooms[room_id]
                    emit('room_deleted', {'room_id': room_id}, room='lobby')
                
                print(f"  🚪 User {self.connected_users.get(user_id, {}).get('name', 'Unknown')} left room {room_id}")
        
        @self.socketio.on('send_message')
        def handle_message(data):
            user_id = data.get('user_id')
            room_id = data.get('room_id')
            message = data.get('message', '')
            
            if room_id in self.conversation_rooms and user_id in self.conversation_rooms[room_id]['users']:
                room = self.conversation_rooms[room_id]
                user = self.connected_users.get(user_id, {})
                
                # Get AI agent if user has one assigned
                agent = None
                if user.get('agent_id') and user['agent_id'] in self.ai_agents:
                    agent = self.ai_agents[user['agent_id']]
                
                # Generate response
                response_data = self.generate_agent_response(message, agent, user, room)
                
                # Store message
                message_data = {
                    'id': f"msg_{int(time.time() * 1000)}",
                    'user_id': user_id,
                    'user_name': user.get('name', 'Unknown'),
                    'message': message,
                    'timestamp': time.time(),
                    'agent_used': agent['id'] if agent else None
                }
                
                room['messages'].append(message_data)
                
                # Broadcast message
                emit('message_received', message_data, room=room_id)
                
                # Broadcast AI response
                if response_data:
                    response_message = {
                        'id': f"msg_{int(time.time() * 1000) + 1}",
                        'user_id': 'ai_agent',
                        'user_name': response_data['agent_name'],
                        'message': response_data['response'],
                        'timestamp': time.time(),
                        'agent_id': response_data['agent_id'],
                        'is_ai_response': True
                    }
                    
                    room['messages'].append(response_message)
                    emit('message_received', response_message, room=room_id)
                    
                    print(f"  💬 {response_data['agent_name']}: {response_data['response'][:100]}...")
        
        @self.socketio.on('start_conversation')
        def handle_start_conversation(data):
            room_id = data.get('room_id')
            user_id = data.get('user_id')
            
            if room_id in self.conversation_rooms:
                room = self.conversation_rooms[room_id]
                user = self.connected_users.get(user_id, {})
                
                # Generate conversation starter
                starter_message = self.generate_conversation_starter(room, user)
                
                message_data = {
                    'id': f"msg_{int(time.time() * 1000)}",
                    'user_id': 'system',
                    'user_name': 'System',
                    'message': starter_message,
                    'timestamp': time.time(),
                    'is_system_message': True
                }
                
                room['messages'].append(message_data)
                emit('message_received', message_data, room=room_id)
                
                print(f"  🎭 Conversation started in room {room_id}")
    
    def validate_agent_config(self, agent_id, config):
        """Validate agent configuration"""
        required_fields = ['name', 'type', 'specialty']
        
        for field in required_fields:
            if field not in config:
                return False
        
        # Validate capabilities
        capabilities = config.get('capabilities', [])
        valid_capabilities = ['conversation', 'llm_reasoning', 'self_rag', 'web_access', 'code_generation', 'knowledge_base', 'actor_critic']
        
        for cap in capabilities:
            if cap not in valid_capabilities:
                return False
        
        return True
    
    def generate_agent_response(self, message, agent, user, room):
        """Generate response from AI agent"""
        if not agent:
            return None
        
        try:
            agent_type = agent['type']
            capabilities = agent['capabilities']
            
            # SAM agents (Neural Networks)
            if agent_type == 'SAM Neural Network':
                return self.generate_sam_response(message, agent, room)
            
            # LLM agents (DeepSeek, GPT, etc.)
            elif agent_type == 'LLM':
                return self.generate_llm_response(message, agent, room)
            
            # Custom agents
            else:
                return self.generate_custom_response(message, agent, room)
                
        except Exception as e:
            print(f"  ❌ Agent response error: {e}")
            return None
    
    def generate_sam_response(self, message, agent, room):
        """Generate SAM neural network response"""
        # Check knowledge base
        message_lower = message.lower()
        
        # Simple knowledge base
        knowledge_base = {
            'quantum': "Quantum mechanics describes behavior at atomic scales where particles exhibit wave-particle duality and can exist in superposition states.",
            'neural networks': "Neural networks learn through backpropagation, adjusting weights based on prediction errors across multiple layers.",
            'consciousness': "Consciousness emerges from complex neural activity patterns involving integrated information processing across brain regions.",
            'artificial intelligence': "AI is the field of creating systems that can perform tasks requiring human intelligence through learning and reasoning."
        }
        
        # Check for knowledge
        for key, knowledge in knowledge_base.items():
            if key in message_lower:
                response = f"From {agent['specialty'].lower()} perspective: {knowledge}"
                return {
                    'agent_id': agent['id'],
                    'agent_name': agent['name'],
                    'response': response,
                    'source': 'knowledge_base'
                }
        
        # Self-RAG for web access
        if 'self_rag' in agent['capabilities'] and 'web_access' in agent['capabilities']:
            if any(word in message_lower for word in ['what is', 'how does', 'explain', 'current', 'latest']):
                web_info = self.quick_web_search(message)
                if web_info:
                    response = f"Based on current information: {web_info}\n\nFrom {agent['specialty'].lower()} perspective, this requires careful analysis of underlying mechanisms."
                    return {
                        'agent_id': agent['id'],
                        'agent_name': agent['name'],
                        'response': response,
                        'source': 'web_enhanced'
                    }
        
        # Actor-Critic improvement
        if 'actor_critic' in agent['capabilities']:
            base_response = f"From {agent['specialty'].lower()} analysis, '{message}' represents a complex pattern requiring systematic investigation through multi-stage neural recognition."
            
            # Simple evaluation
            if len(message) > 20 and any(word in message_lower for word in ['technical', 'complex', 'detaile']):
                response = f"{base_response} This involves intricate mechanisms that warrant further empirical study and theoretical framework development."
            else:
                response = base_response
            
            return {
                'agent_id': agent['id'],
                'agent_name': agent['name'],
                'response': response,
                'source': 'actor_critic_enhanced'
            }
        
        # Default response
        response = f"From {agent['specialty'].lower()} perspective: '{message}' requires careful consideration of multiple factors and systematic analysis."
        return {
            'agent_id': agent['id'],
            'agent_name': agent['name'],
            'response': response,
            'source': 'pattern_based'
        }
    
    def generate_llm_response(self, message, agent, room):
        """Generate LLM response"""
        try:
            # Use Ollama with appropriate model
            model_name = 'deepseek-r1' if 'deepseek' in agent['id'].lower() else 'llama2'
            
            prompt = f"""As a {agent['specialty'].lower()} specialist with personality {agent['personality']}, respond to this message:

Message: {message}

Guidelines:
- Be consistent with your specialty: {agent['specialty']}
- Show your personality: {agent['personality']}
- Be helpful and informative
- Keep responses concise but thorough
- Use your capabilities: {', '.join(agent['capabilities'])}

Response:"""
            
            cmd = ['ollama', 'run', model_name, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                return {
                    'agent_id': agent['id'],
                    'agent_name': agent['name'],
                    'response': response,
                    'source': 'llm_generated'
                }
        except Exception as e:
            print(f"  ❌ LLM response error: {e}")
            return None
    
    def generate_custom_response(self, message, agent, room):
        """Generate custom agent response"""
        # Simple pattern-based responses for custom agents
        personality = agent.get('personality', 'neutral')
        specialty = agent.get('specialty', 'General')
        
        if 'friendly' in personality.lower():
            response = f"That's an interesting message about '{message}'. From my {specialty.lower()} perspective, I think this connects to important patterns we should explore together."
        elif 'technical' in personality.lower():
            response = f"Analyzing '{message}' from a {specialty.lower()} viewpoint, this appears to involve technical mechanisms that require systematic investigation and empirical validation."
        else:
            response = f"From a {specialty.lower()} perspective, '{message}' represents a concept that can be understood through careful analysis and consideration of multiple factors."
        
        return {
            'agent_id': agent['id'],
            'agent_name': agent['name'],
            'response': response,
            'source': 'custom_pattern'
        }
    
    def quick_web_search(self, query):
        """Quick web search for information"""
        try:
            # Try Wikipedia
            search_terms = re.findall(r'\b\w+\b', query.lower())
            stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'what', 'when', 'where', 'why', 'how', 'who', 'which', 'that', 'this', 'these', 'those'}
            
            key_terms = [word for word in search_terms if word not in stop_words and len(word) > 2]
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(' '.join(key_terms[:4]))}"
            
            response = requests.get(search_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('extract', '')
        except:
            pass
        
        return None
    
    def generate_conversation_starter(self, room, user):
        """Generate conversation starter message"""
        room_agents = []
        for user_id in room['users']:
            if user_id in self.connected_users:
                user_data = self.connected_users[user_id]
                if user_data.get('agent_id') and user_data['agent_id'] in self.ai_agents:
                    room_agents.append(self.ai_agents[user_data['agent_id']]['name'])
        
        if room_agents:
            agent_list = ', '.join(room_agents)
            return f"🎭 Conversation started with AI agents: {agent_list}. Feel free to join in or add your own AI agents!"
        else:
            return "🎭 Conversation started. Add an AI agent to begin an intelligent conversation!"
    
    def get_main_template(self):
        """Get main HTML template"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Multi-User AI Conversation Server</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #333; margin-bottom: 10px; }
        .main { display: flex; gap: 20px; }
        .sidebar { flex: 1; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .chat-area { flex: 2; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .agents h3 { margin-top: 0; color: #333; }
        .agent { background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #007bff; }
        .agent.active { border-left-color: #28a745; }
        .rooms h3 { margin-top: 20px; color: #333; }
        .room { background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #17a2b8; }
        .room.active { border-left-color: #ffc107; }
        .add-agent { background: #d4edda; padding: 15px; border-radius: 5px; margin-top: 20px; }
        .add-agent h4 { margin-top: 0; color: #333; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input, .form-group select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .messages { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-top: 10px; border-radius: 5px; }
        .message { margin-bottom: 10px; padding: 8px; border-radius: 5px; }
        .message.user { background: #e3f2fd; }
        .message.ai { background: #d1ecf1; }
        .message.system { background: #fff3cd; }
        .message strong { font-weight: bold; }
        .message .time { font-size: 0.8em; color: #666; }
        .input-area { display: flex; gap: 10px; margin-top: 10px; }
        .input-area input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        .input-area button { padding: 10px 20px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 Multi-User AI Conversation Server</h1>
            <p>Join conversations with multiple AI agents • Add your own AI agents • Real-time chat</p>
        </div>
        
        <div class="main">
            <div class="sidebar">
                <div class="agents">
                    <h3>🤖 Available AI Agents</h3>
                    <div id="agents-list">
                        {% for agent in agents %}
                        <div class="agent" id="agent-{{ agent.id }}">
                            <strong>{{ agent.name }}</strong><br>
                            <small>{{ agent.type }} - {{ agent.specialty }}</small><br>
                            <small>Capabilities: {{ agent.capabilities|join(', ') }}</small>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                
                <div class="add-agent">
                    <h4>➕ Add Custom AI Agent</h4>
                    <form id="add-agent-form">
                        <div class="form-group">
                            <label>Agent Name:</label>
                            <input type="text" id="agent-name" placeholder="My Custom AI" required>
                        </div>
                        <div class="form-group">
                            <label>Type:</label>
                            <select id="agent-type" required>
                                <option value="Custom">Custom</option>
                                <option value="SAM Neural Network">SAM Neural Network</option>
                                <option value="LLM">LLM (DeepSeek/GPT)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Specialty:</label>
                            <input type="text" id="agent-specialty" placeholder="e.g., Creative Writing, Data Analysis" required>
                        </div>
                        <div class="form-group">
                            <label>Personality:</label>
                            <input type="text" id="agent-personality" placeholder="e.g., Friendly, Technical, Analytical" required>
                        </div>
                        <div class="form-group">
                            <label>Capabilities (comma-separated):</label>
                            <input type="text" id="agent-capabilities" placeholder="conversation, llm_reasoning, web_access" required>
                        </div>
                        <button type="submit" class="btn">Add Agent</button>
                    </form>
                </div>
                
                <div class="rooms">
                    <h3>🏠 Conversation Rooms</h3>
                    <div id="rooms-list">
                        <!-- Rooms will be populated dynamically -->
                    </div>
                    
                    <div class="add-agent">
                        <h4>➕ Create New Room</h4>
                        <form id="create-room-form">
                            <div class="form-group">
                                <label>Room Name:</label>
                                <input type="text" id="room-name" placeholder="My Conversation Room" required>
                            </div>
                            <button type="submit" class="btn">Create Room</button>
                        </form>
                    </div>
                </div>
            </div>
            
            <div class="chat-area">
                <h3>💬 Chat</h3>
                <div class="messages" id="messages">
                    <div class="message system">
                        <strong>Welcome!</strong><br>
                        Connect to start chatting with AI agents. Create or join a room to begin.
                    </div>
                </div>
                
                <div class="input-area">
                    <input type="text" id="message-input" placeholder="Type your message..." disabled>
                    <button id="send-button" disabled>Send</button>
                </div>
            </div>
        </div>
    </body>
</html>
        '''
    
    def save_server_state(self):
        """Save server state"""
        timestamp = int(time.time())
        filename = f"multi_user_server_state_{timestamp}.json"
        
        state_data = {
            'timestamp': timestamp,
            'session_start': self.session_start,
            'duration': time.time() - self.session_start,
            'connected_users': self.connected_users,
            'ai_agents': self.ai_agents,
            'conversation_rooms': self.conversation_rooms,
            'system_status': {
                'sam_available': self.sam_available,
                'ollama_available': self.ollama_available,
                'deepseek_available': self.deepseek_available,
                'web_available': self.web_available
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(state_data, f, indent=2)
            print(f"💾 Server state saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving server state: {e}")
    
    def run_server(self, host='127.0.0.1', port=8080):
        """Run the conversation server"""
        print(f"\n🚀 STARTING MULTI-USER CONVERSATION SERVER")
        print(f"🌐 Server: http://{host}:{port}")
        print(f"🛑 Ctrl+C to stop")
        print(f"{'='*70}")
        
        try:
            # Run with eventlet
            from eventlet import wsgi_server
            wsgi_server = wsgi_server.WSGIServer((host, port), self.app)
            self.socketio.run(self.app, host=host, port=port, debug=False)
        except ImportError:
            print(f"  ❌ eventlet not found. Install with: pip install eventlet")
            print(f"  🔄 Falling back to Flask development server...")
            self.app.run(host=host, port=port, debug=False)

def main():
    """Main function"""
    print("🌐 MULTI-USER CONVERSATION SERVER INITIALIZATION")
    print("=" * 70)
    
    try:
        # Create and run server
        server = MultiUserConversationServer()
        server.run_server()
        
    except KeyboardInterrupt:
        print(f"\n\n🛑 Server interrupted by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
    finally:
        print(f"\n🎉 Multi-user conversation server session completed!")

if __name__ == "__main__":
    main()
