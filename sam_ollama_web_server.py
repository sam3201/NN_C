#!/usr/bin/env python3
"""
SAM + Ollama Web Server
Connects the web UI to the trained SAM + Ollama chatbot
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import time
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import our chatbot
sys.path.append(str(Path(__file__).parent))

from sam_ollama_chatbot import SAMOllamaChatbot

app = Flask(__name__)
CORS(app)

# Global chatbot instance
chatbot = None

def initialize_chatbot():
    """Initialize the SAM + Ollama chatbot"""
    global chatbot
    try:
        chatbot = SAMOllamaChatbot()
        print("✅ SAM + Ollama chatbot initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        return False

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('sam_chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        if not chatbot:
            return jsonify({'error': 'Chatbot not initialized'}), 500
        
        # Process the message with SAM + Ollama
        start_time = time.time()
        
        # Generate SAM response
        sam_response = chatbot.generate_sam_response(message)
        sam_time = time.time() - start_time
        
        # Get Ollama evaluation
        eval_start = time.time()
        evaluation = chatbot.query_ollama_evaluation(message, sam_response)
        eval_time = time.time() - eval_start
        
        total_time = time.time() - start_time
        
        # Store conversation
        chatbot.conversation_history.append({
            'timestamp': time.time(),
            'user': message,
            'sam_response': sam_response,
            'sam_time': sam_time,
            'evaluation': evaluation,
            'eval_time': eval_time
        })
        
        response_data = {
            'message': sam_response,
            'evaluation': evaluation,
            'timing': {
                'sam_time': sam_time,
                'eval_time': eval_time,
                'total_time': total_time
            },
            'timestamp': time.time()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/status')
def status():
    """Get system status"""
    try:
        if not chatbot:
            return jsonify({'error': 'Chatbot not initialized'}), 500
        
        status_data = {
            'sam_available': chatbot.sam_available,
            'ollama_available': chatbot.ollama_available,
            'conversation_count': len(chatbot.conversation_history),
            'session_duration': time.time() - chatbot.session_start,
            'system_status': 'active'
        }
        
        return jsonify(status_data)
        
    except Exception as e:
        return jsonify({'error': f'Status error: {str(e)}'}), 500

@app.route('/api/save')
def save_conversation():
    """Save conversation history"""
    try:
        if not chatbot:
            return jsonify({'error': 'Chatbot not initialized'}), 500
        
        filename = chatbot.save_conversation()
        return jsonify({'saved': True, 'filename': filename})
        
    except Exception as e:
        return jsonify({'error': f'Save error: {str(e)}'}), 500

@app.route('/api/clear')
def clear_conversation():
    """Clear conversation history"""
    try:
        if not chatbot:
            return jsonify({'error': 'Chatbot not initialized'}), 500
        
        chatbot.conversation_history = []
        return jsonify({'cleared': True})
        
    except Exception as e:
        return jsonify({'error': f'Clear error: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 SAM + Ollama Web Server Starting...")
    print("=" * 50)
    
    # Initialize chatbot
    if not initialize_chatbot():
        print("❌ Failed to start server - chatbot initialization failed")
        sys.exit(1)
    
    print("🌐 Server starting on http://localhost:8080")
    print("🎯 Open your browser and navigate to the chat interface")
    print("💬 SAM will respond and Ollama will evaluate")
    print("⚡ Ready for web-based conversations!")
    
    try:
        app.run(host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
