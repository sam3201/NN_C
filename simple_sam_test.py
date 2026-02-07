#!/usr/bin/env python3
"""
Simple SAM Hub Test
"""

from flask import Flask
import json

app = Flask(__name__)

@app.route('/')
def index():
    return "SAM Hub is working!"

@app.route('/api/agents')
def get_agents():
    agents = [
        {
            'id': 'sam_researcher',
            'name': 'SAM-Researcher',
            'specialty': 'Web Research & Information Gathering',
            'color': '#3498db',
            'capabilities': ['web_research', 'data_collection', 'source_validation', 'fact_checking'],
            'role': 'Research Specialist'
        },
        {
            'id': 'sam_analyst',
            'name': 'SAM-Analyst',
            'specialty': 'Self-RAG & Data Analysis',
            'color': '#2ecc71',
            'capabilities': ['self_rag', 'data_analysis', 'pattern_recognition', 'insight_generation'],
            'role': 'Analysis Specialist'
        },
        {
            'id': 'sam_augmentor',
            'name': 'SAM-Augmentor',
            'specialty': 'Knowledge Augmentation & Synthesis',
            'color': '#9b59b6',
            'capabilities': ['knowledge_augmentation', 'synthesis', 'integration', 'explanation'],
            'role': 'Knowledge Augmentation Specialist'
        }
    ]
    return json.dumps(agents)

@app.route('/api/status')
def get_status():
    return json.dumps({
        'status': '💬 0 messages exchanged | 🤖 3 SAM agents active',
        'typing_agent': None,
        'timestamp': '17:45:00'
    })

if __name__ == "__main__":
    print("🚀 Starting Simple SAM Hub")
    print("🌐 URL: http://127.0.0.1:8082")
    print("🤖 3 SAM agents ready")
    print("=" * 50)
    app.run(host='127.0.0.1', port=8082, debug=False)
