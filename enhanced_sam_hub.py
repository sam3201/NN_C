#!/usr/bin/env python3
"""
Enhanced SAM Hub with Conversation Coherence Monitoring
Integrates coherence monitoring into the main SAM system
"""

import sys
import json
import time
from datetime import datetime

# Add coherence monitor to imports
try:
    from conversation_coherence_monitor import (
        analyze_conversation_coherence, 
        get_coherence_loss_and_reward,
        get_coherence_report
    )
    COHERENCE_MONITOR_AVAILABLE = True
except ImportError:
    COHERENCE_MONITOR_AVAILABLE = False
    print("⚠️ Coherence monitor not available")

def integrate_coherence_monitoring(sam_hub):
    """Integrate coherence monitoring into SAM hub"""
    
    if not COHERENCE_MONITOR_AVAILABLE:
        print("⚠️ Coherence monitoring not available")
        return sam_hub
    
    # Store original methods
    original_process_message = getattr(sam_hub, 'process_message', None)
    original_save_conversation = getattr(sam_hub, 'save_conversation', None)
    
    def enhanced_process_message(message, agent_id=None, **kwargs):
        """Enhanced message processing with coherence monitoring"""
        
        # Get context (last few messages)
        context = []
        if hasattr(sam_hub, 'conversations') and sam_hub.conversations:
            for conv in sam_hub.conversations[-5:]:  # Last 5 messages
                if 'message' in conv:
                    context.append(conv['message'])
        
        # Analyze coherence
        metrics = analyze_conversation_coherence(message, context)
        
        # Calculate loss and reward signals
        loss, reward = get_coherence_loss_and_reward(message, context)
        
        # Log coherence metrics
        print(f"🔍 Coherence Analysis:")
        print(f"   Score: {metrics.overall_score:.3f}")
        print(f"   Loss: {loss:.3f}, Reward: {reward:.3f}")
        if metrics.issues:
            print(f"   Issues: {len(metrics.issues)} found")
        
        # Store coherence data
        coherence_data = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'coherence_score': metrics.overall_score,
            'loss_signal': loss,
            'reward_signal': reward,
            'issues': metrics.issues,
            'suggestions': metrics.suggestions
        }
        
        # Save to knowledge base
        if hasattr(sam_hub, 'save_to_knowledge_base'):
            sam_hub.save_to_knowledge_base('coherence_metrics', coherence_data)
        
        # Call original method
        if original_process_message:
            return original_process_message(message, agent_id, **kwargs)
        else:
            return {"status": "processed", "coherence": metrics.overall_score}
    
    def enhanced_save_conversation(conversation_data):
        """Enhanced conversation saving with coherence metrics"""
        
        # Add coherence metrics to conversation
        if 'messages' in conversation_data:
            for i, msg in enumerate(conversation_data['messages']):
                if 'message' in msg:
                    # Get context (previous messages)
                    context = [m['message'] for m in conversation_data['messages'][:i] if 'message' in m]
                    
                    # Analyze coherence
                    metrics = analyze_conversation_coherence(msg['message'], context)
                    
                    # Add metrics to message
                    msg['coherence_metrics'] = {
                        'score': metrics.overall_score,
                        'grammar': metrics.grammar_score,
                        'relevance': metrics.relevance_score,
                        'clarity': metrics.clarity_score,
                        'completeness': metrics.completeness_score,
                        'issues': metrics.issues
                    }
        
        # Call original method
        if original_save_conversation:
            return original_save_conversation(conversation_data)
    
    # Replace methods
    sam_hub.process_message = enhanced_process_message
    sam_hub.save_conversation = enhanced_save_conversation
    
    # Add coherence monitoring endpoint
    def add_coherence_routes():
        """Add coherence monitoring API routes"""
        
        @sam_hub.app.route('/api/coherence/report')
        def get_coherence_report_endpoint():
            """Get coherence monitoring report"""
            try:
                report = get_coherence_report()
                return jsonify(report)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @sam_hub.app.route('/api/coherence/analyze', methods=['POST'])
        def analyze_coherence_endpoint():
            """Analyze message coherence"""
            try:
                data = request.get_json()
                message = data.get('message', '')
                context = data.get('context', [])
                
                metrics = analyze_conversation_coherence(message, context)
                
                return jsonify({
                    'coherence_score': metrics.overall_score,
                    'grammar_score': metrics.grammar_score,
                    'relevance_score': metrics.relevance_score,
                    'clarity_score': metrics.clarity_score,
                    'completeness_score': metrics.completeness_score,
                    'issues': metrics.issues,
                    'suggestions': metrics.suggestions
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @sam_hub.app.route('/api/coherence/loss_reward', methods=['POST'])
        def get_loss_reward_endpoint():
            """Get loss and reward signals for message"""
            try:
                data = request.get_json()
                message = data.get('message', '')
                context = data.get('context', [])
                
                loss, reward = get_coherence_loss_and_reward(message, context)
                
                return jsonify({
                    'loss_signal': loss,
                    'reward_signal': reward
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    # Add routes
    add_coherence_routes()
    
    print("✅ Coherence monitoring integrated into SAM hub")
    return sam_hub

def create_coherence_dashboard():
    """Create a simple coherence monitoring dashboard"""
    
    dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>SAM 2.0 - Coherence Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .score { font-size: 24px; font-weight: bold; color: #2196F3; }
        .issue { color: #f44336; margin: 5px 0; }
        .suggestion { color: #4CAF50; margin: 5px 0; }
        .chart { width: 100%; height: 200px; background: #e0e0e0; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🔍 SAM 2.0 Conversation Coherence Monitor</h1>
    
    <div class="metric">
        <h3>Overall Coherence Score</h3>
        <div class="score" id="overall-score">0.00</div>
        <div class="chart" id="coherence-chart"></div>
    </div>
    
    <div class="metric">
        <h3>Grammar Score</h3>
        <div class="score" id="grammar-score">0.00</div>
    </div>
    
    <div class="metric">
        <h3>Relevance Score</h3>
        <div class="score" id="relevance-score">0.00</div>
    </div>
    
    <div class="metric">
        <h3>Clarity Score</h3>
        <div class="score" id="clarity-score">0.00</div>
    </div>
    
    <div class="metric">
        <h3>Completeness Score</h3>
        <div class="score" id="completeness-score">0.00</div>
    </div>
    
    <div class="metric">
        <h3>Recent Issues</h3>
        <div id="issues"></div>
    </div>
    
    <div class="metric">
        <h3>Suggestions</h3>
        <div id="suggestions"></div>
    </div>
    
    <script>
        // Update coherence metrics
        function updateCoherence() {
            fetch('/api/coherence/report')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        console.error(data.error);
                        return;
                    }
                    
                    document.getElementById('overall-score').textContent = 
                        (data.average_scores.overall || 0).toFixed(3);
                    document.getElementById('grammar-score').textContent = 
                        (data.average_scores.grammar || 0).toFixed(3);
                    document.getElementById('relevance-score').textContent = 
                        (data.average_scores.relevance || 0).toFixed(3);
                    document.getElementById('clarity-score').textContent = 
                        (data.average_scores.clarity || 0).toFixed(3);
                    document.getElementById('completeness-score').textContent = 
                        (data.average_scores.completeness || 0).toFixed(3);
                    
                    // Display issues
                    const issuesDiv = document.getElementById('issues');
                    issuesDiv.innerHTML = '';
                    if (data.common_issues) {
                        data.common_issues.forEach(([issue, count]) => {
                            const div = document.createElement('div');
                            div.className = 'issue';
                            div.textContent = `${issue} (${count} times)`;
                            issuesDiv.appendChild(div);
                        });
                    }
                })
                .catch(error => console.error('Error:', error));
        }
        
        // Update every 5 seconds
        setInterval(updateCoherence, 5000);
        
        // Initial update
        updateCoherence();
    </script>
</body>
</html>
    """
    
    with open('coherence_dashboard.html', 'w') as f:
        f.write(dashboard_html)
    
    print("✅ Coherence dashboard created: coherence_dashboard.html")

if __name__ == "__main__":
    # Test coherence monitoring
    print("🧪 Testing Conversation Coherence Monitor")
    print("=" * 50)
    
    # Test messages
    test_messages = [
        "Hello, how are you today?",
        "I am doing well, thank you for asking.",
        "Because the weather is nice today.",
        "Yes, I definitely agree with your point about the weather.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    context = []
    for i, message in enumerate(test_messages):
        print(f"\nMessage {i+1}: {message}")
        
        # Analyze coherence
        metrics = analyze_conversation_coherence(message, context)
        loss, reward = get_coherence_loss_and_reward(message, context)
        
        print(f"  Coherence Score: {metrics.overall_score:.3f}")
        print(f"  Loss Signal: {loss:.3f}")
        print(f"  Reward Signal: {reward:.3f}")
        
        if metrics.issues:
            print(f"  Issues: {metrics.issues}")
        
        if metrics.suggestions:
            print(f"  Suggestions: {metrics.suggestions}")
        
        context.append(message)
    
    # Get report
    print(f"\n📊 Coherence Report:")
    report = get_coherence_report()
    print(json.dumps(report, indent=2))
    
    # Create dashboard
    create_coherence_dashboard()
    
    print("\n✅ Coherence monitoring test complete!")
