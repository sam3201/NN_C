#!/usr/bin/env python3
"""
Simple launcher for Personal AI Conversation Hub
"""

import subprocess
import webbrowser
import time
import os

def main():
    print("🚀 Starting Personal AI Conversation Hub...")
    print("🌐 This will open your web browser automatically")
    print("🔐 Your private AI conversation space")
    print("🤖 Connect to multiple AI agents")
    print("🛑 Close this window to stop the hub")
    print("=" * 50)
    
    # Start the hub in the background
    hub_process = subprocess.Popen([
        'python3', 'personal_ai_conversation_hub.py'
    ], cwd='/Users/samueldasari/Personal/NN_C')
    
    # Wait a moment for server to start
    time.sleep(3)
    
    # Open browser
    try:
        webbrowser.open('http://127.0.0.1:8080')
        print("🌐 Browser opened to http://127.0.0.1:8080")
    except:
        print("❌ Could not open browser automatically")
        print("🌐 Please open http://127.0.0.1:8080 manually")
    
    print("\n📝 Setup Instructions:")
    print("1. Connect AI agents from the sidebar")
    print("2. Type messages and send to specific agents")
    print("3. Use 'Broadcast to All' to message all connected agents")
    print("4. Set API keys for more agents:")
    print("   export ANTHROPIC_API_KEY='your-claude-key'")
    print("   export GOOGLE_API_KEY='your-gemini-key'")
    print("   export OPENAI_API_KEY='your-openai-key'")
    print("\n🛑 Close this window to stop the hub")
    
    try:
        # Wait for the hub process
        hub_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping hub...")
        hub_process.terminate()
        hub_process.wait()
        print("👋 Hub stopped")

if __name__ == "__main__":
    main()
