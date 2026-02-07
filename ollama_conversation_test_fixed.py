#!/usr/bin/env python3
"""
Fixed Ollama Conversation Test
Better timeout handling and more robust conversation
"""

import subprocess
import time
import threading
import queue

def query_ollama_with_timeout(model, prompt, timeout=15):
    """Query Ollama with better timeout handling"""
    result_queue = queue.Queue()
    
    def run_query():
        try:
            result = subprocess.run(
                ['ollama', 'run', model, prompt],
                capture_output=True,
                text=True,
                timeout=timeout + 5  # Give subprocess extra time
            )
            result_queue.put(('success', result))
        except Exception as e:
            result_queue.put(('error', str(e)))
    
    # Start query in thread
    query_thread = threading.Thread(target=run_query)
    query_thread.daemon = True
    query_thread.start()
    
    # Wait for result with timeout
    try:
        status, result = result_queue.get(timeout=timeout)
        return status, result
    except queue.Empty:
        return 'timeout', None

def test_ollama_quick():
    """Quick test with better timeout handling"""
    print("🤖 QUICK OLLAMA TEST")
    print("=" * 30)
    
    # Simple test prompts
    test_prompts = [
        ("Basic Math", "What is 2 + 2?"),
        ("Simple Algebra", "Solve x + 3 = 7"),
        ("P vs NP", "What is P vs NP in one sentence?")
    ]
    
    for name, prompt in test_prompts:
        print(f"\n📝 {name}: {prompt}")
        print("🤔 Thinking...")
        
        status, result = query_ollama_with_timeout('codellama', prompt, timeout=10)
        
        if status == 'success' and result.returncode == 0:
            response = result.stdout.strip()
            print(f"💬 Response: {response}")
        elif status == 'timeout':
            print("⏰ Response took too long")
        else:
            print(f"❌ Error: {result if status == 'error' else 'Unknown error'}")
        
        print("-" * 30)

def interactive_chat_fixed():
    """Interactive chat with fixed timeout handling"""
    print("\n🗣️ INTERACTIVE CHAT (FIXED)")
    print("=" * 30)
    print("Commands: quit, model <name>, timeout <seconds>")
    print("Current model: codellama")
    print("Current timeout: 15 seconds")
    
    current_model = 'codellama'
    current_timeout = 15
    
    while True:
        try:
            user_input = input(f"\n👤 You> ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.startswith('model '):
                current_model = user_input[6:].strip()
                print(f"🔄 Model: {current_model}")
                continue
            elif user_input.startswith('timeout '):
                try:
                    current_timeout = int(user_input[8:].strip())
                    print(f"⏱️ Timeout: {current_timeout} seconds")
                except:
                    print("❌ Invalid timeout value")
                continue
            
            print("🤔 Thinking...")
            
            # Query with current settings
            status, result = query_ollama_with_timeout(current_model, user_input, current_timeout)
            
            if status == 'success' and result.returncode == 0:
                response = result.stdout.strip()
                print(f"🤖 {current_model}: {response}")
            elif status == 'timeout':
                print(f"⏰ Response took longer than {current_timeout} seconds")
                print("💡 Try: 'timeout 30' for longer wait time")
            else:
                error_msg = result if status == 'error' else 'Unknown error'
                print(f"❌ Error: {error_msg}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def test_different_models():
    """Test different available models"""
    print("\n🤖 TESTING DIFFERENT MODELS")
    print("=" * 30)
    
    models = ['codellama', 'llama2', 'llama3.1', 'deepseek-r1']
    test_prompt = "What is 5 + 3?"
    
    for model in models:
        print(f"\n🔍 Testing {model}...")
        print(f"📝 Question: {test_prompt}")
        
        status, result = query_ollama_with_timeout(model, test_prompt, timeout=8)
        
        if status == 'success' and result and result.returncode == 0:
            response = result.stdout.strip()
            print(f"💬 {model}: {response}")
        elif status == 'timeout':
            print(f"⏰ {model}: Timeout")
        else:
            print(f"❌ {model}: Error")
        
        print("-" * 25)

def main():
    """Main function"""
    print("🤖 FIXED OLLAMA CONVERSATION TEST")
    print("=" * 40)
    print("Better timeout handling and robust conversation")
    
    # Quick test first
    test_ollama_quick()
    
    # Test different models
    test_different_models()
    
    # Ask for interactive mode
    try:
        choice = input("\n🎮 Try interactive chat? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_chat_fixed()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
