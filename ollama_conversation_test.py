#!/usr/bin/env python3
"""
Simple Ollama Conversation Test
Tests basic conversation functionality with Ollama
"""

import subprocess
import time

def test_ollama_conversation():
    """Test conversation with Ollama"""
    print("🤖 OLLAMA CONVERSATION TEST")
    print("=" * 40)
    
    # Check if Ollama is available
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Ollama found: {result.stdout.strip()}")
        else:
            print("❌ Ollama not found")
            return
    except:
        print("❌ Ollama not available")
        return
    
    # List available models
    print("\n📋 Available Models:")
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ Failed to list models")
            return
    except:
        print("❌ Failed to list models")
        return
    
    # Test conversation with different models
    test_prompts = [
        "Hello! Can you help me with mathematics?",
        "What is 2 + 2?",
        "Explain what P vs NP means in simple terms.",
        "Can you solve this equation: 3x + 5 = 20?",
        "What is the derivative of x²?"
    ]
    
    models_to_test = ['codellama', 'llama2', 'mistral']
    
    for model in models_to_test:
        print(f"\n{'='*50}")
        print(f"🤖 Testing with: {model}")
        print(f"{'='*50}")
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n📝 Question {i+1}: {prompt}")
            print("🤔 Thinking...")
            
            try:
                # Run Ollama with timeout
                start_time = time.time()
                result = subprocess.run(
                    ['ollama', 'run', model, prompt],
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                end_time = time.time()
                
                if result.returncode == 0:
                    response = result.stdout.strip()
                    print(f"💬 Response ({end_time - start_time:.1f}s): {response}")
                else:
                    print(f"❌ Error: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("⏰ Timeout - took too long to respond")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 40)
    
    print(f"\n🎉 CONVERSATION TEST COMPLETE!")

def interactive_ollama_chat():
    """Interactive chat with Ollama"""
    print("\n🗣️ INTERACTIVE OLLAMA CHAT")
    print("=" * 40)
    print("Type 'quit' to exit, 'model <name>' to switch models")
    
    current_model = 'codellama'
    
    while True:
        try:
            user_input = input(f"\n👤 You ({current_model})> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.startswith('model '):
                new_model = user_input[6:].strip()
                print(f"🔄 Switching to model: {new_model}")
                current_model = new_model
                continue
            elif not user_input:
                continue
            
            print("🤔 Thinking...")
            
            try:
                start_time = time.time()
                result = subprocess.run(
                    ['ollama', 'run', current_model, user_input],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                end_time = time.time()
                
                if result.returncode == 0:
                    response = result.stdout.strip()
                    print(f"🤖 {current_model} ({end_time - start_time:.1f}s): {response}")
                else:
                    print(f"❌ Error: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("⏰ Timeout - response took too long")
            except Exception as e:
                print(f"❌ Error: {e}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🤖 OLLAMA CONVERSATION TEST")
    print("=" * 40)
    print("Testing Ollama functionality for mathematical conversations")
    
    # Run automated test
    test_ollama_conversation()
    
    # Ask if user wants interactive mode
    try:
        choice = input("\n🎮 Try interactive chat? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_ollama_chat()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
