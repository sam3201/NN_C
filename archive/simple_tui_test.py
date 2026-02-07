#!/usr/bin/env python3
"""
Simple TUI Test - Debug version
"""

import time
import threading
from datetime import datetime

class SimpleTUI:
    def __init__(self):
        self.running = True
        self.messages = []
        self.status = "Starting..."
        
    def start_conversation(self):
        """Start conversation thread"""
        def conversation_thread():
            self.status = "Starting conversation..."
            time.sleep(2)
            
            # Add first message
            self.messages.append({
                'sender': 'SAM-Alpha',
                'message': 'Hello everyone! I\'ve been thinking about consciousness and AI.',
                'time': datetime.now().strftime("%H:%M:%S")
            })
            self.status = f"1 message sent"
            
            time.sleep(3)
            
            # Add second message
            self.messages.append({
                'sender': 'SAM-Beta', 
                'message': 'That\'s fascinating! I love exploring these philosophical questions.',
                'time': datetime.now().strftime("%H:%M:%S")
            })
            self.status = f"2 messages sent"
            
            # Continue conversation
            while self.running:
                time.sleep(5)
                self.messages.append({
                    'sender': 'Ollama-DeepSeek',
                    'message': 'From a technical perspective, this raises interesting questions about implementation.',
                    'time': datetime.now().strftime("%H:%M:%S")
                })
                self.status = f"{len(self.messages)} messages sent"
        
        thread = threading.Thread(target=conversation_thread, daemon=True)
        thread.start()
        print("Conversation thread started")
    
    def run(self):
        """Run simple test"""
        print("🚀 Simple TUI Test Starting...")
        print("=" * 50)
        
        # Start conversation
        self.start_conversation()
        
        # Simple display loop
        try:
            while self.running:
                print(f"\n📊 Status: {self.status} | ⏰ {datetime.now().strftime('%H:%M:%S')}")
                print("-" * 50)
                
                # Show recent messages
                for msg in self.messages[-5:]:
                    print(f"[{msg['time']}] {msg['sender']}: {msg['message']}")
                
                print("-" * 50)
                print("Type 'quit' to exit")
                
                try:
                    user_input = input("You: ")
                    if user_input.lower() == 'quit':
                        self.running = False
                        break
                    elif user_input.strip():
                        self.messages.append({
                            'sender': 'User',
                            'message': user_input,
                            'time': datetime.now().strftime("%H:%M:%S")
                        })
                except KeyboardInterrupt:
                    break
                
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    hub = SimpleTUI()
    hub.run()
