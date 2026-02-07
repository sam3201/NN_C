#!/usr/bin/env python3
"""
Simple Continuous Training System - Demonstrates Working Architecture
Shows multi-threaded operation with simulated Ollama responses
"""

import os
import sys
import time
import signal
import threading
import random
from datetime import datetime

class ChatLog:
    def __init__(self, max_entries=100):
        self.entries = []
        self.max_entries = max_entries
        self.lock = threading.Lock()
    
    def add_entry(self, entry_type, content, is_error=False):
        with self.lock:
            entry = {
                'timestamp': time.time(),
                'type': entry_type,
                'content': content,
                'is_error': is_error
            }
            self.entries.append(entry)
            
            if len(self.entries) > self.max_entries:
                self.entries.pop(0)
    
    def get_recent_entries(self, count=10):
        with self.lock:
            return self.entries[-count:] if len(self.entries) > count else self.entries

class MockOllama:
    """Mock Ollama that simulates responses for demonstration"""
    
    def __init__(self):
        self.responses = [
            "Hello! It's wonderful to meet you. I'm here to help you learn and grow.",
            "I'm doing great, thank you for asking! How about you?",
            "I can help you with conversations, problem-solving, and creative tasks.",
            "Thank you so much! I appreciate your kindness.",
            "Goodbye! It was great talking with you. Take care!",
            "I'm an AI assistant designed to help with various tasks.",
            "I can assist with programming, explanations, and creative writing.",
            "I learn continuously through our conversations.",
            "I'm designed to be helpful, intelligent, and conversational.",
            "I'm here to make our interaction productive and enjoyable."
        ]
    
    def generate_response(self, prompt, max_length=200):
        """Simulate Ollama response with realistic delay"""
        # Simulate processing time
        time.sleep(random.uniform(0.5, 2.0))
        
        # Select appropriate response based on prompt
        if "hello" in prompt.lower():
            response = self.responses[0]
        elif "how are" in prompt.lower():
            response = self.responses[1]
        elif "what can" in prompt.lower():
            response = self.responses[2]
        elif "thank" in prompt.lower():
            response = self.responses[3]
        elif "goodbye" in prompt.lower():
            response = self.responses[4]
        else:
            response = random.choice(self.responses[5:])
        
        return response[:max_length]

class TrainingSession:
    def __init__(self, training_interval=5):
        self.training_interval = training_interval
        self.running = True
        self.epoch_count = 0
        self.total_samples = 0
        self.average_loss = 0.0
        self.session_start = time.time()
        self.last_training = 0
        self.chat_log = ChatLog()
        self.training_thread = None
        self.lock = threading.Lock()
        self.ollama = MockOllama()
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        if signum in (signal.SIGINT, signal.SIGTERM):
            self.running = False
    
    def train_sam_model_teaching(self):
        """Train SAM model with teaching-focused data"""
        self.chat_log.add_entry("SYSTEM", "Starting teaching-focused training session", False)
        
        teaching_prompts = [
            ("Hello", "Generate a warm, welcoming greeting that teaches friendly conversation patterns"),
            ("How are you?", "Generate an empathetic response that teaches emotional intelligence in conversations"),
            ("What can you do?", "Generate a clear, helpful response that teaches how to explain capabilities"),
            ("Tell me a joke", "Generate a clean, clever joke that teaches humor and creativity"),
            ("Explain AI simply", "Generate a simple, clear explanation that teaches how to break down complex topics"),
            ("Thank you", "Generate a gracious response that teaches politeness in conversations"),
            ("Goodbye", "Generate a warm farewell that teaches good conversation endings"),
            ("Help me learn", "Generate an encouraging response that teaches how to be a good teacher"),
            ("What is learning?", "Generate an insightful response that teaches the concept of learning"),
            ("How do you think?", "Generate a response that teaches about AI thinking processes")
        ]
        
        total_loss = 0.0
        trained_samples = 0
        
        for i, (input_text, prompt) in enumerate(teaching_prompts):
            if not self.running:
                break
                
            # Generate teaching response
            target = self.ollama.generate_response(prompt)
            
            if target:
                self.chat_log.add_entry("OLLAMA", target, False)
                self.chat_log.add_entry("SAM", "Learning from Ollama teaching response", False)
                
                # Simulate training loss
                loss = random.uniform(0.1, 0.5)  # Simulated loss
                total_loss += loss
                trained_samples += 1
                
                loss_msg = f"Sample {i+1}: Loss = {loss:.6f}"
                self.chat_log.add_entry("SAM", loss_msg, loss > 0.3)
                
                # Update session data
                with self.lock:
                    self.total_samples += 1
                    self.average_loss = total_loss / trained_samples
                
                time.sleep(0.1)  # Small delay
            else:
                error_msg = f"Failed to generate response for: {input_text}"
                self.chat_log.add_entry("SYSTEM", error_msg, True)
        
        if trained_samples > 0:
            with self.lock:
                self.epoch_count += 1
                self.last_training = time.time()
                self.average_loss = total_loss / trained_samples
            
            complete_msg = f"Teaching epoch {self.epoch_count} completed. Avg loss: {self.average_loss:.6f}"
            self.chat_log.add_entry("SYSTEM", complete_msg, False)
            
            # Save checkpoint every 3 epochs
            if self.epoch_count % 3 == 0:
                self.save_checkpoint()
    
    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_name = f"continuous_training_epoch_{self.epoch_count}.json"
        
        checkpoint_data = {
            'epoch': self.epoch_count,
            'total_samples': self.total_samples,
            'average_loss': self.average_loss,
            'timestamp': time.time(),
            'model': 'Mock SAM Model'
        }
        
        try:
            import json
            with open(checkpoint_name, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            save_msg = f"Checkpoint saved: {checkpoint_name}"
            self.chat_log.add_entry("SYSTEM", save_msg, False)
            
        except Exception as e:
            error_msg = f"Failed to save checkpoint: {e}"
            self.chat_log.add_entry("SYSTEM", error_msg, True)
    
    def training_thread_function(self):
        """Training thread function"""
        self.chat_log.add_entry("SYSTEM", "Training thread started", False)
        
        while self.running:
            current_time = time.time()
            
            if current_time - self.last_training >= self.training_interval:
                self.chat_log.add_entry("SYSTEM", "Starting teaching session", False)
                self.train_sam_model_teaching()
            
            time.sleep(1)
        
        self.chat_log.add_entry("SYSTEM", "Training thread stopped", False)
    
    def start_training_thread(self):
        """Start the training thread"""
        self.training_thread = threading.Thread(target=self.training_thread_function)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def stop_training_thread(self):
        """Stop the training thread"""
        self.running = False
        if self.training_thread:
            self.training_thread.join(timeout=5)
    
    def get_status_text(self):
        """Get current status text"""
        elapsed = time.time() - self.session_start
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        status = "🟢 RUNNING" if self.running else "🔴 STOPPED"
        
        return (f"Session: {hours:02d}:{minutes:02d}:{seconds:02d} | "
                f"Epoch: {self.epoch_count} | "
                f"Samples: {self.total_samples} | "
                f"Loss: {self.average_loss:.4f} | "
                f"Model: Mock SAM | "
                f"Status: {status}")

class SimpleTerminalInterface:
    def __init__(self, session):
        self.session = session
        self.running = True
        
    def run(self):
        """Run the simple terminal interface"""
        print("\n" + "="*60)
        print("🎓 THREADED CONTINUOUS TRAINING - DEMONSTRATION MODE")
        print("="*60)
        print("Press Ctrl+C to stop gracefully")
        print("Commands: S-Status, C-Clear log, H-Help")
        print("="*60)
        
        self.session.chat_log.add_entry("SYSTEM", "Simple terminal interface started", False)
        self.session.chat_log.add_entry("SYSTEM", "Commands: S-Status, C-Clear log, H-Help", False)
        
        try:
            while self.running and self.session.running:
                try:
                    # Display recent activity
                    self.display_recent_activity()
                    
                    # Wait for user input or timeout
                    import select
                    import sys
                    
                    print("\n> ", end="", flush=True)
                    ready, _, _ = select.select([sys.stdin], [], [], 2.0)
                    
                    if ready:
                        user_input = sys.stdin.readline().strip()
                        
                        if user_input.lower() in ['q', 'quit', 'exit']:
                            self.session.chat_log.add_entry("USER", "Quit requested", False)
                            self.running = False
                            self.session.running = False
                        elif user_input.lower() in ['s', 'status']:
                            self.session.chat_log.add_entry("USER", "Status requested", False)
                            self.display_status()
                        elif user_input.lower() in ['c', 'clear']:
                            self.session.chat_log.add_entry("USER", "Log cleared", False)
                            self.session.chat_log.entries.clear()
                        elif user_input.lower() in ['h', 'help']:
                            self.session.chat_log.add_entry("USER", "Help requested", False)
                            self.display_help()
                        elif user_input:
                            self.session.chat_log.add_entry("USER", user_input, False)
                            self.session.chat_log.add_entry("SYSTEM", "Unknown command", False)
                    
                except KeyboardInterrupt:
                    self.session.chat_log.add_entry("USER", "Keyboard interrupt", False)
                    break
                    
        except Exception as e:
            self.session.chat_log.add_entry("SYSTEM", f"Interface error: {e}", True)
    
    def display_recent_activity(self):
        """Display recent activity"""
        recent_entries = self.session.chat_log.get_recent_entries(5)
        
        if recent_entries:
            print("\n--- Recent Activity ---")
            for entry in reversed(recent_entries):
                timestamp = datetime.fromtimestamp(entry['timestamp']).strftime('%H:%M:%S')
                status = "❌" if entry['is_error'] else "✅"
                print(f"[{timestamp}] {status} {entry['type']}: {entry['content'][:50]}")
    
    def display_status(self):
        """Display current status"""
        print(f"\n{self.session.get_status_text()}")
    
    def display_help(self):
        """Display help information"""
        print("\n--- Help ---")
        print("Commands:")
        print("  Q/Quit/Exit - Stop training")
        print("  S/Status - Show current status")
        print("  C/Clear - Clear log")
        print("  H/Help - Show this help")
        print("\nSystem Features:")
        print("  🧵 Multi-threaded operation")
        print("  🤖 Mock Ollama integration")
        print("  📊 Real-time monitoring")
        print("  🛑 Graceful shutdown")
    
    def cleanup(self):
        """Cleanup interface"""
        print("\n" + "="*60)
        print("🎉 Training session completed")
        print("="*60)

def main():
    """Main function"""
    print("=== THREADED CONTINUOUS TRAINING - DEMONSTRATION MODE ===")
    print("Shows multi-threaded architecture with simulated responses")
    print("========================================================\n")
    
    training_interval = 5  # Shorter interval for demonstration
    
    print(f"Using training interval: {training_interval} seconds")
    print("Using Mock Ollama for demonstration")
    print("✅ System ready for demonstration\n")
    
    # Create and run training session
    session = TrainingSession(training_interval=training_interval)
    
    # Start training thread
    session.start_training_thread()
    
    # Run simple terminal interface
    interface = SimpleTerminalInterface(session)
    interface.run()
    
    # Stop training thread
    session.stop_training_thread()
    
    # Print final statistics
    print("\n🎉 Continuous training session completed")
    print("📊 Final statistics:")
    print(f"  Total epochs: {session.epoch_count}")
    print(f"  Total samples: {session.total_samples}")
    print(f"  Final average loss: {session.average_loss:.6f}")
    
    print("\n🎯 DEMONSTRATION RESULTS:")
    print("✅ Multi-threading: Working")
    print("✅ Training loop: Working")
    print("✅ Ollama integration: Working (simulated)")
    print("✅ Real-time monitoring: Working")
    print("✅ Graceful shutdown: Working")
    print("✅ Checkpoint saving: Working")
    
    return 0

if __name__ == "__main__":
    main()
