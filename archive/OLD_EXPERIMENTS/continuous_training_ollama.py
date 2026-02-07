#!/usr/bin/env python3
"""
Continuous Training System with Ollama Integration
Uses Ollama to generate training data for SAM model continuously
"""

import os
import sys
import time
import signal
import subprocess
import json
import logging
from datetime import datetime
from pathlib import Path

# Add SAM utilities to path
sys.path.append('ORGANIZED/UTILS/SAM')

try:
    import ctypes
    import numpy as np
except ImportError:
    print("❌ Required Python packages not found. Install with:")
    print("   pip install numpy")
    sys.exit(1)

class ContinuousTrainingSession:
    def __init__(self, ollama_model="llama2", training_interval=30):
        self.ollama_model = ollama_model
        self.training_interval = training_interval
        self.running = True
        self.epoch_count = 0
        self.total_samples = 0
        self.average_loss = 0.0
        self.session_start = time.time()
        self.last_training = 0
        self.sam_model = None
        self.log_file = None
        
        # Setup logging
        self.setup_logging()
        
        # Initialize signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_filename = f"continuous_training_{int(time.time())}.log"
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Setup file handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger('continuous_training')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Continuous training session started")
        self.logger.info(f"Ollama model: {self.ollama_model}")
        
    def signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        if signum in (signal.SIGINT, signal.SIGTERM):
            self.logger.info("Received interrupt signal. Shutting down gracefully...")
            self.running = False
            
    def check_ollama_available(self):
        """Check if Ollama is available"""
        self.logger.info("Checking Ollama availability...")
        
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            
            if result.returncode == 0:
                self.logger.info("✅ Ollama is available")
                return True
            else:
                self.logger.error("❌ Ollama command failed")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Ollama command timed out")
            return False
        except FileNotFoundError:
            self.logger.error("❌ Ollama not found in PATH")
            self.logger.info("💡 Please install Ollama: https://ollama.ai/")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error checking Ollama: {e}")
            return False
            
    def generate_ollama_response(self, prompt, max_length=500):
        """Generate response using Ollama"""
        try:
            # Build command
            cmd = ['ollama', 'run', self.ollama_model, prompt]
            
            # Execute command
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=30)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                # Truncate if too long
                if len(response) > max_length:
                    response = response[:max_length]
                return response
            else:
                self.logger.error(f"❌ Ollama command failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Ollama command timed out")
            return None
        except Exception as e:
            self.logger.error(f"❌ Error generating Ollama response: {e}")
            return None
            
    def generate_training_samples(self, num_samples=20):
        """Generate training samples using Ollama"""
        self.logger.info(f"🎯 Generating {num_samples} training samples using Ollama ({self.ollama_model})")
        
        # Training prompts for different types of responses
        training_prompts = [
            ("Hello", "Generate a friendly greeting response"),
            ("How are you?", "Generate a response to 'How are you?'"),
            ("What can you do?", "Generate a response explaining capabilities"),
            ("Tell me a joke", "Generate a funny joke response"),
            ("Explain AI", "Generate a simple explanation of AI"),
            ("Thank you", "Generate a response to 'Thank you'"),
            ("Goodbye", "Generate a farewell response"),
            ("Help programming", "Generate a helpful programming response"),
            ("Machine learning", "Generate a response about machine learning"),
            ("How do you work?", "Generate a response explaining how you work"),
            ("What is your purpose?", "Generate a response about your purpose"),
            ("Can you help me?", "Generate a helpful response"),
            ("Tell me something interesting", "Generate an interesting fact response"),
            ("How old are you?", "Generate a response about your age"),
            ("Where are you from?", "Generate a response about your origin"),
            ("What do you think?", "Generate a thoughtful response"),
            ("Are you real?", "Generate a response about your reality"),
            ("Can you learn?", "Generate a response about learning"),
            ("What's your name?", "Generate a response about your name"),
            ("Do you have feelings?", "Generate a response about emotions")
        ]
        
        samples = []
        
        for i, (input_text, prompt) in enumerate(training_prompts[:num_samples]):
            self.logger.info(f"  📝 Generating sample {i+1}/{num_samples}: '{input_text}'")
            
            # Generate response using Ollama
            response = self.generate_ollama_response(prompt)
            
            if response:
                sample = {
                    'input': input_text,
                    'target': response,
                    'loss': 0.0,
                    'processed': False
                }
                samples.append(sample)
                
                self.logger.info(f"    ✅ Generated: '{response[:50]}...'")
            else:
                self.logger.warning(f"    ❌ Failed to generate response for: '{input_text}'")
            
            # Small delay to avoid overwhelming Ollama
            time.sleep(0.5)
        
        self.logger.info(f"📊 Generated {len(samples)} training samples")
        return samples
        
    def encode_text_to_vector(self, text, max_length=128):
        """Encode text to vector (simple character encoding)"""
        vector = np.zeros(max_length, dtype=np.float64)
        
        for i, char in enumerate(text[:max_length]):
            vector[i] = ord(char) / 255.0
            
        return vector
        
    def train_sam_model(self, samples):
        """Train SAM model with generated samples"""
        self.logger.info(f"🎓 Training SAM model with {len(samples)} samples")
        
        if not self.sam_model:
            self.logger.error("❌ SAM model not initialized")
            return
            
        total_loss = 0.0
        trained_samples = 0
        
        for i, sample in enumerate(samples):
            if not sample['processed']:
                self.logger.info(f"  🔄 Training sample {i+1}/{len(samples)}: '{sample['input']}'")
                
                try:
                    # Encode input and target
                    input_vector = self.encode_text_to_vector(sample['input'])
                    target_vector = self.encode_text_to_vector(sample['target'])
                    
                    # Simple forward pass (simplified - would need actual SAM integration)
                    # This is a placeholder for actual SAM training
                    output_vector = input_vector + np.random.normal(0, 0.1, input_vector.shape)
                    
                    # Calculate loss (MSE)
                    loss = np.mean((output_vector - target_vector) ** 2)
                    
                    sample['loss'] = loss
                    total_loss += loss
                    trained_samples += 1
                    
                    self.logger.info(f"    ✅ Loss: {loss:.6f}")
                    
                    # Simple gradient descent update (placeholder)
                    # In real implementation, this would use SAM_backprop
                    
                    sample['processed'] = True
                    
                except Exception as e:
                    self.logger.error(f"    ❌ Error training sample: {e}")
                
                # Small delay
                time.sleep(0.1)
        
        if trained_samples > 0:
            self.average_loss = total_loss / trained_samples
            self.total_samples += trained_samples
            
            self.logger.info(f"📊 Training completed:")
            self.logger.info(f"  Trained samples: {trained_samples}")
            self.logger.info(f"  Average loss: {self.average_loss:.6f}")
            self.logger.info(f"  Total samples: {self.total_samples}")
        else:
            self.logger.warning("⚠️  No samples were trained")
            
    def save_model_checkpoint(self):
        """Save model checkpoint"""
        self.logger.info("💾 Saving model checkpoint...")
        
        checkpoint_name = f"continuous_training_epoch_{self.epoch_count}.json"
        
        checkpoint_data = {
            'epoch': self.epoch_count,
            'total_samples': self.total_samples,
            'average_loss': self.average_loss,
            'timestamp': time.time(),
            'ollama_model': self.ollama_model
        }
        
        try:
            with open(checkpoint_name, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.logger.info(f"✅ Checkpoint saved: {checkpoint_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint: {e}")
            
    def display_training_status(self):
        """Display current training status"""
        elapsed = time.time() - self.session_start
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        status = "🟢 Running" if self.running else "🔴 Stopping"
        
        print("\n" + "=" * 60)
        print("🎓 CONTINUOUS TRAINING STATUS")
        print("=" * 60)
        print(f"Session Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"Epoch: {self.epoch_count}")
        print(f"Total Samples: {self.total_samples}")
        print(f"Average Loss: {self.average_loss:.6f}")
        print(f"Ollama Model: {self.ollama_model}")
        print(f"Status: {status}")
        print("=" * 60)
        print()
        
    def continuous_training_loop(self):
        """Main continuous training loop"""
        self.logger.info("🚀 Starting continuous training loop...")
        self.logger.info("💡 Press Ctrl+C to stop gracefully")
        
        while self.running:
            current_time = time.time()
            
            # Check if it's time to train
            if current_time - self.last_training >= self.training_interval:
                self.logger.info(f"⏰ Training interval reached ({self.training_interval} seconds)")
                
                # Generate new training samples
                samples = self.generate_training_samples(num_samples=20)
                
                # Train the model
                if samples:
                    self.train_sam_model(samples)
                    self.epoch_count += 1
                    self.last_training = current_time
                    
                    # Save checkpoint every 5 epochs
                    if self.epoch_count % 5 == 0:
                        self.save_model_checkpoint()
                    
                    # Display status
                    self.display_training_status()
            
            # Sleep for a short time
            time.sleep(1)
            
    def run(self):
        """Run the continuous training session"""
        # Check Ollama availability
        if not self.check_ollama_available():
            self.logger.error("❌ Cannot proceed without Ollama")
            return False
            
        # Initialize SAM model (placeholder)
        self.logger.info("🤖 Initializing SAM model...")
        # In real implementation, this would load the actual SAM model
        self.sam_model = True  # Placeholder
        self.logger.info("✅ SAM model initialized (placeholder)")
        
        # Display initial status
        self.display_training_status()
        
        # Start continuous training
        try:
            self.continuous_training_loop()
        except KeyboardInterrupt:
            self.logger.info("🛑 Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"❌ Training loop error: {e}")
        finally:
            # Cleanup
            self.cleanup()
            
        return True
        
    def cleanup(self):
        """Cleanup training session"""
        self.logger.info("🧹 Cleaning up training session...")
        
        # Save final checkpoint
        if self.epoch_count > 0:
            self.save_model_checkpoint()
        
        # Log final statistics
        self.logger.info("🎉 Continuous training session completed")
        self.logger.info(f"📊 Final statistics:")
        self.logger.info(f"  Total epochs: {self.epoch_count}")
        self.logger.info(f"  Total samples: {self.total_samples}")
        self.logger.info(f"  Final average loss: {self.average_loss:.6f}")
        
        self.logger.info("✅ Cleanup completed")

def main():
    """Main function"""
    print("=== CONTINUOUS TRAINING WITH OLLAMA ===")
    print("Using Ollama to generate training data for SAM model")
    print("=========================================\n")
    
    # Parse command line arguments
    ollama_model = "llama2"
    training_interval = 30
    
    if len(sys.argv) > 1:
        ollama_model = sys.argv[1]
        print(f"Using Ollama model: {ollama_model}")
    else:
        print(f"Using default Ollama model: {ollama_model}")
        
    if len(sys.argv) > 2:
        try:
            training_interval = int(sys.argv[2])
            print(f"Training interval: {training_interval} seconds")
        except ValueError:
            print("⚠️  Invalid training interval, using default: 30 seconds")
    
    # Create and run training session
    session = ContinuousTrainingSession(
        ollama_model=ollama_model,
        training_interval=training_interval
    )
    
    success = session.run()
    
    if success:
        print("\n🎉 Continuous training completed successfully!")
    else:
        print("\n❌ Continuous training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
