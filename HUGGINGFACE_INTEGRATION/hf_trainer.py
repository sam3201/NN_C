#!/usr/bin/env python3
"""
Hugging Face Trainer for SAM Model
Uses HF models as teachers to train the SAM model through knowledge distillation
"""

import sys
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch

class HFTrainer:
    def __init__(self, model_name="gpt2"):
        """Initialize Hugging Face model"""
        print(f"Loading Hugging Face model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval()
            
            # Get model dimension
            if hasattr(self.model.config, 'n_embd'):
                self.model_dim = self.model.config.n_embd
            elif hasattr(self.model.config, 'hidden_size'):
                self.model_dim = self.model.config.hidden_size
            else:
                self.model_dim = 768
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Model loaded. Dimension: {self.model_dim}")
            
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            sys.exit(1)
    
    def get_embeddings(self, text, max_length=512):
        """Get embeddings from HF model"""
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get last hidden state
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    hidden_states = outputs.hidden_states[-1]
                    # Average over sequence length
                    embeddings = hidden_states.mean(dim=1).squeeze().numpy()
                else:
                    # Fallback to logits
                    logits = outputs.logits
                    embeddings = logits.mean(dim=1).squeeze().numpy()
            
            # Normalize to [0, 1] range
            if embeddings.max() > embeddings.min():
                embeddings = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())
            else:
                embeddings = np.zeros_like(embeddings)
            
            return embeddings.tolist()
            
        except Exception as e:
            print(f"Error getting embeddings: {e}", file=sys.stderr)
            return None
    
    def generate_teacher_output(self, text, max_length=50):
        """Generate text using HF model (teacher output)"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated
            
        except Exception as e:
            print(f"Error generating text: {e}", file=sys.stderr)
            return text
    
    def train_sam(self, training_texts, epochs=10):
        """Train SAM model using HF model as teacher"""
        print(f"\nTraining SAM with {len(training_texts)} samples for {epochs} epochs...")
        
        training_data = []
        
        for i, text in enumerate(training_texts):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(training_texts)}...")
            
            # Get teacher embeddings
            teacher_embeddings = self.get_embeddings(text)
            if teacher_embeddings is None:
                continue
            
            # Generate teacher output
            teacher_output = self.generate_teacher_output(text[:100])  # Use first 100 chars
            
            training_data.append({
                'input_text': text,
                'teacher_embeddings': teacher_embeddings,
                'teacher_output': teacher_output,
                'model_dim': self.model_dim
            })
        
        # Save training data as JSON for C program to use
        with open('hf_training_data.json', 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"\nSaved {len(training_data)} training samples to hf_training_data.json")
        return training_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python hf_trainer.py <model_name> [epochs] [data_file]")
        print("Example: python hf_trainer.py bert-base-uncased 10 ../utils/DATASETS/RomeoAndJuliet.txt")
        sys.exit(1)
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    data_file = sys.argv[3] if len(sys.argv) > 3 else "../utils/DATASETS/RomeoAndJuliet.txt"
    
    # Initialize trainer
    trainer = HFTrainer(model_name)
    
    # Load training data
    print(f"\nLoading training data from: {data_file}")
    training_texts = []
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and len(line) > 10:  # Skip very short lines
                    training_texts.append(line)
                    if len(training_texts) >= 1000:  # Limit for testing
                        break
    except Exception as e:
        print(f"Error loading data file: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(training_texts)} training texts")
    
    # Train SAM
    trainer.train_sam(training_texts, epochs)
    
    print("\n✓ Training data prepared for SAM model")
    print("Run the C trainer to complete training: ./hf_trainer")

if __name__ == "__main__":
    main()

