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
    def __init__(self, model_name="distilbert-base-uncased"):
        """Initialize Hugging Face model"""
        print(f"Loading Hugging Face model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Handle different model types
            if any(x in model_name.lower() for x in ['bert', 'distilbert', 'roberta', 'albert']):
                # For BERT-like models, use AutoModel for embeddings
                self.model = AutoModel.from_pretrained(model_name)
                self.is_encoder = True
            else:
                # For generative models, use AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.is_encoder = False
            
            self.model.eval()
            
            # Get model dimension
            if hasattr(self.model.config, 'n_embd'):
                self.model_dim = self.model.config.n_embd
            elif hasattr(self.model.config, 'hidden_size'):
                self.model_dim = self.model.config.hidden_size
            elif hasattr(self.model.config, 'd_model'):
                self.model_dim = self.model.config.d_model
            else:
                self.model_dim = 768
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = "[PAD]"
            
            print(f"Model loaded. Dimension: {self.model_dim}")
            print(f"Model type: {'Encoder' if self.is_encoder else 'Decoder'}")
            
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
                if self.is_encoder:
                    # For encoder models (BERT, DistilBERT, etc.)
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    # Average over sequence length
                    embeddings = hidden_states.mean(dim=1).squeeze().numpy()
                else:
                    # For decoder models (GPT, etc.)
                    outputs = self.model(**inputs, output_hidden_states=True)
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        hidden_states = outputs.hidden_states[-1]
                        embeddings = hidden_states.mean(dim=1).squeeze().numpy()
                    else:
                        # Fallback to logits
                        logits = outputs.logits
                        embeddings = logits.mean(dim=1).squeeze().numpy()
            
            # Normalize to [0, 1] range
            if len(embeddings.shape) > 0 and embeddings.max() > embeddings.min():
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
    
    def load_vocabulary(self, words_file="../utils/DATASETS/words.txt"):
        """Load vocabulary from words.txt file"""
        vocabulary = []
        try:
            with open(words_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and len(word) > 0:
                        vocabulary.append(word)
            print(f"Loaded {len(vocabulary)} words from vocabulary")
            return vocabulary
        except Exception as e:
            print(f"Error loading vocabulary: {e}", file=sys.stderr)
            return []
    
    def create_vocabulary_training_data(self, vocabulary, num_samples=500):
        """Create training data using vocabulary words"""
        training_data = []
        
        # Filter to reasonable words
        filtered_vocab = [word for word in vocabulary 
                          if len(word) > 1 and len(word) < 20 
                          and not word.replace('-', '').replace('.', '').replace(',', '').isdigit()]
        
        print(f"Using {len(filtered_vocab)} filtered words for training")
        
        for i in range(min(num_samples, len(filtered_vocab))):
            if i % 50 == 0:
                print(f"Creating vocabulary sample {i}/{min(num_samples, len(filtered_vocab))}...")
            
            word = filtered_vocab[i % len(filtered_vocab)]
            
            # Create conversation examples
            examples = [
                f"Hello, how are you?",
                f"What is {word}?",
                f"Tell me about {word}.",
                f"Thank you for your help.",
                f"Goodbye!"
            ]
            
            input_text = examples[i % len(examples)]
            
            # Generate responses
            if "how are you" in input_text.lower():
                teacher_output = "I'm doing well, thank you for asking!"
            elif "what is" in input_text.lower():
                teacher_output = f"{word} is something I can help you learn about."
            elif "tell me" in input_text.lower():
                teacher_output = f"{word} is an interesting topic worth discussing."
            elif "thank" in input_text.lower():
                teacher_output = "You're welcome! Is there anything else I can help with?"
            elif "goodbye" in input_text.lower():
                teacher_output = "Goodbye! Have a great day!"
            else:
                teacher_output = "I'm here to help with your questions."
            
            # Get embeddings
            input_embeddings = self.get_embeddings(input_text)
            
            if input_embeddings:
                training_data.append({
                    'input_text': input_text,
                    'teacher_embeddings': input_embeddings,
                    'teacher_output': teacher_output,
                    'model_dim': self.model_dim
                })
        
        return training_data
    
    def train_sam_with_vocabulary(self, epochs=5, num_samples=500):
        """Train SAM model using vocabulary words"""
        print(f"\n=== Vocabulary-based Training for SAM ===")
        
        # Load vocabulary
        vocabulary = self.load_vocabulary()
        if not vocabulary:
            print("No vocabulary loaded!", file=sys.stderr)
            return []
        
        # Create training data
        training_data = self.create_vocabulary_training_data(vocabulary, num_samples)
        
        # Save training data
        with open('hf_training_data.json', 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"\nSaved {len(training_data)} vocabulary training samples to hf_training_data.json")
        return training_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python hf_trainer.py <model_name> [epochs] [data_file|vocabulary]")
        print("Examples:")
        print("  python hf_trainer.py bert-base-uncased 10 ../utils/DATASETS/RomeoAndJuliet.txt")
        print("  python hf_trainer.py bert-base-uncased 5 vocabulary  # Uses words.txt for conversation")
        sys.exit(1)
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    data_arg = sys.argv[3] if len(sys.argv) > 3 else "../utils/DATASETS/RomeoAndJuliet.txt"
    
    # Initialize trainer
    trainer = HFTrainer(model_name)
    
    # Check training mode
    if data_arg.lower() == "vocabulary":
        # Vocabulary-based training for conversation
        trainer.train_sam_with_vocabulary(epochs, 500)
    else:
        # Check if this is already training data (JSON format)
        try:
            with open(data_arg, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Read first 1000 characters
                if content.startswith('[') and ('input_text' in content or 'teacher_embeddings' in content):
                    print(f"\nDetected existing training data in {data_arg}")
                    print("Training data is already prepared - no processing needed")
                    print(f"Run the C trainer directly: ./hf_trainer {model_name} {epochs} {data_arg}")
                    sys.exit(0)
        except Exception as e:
            pass  # Continue with normal processing
        
        # Load training data from file
        print(f"\nLoading training data from: {data_arg}")
        training_texts = []
        
        try:
            with open(data_arg, 'r', encoding='utf-8') as f:
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
        trainer.train_sam(training_texts, epochs)
    
    print("\n✓ Training data prepared for SAM model")
    print("Run the C trainer to complete training: ./hf_trainer")

if __name__ == "__main__":
    main()

