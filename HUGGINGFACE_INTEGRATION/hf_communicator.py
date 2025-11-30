#!/usr/bin/env python3
"""
Hugging Face Communicator for SAM Model
Allows SAM to communicate with and query Hugging Face models
"""

import sys
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

class HFCommunicator:
    def __init__(self, model_name="gpt2"):
        """Initialize Hugging Face model for communication"""
        print(f"Initializing Hugging Face model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval()
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_name = model_name
            print(f"Model {model_name} ready for communication")
            
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            sys.exit(1)
    
    def communicate(self, prompt, max_length=200, temperature=0.7, top_p=0.9):
        """Send a prompt to the HF model and get a response"""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.1
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from response
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            return response
            
        except Exception as e:
            print(f"Error in communication: {e}", file=sys.stderr)
            return f"Error: {str(e)}"
    
    def get_embeddings(self, text):
        """Get embeddings for text"""
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    hidden_states = outputs.hidden_states[-1]
                    embeddings = hidden_states.mean(dim=1).squeeze().numpy()
                else:
                    logits = outputs.logits
                    embeddings = logits.mean(dim=1).squeeze().numpy()
            
            # Normalize
            if embeddings.max() > embeddings.min():
                embeddings = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())
            
            return embeddings.tolist()
            
        except Exception as e:
            print(f"Error getting embeddings: {e}", file=sys.stderr)
            return None
    
    def conversation(self, messages, max_length=200):
        """Have a conversation with the model"""
        # Format messages as a single prompt
        prompt = ""
        for msg in messages:
            prompt += f"{msg}\n"
        
        return self.communicate(prompt, max_length=max_length)

def main():
    if len(sys.argv) < 2:
        print("Usage: python hf_communicator.py <model_name> [prompt]")
        print("Example: python hf_communicator.py gpt2 'How can we create a model that self actualizes?'")
        sys.exit(1)
    
    model_name = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "How can we create a model that self actualizes?"
    
    # Initialize communicator
    comm = HFCommunicator(model_name)
    
    # Send prompt
    print(f"\nPrompt: {prompt}\n")
    print("HF Model Response:")
    print("-" * 60)
    response = comm.communicate(prompt, max_length=300)
    print(response)
    print("-" * 60)
    
    # Save response for SAM
    result = {
        "prompt": prompt,
        "response": response,
        "model": model_name
    }
    
    with open("hf_response.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResponse saved to hf_response.json")

if __name__ == "__main__":
    main()

