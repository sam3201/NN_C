#!/usr/bin/env python3
"""
Python bridge for Hugging Face model integration with SAM
This module provides a C-compatible interface to Hugging Face models
"""

import ctypes
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import torch
import json
import sys
import os

# Global model storage
_models = {}
_model_dims = {}

def init_model(model_name):
    """Initialize a Hugging Face model"""
    try:
        print(f"Loading Hugging Face model: {model_name}")
        
        # Try to load as causal LM first (for generation)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_type = "causal"
        except:
            # Fall back to base model
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_type = "base"
        
        # Get model dimension
        if hasattr(model.config, 'hidden_size'):
            model_dim = model.config.hidden_size
        elif hasattr(model.config, 'd_model'):
            model_dim = model.config.d_model
        elif hasattr(model.config, 'n_embd'):
            model_dim = model.config.n_embd
        else:
            model_dim = 768  # Default
        
        model.eval()  # Set to evaluation mode
        
        # Store model
        model_id = id(model)
        _models[model_id] = {
            'model': model,
            'tokenizer': tokenizer,
            'type': model_type,
            'dim': model_dim
        }
        _model_dims[model_id] = model_dim
        
        print(f"Model loaded successfully. Dimension: {model_dim}")
        return model_id
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}", file=sys.stderr)
        return 0

def free_model(model_id):
    """Free a Hugging Face model"""
    if model_id in _models:
        del _models[model_id]
        if model_id in _model_dims:
            del _model_dims[model_id]
        return 1
    return 0

def get_embeddings(model_id, text, embeddings_ptr, model_dim):
    """Get embeddings from HF model"""
    if model_id not in _models:
        return 0
    
    try:
        model_data = _models[model_id]
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        
        # Tokenize and get embeddings
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            if model_data['type'] == 'causal':
                outputs = model(**inputs, output_hidden_states=True)
                # Use last hidden state, average over sequence
                hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.logits
                if len(hidden_states.shape) > 2:
                    embeddings = hidden_states.mean(dim=1).squeeze().numpy()
                else:
                    embeddings = hidden_states.mean(dim=0).numpy()
            else:
                outputs = model(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                elif hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output.squeeze().numpy()
                else:
                    embeddings = outputs[0].mean(dim=1).squeeze().numpy()
        
        # Normalize to [0, 1] range for compatibility with SAM
        embeddings = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min() + 1e-8)
        
        # Copy to C array
        actual_dim = min(len(embeddings), model_dim)
        embeddings_array = (ctypes.c_double * model_dim).from_address(embeddings_ptr)
        for i in range(actual_dim):
            embeddings_array[i] = float(embeddings[i])
        for i in range(actual_dim, model_dim):
            embeddings_array[i] = 0.0
        
        return 1
        
    except Exception as e:
        print(f"Error getting embeddings: {e}", file=sys.stderr)
        return 0

def generate_text(model_id, prompt, max_length):
    """Generate text using HF model"""
    if model_id not in _models:
        return ""
    
    try:
        model_data = _models[model_id]
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        
        if model_data['type'] == 'causal':
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
        else:
            return prompt  # Non-generative models just return input
            
    except Exception as e:
        print(f"Error generating text: {e}", file=sys.stderr)
        return ""

def get_model_dim(model_id):
    """Get model dimension"""
    return _model_dims.get(model_id, 0)

# C interface functions for ctypes
def _c_init_model(model_name_bytes):
    model_name = model_name_bytes.decode('utf-8')
    return init_model(model_name)

def _c_get_embeddings(model_id, text_bytes, embeddings_ptr, model_dim):
    text = text_bytes.decode('utf-8')
    return get_embeddings(model_id, text, embeddings_ptr, model_dim)

def _c_generate_text(model_id, prompt_bytes, max_length):
    prompt = prompt_bytes.decode('utf-8')
    result = generate_text(model_id, prompt, max_length)
    return result.encode('utf-8')

if __name__ == "__main__":
    # Test the interface
    print("Testing Hugging Face interface...")
    model_id = init_model("gpt2")
    if model_id:
        print(f"Model initialized with ID: {model_id}")
        print(f"Model dimension: {get_model_dim(model_id)}")
        free_model(model_id)
        print("Test completed successfully!")

