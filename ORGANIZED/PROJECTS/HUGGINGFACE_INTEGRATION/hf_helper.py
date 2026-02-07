#!/usr/bin/env python3
"""
Helper script to parse JSON and provide embeddings to C program
"""

import json
import sys

def extract_embeddings(json_file):
    """Extract embeddings from JSON file and output in C-friendly format"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"NUM_SAMPLES:{len(data)}")
    
    for i, sample in enumerate(data):
        embeddings = sample['teacher_embeddings']
        model_dim = sample.get('model_dim', len(embeddings))
        
        print(f"SAMPLE:{i}")
        print(f"DIM:{model_dim}")
        print("EMBEDDINGS:", end="")
        for j, val in enumerate(embeddings):
            if j < model_dim:
                print(f"{val:.10f}", end=" ")
        print()
        print(f"TEXT:{sample['input_text'][:100]}")  # First 100 chars
        print("END_SAMPLE")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hf_helper.py <json_file>", file=sys.stderr)
        sys.exit(1)
    
    extract_embeddings(sys.argv[1])

