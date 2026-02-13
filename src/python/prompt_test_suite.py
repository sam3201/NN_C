# SAM Prompt Testing Suite
# Autonomous evaluation and optimization of agent prompts

import time as sam_time_ref
import json
import random
from typing import Dict, List, Any

class PromptTestSuite:
    """
    Automated Prompt Testing (APT) for SAM-D.
    - Maintains a registry of agent prompts.
    - Runs A/B tests on prompt variants.
    - Tracks performance (latency, token count, goal success).
    """
    
    def __init__(self, system=None):
        self.system = system
        self.prompt_registry = {}
        self.test_results = []
        
    def register_prompt(self, prompt_id: str, base_template: str):
        self.prompt_registry[prompt_id] = {
            "base": base_template,
            "variants": [base_template],
            "active_index": 0
        }
        
    def get_prompt(self, prompt_id: str) -> str:
        if prompt_id not in self.prompt_registry:
            return ""
        reg = self.prompt_registry[prompt_id]
        return reg["variants"][reg["active_index"]]
        
    def run_ab_test(self, prompt_id: str, input_val: str):
        """Run an autonomous A/B test on a prompt variant"""
        if prompt_id not in self.prompt_registry: return
        
        reg = self.prompt_registry[prompt_id]
        if len(reg["variants"]) < 2:
            # Generate a variant if we only have one
            self._generate_variant(prompt_id)
            
        # Select variant to test
        variant_idx = random.randint(0, len(reg["variants"]) - 1)
        variant = reg["variants"][variant_idx]
        
        # In a real system, this would call the model
        # For now, we simulate the performance tracking
        start = sam_time_ref.time()
        success = True # Simulated
        latency = sam_time_ref.time() - start
        
        self.test_results.append({
            "prompt_id": prompt_id,
            "variant_index": variant_idx,
            "latency": latency,
            "success": success,
            "ts": sam_time_ref.time()
        })
        
        print(f"🧪 APT: Tested variant {variant_idx} for prompt '{prompt_id}'. Latency: {latency:.3f}s")

    def _generate_variant(self, prompt_id: str):
        """Use system intelligence to propose a new prompt variant"""
        base = self.prompt_registry[prompt_id]["base"]
        # Mutation: Add emphasis or detail
        variants = [
            f"{base}\nFocus on precision and efficiency.",
            f"Think step-by-step: {base}",
            f"Act as an expert: {base}"
        ]
        self.prompt_registry[prompt_id]["variants"].extend(variants)

def create_prompt_test_suite(system=None):
    return PromptTestSuite(system)
