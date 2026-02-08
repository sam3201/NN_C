#!/usr/bin/env python3
"""
RAM-Aware Model Switcher for Intelligent Provider Selection
Dynamically switches between Ollama, Hugging Face, and SWE models based on RAM consumption
"""

import psutil
import os
import time
import subprocess
import json
from typing import Dict, List, Any, Tuple, Optional
import threading

class RAMAwareModelSwitcher:
    """Intelligent model switcher that monitors RAM and switches providers accordingly"""
    
    def __init__(self, system_instance):
        self.system = system_instance
        self.ram_threshold_high = 80.0  # Switch to lightweight models above 80% RAM
        self.ram_threshold_critical = 90.0  # Emergency switch above 90% RAM
        self.monitoring_interval = 30  # Check every 30 seconds
        self.current_provider = "ollama"  # Start with Ollama
        self.current_model = None
        
        # Model preferences by RAM usage (lower RAM first)
        self.model_hierarchy = {
            "lightweight": [
                {"provider": "ollama", "models": ["phi:latest", "orca-mini:3b"]},
                {"provider": "huggingface", "models": ["microsoft/DialoGPT-small", "distilgpt2"]},
            ],
            "medium": [
                {"provider": "ollama", "models": ["codellama:latest", "deepseek-coder:6b", "llama3.1:latest"]},
                {"provider": "huggingface", "models": ["microsoft/DialoGPT-medium", "gpt2-medium"]},
                {"provider": "swe", "models": ["basic_swe_model"]},
            ],
            "heavy": [
                {"provider": "ollama", "models": ["qwen2.5-coder:7b", "deepseek-coder:33b", "codellama:13b"]},
                {"provider": "huggingface", "models": ["microsoft/DialoGPT-large", "gpt2-large"]},
            ]
        }
        
        # Start RAM monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._ram_monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        print("🧠 RAM-Aware Model Switcher initialized")
        print(f"   📊 RAM thresholds: High={self.ram_threshold_high}%, Critical={self.ram_threshold_critical}%")
    
    def get_optimal_model(self, task_type: str = "general") -> Dict[str, Any]:
        """Get the optimal model based on current RAM usage and task type"""
        ram_percent = self._get_current_ram_percent()
        
        # Determine RAM category
        if ram_percent >= self.ram_threshold_critical:
            ram_category = "lightweight"
            print(f"🚨 CRITICAL RAM usage ({ram_percent:.1f}%) - switching to lightweight models")
        elif ram_percent >= self.ram_threshold_high:
            ram_category = "lightweight" if ram_percent > 85 else "medium"
            print(f"⚠️ High RAM usage ({ram_percent:.1f}%) - using {ram_category} models")
        else:
            ram_category = "medium"  # Default to medium for balance
            if ram_percent < 50:
                ram_category = "heavy"  # Allow heavy models if RAM is plentiful
        
        # Get available models for this RAM category
        available_models = self.model_hierarchy.get(ram_category, [])
        
        # Find the best available model
        for provider_config in available_models:
            provider = provider_config["provider"]
            models = provider_config["models"]
            
            for model in models:
                if self._is_model_available(provider, model):
                    return {
                        "provider": provider,
                        "model": model,
                        "ram_category": ram_category,
                        "ram_percent": ram_percent,
                        "task_type": task_type
                    }
        
        # Fallback to any available lightweight model
        return self._get_fallback_model()
    
    def switch_model_if_needed(self, current_model: str, current_provider: str) -> Optional[Dict[str, Any]]:
        """Check if model switch is needed based on RAM usage"""
        ram_percent = self._get_current_ram_percent()
        
        if ram_percent >= self.ram_threshold_critical:
            optimal = self.get_optimal_model()
            if optimal["model"] != current_model or optimal["provider"] != current_provider:
                print(f"🔄 RAM-critical switch: {current_provider}/{current_model} → {optimal['provider']}/{optimal['model']}")
                return optimal
        
        elif ram_percent >= self.ram_threshold_high and current_provider in ["ollama"] and "qwen2.5-coder" in current_model:
            # Specifically switch away from memory-intensive qwen2.5-coder when RAM is high
            optimal = self.get_optimal_model()
            if optimal["model"] != current_model:
                print(f"🔄 High RAM switch: {current_provider}/{current_model} → {optimal['provider']}/{optimal['model']}")
                return optimal
        
        return None
    
    def _ram_monitoring_loop(self):
        """Continuous RAM monitoring loop"""
        while self.monitoring_active:
            try:
                ram_percent = self._get_current_ram_percent()
                
                # Log warnings for high RAM usage
                if ram_percent >= self.ram_threshold_critical:
                    print(f"🚨 CRITICAL RAM ALERT: {ram_percent:.1f}% - Immediate model switch recommended")
                elif ram_percent >= self.ram_threshold_high:
                    print(f"⚠️ HIGH RAM WARNING: {ram_percent:.1f}% - Consider switching to lightweight models")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"⚠️ RAM monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _get_current_ram_percent(self) -> float:
        """Get current RAM usage percentage"""
        try:
            return psutil.virtual_memory().percent
        except Exception:
            # Fallback if psutil not available
            return 50.0  # Assume medium usage
    
    def _is_model_available(self, provider: str, model: str) -> bool:
        """Check if a specific model is available"""
        try:
            if provider == "ollama":
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return model in result.stdout
            
            elif provider == "huggingface":
                # For Hugging Face, we assume models are available if the library is installed
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    return True
                except ImportError:
                    return False
            
            elif provider == "swe":
                # SWE models would need custom checking
                return model in ["basic_swe_model"]  # Placeholder
            
            return False
            
        except Exception:
            return False
    
    def _get_fallback_model(self) -> Dict[str, Any]:
        """Get a fallback model when no optimal model is available"""
        return {
            "provider": "ollama",
            "model": "phi:latest",  # Very lightweight fallback
            "ram_category": "lightweight",
            "ram_percent": self._get_current_ram_percent(),
            "task_type": "fallback"
        }

class ConversationDiversityManager:
    """Manages conversation diversity to prevent MetaAgent help requests from dominating chat"""
    
    def __init__(self, system_instance):
        self.system = system_instance
        self.conversation_history = []
        self.meta_agent_message_count = 0
        self.total_message_count = 0
        self.diversity_threshold = 0.3  # Max 30% MetaAgent messages
        self.time_window = 300  # 5 minutes
        
        print("🎭 Conversation Diversity Manager initialized")
        print(f"   📊 Diversity threshold: {self.diversity_threshold:.1%} MetaAgent messages allowed")
    
    def should_allow_meta_agent_message(self, message_type: str = "help_request") -> bool:
        """Check if MetaAgent message should be allowed based on diversity rules"""
        current_time = time.time()
        
        # Clean old messages from history
        self.conversation_history = [
            msg for msg in self.conversation_history 
            if current_time - msg["timestamp"] < self.time_window
        ]
        
        # Count recent MetaAgent messages
        recent_meta_messages = sum(
            1 for msg in self.conversation_history 
            if msg.get("sender") == "meta_agent"
        )
        
        total_recent_messages = len(self.conversation_history)
        
        if total_recent_messages == 0:
            return True  # Allow first message
        
        meta_ratio = recent_meta_messages / total_recent_messages
        
        if meta_ratio >= self.diversity_threshold:
            print(f"🎭 Diversity control: MetaAgent ratio {meta_ratio:.1%} exceeds threshold {self.diversity_threshold:.1%}")
            print("   ⏸️ Delaying MetaAgent message to maintain conversation diversity")
            return False
        
        return True
    
    def record_message(self, sender: str, message_type: str, content: str):
        """Record a message in the conversation history"""
        self.conversation_history.append({
            "sender": sender,
            "type": message_type,
            "content": content[:100] + "..." if len(content) > 100 else content,
            "timestamp": time.time()
        })
        
        # Maintain history size
        if len(self.conversation_history) > 1000:
            self.conversation_history = self.conversation_history[-500:]  # Keep last 500
    
    def get_diversity_stats(self) -> Dict[str, Any]:
        """Get current conversation diversity statistics"""
        current_time = time.time()
        recent_messages = [
            msg for msg in self.conversation_history 
            if current_time - msg["timestamp"] < self.time_window
        ]
        
        meta_messages = sum(1 for msg in recent_messages if msg.get("sender") == "meta_agent")
        total_messages = len(recent_messages)
        
        return {
            "total_recent_messages": total_messages,
            "meta_agent_messages": meta_messages,
            "meta_ratio": meta_messages / total_messages if total_messages > 0 else 0,
            "diversity_threshold": self.diversity_threshold,
            "within_limits": (meta_messages / total_messages if total_messages > 0 else 0) < self.diversity_threshold
        }

# Integration functions for the main system
def initialize_ram_aware_switching(system_instance):
    """Initialize RAM-aware model switching for the system"""
    if not hasattr(system_instance, 'model_switcher'):
        system_instance.model_switcher = RAMAwareModelSwitcher(system_instance)
        print("✅ RAM-aware model switching initialized")
    
    if not hasattr(system_instance, 'diversity_manager'):
        system_instance.diversity_manager = ConversationDiversityManager(system_instance)
        print("✅ Conversation diversity manager initialized")
    
    return True

def get_optimal_model_for_task(system_instance, task_type: str = "general"):
    """Get optimal model for a task based on RAM usage"""
    if hasattr(system_instance, 'model_switcher'):
        return system_instance.model_switcher.get_optimal_model(task_type)
    else:
        # Fallback to basic model selection
        return {"provider": "ollama", "model": "codellama:latest", "fallback": True}

def check_model_switch_needed(system_instance, current_model: str, current_provider: str):
    """Check if model switch is needed based on RAM"""
    if hasattr(system_instance, 'model_switcher'):
        switch_info = system_instance.model_switcher.switch_model_if_needed(current_model, current_provider)
        return switch_info
    return None

def should_allow_meta_agent_message(system_instance, message_type: str = "help_request"):
    """Check if MetaAgent message should be allowed for diversity"""
    if hasattr(system_instance, 'diversity_manager'):
        return system_instance.diversity_manager.should_allow_meta_agent_message(message_type)
    return True  # Allow if no diversity manager

def record_conversation_message(system_instance, sender: str, message_type: str, content: str):
    """Record a message for diversity tracking"""
    if hasattr(system_instance, 'diversity_manager'):
        system_instance.diversity_manager.record_message(sender, message_type, content)

if __name__ == "__main__":
    print("🧠 RAM-Aware Model Switcher & Conversation Diversity Manager")
    print("   �� RAM monitoring + intelligent model switching")
    print("   🎭 Conversation diversity controls")
    print("   🔄 Automatic optimization based on system resources")
