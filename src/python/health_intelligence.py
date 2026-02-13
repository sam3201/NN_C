# SAM Health Intelligence Module
# Automated prompt testing, health baselines, and error anomaly detection

import time as sam_time_ref
import json
import os
import re
import traceback
from typing import Dict, Any, List, Optional
from pathlib import Path

class HealthIntelligence:
    """
    Advanced health monitoring and self-optimization for SAM-D.
    - Automated Prompt Testing (APT): Evaluates prompt effectiveness.
    - Health Baselines: Establishes 'normal' behavior metrics.
    - Anomaly Detection: Identifies unusual error patterns in real-time.
    """
    
    def __init__(self, system=None):
        self.system = system
        self.project_root = Path(system.project_root) if system else Path(".")
        self.error_log_path = self.project_root / "logs" / "error_anomalies.jsonl"
        self.baseline_path = self.project_root / "sam_data" / "health_baseline.json"
        
        # APT State
        self.prompt_templates = {}
        self.prompt_performance = {}
        
        # Anomaly Detection State
        self.error_counts = {}
        self.anomaly_threshold = 5 # Occurrences per minute
        
        # Health Metrics
        self.health_baseline = {
            "avg_latency": 0.5,
            "error_rate": 0.01,
            "survival_stability": 0.9,
            "memory_usage_stable": 0.5
        }
        self._load_baseline()

    def record_error(self, error_type: str, message: str, traceback_str: str):
        """Record an error and check for anomalies"""
        timestamp = sam_time_ref.time()
        error_entry = {
            "ts": timestamp,
            "type": error_type,
            "msg": message,
            "trace": traceback_str
        }
        
        # Persistent logging
        try:
            with open(self.error_log_path, "a") as f:
                f.write(json.dumps(error_entry) + "\n")
        except:
            pass
            
        # Real-time anomaly detection
        minute_bucket = int(timestamp / 60)
        key = (error_type, minute_bucket)
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        if self.error_counts[key] >= self.anomaly_threshold:
            print(f"🚨 ANOMALY DETECTED: Error '{error_type}' occurred {self.error_counts[key]} times in the last minute!")
            self._trigger_emergency_healing(error_type, message)

    def evaluate_prompt(self, prompt_id: str, input_data: str, response: str, success: bool):
        """Evaluate the performance of a prompt"""
        if prompt_id not in self.prompt_performance:
            self.prompt_performance[prompt_id] = {"success": 0, "total": 0, "avg_length": 0}
            
        perf = self.prompt_performance[prompt_id]
        perf["total"] += 1
        if success:
            perf["success"] += 1
        
        # Update moving average of response quality (proxy: length and structure)
        perf["avg_length"] = (perf["avg_length"] * (perf["total"] - 1) + len(response)) / perf["total"]
        
        # Auto-optimization: If success rate drops, propose prompt change
        if perf["total"] > 10 and (perf["success"] / perf["total"]) < 0.6:
            self._propose_prompt_optimization(prompt_id)

    def check_system_health(self) -> Dict[str, Any]:
        """Verify current health against baseline"""
        if not self.system: return {"status": "detached"}
        
        current_metrics = self.system.system_metrics
        survival = current_metrics.get("survival_score", 1.0)
        latency = current_metrics.get("latency", 0.5)
        
        health_status = "excellent"
        issues = []
        
        if survival < self.health_baseline["survival_stability"] * 0.8:
            health_status = "critical"
            issues.append("Survival score below critical baseline")
            
        if latency > self.health_baseline["avg_latency"] * 3:
            health_status = "degraded"
            issues.append("Latency anomaly detected")
            
        return {
            "status": health_status,
            "issues": issues,
            "baseline_drift": survival - self.health_baseline["survival_stability"]
        }

    def _load_baseline(self):
        if self.baseline_path.exists():
            try:
                self.health_baseline = json.loads(self.baseline_path.read_text())
            except:
                pass

    def _trigger_emergency_healing(self, error_type: str, message: str):
        """Autonomously attempt to fix a repeating error"""
        if "time" in message.lower() or "time" in error_type.lower():
            print("🛠️ APT: Identified persistent 'time' namespace issue. Proposing architectural shielding.")
            # This would trigger a specific repair task in GoalManager
            if self.system and hasattr(self.system, "goal_manager"):
                self.system.goal_manager.add_goal(
                    f"Fix repeating anomaly: {error_type}",
                    priority="critical",
                    goal_id="repair_anomaly"
                )

    def _propose_prompt_optimization(self, prompt_id: str):
        """Use MetaAgent to optimize a failing prompt template"""
        print(f"🧪 APT: Prompt '{prompt_id}' performance degraded. Proposing optimization.")
        # Trigger autonomous research task for prompt engineering

def create_health_intelligence(system=None):
    return HealthIntelligence(system)
