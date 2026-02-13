# SAM Multimodal Sensory Controller
# Handles internal and external stimuli, mapping them to system responses

import time as sam_time_ref
import json
import os
from typing import Dict, Any, List, Optional

class SensoryController:
    """
    Multimodal Sensory Controller for SAM-D.
    - Vision: File explorer, image analysis.
    - Audit: Log monitoring, anomaly detection.
    - Proprioception: Internal state, telemetry.
    - Haptics: System pressure, UI feedback.
    """
    
    def __init__(self, system=None):
        self.system = system
        self.modalities = ["vision", "audit", "proprioception", "haptics"]
        self.sensory_history = []
        self.max_history = 100
        
        # Stimuli thresholds
        self.stimuli_sensitivity = 0.5
        
    def process_stimulus(self, modality: str, source: str, data: Any):
        """Process an incoming stimulus and determine response"""
        timestamp = sam_time_ref.time()
        
        stimulus = {
            "ts": timestamp,
            "modality": modality,
            "source": source,
            "intensity": self._calculate_intensity(data),
            "data_summary": str(data)[:100]
        }
        
        self.sensory_history.append(stimulus)
        if len(self.sensory_history) > self.max_history:
            self.sensory_history.pop(0)
            
        # Map stimuli to system response
        if stimulus["intensity"] > self.stimuli_sensitivity:
            self._trigger_response(stimulus)
            
        return stimulus

    def _calculate_intensity(self, data: Any) -> float:
        """Heuristic for stimulus intensity (0.0 to 1.0)"""
        if isinstance(data, dict):
            # If it's a telemetry signal, use normalized value
            return float(data.get("value", 0.5))
        if isinstance(data, (int, float)):
            return min(1.0, float(data))
        if isinstance(data, str):
            # Complexity/length as proxy for intensity
            return min(1.0, len(data) / 1000.0)
        return 0.5

    def _trigger_response(self, stimulus: Dict[str, Any]):
        """Directly influence system behavior based on sense input"""
        if not self.system: return
        
        modality = stimulus["modality"]
        intensity = stimulus["intensity"]
        
        # Proprioception: High pressure triggers growth evaluation
        if modality == "proprioception" and intensity > 0.8:
            print(f"🧠 SENSE (Proprioception): High internal pressure ({intensity:.2f}). Triggering growth.")
            if hasattr(self.system, "_trigger_growth_system"):
                self.system._trigger_growth_system()
                
        # Audit (Hearing): Anomalies trigger self-healing
        if modality == "audit" and intensity > 0.7:
            print(f"👂 SENSE (Hearing): High-intensity audit event detected. Investigating system integrity.")
            if hasattr(self.system, "_perform_self_healing_check"):
                self.system._perform_self_healing_check()
                
        # Haptics (Touch): High friction triggers adaptive resource allocation
        if modality == "haptics" and intensity > 0.7:
            print(f"✋ SENSE (Haptics): High operational friction ({intensity:.2f}). Adapting resource strategy.")
            # Trigger dynamic resource scaling
            if hasattr(self.system, "_wire_regulator_knobs"):
                # Simulate a nudge to the regulator
                self.system.system_metrics["last_haptic_event"] = sam_time_ref.time()

        # Vision: Large file changes trigger code scanning
        if modality == "vision" and "file_change" in stimulus["source"] and intensity > 0.5:
            print(f"👁️ SENSE (Vision): Codebase modification detected. Initiating autonomous scan.")
            if hasattr(self.system, "code_scanner") and self.system.code_scanner:
                self.system.code_scanner.scan_next()

    def get_sensory_state(self) -> Dict[str, Any]:
        """Summary of current sensory input"""
        return {
            "active_modalities": self.modalities,
            "recent_events": len(self.sensory_history),
            "last_stimulus": self.sensory_history[-1] if self.sensory_history else None
        }

def create_sensory_controller(system=None):
    return SensoryController(system)
