#!/usr/bin/env python3
"""
OpenClaw Integration Bridge for SAM-D AGI
Connects OpenClaw with OpenCode workflows and tri-cameral governance.
Wired to the REAL automation core.
"""

import json
import sys
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add automation core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/python")))
try:
    from automation.core import FileProcessor, TriCameralGovernance, ProcessingResult
except ImportError:
    # Fallback if structure is slightly different
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".." )))
    from src.python.automation.core import FileProcessor, TriCameralGovernance, ProcessingResult

class Branch(Enum):
    CIC = "Constructive Intelligence Core"
    AEE = "Adversarial Exploration Engine"
    CSF = "Coherence Stabilization Field"

class Phase(Enum):
    PLAN = "plan"
    ANALYZE = "analyze"
    BUILD = "build"
    TEST = "test"
    COMPLETE = "complete"

class OpenClawOrchestrator:
    """
    Orchestrates tri-cameral governance with cyclic development workflow
    Powered by the Real Automation Core.
    """
    
    def __init__(self, config_path: str = ".openclaw/config.json"):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
            
        self.governance = TriCameralGovernance()
        self.processor = None
        
    def start_tri_cameral_cycle(self, task: str, target_file: str):
        """Start a new tri-cameral decision cycle for a real task"""
        print(f"🔱 Starting REAL Tri-Cameral Cycle for: {task}")
        
        if not os.path.exists(target_file):
            return {"error": f"Target file {target_file} not found"}
            
        self.processor = FileProcessor(target_file)
        if not self.processor.load_file():
            return {"error": "Failed to load file"}
            
        self.processor.split_into_chunks()
        
        # 1. Real Planning Phase
        print("\n📋 PHASE: PLANNING")
        plan_result = self.processor.planning_phase()
        decision = self.governance.evaluate("planning", plan_result)
        
        if not decision.proceed:
            return {
                "decision": "🛑 REJECT (Planning failed)",
                "confidence": decision.confidence,
                "reasoning": decision.concerns
            }
            
        # 2. Real Building Phase
        print("\n🔨 PHASE: BUILDING")
        build_result = self.processor.building_phase()
        decision = self.governance.evaluate("building", build_result)
        
        if not decision.proceed:
            return {
                "decision": "⚠️ REVISE (Building needs improvement)",
                "confidence": decision.confidence,
                "reasoning": decision.concerns
            }
            
        # 3. Real Testing Phase
        print("\n🧪 PHASE: TESTING")
        test_result = self.processor.testing_phase()
        decision = self.governance.evaluate("testing", test_result)
        
        status = "✅ PROCEED" if decision.proceed else "🛑 REJECT"
        
        return {
            "decision": status,
            "consensus": decision.proceed,
            "confidence": decision.confidence,
            "votes": {
                "CIC": decision.cic_vote,
                "AEE": decision.aee_vote,
                "CSF": decision.csf_vote
            },
            "metrics": self.processor.metrics
        }

    def run_cyclic_workflow(self, task: str, target_file: str):
        """Run the real cyclic workflow using the automation core's logic"""
        print(f"\n🔄 Starting REAL Cyclic Workflow: {task}")
        
        # Since the automation core's main() already implements the cyclic loop,
        # we delegate to a more integrated execution here.
        
        if not os.path.exists(target_file):
            return {"status": "error", "message": f"File {target_file} not found"}
            
        # Initialize
        self.processor = FileProcessor(target_file)
        if not self.processor.load_file():
            return {"status": "error", "message": "Load failed"}
            
        self.processor.split_into_chunks()
        
        iteration = 0
        max_iters = self.config.get("workflows", {}).get("cyclic_development", {}).get("max_iterations", 3)
        
        while iteration < max_iters:
            iteration += 1
            print(f"\n--- Workflow Iteration {iteration} ---")
            
            # Run phases
            p_res = self.processor.planning_phase()
            if not self.governance.evaluate("planning", p_res).proceed:
                print("   ↩️  Planning failed, retrying...")
                continue
                
            b_res = self.processor.building_phase()
            if not self.governance.evaluate("building", b_res).proceed:
                print("   ↩️  Building failed, entering revision...")
                self.processor.revision_phase(b_res)
                continue
                
            t_res = self.processor.testing_phase()
            decision = self.governance.evaluate("testing", t_res)
            
            if decision.proceed:
                print(f"\n✨ Workflow completed successfully after {iteration} iterations!")
                return {"status": "completed", "iterations": iteration, "quality": t_res.quality_score}
            else:
                print("   ↩️  Testing failed, looping back...")
                
        return {"status": "max_iterations_reached", "quality": self.processor._calculate_quality()}

def main():
    if len(sys.argv) < 3:
        print("Usage: openclaw_bridge.py <command> <target_file> [args...]")
        print("Commands:")
        print("  tri-cameral <target_file>")
        print("  cyclic <target_file>")
        sys.exit(1)
    
    command = sys.argv[1]
    target_file = sys.argv[2]
    orchestrator = OpenClawOrchestrator()
    
    if command == "tri-cameral":
        result = orchestrator.start_tri_cameral_cycle("Refine codebase", target_file)
        print("\n" + json.dumps(result, indent=2))
    
    elif command == "cyclic":
        result = orchestrator.run_cyclic_workflow("Production hardening", target_file)
        print("\n" + json.dumps(result, indent=2))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
