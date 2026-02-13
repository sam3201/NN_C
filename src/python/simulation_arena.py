# SAM Simulation Arena
# Revenue-generating autonomous simulations

import time as sam_time_ref
import random
from typing import Dict, Any, List

class SimulationArena:
    """
    Autonomous Simulation Arena for SAM-D.
    - Runs virtual environments (trading, games, resource management).
    - Generates virtual revenue (which can be mapped to real revenue goals).
    - Provides a training ground for submodels/shards.
    """
    
    def __init__(self, system=None):
        self.system = system
        self.active_simulations = {}
        self.total_virtual_revenue = 0.0
        self.sim_count = 0
        
    def spawn_simulation(self, sim_type: str = "trading") -> str:
        sim_id = f"sim_{int(sam_time_ref.time())}_{random.randint(100, 999)}"
        self.active_simulations[sim_id] = {
            "type": sim_type,
            "start_time": sam_time_ref.time(),
            "revenue": 0.0,
            "status": "running",
            "complexity": random.uniform(0.1, 0.9)
        }
        self.sim_count += 1
        print(f"🎮 SPAWNED Simulation: {sim_type} (ID: {sim_id})")
        return sim_id
        
    def update(self):
        """Update all active simulations"""
        for sim_id, sim in self.active_simulations.items():
            if sim["status"] != "running":
                continue
                
            # Random revenue generation based on complexity
            gain = random.uniform(0, sim["complexity"]) * 0.1
            sim["revenue"] += gain
            self.total_virtual_revenue += gain
            
            # Periodic completion
            if random.random() < 0.01:
                sim["status"] = "completed"
                print(f"🎮 COMPLETED Simulation: {sim_id} (Yield: {sim['revenue']:.2f})")
                
        # Sync with system banking if available
        if self.system and hasattr(self.system, "banking"):
            # Map virtual revenue to system metrics (Phase 5.3)
            self.system.system_metrics["virtual_revenue"] = self.total_virtual_revenue

    def get_status(self):
        return {
            "active": len([s for s in self.active_simulations.values() if s["status"] == "running"]),
            "total_yield": self.total_virtual_revenue,
            "sim_count": self.sim_count
        }

def create_simulation_arena(system=None):
    return SimulationArena(system)
