import sys
import os
import traceback
from pathlib import Path

# Add src/python to path
sys.path.append(os.path.abspath("src/python"))

try:
    from complete_sam_unified import UnifiedSAMSystem
    
    system = UnifiedSAMSystem()
    print("System initialized. Running one cycle of autonomous loop...")
    
    # Manually run one iteration of the loop logic to catch the error
    system._update_system_metrics()
    system._generate_unsupervised_metrics()
    system._generate_autonomous_goals()
    system._generate_autonomous_conversations()
    system._execute_autonomous_tasks()
    system._run_survival_evaluation()
    system._execute_goal_cycle()
    
    print("Cycle complete. No error found.")
    
except Exception:
    print("Caught expected error:")
    traceback.print_exc()
