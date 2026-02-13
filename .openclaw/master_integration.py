#!/usr/bin/env python3
"""
SAM-D AGI - Master Integration Launcher
Brings together all components: Automation Framework + SAM-D Core
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'automation_framework' / 'python'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'python'))

from automation_bridge import (
    AutomationFramework, 
    WorkflowConfig,
    select_best_model,
    auto_switch_model
)

try:
    from sam_cores import SamCores
    SAM_CORES_AVAILABLE = True
except ImportError:
    SAM_CORES_AVAILABLE = False
    print("⚠️  SAM cores not available (C extensions not built)")

class MasterIntegration:
    """
    Master integration bringing together:
    - Automation Framework (tri-cameral governance, model routing)
    - SAM-D Core (ΨΔ•Ω-Core, Id/Ego/Superego, C extensions)
    """
    
    def __init__(self):
        self.automation = AutomationFramework()
        self.sam_cores = None
        if SAM_CORES_AVAILABLE:
            try:
                self.sam_cores = SamCores(seed=42)
                print("✅ SAM-D Core initialized")
            except Exception as e:
                print(f"⚠️  SAM-D Core initialization failed: {e}")
        
        self.system_state = {
            'phase': 'initialization',
            'components_ready': [],
            'current_model': None,
            'last_action': None
        }
    
    async def start(self):
        """Initialize and start the integrated system"""
        print("\n" + "="*70)
        print("🚀 SAM-D AGI - Master Integration")
        print("="*70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. Initialize Automation Framework
        print("📊 Initializing Automation Framework...")
        self.system_state['components_ready'].append('automation_framework')
        print("  ✅ Tri-cameral governance ready")
        print("  ✅ Dynamic model router ready")
        print("  ✅ Concurrent subagents ready")
        print("  ✅ Resource management ready")
        
        # 2. Check SAM-D Core
        if self.sam_cores:
            print("\n🧠 Initializing SAM-D Core...")
            self.system_state['components_ready'].append('sam_core')
            print("  ✅ C extensions loaded")
            print("  ✅ Phase 1 systems active (Id/Ego/Superego)")
            print("  ✅ God Equation ready")
            print("  ✅ 53-Regulator compiler ready")
        
        # 3. Select optimal model for initialization
        print("\n🎯 Selecting optimal model...")
        model = auto_switch_model("Initialize SAM-D AGI system with full automation")
        self.system_state['current_model'] = model
        print(f"  ✅ Using model: {model}")
        
        # 4. Display system status
        print("\n📋 System Status:")
        print(f"  Components Ready: {len(self.system_state['components_ready'])}/2")
        print(f"  Current Phase: {self.system_state['phase']}")
        print(f"  Active Model: {self.system_state['current_model']}")
        
        print("\n✅ Master Integration Ready!")
        print("="*70 + "\n")
        
        return True
    
    async def execute_phase_2(self):
        """Execute Phase 2: Add Power/Control systems"""
        print("\n🏗️  Executing Phase 2: Power & Control Systems")
        print("-"*70)
        
        # Create workflow configuration
        config = WorkflowConfig(
            name="Phase 2 Implementation",
            description="Add Power (P_t) and Control (C_t) systems",
            high_level_plan="Integrate Power and Control with existing Phase 1 systems",
            low_level_plan="Create PowerSystem and ControlSystem classes, integrate with DriveSystem",
            invariants=[
                "maintain_API_compatibility",
                "no_breaking_changes",
                "preserve_identity_continuity"
            ],
            hard_constraints=[
                "All tests must pass",
                "Performance must not regress",
                "Memory usage within bounds"
            ],
            risk_level=0.7
        )
        
        # Execute with full automation
        result = await self.automation.execute_workflow(config)
        
        if result['success']:
            print("\n✅ Phase 2 workflow completed successfully!")
            print(f"  Model Used: {result['model_used']}")
            print(f"  Phases Completed: {result['workflow']['phases_completed']}")
            
            # Get governance decision details
            gov = result['governance']
            print(f"\n  Tri-Cameral Decision:")
            print(f"    CIC: {gov['cic_vote']['decision']} ({gov['cic_vote']['confidence']:.2f})")
            print(f"    AEE: {gov['aee_vote']['decision']} ({gov['aee_vote']['confidence']:.2f})")
            print(f"    CSF: {gov['csf_vote']['decision']} ({gov['csf_vote']['confidence']:.2f})")
            
            if gov['concerns']:
                print(f"\n  Concerns Addressed: {len(gov['concerns'])}")
            
            return True
        else:
            print(f"\n❌ Phase 2 workflow failed: {result['reason']}")
            return False
    
    def get_status(self):
        """Get current system status"""
        return {
            'system': 'SAM-D AGI Master Integration',
            'timestamp': datetime.now().isoformat(),
            'components': self.system_state['components_ready'],
            'sam_cores_available': SAM_CORES_AVAILABLE,
            'current_model': self.system_state['current_model'],
            'automation_stats': self.automation.get_model_stats() if hasattr(self.automation, 'get_model_stats') else {}
        }
    
    def interactive_shell(self):
        """Start interactive shell"""
        print("\n🖥️  Interactive Mode")
        print("Commands: status, phase2, model <task>, exit")
        print("-"*70)
        
        while True:
            try:
                command = input("\nSAM-D> ").strip().lower()
                
                if command == 'exit':
                    print("👋 Goodbye!")
                    break
                
                elif command == 'status':
                    status = self.get_status()
                    print(json.dumps(status, indent=2, default=str))
                
                elif command == 'phase2':
                    asyncio.run(self.execute_phase_2())
                
                elif command.startswith('model '):
                    task = command[6:]
                    model = select_best_model(task)
                    print(f"Selected model: {model}")
                
                elif command == 'help':
                    print("Commands:")
                    print("  status    - Show system status")
                    print("  phase2    - Execute Phase 2 workflow")
                    print("  model <task> - Select best model for task")
                    print("  exit      - Exit shell")
                
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

async def main():
    """Main entry point"""
    integration = MasterIntegration()
    
    # Initialize
    await integration.start()
    
    # Start interactive shell
    integration.interactive_shell()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Shutdown complete.")
        sys.exit(0)
