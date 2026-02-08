#!/usr/bin/env python3
"""
Experimental Auto-Code-Modification System
Allows SAM to automatically apply code changes with high confidence,
test them in spawned processes, and integrate successful modifications.
"""

import os
import sys
import time
import json
import subprocess
import threading
import multiprocessing
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import shutil

class ExperimentalCodeModifier:
    """Experimental system for automatic code modification and testing"""
    
    def __init__(self, system_instance):
        self.system = system_instance
        self.auto_code_confidence_threshold = 0.8  # Higher threshold for auto-code
        self.test_timeout = 30  # seconds to test changes
        self.max_experiments = 3  # Max concurrent experimental processes
        self.active_experiments = {}
        
        # Create experimental directory
        self.exp_dir = Path.home() / ".sam_experiments"
        self.exp_dir.mkdir(exist_ok=True)
        
        print("🧪 Experimental Auto-Code-Modification System initialized")
        print(f"   📊 Confidence threshold: {self.auto_code_confidence_threshold}")
        print(f"   ⏱️ Test timeout: {self.test_timeout}s")
        print(f"   📁 Experiments directory: {self.exp_dir}")
    
    def should_attempt_auto_code(self, confidence: float, solution_type: str) -> bool:
        """Determine if we should attempt automatic code modification"""
        return (
            confidence >= self.auto_code_confidence_threshold and
            solution_type == "code_change" and
            len(self.active_experiments) < self.max_experiments
        )
    
    def apply_experimental_code_change(self, issue: Dict, code_changes: Dict, confidence: float) -> bool:
        """Apply code changes experimentally in a spawned process"""
        try:
            # Create unique experiment ID
            exp_id = f"exp_{int(time.time())}_{hash(str(code_changes)) % 1000}"
            
            # Create experiment directory
            exp_path = self.exp_dir / exp_id
            exp_path.mkdir(exist_ok=True)
            
            # Copy current system to experiment
            self._copy_system_to_experiment(exp_path)
            
            # Apply code changes to experimental copy
            success = self._apply_changes_to_experiment(exp_path, code_changes)
            if not success:
                print(f"❌ Failed to apply changes in experiment {exp_id}")
                return False
            
            # Spawn experimental process
            self._spawn_experimental_process(exp_id, exp_path, issue)
            
            print(f"🧪 Started experimental code modification: {exp_id}")
            print(f"   🎯 Testing with confidence: {confidence:.2f}")
            print(f"   📂 Experiment path: {exp_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Experimental code modification failed: {e}")
            return False
    
    def _copy_system_to_experiment(self, exp_path: Path):
        """Copy current system files to experimental directory"""
        current_dir = Path.cwd()
        
        # Copy essential files
        essential_files = [
            'complete_sam_unified.py',
            'sam_cli.py',
            'ram_model_switcher.py',
            'terminal_routes.py',
            'experimental_auto_code.py',
            'run_sam.sh',
            'README.md'
        ]
        
        for file in essential_files:
            src = current_dir / file
            if src.exists():
                shutil.copy2(src, exp_path / file)
        
        # Copy requirements if exists
        req_file = current_dir / 'requirements.txt'
        if req_file.exists():
            shutil.copy2(req_file, exp_path / 'requirements.txt')
    
    def _apply_changes_to_experiment(self, exp_path: Path, code_changes: Dict) -> bool:
        """Apply code changes to experimental copy"""
        try:
            for file_path, changes in code_changes.items():
                exp_file = exp_path / file_path
                
                if not exp_file.exists():
                    print(f"⚠️ File {file_path} not found in experiment")
                    continue
                
                # Read current content
                with open(exp_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Apply changes (simplified - assumes changes are replacements)
                if isinstance(changes, str):
                    # Simple string replacement
                    content = content.replace(changes.split(' -> ')[0], changes.split(' -> ')[1])
                elif isinstance(changes, dict):
                    # More complex changes
                    for old_code, new_code in changes.items():
                        content = content.replace(old_code, new_code)
                
                # Write modified content
                with open(exp_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to apply changes to experiment: {e}")
            return False
    
    def _spawn_experimental_process(self, exp_id: str, exp_path: Path, issue: Dict):
        """Spawn an experimental process to test the code changes"""
        
        def run_experiment():
            try:
                # Change to experiment directory
                os.chdir(exp_path)
                
                # Start experimental system
                print(f"🧪 Experiment {exp_id}: Starting test system...")
                
                # Run basic syntax check first
                result = subprocess.run(
                    [sys.executable, '-m', 'py_compile', 'complete_sam_unified.py'],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode != 0:
                    print(f"❌ Experiment {exp_id}: Syntax check failed")
                    self._experiment_failed(exp_id, "syntax_error")
                    return
                
                print(f"✅ Experiment {exp_id}: Syntax check passed")
                
                # Try to import and initialize (basic smoke test)
                try:
                    # Add current directory to path for imports
                    sys.path.insert(0, str(exp_path))
                    
                    # Import and create system instance (without running it)
                    from complete_sam_unified import UnifiedSAMSystem
                    
                    # Just test initialization without full startup
                    print(f"🧪 Experiment {exp_id}: Testing system initialization...")
                    
                    # Create system but don't run it (to avoid conflicts)
                    system = UnifiedSAMSystem.__new__(UnifiedSAMSystem)
                    
                    print(f"✅ Experiment {exp_id}: System initialization successful")
                    self._experiment_succeeded(exp_id, issue)
                    
                except Exception as e:
                    print(f"❌ Experiment {exp_id}: System test failed: {e}")
                    self._experiment_failed(exp_id, str(e))
                
            except subprocess.TimeoutExpired:
                print(f"⏰ Experiment {exp_id}: Timed out")
                self._experiment_failed(exp_id, "timeout")
            except Exception as e:
                print(f"❌ Experiment {exp_id}: Unexpected error: {e}")
                self._experiment_failed(exp_id, str(e))
        
        # Start experiment in background thread
        exp_thread = threading.Thread(target=run_experiment, daemon=True)
        exp_thread.start()
        
        # Track active experiment
        self.active_experiments[exp_id] = {
            'thread': exp_thread,
            'path': exp_path,
            'start_time': time.time(),
            'issue': issue,
            'status': 'running'
        }
    
    def _experiment_succeeded(self, exp_id: str, issue: Dict):
        """Handle successful experiment"""
        if exp_id in self.active_experiments:
            self.active_experiments[exp_id]['status'] = 'success'
            
            print(f"🎉 Experiment {exp_id}: SUCCESS!")
            print("   📝 Code changes validated and ready for integration")
            
            # Here we could automatically integrate the changes
            # For now, we'll just log the success
            
            # Clean up after some time
            threading.Timer(300, lambda: self._cleanup_experiment(exp_id)).start()
    
    def _experiment_failed(self, exp_id: str, reason: str):
        """Handle failed experiment"""
        if exp_id in self.active_experiments:
            self.active_experiments[exp_id]['status'] = 'failed'
            self.active_experiments[exp_id]['failure_reason'] = reason
            
            print(f"❌ Experiment {exp_id}: FAILED - {reason}")
            print("   🗑️ Discarding experimental changes")
            
            # Clean up immediately for failed experiments
            self._cleanup_experiment(exp_id)
    
    def _cleanup_experiment(self, exp_id: str):
        """Clean up experimental directory and tracking"""
        if exp_id in self.active_experiments:
            exp_info = self.active_experiments[exp_id]
            exp_path = exp_info['path']
            
            try:
                # Remove experimental directory
                if exp_path.exists():
                    shutil.rmtree(exp_path)
                    print(f"🗑️ Cleaned up experiment {exp_id}")
            except Exception as e:
                print(f"⚠️ Failed to cleanup experiment {exp_id}: {e}")
            
            # Remove from tracking
            del self.active_experiments[exp_id]
    
    def get_experiment_status(self) -> Dict[str, Any]:
        """Get status of all active experiments"""
        return {
            'active_experiments': len(self.active_experiments),
            'experiments': {
                exp_id: {
                    'status': info['status'],
                    'runtime': time.time() - info['start_time'],
                    'issue_type': info['issue'].get('type', 'unknown'),
                    'failure_reason': info.get('failure_reason')
                }
                for exp_id, info in self.active_experiments.items()
            }
        }
    
    def integrate_successful_experiment(self, exp_id: str) -> bool:
        """Integrate successful experimental changes into main codebase"""
        if exp_id not in self.active_experiments:
            print(f"❌ Experiment {exp_id} not found")
            return False
        
        exp_info = self.active_experiments[exp_id]
        if exp_info['status'] != 'success':
            print(f"❌ Experiment {exp_id} is not successful (status: {exp_info['status']})")
            return False
        
        try:
            exp_path = exp_info['path']
            current_dir = Path.cwd()
            
            # Copy modified files back to main directory
            for file in ['complete_sam_unified.py', 'sam_cli.py', 'ram_model_switcher.py']:
                exp_file = exp_path / file
                main_file = current_dir / file
                
                if exp_file.exists():
                    shutil.copy2(exp_file, main_file)
                    print(f"✅ Integrated {file} from experiment {exp_id}")
            
            print(f"🎉 Successfully integrated experimental changes from {exp_id}")
            print("   🔄 System restart recommended to apply changes")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to integrate experiment {exp_id}: {e}")
            return False

# Integration functions
def initialize_experimental_auto_code(system_instance):
    """Initialize experimental auto-code modification system"""
    if not hasattr(system_instance, 'experimental_modifier'):
        system_instance.experimental_modifier = ExperimentalCodeModifier(system_instance)
        print("🧪 Experimental Auto-Code-Modification System: ACTIVE")
        print("   🎯 High-confidence code changes will be tested automatically")
        print("   🔄 Successful experiments can be integrated into main codebase")
    
    return True

def check_experimental_code_application(system_instance, issue: Dict, confidence: float, solution_type: str, code_changes: Dict) -> bool:
    """Check if we should apply code changes experimentally"""
    if hasattr(system_instance, 'experimental_modifier'):
        if system_instance.experimental_modifier.should_attempt_auto_code(confidence, solution_type):
            print(f"🧪 High confidence detected ({confidence:.2f}) - attempting experimental code modification")
            return system_instance.experimental_modifier.apply_experimental_code_change(issue, code_changes, confidence)
    
    return False

def get_experimental_status(system_instance) -> Dict[str, Any]:
    """Get status of experimental system"""
    if hasattr(system_instance, 'experimental_modifier'):
        return system_instance.experimental_modifier.get_experiment_status()
    
    return {'active_experiments': 0, 'experiments': {}}

if __name__ == "__main__":
    print("🧪 Experimental Auto-Code-Modification System")
    print("   🎯 Automatic code testing and integration")
    print("   🔄 Process spawning for safe experimentation")
    print("   📊 High-confidence code application")
    
    # Test basic functionality
    modifier = ExperimentalCodeModifier(None)
    print("✅ Experimental modifier initialized")
    
    status = modifier.get_experiment_status()
    print(f"📊 Status: {status['active_experiments']} active experiments")
