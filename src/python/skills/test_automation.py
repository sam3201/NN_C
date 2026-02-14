# SAM Test Automation Skill
# Manages test generation and execution

import subprocess
import os
from pathlib import Path
from typing import Dict, Any

class TestAutomator:
    """
    Automates the testing lifecycle.
    - Discovers existing tests.
    - Executes test suites via pytest.
    - (Future) Generates missing tests for uncovered paths.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
    def run_tests(self, target: str = "tests/") -> Dict[str, Any]:
        """Run pytest on the specified target."""
        print(f"🧪 Running tests: {target}")
        try:
            result = subprocess.run(
                ["pytest", "--json-report", "--json-report-file=logs/test_results.json", target],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[-500:], # Return last 500 chars
                "exit_code": result.returncode
            }
        except Exception as e:
            return {"error": str(e)}

def create_automator(project_root: Path):
    return TestAutomator(project_root)
