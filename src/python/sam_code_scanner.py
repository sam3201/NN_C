# SAM Code Scanner
# Autonomous codebase analysis and task generation

import os
import ast
from pathlib import Path
from typing import List, Dict, Any

class CodeScanner:
    """
    Autonomous Code Scanner for SAM-D.
    - Scans project directory for Python/C files.
    - Analyzes complexity, TODOs, and potential bugs.
    - Generates tasks for the GoalManager.
    """
    
    def __init__(self, project_root: str, system=None):
        self.project_root = Path(project_root)
        self.system = system
        self.last_scanned_file = None
        
    def scan_next(self) -> List[Dict[str, Any]]:
        """Scan a random file and return findings"""
        try:
            all_files = []
            for ext in ['*.py', '*.c', '*.h']:
                all_files.extend(list(self.project_root.rglob(ext)))
            
            if not all_files:
                return []
                
            import random
            target = random.choice(all_files)
            self.last_scanned_file = str(target)
            
            findings = []
            content = target.read_text(encoding='utf-8', errors='ignore')
            
            # Simple Pattern Matching
            if "TODO" in content:
                findings.append({
                    "type": "improvement",
                    "name": f"Address TODO in {target.name}",
                    "description": f"Code contains TODO markers in {target.relative_to(self.project_root)}"
                })
                
            if len(content.splitlines()) > 500:
                findings.append({
                    "type": "code",
                    "name": f"Refactor large file {target.name}",
                    "description": f"File {target.name} has over 500 lines. Consider modularizing."
                })
                
            # Python specific AST analysis
            if target.suffix == '.py':
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if len(node.body) > 20:
                                findings.append({
                                    "type": "code",
                                    "name": f"Refactor class {node.name}",
                                    "description": f"Class {node.name} in {target.name} is too complex."
                                })
                except:
                    pass
                    
            return findings
            
        except Exception as e:
            print(f"⚠️ Code scanning error: {e}")
            return []

def create_code_scanner(project_root: str, system=None):
    return CodeScanner(project_root, system)
