# SAM Code Analysis Skill
# Deep structural and security analysis of the codebase

import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Any

class CodeAnalyzer:
    """
    Performs deep static analysis on Python/C source files.
    - Detects dangerous patterns (injection, hardcoded keys).
    - Analyzes structural complexity.
    - Verifies docstring coverage.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """Deep scan a single file."""
        abs_path = self.project_root / filepath
        if not abs_path.exists():
            return {"error": "File not found"}
            
        content = abs_path.read_text(encoding="utf-8", errors="ignore")
        
        findings = {
            "complexity": self._estimate_complexity(content),
            "security_issues": self._scan_security(content),
            "doc_coverage": self._check_docs(content)
        }
        
        return findings

    def _estimate_complexity(self, content: str) -> float:
        """Heuristic complexity score based on nesting and line count."""
        lines = content.split('\n')
        indent_levels = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
        if not indent_levels: return 0.0
        return sum(indent_levels) / len(indent_levels) / 4.0

    def _scan_security(self, content: str) -> List[str]:
        """Scan for hardcoded keys and dangerous calls."""
        issues = []
        if re.search(r'sk-[a-zA-Z0-9]{20,}', content):
            issues.append("HARDCODED_API_KEY")
        if "eval(" in content or "exec(" in content:
            issues.append("INSECURE_EXECUTION")
        if "subprocess.run(shell=True)" in content:
            issues.append("SHELL_INJECTION_RISK")
        return issues

    def _check_docs(self, content: str) -> float:
        """Percentage of functions with docstrings."""
        try:
            tree = ast.parse(content)
            functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if not functions: return 1.0
            with_docs = [f for f in functions if ast.get_docstring(f)]
            return len(with_docs) / len(functions)
        except:
            return 0.0

def create_analyzer(project_root: Path):
    return CodeAnalyzer(project_root)
