#!/usr/bin/env python3
"""
SAM-D Experiment Framework
Systematic experimentation to find missing pieces and validate integrations
"""

import os
import sys
import json
import subprocess
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src" / "python"))

class ExperimentStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    PENDING = "pending"

@dataclass
class ExperimentResult:
    name: str
    status: ExperimentStatus
    duration_ms: float
    output: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class ExperimentFramework:
    """Framework for running systematic experiments on SAM-D"""
    
    # Default model for experiments - Kimi K2.5 (best free model)
    DEFAULT_MODEL = "kimi:kimi-k2.5-flash"
    FALLBACK_MODEL = "ollama:qwen2.5-coder:7b"
    
    def __init__(self):
        self.results: List[ExperimentResult] = []
        self.start_time = time.time()
        self.current_model = self._select_model()
    
    def _select_model(self) -> str:
        """Select the best available model for experiments"""
        # Priority: Kimi K2.5 > Ollama > Local
        kimi_key = os.getenv("KIMI_API_KEY", "")
        if kimi_key:
            return self.DEFAULT_MODEL
        
        if self._check_ollama():
            return self.FALLBACK_MODEL
        
        return "local:rules"
    
    def run_experiment(self, name: str, experiment_fn) -> ExperimentResult:
        """Run a single experiment"""
        start = time.time()
        try:
            result = experiment_fn()
            duration_ms = (time.time() - start) * 1000
            status = ExperimentStatus.PASS if result.get("success", False) else ExperimentStatus.FAIL
            return ExperimentResult(
                name=name,
                status=status,
                duration_ms=duration_ms,
                output=result.get("output", ""),
                error=result.get("error", ""),
                metadata=result.get("metadata", {})
            )
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return ExperimentResult(
                name=name,
                status=ExperimentStatus.FAIL,
                duration_ms=duration_ms,
                error=str(e)
            )
    
    def check_c_extensions(self) -> Dict[str, Any]:
        """Verify all C extensions are built and importable"""
        c_extensions = [
            "consciousness_algorithmic",
            "orchestrator_and_agents",
            "sam_regulator_c",
            "sam_meta_controller_c",
            "sav_core_c",
            "sam_sav_dual_system",
            "sam_fast_rng",
            "sam_telemetry_core",
            "sam_god_equation",
            "sam_regulator_compiler_c",
            "sam_consciousness",
            "sam_memory"
        ]
        
        missing = []
        available = []
        
        for ext in c_extensions:
            try:
                __import__(ext)
                available.append(ext)
            except ImportError:
                missing.append(ext)
        
        return {
            "success": len(missing) == 0,
            "output": f"Available: {len(available)}, Missing: {len(missing)}",
            "metadata": {"available": available, "missing": missing}
        }
    
    def check_python_syntax(self) -> Dict[str, Any]:
        """Check Python files for syntax errors"""
        python_files = [
            ROOT_DIR / "src" / "python" / "complete_sam_unified.py",
            ROOT_DIR / "src" / "python" / "sam_cores.py",
            ROOT_DIR / "src" / "python" / "SAM_AGI.py",
        ]
        
        errors = []
        for pf in python_files:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(pf)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                errors.append(f"{pf.name}: {result.stderr[:200]}")
        
        return {
            "success": len(errors) == 0,
            "output": f"Checked {len(python_files)} files",
            "error": "\n".join(errors) if errors else "",
            "metadata": {"files_checked": len(python_files), "errors": len(errors)}
        }
    
    def check_system_imports(self) -> Dict[str, Any]:
        """Check if system can be imported"""
        try:
            os.environ["PYTHONPATH"] = "src/python:."
            result = subprocess.run(
                [sys.executable, "-c", "from complete_sam_unified import UnifiedSAMSystem; print('OK')"],
                cwd=str(ROOT_DIR / "src" / "python"),
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "PYTHONPATH": "src/python:."}
            )
            success = result.returncode == 0 and "OK" in result.stdout
            return {
                "success": success,
                "output": result.stdout[:200] if result.stdout else "",
                "error": result.stderr[:200] if result.stderr else ""
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def check_api_providers(self) -> Dict[str, Any]:
        """Check available API providers"""
        # Default model for experiments
        default_model = os.getenv("SAM_EXPERIMENT_MODEL", self.DEFAULT_MODEL)
        
        providers = {
            "kimi": bool(os.getenv("KIMI_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "ollama": self._check_ollama()
        }
        
        available = [k for k, v in providers.items() if v]
        
        # Determine selected model
        if providers.get("kimi"):
            selected = "kimi:kimi-k2.5-flash"
        elif providers.get("openai"):
            selected = "openai:gpt-4o"
        elif providers.get("anthropic"):
            selected = "anthropic:claude-3-5-sonnet"
        elif providers.get("ollama"):
            selected = self.FALLBACK_MODEL
        else:
            selected = "local:rules"
        
        return {
            "success": len(available) > 0,
            "output": f"Available: {available}, Selected: {selected}",
            "metadata": {**providers, "selected_model": selected, "default_model": self.DEFAULT_MODEL}
        }
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False
    
    def check_fallback_usage(self) -> Dict[str, Any]:
        """Find all fallback patterns in the code"""
        fallback_patterns = [
            ("fallback", "src/python/complete_sam_unified.py"),
            ("_fallback", "src/python/complete_sam_unified.py"),
            ("simulated", "src/python/complete_sam_unified.py"),
            ("stub", "src/python/"),
        ]
        
        findings = []
        for pattern, path in fallback_patterns:
            result = subprocess.run(
                ["grep", "-r", "-n", pattern, path, "--include=*.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                count = len(result.stdout.strip().split("\n"))
                findings.append(f"{pattern}: {count} occurrences")
        
        return {
            "success": True,
            "output": "\n".join(findings) if findings else "No fallbacks found",
            "metadata": {"findings": findings}
        }
    
    def check_security_patterns(self) -> Dict[str, Any]:
        """Check for security issues"""
        issues = []
        
        # Check for unsafe eval/exec usage
        result = subprocess.run(
            ["grep", "-r", "-n", "eval(", "src/python/", "--include=*.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            issues.append(f"unsafe eval usage: {len(result.stdout.strip().split(chr(10)))} occurrences")
        
        # Check for pickle usage
        result = subprocess.run(
            ["grep", "-r", "-n", "pickle.load", "src/python/", "--include=*.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            issues.append(f"pickle.load usage: {len(result.stdout.strip().split(chr(10)))} occurrences")
        
        # Check for shell=True in subprocess
        result = subprocess.run(
            ["grep", "-r", "-n", "shell=True", "src/python/", "--include=*.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            issues.append(f"subprocess shell=True: {len(result.stdout.strip().split(chr(10)))} occurrences")
        
        return {
            "success": len(issues) == 0,
            "output": "No security issues found" if not issues else "\n".join(issues),
            "metadata": {"issues": issues}
        }
    
    def run_all_experiments(self) -> List[ExperimentResult]:
        """Run all experiments"""
        experiments = [
            ("C Extensions Check", self.check_c_extensions),
            ("Python Syntax Check", self.check_python_syntax),
            ("System Import Check", self.check_system_imports),
            ("API Providers Check", self.check_api_providers),
            ("Fallback Patterns Check", self.check_fallback_usage),
            ("Security Patterns Check", self.check_security_patterns),
        ]
        
        for name, fn in experiments:
            result = self.run_experiment(name, fn)
            self.results.append(result)
            print(f"[{'PASS' if result.status == ExperimentStatus.PASS else 'FAIL'}] {name}: {result.output[:80]}")
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate experiment report"""
        passed = sum(1 for r in self.results if r.status == ExperimentStatus.PASS)
        failed = sum(1 for r in self.results if r.status == ExperimentStatus.FAIL)
        total = len(self.results)
        
        report = f"""
SAM-D Experiment Report
=======================
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {(time.time() - self.start_time):.2f}s

Summary: {passed}/{total} passed, {failed}/{total} failed

Details:
"""
        for r in self.results:
            report += f"\n{r.name}: {r.status.value}\n"
            if r.error:
                report += f"  Error: {r.error[:200]}\n"
            if r.output:
                report += f"  Output: {r.output[:200]}\n"
        
        return report

def main():
    """Run experiment framework"""
    framework = ExperimentFramework()
    print("Running SAM-D Experiment Framework...")
    print("=" * 50)
    
    results = framework.run_all_experiments()
    report = framework.generate_report()
    
    print(report)
    
    # Save report
    report_path = ROOT_DIR / "logs" / "experiment_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")
    
    # Return exit code based on results
    failed = sum(1 for r in results if r.status == ExperimentStatus.FAIL)
    return 1 if failed > 0 else 0

if __name__ == "__main__":
    sys.exit(main())
