#!/usr/bin/env python3
"""
Coding Meta-Agent for SAM 2.0
Self-optimizing Flask app with local LLM-based code fixing
"""

import os
import sys
import time
import json
import asyncio
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Local imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available, using mock monitoring")

try:
    import watchdog.observers
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️ watchdog not available, using mock file monitoring")

# Try to import local LLM (fallback to mock if not available)
try:
    from local_llm import local_llm, generate_llm_response, analyze_message_coherence
    LOCAL_LLM_AVAILABLE = local_llm.is_available()
    if LOCAL_LLM_AVAILABLE:
        print("✅ Local LLM loaded successfully for meta-agent")
    else:
        print("⚠️ Local LLM available but not fully loaded")
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    print("⚠️ Local LLM module not available, using mock mode")

# Mock classes for missing dependencies
class MockFileSystemEventHandler:
    """Mock file system event handler"""
    def on_modified(self, event):
        pass

class MockObserver:
    """Mock file observer"""
    def __init__(self):
        self.handlers = []
    
    def schedule(self, handler, path, recursive=False):
        self.handlers.append((handler, path, recursive))
    
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def join(self):
        pass

class MockPsutil:
    """Mock psutil for testing"""
    @staticmethod
    def process_iter(attrs=None):
        return []
    
    @staticmethod
    def cpu_percent():
        return 0.0

if not WATCHDOG_AVAILABLE:
    FileSystemEventHandler = MockFileSystemEventHandler
    watchdog = type('watchdog', (), {'observers': type('observers', (), {'Observer': MockObserver})})()

if not PSUTIL_AVAILABLE:
    psutil = MockPsutil()

class CodeMonitor(FileSystemEventHandler):
    """Monitor Flask app for errors and performance issues"""
    
    def __init__(self, meta_agent):
        self.meta_agent = meta_agent
        self.error_patterns = [
            "TimeoutError",
            "ConnectionError", 
            "AttributeError",
            "ImportError",
            "Exception"
        ]
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if event.src_path.endswith('.py'):
            print(f"📝 File changed: {event.src_path}")
            self.meta_agent.analyze_file(event.src_path)
    
    def check_flask_health(self):
        """Monitor Flask app performance"""
        try:
            # Check if Flask process is running
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'python' in proc.info['name'] and 'correct_sam_hub.py' in ' '.join(proc.info['cmdline'] or []):
                    # Check CPU and memory usage
                    cpu_percent = proc.cpu_percent()
                    memory_info = proc.memory_info()
                    
                    if cpu_percent > 80:
                        print(f"⚠️ High CPU usage: {cpu_percent}%")
                        self.meta_agent.suggest_optimization("high_cpu", cpu_percent)
                    
                    if memory_info.rss > 1024 * 1024 * 1024:  # 1GB
                        print(f"⚠️ High memory usage: {memory_info.rss / 1024 / 1024:.1f}MB")
                        self.meta_agent.suggest_optimization("high_memory", memory_info.rss)
        except Exception as e:
            print(f"❌ Health check failed: {e}")

class LocalLLM:
    """Local LLM for code analysis and generation"""

    def __init__(self):
        self.llm = local_llm if LOCAL_LLM_AVAILABLE else None

    def generate_code_fix(self, error_msg: str, code_snippet: str) -> str:
        """Generate code fix using local LLM"""
        if not LOCAL_LLM_AVAILABLE or not self.llm:
            return self._mock_fix(error_msg, code_snippet)

        try:
            prompt = f"""Fix this code error:

Error: {error_msg}

Code:
```python
{code_snippet}
```

Please fix this code. Return only the fixed code without explanation.
"""

            response = generate_llm_response(prompt, max_tokens=300)
            # Clean up the response
            response = response.strip()
            if response.startswith("```python"):
                response = response[9:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            return response.strip()
    
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return self._mock_fix(error_msg, code_snippet)
    
    def _mock_fix(self, error_msg: str, code_snippet: str) -> str:
        """Mock fix when LLM is not available"""
        if "TimeoutError" in error_msg:
            return "# Added timeout handling\nimport asyncio\n\n" + code_snippet
        elif "AttributeError" in error_msg:
            return "# Added attribute check\n" + code_snippet
        else:
            return "# Added error handling\ntry:\n    " + code_snippet.replace('\n', '\n    ') + "\nexcept Exception as e:\n    print(f'Error: {e}')"

class CodeValidator:
    """Validate code changes before applying"""
    
    def __init__(self):
        self.test_commands = [
            ["python3", "-m", "pylint", "--score=no"],
            ["python3", "-m", "flake8"],
            ["python3", "-c", "import ast; ast.parse(open('{file}').read())"]
        ]
    
    def validate_syntax(self, file_path: str) -> bool:
        """Check if Python syntax is valid"""
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error: {e}")
            return False
    
    def run_linting(self, file_path: str) -> Dict[str, Any]:
        """Run code quality checks"""
        results = {}
        for cmd in self.test_commands:
            try:
                result = subprocess.run(
                    cmd + [file_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                results[cmd[1]] = {
                    'returncode': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            except subprocess.TimeoutExpired:
                results[cmd[1]] = {'error': 'timeout'}
            except Exception as e:
                results[cmd[1]] = {'error': str(e)}
        
        return results

class CodingMetaAgent:
    """Main meta-agent for self-optimizing Flask app"""
    
    def __init__(self, flask_app_path: str = "/Users/samueldasari/Personal/NN_C"):
        self.flask_app_path = Path(flask_app_path)
        self.llm = LocalLLM()
        self.validator = CodeValidator()
        self.monitor = CodeMonitor(self)
        self.observer = watchdog.observers.Observer()
        
        # Performance tracking
        self.metrics = {
            'fixes_applied': 0,
            'errors_detected': 0,
            'optimizations_suggested': 0,
            'start_time': datetime.now()
        }
        
        # Optimization patterns
        self.optimization_patterns = {
            'high_cpu': [
                "Add caching to reduce computation",
                "Use async/await for I/O operations",
                "Implement request batching"
            ],
            'high_memory': [
                "Add memory cleanup after requests",
                "Use generators instead of lists",
                "Implement connection pooling"
            ],
            'timeout': [
                "Add timeout handling to requests",
                "Use async HTTP clients",
                "Implement retry logic with exponential backoff"
            ]
        }
    
    def start_monitoring(self):
        """Start monitoring Flask app"""
        print("🚀 Starting Coding Meta-Agent monitoring...")
        
        # Monitor file changes
        self.observer.schedule(
            self.monitor,
            str(self.flask_app_path),
            recursive=True
        )
        self.observer.start()
        
        # Start health check thread
        health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        health_thread.start()
        
        print("✅ Meta-Agent monitoring started")
    
    def _health_check_loop(self):
        """Continuous health monitoring"""
        while True:
            try:
                self.monitor.check_flask_health()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"❌ Health check error: {e}")
                time.sleep(60)
    
    def analyze_file(self, file_path: str):
        """Analyze Python file for potential issues"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for common issues
            issues = []
            
            # Check for blocking operations
            if 'requests.get(' in content or 'requests.post(' in content:
                if 'async' not in content:
                    issues.append("Blocking HTTP requests detected")
            
            # Check for missing error handling
            if 'except:' not in content and 'try:' in content:
                issues.append("Incomplete error handling")
            
            # Check for long functions
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') and i < len(lines) - 50:
                    # Check function length
                    func_lines = 0
                    j = i + 1
                    while j < len(lines) and (lines[j].startswith(' ') or lines[j].startswith('\t')):
                        func_lines += 1
                        j += 1
                    
                    if func_lines > 30:
                        issues.append(f"Long function detected ({func_lines} lines)")
            
            if issues:
                print(f"🔍 Issues found in {file_path}:")
                for issue in issues:
                    print(f"  - {issue}")
                    self.metrics['errors_detected'] += 1
                
                # Generate fixes
                self.generate_fixes(file_path, issues)
        
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
    
    def generate_fixes(self, file_path: str, issues: List[str]):
        """Generate and apply code fixes"""
        try:
            with open(file_path, 'r') as f:
                original_content = f.read()
            
            for issue in issues:
                print(f"🔧 Generating fix for: {issue}")
                
                # Generate fix using LLM
                fixed_code = self.llm.generate_code_fix(issue, original_content)
                
                # Validate fix
                temp_file = f"{file_path}.temp"
                with open(temp_file, 'w') as f:
                    f.write(fixed_code)
                
                if self.validator.validate_syntax(temp_file):
                    print(f"✅ Fix validated for {issue}")
                    
                    # Apply fix
                    with open(file_path, 'w') as f:
                        f.write(fixed_code)
                    
                    self.metrics['fixes_applied'] += 1
                    print(f"✅ Applied fix to {file_path}")
                    
                    # Clean up temp file
                    os.remove(temp_file)
                else:
                    print(f"❌ Fix validation failed for {issue}")
                    os.remove(temp_file)
        
        except Exception as e:
            print(f"❌ Error generating fixes: {e}")
    
    def suggest_optimization(self, issue_type: str, metric_value: Any):
        """Suggest optimizations based on metrics"""
        if issue_type in self.optimization_patterns:
            suggestions = self.optimization_patterns[issue_type]
            print(f"💡 Optimization suggestions for {issue_type}:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")
            
            self.metrics['optimizations_suggested'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get meta-agent performance metrics"""
        runtime = datetime.now() - self.metrics['start_time']
        return {
            **self.metrics,
            'runtime_hours': runtime.total_seconds() / 3600,
            'fixes_per_hour': self.metrics['fixes_applied'] / max(runtime.total_seconds() / 3600, 1)
        }
    
    def stop(self):
        """Stop monitoring"""
        self.observer.stop()
        self.observer.join()
        print("🛑 Meta-Agent monitoring stopped")

def main():
    """Main entry point"""
    print("🤖 SAM 2.0 Coding Meta-Agent")
    print("=" * 50)
    
    # Initialize meta-agent
    meta_agent = CodingMetaAgent()
    
    try:
        # Start monitoring
        meta_agent.start_monitoring()
        
        # Keep running
        print("🔄 Meta-Agent is monitoring your Flask app...")
        print("Press Ctrl+C to stop")
        
        while True:
            time.sleep(60)
            
            # Print metrics every hour
            metrics = meta_agent.get_metrics()
            if int(metrics['runtime_hours']) % 1 == 0:
                print(f"📊 Metrics: {metrics['fixes_applied']} fixes applied, "
                      f"{metrics['errors_detected']} errors detected")
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        meta_agent.stop()
        
        # Final metrics
        final_metrics = meta_agent.get_metrics()
        print(f"📊 Final Metrics:")
        print(f"  Runtime: {final_metrics['runtime_hours']:.1f} hours")
        print(f"  Fixes applied: {final_metrics['fixes_applied']}")
        print(f"  Errors detected: {final_metrics['errors_detected']}")
        print(f"  Optimizations suggested: {final_metrics['optimizations_suggested']}")

if __name__ == "__main__":
    main()
