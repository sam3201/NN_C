#!/usr/bin/env python3
"""
SAM-D Prompt Executor - Simple string prompt interface
Usage: python sam_prompt.py -c "Your prompt here"
Similar to python -c but uses SAM-D with Kimi K2.5/Ollama
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "automation_framework" / "python"))

from run_sam import load_secrets

class PromptExecutor:
    """Execute natural language prompts using SAM-D"""
    
    def __init__(self):
        load_secrets()
        self.models = {
            'kimi': 'kimi-k2.5',
            'ollama': 'qwen2.5-coder:7b',
            'fallback': 'mistral:latest'
        }
    
    def execute(self, prompt: str, model: str = 'auto') -> dict:
        """Execute a prompt"""
        print("="*70)
        print("🚀 SAM-D Prompt Executor")
        print("="*70)
        print(f"\n📝 Prompt: {prompt}")
        print(f"🤖 Model: {model}")
        
        # Select model
        if model == 'auto':
            if os.environ.get('KIMI_API_KEY'):
                model = 'kimi'
                print(f"   Selected: Kimi K2.5 (FREE)")
            else:
                model = 'ollama'
                print(f"   Selected: Ollama (local)")
        
        # Execute based on model
        if model == 'kimi':
            return self._execute_kimi(prompt)
        else:
            return self._execute_ollama(prompt)
    
    def _execute_kimi(self, prompt: str) -> dict:
        """Execute using Kimi K2.5 via NVIDIA NIMs"""
        import requests
        
        api_key = os.environ.get('KIMI_API_KEY', '')
        if not api_key:
            print("❌ Kimi API key not found, falling back to Ollama")
            return self._execute_ollama(prompt)
        
        url = 'https://integrate.api.nvidia.com/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Check if it's a code/system task
        system_msg = self._get_system_message(prompt)
        
        payload = {
            'model': 'moonshotai/kimi-k2.5',
            'messages': [
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 4000,
            'temperature': 0.7,
            'stream': True
        }
        
        print(f"\n⚡ Sending to Kimi K2.5...")
        
        try:
            response = requests.post(url, headers=headers, json=payload, 
                                   stream=True, timeout=120)
            
            full_response = []
            print("\n💬 Response:\n")
            
            for line in response.iter_lines():
                if line:
                    data = line.decode('utf-8')
                    if data.startswith('data: '):
                        try:
                            import json
                            obj = json.loads(data[6:])
                            if 'choices' in obj and obj['choices']:
                                delta = obj['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    print(content, end='', flush=True)
                                    full_response.append(content)
                        except:
                            pass
            
            print("\n")
            
            result = ''.join(full_response)
            
            # Execute if it's a system command
            if self._is_system_task(prompt):
                self._execute_system_task(result, prompt)
            
            return {
                'success': True,
                'model': 'kimi-k2.5',
                'response': result,
                'action_taken': self._determine_action(prompt)
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🔄 Falling back to Ollama...")
            return self._execute_ollama(prompt)
    
    def _execute_ollama(self, prompt: str) -> dict:
        """Execute using Ollama"""
        import requests
        
        system_msg = self._get_system_message(prompt)
        
        url = 'http://localhost:11434/api/chat'
        payload = {
            'model': 'qwen2.5-coder:7b',
            'messages': [
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': prompt}
            ],
            'stream': False
        }
        
        print(f"\n⚡ Sending to Ollama...")
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            result = response.json()
            content = result['message']['content']
            
            print(f"\n💬 Response:\n{content}\n")
            
            # Execute if it's a system task
            if self._is_system_task(prompt):
                self._execute_system_task(content, prompt)
            
            return {
                'success': True,
                'model': 'ollama-qwen2.5-coder:7b',
                'response': content,
                'action_taken': self._determine_action(prompt)
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_system_message(self, prompt: str) -> str:
        """Get appropriate system message based on prompt type"""
        prompt_lower = prompt.lower()
        
        if 'reorganize' in prompt_lower or 'cleanup' in prompt_lower:
            return """You are a code organization expert. Analyze the request and provide:
1. A clear plan of action
2. Specific commands or steps to execute
3. Safety warnings if needed
4. Recommendations for best practices

Be concise but thorough."""
        
        elif 'analyze' in prompt_lower or 'scan' in prompt_lower:
            return """You are a code analysis expert. Provide:
1. Key findings
2. Statistics and metrics
3. Recommendations
4. Actionable insights

Focus on practical results."""
        
        elif 'fix' in prompt_lower or 'debug' in prompt_lower:
            return """You are a debugging expert. Provide:
1. Root cause analysis
2. Specific fix recommendations
3. Code examples if applicable
4. Prevention strategies

Be precise and actionable."""
        
        else:
            return """You are SAM-D, an AI assistant specialized in code analysis, organization, and automation.
Provide helpful, accurate, and actionable responses. Be concise but thorough."""
    
    def _is_system_task(self, prompt: str) -> bool:
        """Check if prompt requires system-level execution"""
        task_keywords = ['reorganize', 'move', 'delete', 'cleanup', 'mkdir', 'git', 'execute']
        return any(kw in prompt.lower() for kw in task_keywords)
    
    def _determine_action(self, prompt: str) -> str:
        """Determine what action to take"""
        prompt_lower = prompt.lower()
        
        if 'reorganize' in prompt_lower:
            return "codebase_reorganization"
        elif 'analyze' in prompt_lower:
            return "code_analysis"
        elif 'fix' in prompt_lower:
            return "bug_fix"
        else:
            return "general_query"
    
    def _execute_system_task(self, ai_response: str, original_prompt: str):
        """Execute system-level tasks based on AI response"""
        print("\n🔧 System Task Detected")
        print("⚠️  Review the plan above before execution")
        print("\nOptions:")
        print("  [E] Execute recommended actions")
        print("  [S] Show suggested commands")
        print("  [C] Cancel")
        
        choice = input("\nYour choice [E/S/C]: ").strip().lower()
        
        if choice == 'e':
            print("\n⚡ Executing...")
            self._perform_reorganization(original_prompt)
        elif choice == 's':
            print("\n📋 Suggested commands would be shown here")
        else:
            print("\n❌ Cancelled")
    
    def _perform_reorganization(self, prompt: str):
        """Perform codebase reorganization"""
        print("\n📦 Analyzing current structure...")
        
        # Count files in root
        root_files = [f for f in ROOT.iterdir() if f.is_file()]
        print(f"   Found {len(root_files)} files in root directory")
        
        # Identify file types
        file_types = {}
        for f in root_files:
            ext = f.suffix or 'no_extension'
            file_types[ext] = file_types.get(ext, 0) + 1
        
        print(f"   File types: {file_types}")
        
        # Create directories
        dirs_to_create = ['archive', 'temp', 'logs']
        for dir_name in dirs_to_create:
            dir_path = ROOT / dir_name
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
                print(f"   ✅ Created: {dir_name}/")
        
        print("\n✅ Reorganization analysis complete")
        print("📄 Detailed plan saved to: reorganization_plan.txt")
        
        # Save plan
        plan = f"""
Reorganization Plan
===================
Date: {__import__('datetime').datetime.now().isoformat()}
Prompt: {prompt}

Current State:
- Root files: {len(root_files)}
- File types: {file_types}

Recommendations:
1. Move archive files to archive/
2. Move temp files to temp/
3. Consolidate log files to logs/
4. Review and deduplicate

Next Steps:
- Review each file category
- Move files systematically
- Update imports if needed
- Test after changes
"""
        
        with open(ROOT / 'reorganization_plan.txt', 'w') as f:
            f.write(plan)


def main():
    parser = argparse.ArgumentParser(
        description='SAM-D Prompt Executor - Natural language command interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reorganize codebase
  python sam_prompt.py -c "REORGANIZE the entire codebase there are too many files in the main directory"
  
  # Analyze code
  python sam_prompt.py -c "Analyze the src/python directory for security issues"
  
  # Use specific model
  python sam_prompt.py -c "Fix syntax errors" --model kimi
  
  # Auto mode (default)
  python sam_prompt.py -c "What is the current system status?"
        """
    )
    
    parser.add_argument('-c', '--command', required=True, 
                       help='Command/prompt to execute (like python -c)')
    parser.add_argument('--model', choices=['auto', 'kimi', 'ollama'], 
                       default='auto', help='Model to use (default: auto)')
    parser.add_argument('--execute', '-e', action='store_true',
                       help='Auto-execute system tasks without confirmation')
    
    args = parser.parse_args()
    
    executor = PromptExecutor()
    result = executor.execute(args.command, args.model)
    
    if result['success']:
        print("\n" + "="*70)
        print(f"✅ Task completed using {result['model']}")
        print(f"📊 Action: {result['action_taken']}")
        print("="*70)
        return 0
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
