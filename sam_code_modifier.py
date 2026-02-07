#!/usr/bin/env python3
"""
SAM Codebase Modification System with Safe Environment
Allows the SAM system to safely modify its own codebase for self-improvement
Works in cloned environments to prevent breaking the main codebase
"""

import os
import re
import ast
import ast
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import shutil

class SAMCodeModifier:
    """Safe codebase modification system for SAM self-improvement with isolated testing"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.backup_dir = self.project_root / "SAM_Code_Backups"
        self.backup_dir.mkdir(exist_ok=True)

        # Safe testing environment
        self.safe_env_dir = self.project_root / "SAM_Safe_Test_Env"
        self.safe_env_dir.mkdir(exist_ok=True)

        # Define safe files that can be modified
        self.safe_files = {
            'complete_sam_unified.py',
            'sam_web_search.py',
            'google_drive_integration.py',
            'sam_code_modifier.py',  # Allow self-modification
            'README.md',
            'ARCHITECTURE.md',
            'CHANGELOG.md'
        }

        # Define dangerous patterns to avoid
        self.dangerous_patterns = [
            r'import\s+os\s*$',  # Don't allow os import modifications
            r'os\.system\s*\(',  # Don't allow system calls
            r'os\.popen\s*\(',   # Don't allow subprocess calls
            r'subprocess\.',     # Don't allow subprocess usage
            r'eval\s*\(',        # Don't allow eval
            r'exec\s*\(',        # Don't allow exec
            r'__import__\s*\(',  # Don't allow dynamic imports
            r'open\s*\([^)]*w',  # Careful with file writes
        ]

        print("🛠️ SAM Code Modifier initialized with safe environment")
        print(f"   📁 Project root: {self.project_root}")
        print(f"   �️ Safe test environment: {self.safe_env_dir}")
        print(f"   �💾 Backup directory: {self.backup_dir}")
        print(f"   🔒 Safe files: {len(self.safe_files)} allowed")

    def create_safe_environment(self) -> Optional[str]:
        """Create a safe cloned environment for testing code changes"""
        try:
            # Clean up any existing safe environment
            if self.safe_env_dir.exists():
                shutil.rmtree(self.safe_env_dir)

            self.safe_env_dir.mkdir(exist_ok=True)

            # Clone/copy essential files to safe environment
            essential_files = [
                'complete_sam_unified.py',
                'sam_code_modifier.py',
                'sam_web_search.py',
                'google_drive_integration.py',
                'requirements.txt',
                'run_sam.sh'
            ]

            for file in essential_files:
                src = self.project_root / file
                if src.exists():
                    shutil.copy2(src, self.safe_env_dir / file)

            print(f"🛡️ Safe environment created: {self.safe_env_dir}")
            return str(self.safe_env_dir)

        except Exception as e:
            print(f"❌ Failed to create safe environment: {e}")
            return None

    def test_code_in_safe_environment(self, modified_file: str, modification_desc: str) -> Dict:
        """Test code changes in the safe environment before applying to main codebase"""
        test_results = {
            'success': False,
            'syntax_check': False,
            'import_check': False,
            'basic_functionality': False,
            'message': '',
            'safe_env_path': None
        }

        try:
            # Create safe environment
            safe_env_path = self.create_safe_environment()
            if not safe_env_path:
                test_results['message'] = "Failed to create safe environment"
                return test_results

            test_results['safe_env_path'] = safe_env_path
            safe_env = Path(safe_env_path)

            # Copy the modified file to safe environment
            modified_path = Path(modified_file)
            safe_file = safe_env / modified_path.name
            shutil.copy2(modified_file, safe_file)

            # Test 1: Syntax check
            try:
                with open(safe_file, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                ast.parse(code_content)
                test_results['syntax_check'] = True
                print("  ✅ Syntax check passed")
            except SyntaxError as e:
                test_results['message'] = f"Syntax error in modified code: {e}"
                return test_results
            except Exception as e:
                test_results['message'] = f"Syntax validation failed: {e}"
                return test_results

            # Test 2: Import check
            try:
                # Try to import the modified module
                if modified_path.name.endswith('.py'):
                    module_name = modified_path.name[:-3]  # Remove .py extension

                    # Change to safe environment directory for import
                    old_cwd = os.getcwd()
                    os.chdir(safe_env_path)

                    try:
                        # Clear any cached imports
                        import sys
                        if module_name in sys.modules:
                            del sys.modules[module_name]

                        # Try to import
                        __import__(module_name)
                        test_results['import_check'] = True
                        print("  ✅ Import check passed")
                    except ImportError as e:
                        test_results['message'] = f"Import failed: {e}"
                        return test_results
                    except Exception as e:
                        test_results['message'] = f"Import error: {e}"
                        return test_results
                    finally:
                        os.chdir(old_cwd)

            except Exception as e:
                test_results['message'] = f"Import test failed: {e}"
                return test_results

            # Test 3: Basic functionality check
            try:
                # For Python files, try to instantiate main classes if they exist
                if modified_path.name == 'complete_sam_unified.py':
                    # Test basic class instantiation
                    old_cwd = os.getcwd()
                    os.chdir(safe_env_path)

                    try:
                        # Import and try basic instantiation
                        import complete_sam_unified
                        # Don't actually instantiate as it might start the full system
                        # Just check that the class exists
                        if hasattr(complete_sam_unified, 'UnifiedSAMSystem'):
                            test_results['basic_functionality'] = True
                            print("  ✅ Basic functionality check passed")
                        else:
                            test_results['message'] = "Main class not found in modified file"
                            return test_results
                    finally:
                        os.chdir(old_cwd)

                elif modified_path.name == 'sam_code_modifier.py':
                    # Test code modifier functionality
                    old_cwd = os.getcwd()
                    os.chdir(safe_env_path)

                    try:
                        import sam_code_modifier
                        modifier = sam_code_modifier.SAMCodeModifier()
                        test_results['basic_functionality'] = True
                        print("  ✅ Code modifier functionality check passed")
                    finally:
                        os.chdir(old_cwd)

                else:
                    # For other files, just mark as passed if syntax and import work
                    test_results['basic_functionality'] = True
                    print("  ✅ Basic functionality check passed (syntax + import)")

            except Exception as e:
                test_results['message'] = f"Functionality test failed: {e}"
                return test_results

            # All tests passed
            test_results['success'] = True
            test_results['message'] = f"✅ All safety tests passed for {modification_desc}"

            print(f"🛡️ Safe environment testing completed successfully")
            print(f"   📝 Modification: {modification_desc}")
            print(f"   📁 Safe environment: {safe_env_path}")

        except Exception as e:
            test_results['message'] = f"Safe environment testing failed: {e}"

        return test_results

    def can_modify_file(self, filepath: str) -> bool:
        """Check if a file can be safely modified"""
        filename = Path(filepath).name
        return filename in self.safe_files

    def create_backup(self, filepath: str) -> Optional[str]:
        """Create a backup of the file before modification"""
        try:
            source_path = Path(filepath)
            if not source_path.exists():
                return None

            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source_path.name}.{timestamp}.backup"
            backup_path = self.backup_dir / backup_name

            # Copy file to backup
            import shutil
            shutil.copy2(source_path, backup_path)

            print(f"💾 Backup created: {backup_path}")
            return str(backup_path)

        except Exception as e:
            print(f"⚠️ Failed to create backup: {e}")
            return None

    def validate_code_safety(self, code: str) -> Tuple[bool, str]:
        """Validate that code modifications are safe"""
        try:
            # Parse the code to check syntax
            ast.parse(code)

            # Check for dangerous patterns
            for pattern in self.dangerous_patterns:
                if re.search(pattern, code, re.MULTILINE):
                    return False, f"Dangerous pattern detected: {pattern}"

            return True, "Code appears safe"

        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"

    def apply_code_change(self, filepath: str, old_code: str, new_code: str,
                         description: str = "") -> Dict:
        """Apply a code change with safety checks and isolated testing"""
        result = {
            'success': False,
            'message': '',
            'backup_path': None,
            'change_description': description,
            'safe_test_passed': False,
            'safe_env_path': None
        }

        try:
            # Check if file can be modified
            if not self.can_modify_file(filepath):
                result['message'] = f"File '{filepath}' is not in the safe modification list"
                return result

            # Validate new code safety
            is_safe, safety_message = self.validate_code_safety(new_code)
            if not is_safe:
                result['message'] = f"Unsafe code detected: {safety_message}"
                return result

            # FIRST: Test the change in safe environment
            print(f"🛡️ Testing code change in safe environment...")
            print(f"   📝 Change: {description}")
            
            # Create a temporary version of the file with the change for testing
            temp_file = self.create_modified_file_for_testing(filepath, old_code, new_code)
            if not temp_file:
                result['message'] = "Failed to create temporary file for testing"
                return temp_file

            # Test in safe environment
            test_result = self.test_code_in_safe_environment(temp_file, description)
            result['safe_test_passed'] = test_result['success']
            result['safe_env_path'] = test_result['safe_env_path']

            if not test_result['success']:
                result['message'] = f"Safe environment testing failed: {test_result['message']}"
                print(f"❌ Safe environment testing failed: {test_result['message']}")
                return result

            print(f"✅ Safe environment testing passed - proceeding with main codebase modification")

            # SECOND: Apply to main codebase only after safe testing passes
            # Create backup
            backup_path = self.create_backup(filepath)
            if not backup_path:
                result['message'] = "Failed to create backup"
                return result

            # Read current file content
            with open(filepath, 'r', encoding='utf-8') as f:
                current_content = f.read()

            # Verify old_code exists in current content
            if old_code not in current_content:
                result['message'] = "Old code not found in file (may have changed)"
                return result

            # Apply the change
            new_content = current_content.replace(old_code, new_code, 1)

            # Validate the entire new file one more time
            is_safe, safety_message = self.validate_code_safety(new_content)
            if not is_safe:
                result['message'] = f"Modified file would be unsafe: {safety_message}"
                return result

            # Write the modified content to main codebase
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)

            result['success'] = True
            result['message'] = f"Successfully modified {filepath} after safe environment validation"
            result['backup_path'] = backup_path
            result['lines_changed'] = len(new_code.split('\n')) - len(old_code.split('\n'))

            print(f"✅ Code modification applied to main codebase: {filepath}")
            print(f"   📝 Description: {description}")
            print(f"   💾 Backup: {backup_path}")
            print(f"   🛡️ Safe environment validated: {result['safe_env_path']}")

        except Exception as e:
            result['message'] = f"Modification failed: {str(e)}"

        return result

    def create_modified_file_for_testing(self, filepath: str, old_code: str, new_code: str) -> Optional[str]:
        """Create a temporary modified file for safe environment testing"""
        try:
            # Read current file
            with open(filepath, 'r', encoding='utf-8') as f:
                current_content = f.read()

            # Apply the change
            new_content = current_content.replace(old_code, new_code, 1)

            # Write to temporary file
            temp_file = self.safe_env_dir / f"temp_modified_{Path(filepath).name}"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return str(temp_file)

        except Exception as e:
            print(f"❌ Failed to create temporary modified file: {e}")
            return None

    def analyze_code_for_improvements(self, filepath: str) -> Dict:
        """Analyze code for potential improvements"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            improvements = {
                'file': filepath,
                'suggestions': [],
                'metrics': {}
            }

            # Basic code analysis
            lines = content.split('\n')
            improvements['metrics']['total_lines'] = len(lines)
            improvements['metrics']['code_lines'] = len([l for l in lines if l.strip() and not l.strip().startswith('#')])

            # Look for common improvement patterns
            if 'print(' in content and 'logger' not in content:
                improvements['suggestions'].append({
                    'type': 'logging',
                    'description': 'Consider using proper logging instead of print statements',
                    'priority': 'medium'
                })

            if 'TODO' in content or 'FIXME' in content:
                improvements['suggestions'].append({
                    'type': 'todos',
                    'description': 'Found TODO/FIXME comments that could be addressed',
                    'priority': 'low'
                })

            if len(content) > 50000:  # Large files
                improvements['suggestions'].append({
                    'type': 'refactoring',
                    'description': 'File is quite large, consider splitting into modules',
                    'priority': 'low'
                })

            return improvements

        except Exception as e:
            return {'error': str(e)}

    def suggest_self_improvements(self) -> List[Dict]:
        """Suggest self-improvements for the SAM system"""
        suggestions = []

        # Check each safe file for improvements
        for filename in self.safe_files:
            filepath = self.project_root / filename
            if filepath.exists():
                analysis = self.analyze_code_for_improvements(str(filepath))
                if 'suggestions' in analysis:
                    for suggestion in analysis['suggestions']:
                        suggestion['file'] = filename
                        suggestions.append(suggestion)

        return suggestions

    def get_modification_history(self) -> List[Dict]:
        """Get history of code modifications"""
        history = []

        try:
            backup_files = list(self.backup_dir.glob("*.backup"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for backup_file in backup_files[:20]:  # Last 20 modifications
                history.append({
                    'file': backup_file.name.replace('.backup', '').rsplit('.', 2)[0],
                    'backup_path': str(backup_file),
                    'timestamp': datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                    'size': backup_file.stat().st_size
                })

        except Exception as e:
            print(f"⚠️ Failed to read modification history: {e}")

        return history

    def rollback_modification(self, backup_path: str) -> Dict:
        """Rollback a modification using a backup file"""
        result = {
            'success': False,
            'message': '',
            'rolled_back_file': None
        }

        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                result['message'] = f"Backup file not found: {backup_path}"
                return result

            # Extract original filename from backup name
            # Format: filename.timestamp.backup -> filename
            backup_name = backup_path.name
            original_name = backup_name.rsplit('.', 2)[0]  # Remove .timestamp.backup

            original_path = self.project_root / original_name
            if not original_path.exists():
                result['message'] = f"Original file not found: {original_path}"
                return result

            # Create backup of current state before rollback
            current_backup = self.create_backup(str(original_path))

            # Restore from backup
            import shutil
            shutil.copy2(backup_path, original_path)

            result['success'] = True
            result['message'] = f"Successfully rolled back {original_name}"
            result['rolled_back_file'] = str(original_path)
            result['current_backup'] = current_backup

            print(f"🔄 Rolled back modification: {original_name}")
            print(f"   💾 Current state backed up to: {current_backup}")

        except Exception as e:
            result['message'] = f"Rollback failed: {str(e)}"

        return result

# Global instance for SAM system integration
sam_code_modifier = None

def initialize_sam_code_modifier(project_root: str = None):
    """Initialize SAM code modification system"""
    global sam_code_modifier
    sam_code_modifier = SAMCodeModifier(project_root)
    return sam_code_modifier

def modify_code_safely(filepath: str, old_code: str, new_code: str, description: str = "") -> Dict:
    """Safely modify SAM codebase"""
    global sam_code_modifier
    if not sam_code_modifier:
        sam_code_modifier = SAMCodeModifier()

    return sam_code_modifier.apply_code_change(filepath, old_code, new_code, description)

def analyze_codebase() -> Dict:
    """Analyze SAM codebase for improvements"""
    global sam_code_modifier
    if not sam_code_modifier:
        sam_code_modifier = SAMCodeModifier()

    return {
        'improvements': sam_code_modifier.suggest_self_improvements(),
        'modification_history': sam_code_modifier.get_modification_history()
    }
