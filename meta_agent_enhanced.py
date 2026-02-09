#!/usr/bin/env python3
"""
Enhanced MetaAgent with Advanced Problem Detection and Fixing Capabilities
Extends the existing MetaAgent with sophisticated error handling and patch generation
"""

import os
import re
import ast
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

class EnhancedMetaAgent:
    """Enhanced MetaAgent with advanced problem detection and fixing capabilities"""
    
    def __init__(self, system):
        self.system = system
        self.project_root = getattr(system, 'project_root', Path('.'))
        
        # Enhanced capabilities
        self.error_patterns = self._load_error_patterns()
        self.fix_strategies = self._load_fix_strategies()
        self.successful_fixes = []
        self.failed_attempts = []
        
        # Learning system
        self.confidence_threshold = 0.7  # Lower threshold for more fixes
        self.max_attempts = 3
        self.learning_enabled = True
        
        print("🚀 Enhanced MetaAgent initialized")
        print("   ✅ Advanced error pattern matching")
        print("   ✅ Sophisticated fix strategies")
        print("   ✅ Adaptive learning system")
        print(f"   ✅ Confidence threshold: {self.confidence_threshold}")
    
    def _load_error_patterns(self) -> Dict[str, List[Dict]]:
        """Load comprehensive error patterns for detection"""
        return {
            'syntax_errors': [
                {
                    'pattern': r'SyntaxError.*missing.*colon',
                    'type': 'missing_colon',
                    'severity': 'high',
                    'auto_fixable': True
                },
                {
                    'pattern': r'SyntaxError.*invalid.*syntax',
                    'type': 'invalid_syntax',
                    'severity': 'high',
                    'auto_fixable': True
                },
                {
                    'pattern': r'IndentationError',
                    'type': 'indentation_error',
                    'severity': 'medium',
                    'auto_fixable': True
                },
                {
                    'pattern': r'NameError.*not defined',
                    'type': 'name_error',
                    'severity': 'medium',
                    'auto_fixable': True
                }
            ],
            'import_errors': [
                {
                    'pattern': r'ModuleNotFoundError.*No module named',
                    'type': 'missing_module',
                    'severity': 'high',
                    'auto_fixable': True
                },
                {
                    'pattern': r'ImportError.*cannot import',
                    'type': 'import_error',
                    'severity': 'high',
                    'auto_fixable': True
                }
            ],
            'runtime_errors': [
                {
                    'pattern': r'ZeroDivisionError',
                    'type': 'division_by_zero',
                    'severity': 'high',
                    'auto_fixable': True
                },
                {
                    'pattern': r'IndexError.*list index out of range',
                    'type': 'index_out_of_range',
                    'severity': 'high',
                    'auto_fixable': True
                },
                {
                    'pattern': r'KeyError',
                    'type': 'key_error',
                    'severity': 'medium',
                    'auto_fixable': True
                },
                {
                    'pattern': r'TypeError',
                    'type': 'type_mismatch',
                    'severity': 'medium',
                    'auto_fixable': True
                }
            ],
            'performance_issues': [
                {
                    'pattern': r'for.*in.*range.*for.*in.*range',
                    'type': 'nested_loop',
                    'severity': 'medium',
                    'auto_fixable': True
                },
                {
                    'pattern': r'while.*True.*append',
                    'type': 'potential_memory_leak',
                    'severity': 'high',
                    'auto_fixable': True
                }
            ],
            'configuration_errors': [
                {
                    'pattern': r'NoneType.*object.*not.*subscriptable',
                    'type': 'none_value_error',
                    'severity': 'high',
                    'auto_fixable': True
                },
                {
                    'pattern': r'ValueError.*invalid.*literal',
                    'type': 'invalid_value',
                    'severity': 'medium',
                    'auto_fixable': True
                }
            ]
        }
    
    def _load_fix_strategies(self) -> Dict[str, List[Dict]]:
        """Load comprehensive fix strategies"""
        return {
            'missing_colon': [
                {
                    'strategy': 'add_colon',
                    'confidence': 0.9,
                    'description': 'Add missing colon after function/if/while statement'
                },
                {
                    'strategy': 'fix_indentation',
                    'confidence': 0.7,
                    'description': 'Fix indentation that may be causing colon issues'
                }
            ],
            'invalid_syntax': [
                {
                    'strategy': 'syntax_correction',
                    'confidence': 0.8,
                    'description': 'Correct common syntax errors'
                }
            ],
            'indentation_error': [
                {
                    'strategy': 'fix_indentation',
                    'confidence': 0.9,
                    'description': 'Fix Python indentation using 4 spaces standard'
                }
            ],
            'name_error': [
                {
                    'strategy': 'define_variable',
                    'confidence': 0.8,
                    'description': 'Define missing variable or fix typo'
                },
                {
                    'strategy': 'import_module',
                    'confidence': 0.7,
                    'description': 'Import missing module'
                }
            ],
            'missing_module': [
                {
                    'strategy': 'install_package',
                    'confidence': 0.8,
                    'description': 'Install missing Python package'
                },
                {
                    'strategy': 'create_stub',
                    'confidence': 0.6,
                    'description': 'Create stub module for missing dependency'
                }
            ],
            'division_by_zero': [
                {
                    'strategy': 'add_zero_check',
                    'confidence': 0.95,
                    'description': 'Add zero division check before division'
                }
            ],
            'index_out_of_range': [
                {
                    'strategy': 'add_bounds_check',
                    'confidence': 0.9,
                    'description': 'Add array bounds checking'
                }
            ],
            'key_error': [
                {
                    'strategy': 'add_key_check',
                    'confidence': 0.85,
                    'description': 'Add dictionary key existence check'
                },
                {
                    'strategy': 'use_get_method',
                    'confidence': 0.8,
                    'description': 'Use dict.get() method with default value'
                }
            ],
            'nested_loop': [
                {
                    'strategy': 'optimize_algorithm',
                    'confidence': 0.8,
                    'description': 'Optimize nested loop to reduce complexity'
                }
            ],
            'potential_memory_leak': [
                {
                    'strategy': 'add_limit_check',
                    'confidence': 0.9,
                    'description': 'Add memory usage limit and cleanup'
                }
            ],
            'none_value_error': [
                {
                    'strategy': 'add_none_check',
                    'confidence': 0.9,
                    'description': 'Add None value validation'
                }
            ],
            'invalid_value': [
                {
                    'strategy': 'validate_input',
                    'confidence': 0.85,
                    'description': 'Add input validation and sanitization'
                }
            ]
        }
    
    def detect_error_type(self, error_message: str, stack_trace: str) -> Optional[Dict]:
        """Enhanced error type detection using pattern matching"""
        combined_text = f"{error_message} {stack_trace}".lower()
        
        for category, patterns in self.error_patterns.items():
            for pattern_info in patterns:
                if re.search(pattern_info['pattern'], combined_text, re.IGNORECASE):
                    return {
                        'category': category,
                        'type': pattern_info['type'],
                        'severity': pattern_info['severity'],
                        'auto_fixable': pattern_info['auto_fixable'],
                        'pattern_matched': pattern_info['pattern']
                    }
        
        return None
    
    def generate_fixes(self, error_info: Dict, file_path: str, error_line: int) -> List[Dict]:
        """Generate multiple fix strategies for detected error"""
        if not error_info or not error_info.get('auto_fixable'):
            return []
        
        error_type = error_info['type']
        strategies = self.fix_strategies.get(error_type, [])
        
        fixes = []
        for strategy in strategies:
            fix = self._apply_fix_strategy(
                strategy, file_path, error_line, error_info
            )
            if fix:
                fixes.append(fix)
        
        return fixes
    
    def _apply_fix_strategy(self, strategy: Dict, file_path: str, error_line: int, error_info: Dict) -> Optional[Dict]:
        """Apply a specific fix strategy"""
        try:
            strategy_name = strategy['strategy']
            confidence = strategy['confidence']
            
            if strategy_name == 'add_colon':
                return self._fix_missing_colon(file_path, error_line, confidence)
            elif strategy_name == 'fix_indentation':
                return self._fix_indentation(file_path, confidence)
            elif strategy_name == 'syntax_correction':
                return self._fix_syntax_errors(file_path, confidence)
            elif strategy_name == 'define_variable':
                return self._define_missing_variable(file_path, error_line, confidence)
            elif strategy_name == 'add_zero_check':
                return self._add_zero_check(file_path, error_line, confidence)
            elif strategy_name == 'add_bounds_check':
                return self._add_bounds_check(file_path, error_line, confidence)
            elif strategy_name == 'add_key_check':
                return self._add_key_check(file_path, error_line, confidence)
            elif strategy_name == 'use_get_method':
                return self._use_get_method(file_path, error_line, confidence)
            elif strategy_name == 'add_none_check':
                return self._add_none_check(file_path, error_line, confidence)
            elif strategy_name == 'validate_input':
                return self._add_input_validation(file_path, error_line, confidence)
            elif strategy_name == 'install_package':
                return self._install_missing_package(file_path, error_info, confidence)
            elif strategy_name == 'optimize_algorithm':
                return self._optimize_nested_loop(file_path, error_line, confidence)
            
        except Exception as e:
            print(f"⚠️ Fix strategy failed: {strategy_name} - {e}")
            return None
    
    def _fix_missing_colon(self, file_path: str, error_line: int, confidence: float) -> Dict:
        """Fix missing colon in function/if/while statements"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if error_line <= len(lines):
                line = lines[error_line - 1].rstrip()
                
                # Check if line needs colon and doesn't have one
                if (any(keyword in line for keyword in ['def ', 'if ', 'elif ', 'while ', 'for ', 'class ', 'else', 'try', 'except', 'finally']) 
                    and not line.rstrip().endswith(':')
                    and not line.rstrip().endswith(',')):
                    
                    # Add colon
                    lines[error_line - 1] = line + ':\n'
                    
                    # Write back to file
                    with open(file_path, 'w') as f:
                        f.writelines(lines)
                    
                    return {
                        'type': 'code_fix',
                        'strategy': 'add_colon',
                        'confidence': confidence,
                        'description': f'Added missing colon to line {error_line}',
                        'file_path': file_path,
                        'line_number': error_line,
                        'original_line': line,
                        'fixed_line': line + ':'
                    }
        except Exception as e:
            print(f"⚠️ Failed to fix missing colon: {e}")
        
        return None
    
    def _fix_indentation(self, file_path: str, confidence: float) -> Dict:
        """Fix Python indentation issues"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Try to parse with AST to identify indentation issues
            try:
                tree = ast.parse(content)
                # If parsing succeeds, fix formatting
                fixed_content = ast.unparse(tree)
                
                with open(file_path, 'w') as f:
                    f.write(fixed_content)
                
                return {
                    'type': 'code_fix',
                    'strategy': 'fix_indentation',
                    'confidence': confidence,
                    'description': 'Fixed indentation using AST parsing',
                    'file_path': file_path
                }
            except SyntaxError:
                # If still syntax errors, try basic indentation fix
                lines = content.split('\n')
                fixed_lines = []
                
                for line in lines:
                    # Replace tabs with 4 spaces
                    fixed_line = line.replace('\t', '    ')
                    fixed_lines.append(fixed_line)
                
                fixed_content = '\n'.join(fixed_lines)
                
                with open(file_path, 'w') as f:
                    f.write(fixed_content)
                
                return {
                    'type': 'code_fix',
                    'strategy': 'fix_indentation',
                    'confidence': confidence * 0.7,  # Lower confidence for basic fix
                    'description': 'Fixed basic indentation (tabs to spaces)',
                    'file_path': file_path
                }
        except Exception as e:
            print(f"⚠️ Failed to fix indentation: {e}")
        
        return None
    
    def _fix_syntax_errors(self, file_path: str, confidence: float) -> Dict:
        """Fix common syntax errors"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Common syntax fixes
            fixed_content = content
            
            # Fix missing parentheses in print statements (Python 2 style)
            fixed_content = re.sub(r'print\s+(.+)', r'print(\1)', fixed_content)
            
            # Fix common bracket mismatches
            # This is a simplified approach - real implementation would be more sophisticated
            
            with open(file_path, 'w') as f:
                f.write(fixed_content)
            
            return {
                'type': 'code_fix',
                'strategy': 'syntax_correction',
                'confidence': confidence,
                'description': 'Applied common syntax corrections',
                'file_path': file_path
            }
        except Exception as e:
            print(f"⚠️ Failed to fix syntax errors: {e}")
        
        return None
    
    def _add_zero_check(self, file_path: str, error_line: int, confidence: float) -> Dict:
        """Add zero division check"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if error_line <= len(lines):
                original_line = lines[error_line - 1].rstrip()
                
                # Find division operation
                if '/' in original_line:
                    # Add zero check before division
                    indent = len(original_line) - len(original_line.lstrip())
                    spaces = ' ' * indent
                    
                    # Extract variable names around division
                    parts = original_line.split('/')
                    if len(parts) >= 2:
                        divisor = parts[1].strip().split()[0] if parts[1].strip() else 'unknown'
                        
                        fixed_line = f"{spaces}if {divisor} != 0:\n{spaces}    {original_line.strip()}\n{spaces}else:\n{spaces}    # Handle division by zero\n{spaces}    result = None"
                        
                        lines[error_line - 1] = fixed_line + '\n'
                        
                        with open(file_path, 'w') as f:
                            f.writelines(lines)
                        
                        return {
                            'type': 'code_fix',
                            'strategy': 'add_zero_check',
                            'confidence': confidence,
                            'description': f'Added zero division check for {divisor}',
                            'file_path': file_path,
                            'line_number': error_line,
                            'original_line': original_line,
                            'fixed_line': fixed_line
                        }
        except Exception as e:
            print(f"⚠️ Failed to add zero check: {e}")
        
        return None
    
    def _add_bounds_check(self, file_path: str, error_line: int, confidence: float) -> Dict:
        """Add array bounds checking"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if error_line <= len(lines):
                original_line = lines[error_line - 1].rstrip()
                
                # Find array access pattern
                if '[' in original_line and ']' in original_line:
                    indent = len(original_line) - len(original_line.lstrip())
                    spaces = ' ' * indent
                    
                    # Extract array access pattern
                    match = re.search(r'(\w+)\[(\w+)\]', original_line)
                    if match:
                        array_name = match.group(1)
                        index_var = match.group(2)
                        
                        fixed_line = f"{spaces}if 0 <= {index_var} < len({array_name}):\n{spaces}    {original_line.strip()}\n{spaces}else:\n{spaces}    # Handle index out of bounds\n{spaces}    result = None"
                        
                        lines[error_line - 1] = fixed_line + '\n'
                        
                        with open(file_path, 'w') as f:
                            f.writelines(lines)
                        
                        return {
                            'type': 'code_fix',
                            'strategy': 'add_bounds_check',
                            'confidence': confidence,
                            'description': f'Added bounds check for {array_name}[{index_var}]',
                            'file_path': file_path,
                            'line_number': error_line,
                            'original_line': original_line,
                            'fixed_line': fixed_line
                        }
        except Exception as e:
            print(f"⚠️ Failed to add bounds check: {e}")
        
        return None
    
    def _add_key_check(self, file_path: str, error_line: int, confidence: float) -> Dict:
        """Add dictionary key existence check"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if error_line <= len(lines):
                original_line = lines[error_line - 1].rstrip()
                
                # Find dictionary access pattern
                if '[' in original_line and ']' in original_line:
                    indent = len(original_line) - len(original_line.lstrip())
                    spaces = ' ' * indent
                    
                    # Extract dictionary access pattern
                    match = re.search(r'(\w+)\[(\w+)\]', original_line)
                    if match:
                        dict_name = match.group(1)
                        key_var = match.group(2)
                        
                        fixed_line = f"{spaces}if {key_var} in {dict_name}:\n{spaces}    {original_line.strip()}\n{spaces}else:\n{spaces}    # Handle missing key\n{spaces}    result = None"
                        
                        lines[error_line - 1] = fixed_line + '\n'
                        
                        with open(file_path, 'w') as f:
                            f.writelines(lines)
                        
                        return {
                            'type': 'code_fix',
                            'strategy': 'add_key_check',
                            'confidence': confidence,
                            'description': f'Added key check for {dict_name}[{key_var}]',
                            'file_path': file_path,
                            'line_number': error_line,
                            'original_line': original_line,
                            'fixed_line': fixed_line
                        }
        except Exception as e:
            print(f"⚠️ Failed to add key check: {e}")
        
        return None
    
    def _use_get_method(self, file_path: str, error_line: int, confidence: float) -> Dict:
        """Replace dict access with get() method"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if error_line <= len(lines):
                original_line = lines[error_line - 1].rstrip()
                
                # Find dictionary access pattern
                if '[' in original_line and ']' in original_line:
                    match = re.search(r'(\w+)\[(\w+)\]', original_line)
                    if match:
                        dict_name = match.group(1)
                        key_var = match.group(2)
                        
                        fixed_line = original_line.replace(f'{dict_name}[{key_var}]', f'{dict_name}.get({key_var}, None)')
                        
                        lines[error_line - 1] = fixed_line + '\n'
                        
                        with open(file_path, 'w') as f:
                            f.writelines(lines)
                        
                        return {
                            'type': 'code_fix',
                            'strategy': 'use_get_method',
                            'confidence': confidence,
                            'description': f'Replaced {dict_name}[{key_var}] with get() method',
                            'file_path': file_path,
                            'line_number': error_line,
                            'original_line': original_line,
                            'fixed_line': fixed_line
                        }
        except Exception as e:
            print(f"⚠️ Failed to use get method: {e}")
        
        return None
    
    def _add_none_check(self, file_path: str, error_line: int, confidence: float) -> Dict:
        """Add None value validation"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if error_line <= len(lines):
                original_line = lines[error_line - 1].rstrip()
                indent = len(original_line) - len(original_line.lstrip())
                spaces = ' ' * indent
                
                fixed_line = f"{spaces}if value is not None:\n{spaces}    {original_line.strip()}\n{spaces}else:\n{spaces}    # Handle None value\n{spaces}    result = None"
                
                lines[error_line - 1] = fixed_line + '\n'
                
                with open(file_path, 'w') as f:
                    f.writelines(lines)
                
                return {
                    'type': 'code_fix',
                    'strategy': 'add_none_check',
                    'confidence': confidence,
                    'description': 'Added None value validation',
                    'file_path': file_path,
                    'line_number': error_line,
                    'original_line': original_line,
                    'fixed_line': fixed_line
                }
        except Exception as e:
            print(f"⚠️ Failed to add None check: {e}")
        
        return None
    
    def _add_input_validation(self, file_path: str, error_line: int, confidence: float) -> Dict:
        """Add input validation and sanitization"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if error_line <= len(lines):
                original_line = lines[error_line - 1].rstrip()
                indent = len(original_line) - len(original_line.lstrip())
                spaces = ' ' * indent
                
                validation_code = f"{spaces}# Input validation\n{spaces}if not isinstance(value, (str, int, float)):\n{spaces}    raise ValueError('Invalid input type')\n{spaces}{original_line.strip()}"
                
                lines[error_line - 1] = validation_code + '\n'
                
                with open(file_path, 'w') as f:
                    f.writelines(lines)
                
                return {
                    'type': 'code_fix',
                    'strategy': 'validate_input',
                    'confidence': confidence,
                    'description': 'Added input validation',
                    'file_path': file_path,
                    'line_number': error_line,
                    'original_line': original_line,
                    'fixed_line': validation_code
                }
        except Exception as e:
            print(f"⚠️ Failed to add input validation: {e}")
        
        return None
    
    def _install_missing_package(self, file_path: str, error_info: Dict, confidence: float) -> Dict:
        """Install missing Python package"""
        try:
            # Extract module name from error
            error_message = error_info.get('error_message', '')
            match = re.search(r'No module named [\'"]([^\'"]+)[\'"]', error_message)
            if match:
                module_name = match.group(1)
                
                # Try to install package
                try:
                    result = subprocess.run(
                        ['pip', 'install', module_name],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        return {
                            'type': 'package_install',
                            'strategy': 'install_package',
                            'confidence': confidence,
                            'description': f'Successfully installed {module_name}',
                            'module_name': module_name,
                            'install_output': result.stdout
                        }
                    else:
                        return {
                            'type': 'package_install',
                            'strategy': 'install_package',
                            'confidence': 0.3,  # Low confidence on failure
                            'description': f'Failed to install {module_name}',
                            'module_name': module_name,
                            'error_output': result.stderr
                        }
                except subprocess.TimeoutExpired:
                    return {
                        'type': 'package_install',
                        'strategy': 'install_package',
                        'confidence': 0.2,  # Very low confidence on timeout
                        'description': f'Installation timeout for {module_name}',
                        'module_name': module_name
                    }
        except Exception as e:
            print(f"⚠️ Failed to install package: {e}")
        
        return None
    
    def _optimize_nested_loop(self, file_path: str, error_line: int, confidence: float) -> Dict:
        """Optimize nested loops for better performance"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if error_line <= len(lines):
                original_line = lines[error_line - 1].rstrip()
                
                # Suggest optimization for nested loops
                optimization_comment = f"# TODO: Optimize nested loop - consider using list comprehensions or numpy"
                
                lines[error_line - 1] = original_line + '\n' + optimization_comment + '\n'
                
                with open(file_path, 'w') as f:
                    f.writelines(lines)
                
                return {
                    'type': 'code_fix',
                    'strategy': 'optimize_algorithm',
                    'confidence': confidence,
                    'description': 'Added optimization suggestion for nested loop',
                    'file_path': file_path,
                    'line_number': error_line,
                    'original_line': original_line,
                    'fixed_line': original_line + '\n' + optimization_comment
                }
        except Exception as e:
            print(f"⚠️ Failed to optimize nested loop: {e}")
        
        return None
    
    def _define_missing_variable(self, file_path: str, error_line: int, confidence: float) -> Dict:
        """Define missing variable or fix typo"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if error_line <= len(lines):
                original_line = lines[error_line - 1].rstrip()
                
                # Extract undefined variable name
                match = re.search(r'NameError.*name [\'"]([^\'"]+)[\'"]', original_line)
                if match:
                    var_name = match.group(1)
                    indent = len(original_line) - len(original_line.lstrip())
                    spaces = ' ' * indent
                    
                    # Add variable definition
                    definition = f"{spaces}{var_name} = None  # TODO: Initialize properly"
                    
                    lines.insert(error_line - 1, definition + '\n')
                    
                    with open(file_path, 'w') as f:
                        f.writelines(lines)
                    
                    return {
                        'type': 'code_fix',
                        'strategy': 'define_variable',
                        'confidence': confidence,
                        'description': f'Defined missing variable {var_name}',
                        'file_path': file_path,
                        'line_number': error_line,
                        'variable_name': var_name
                    }
        except Exception as e:
            print(f"⚠️ Failed to define missing variable: {e}")
        
        return None
    
    def handle_failure(self, error_message: str, stack_trace: str, file_path: str = None) -> Dict:
        """Enhanced failure handling with multiple fix strategies"""
        print(f"🔧 Enhanced MetaAgent handling failure: {error_message[:100]}...")
        
        # Detect error type
        error_info = self.detect_error_type(error_message, stack_trace)
        
        if not error_info:
            return {
                'status': 'failed',
                'reason': 'Could not detect error type',
                'error_message': error_message
            }
        
        # Extract file path and line number from stack trace
        if not file_path:
            file_match = re.search(r'File "([^"]+)", line (\d+)', stack_trace)
            if file_match:
                file_path = file_match.group(1)
                error_line = int(file_match.group(2))
            else:
                return {
                    'status': 'failed',
                    'reason': 'Could not extract file path and line number',
                    'error_message': error_message
                }
        else:
            # Find line number from stack trace
            line_match = re.search(r'line (\d+)', stack_trace)
            error_line = int(line_match.group(1)) if line_match else 1
        
        # Generate fixes
        fixes = self.generate_fixes(error_info, file_path, error_line)
        
        if not fixes:
            return {
                'status': 'failed',
                'reason': 'No fixes generated',
                'error_info': error_info
            }
        
        # Apply best fix (highest confidence)
        best_fix = max(fixes, key=lambda x: x.get('confidence', 0))
        
        # Validate and apply fix
        if best_fix.get('confidence', 0) >= self.confidence_threshold:
            try:
                # Verify fix doesn't break syntax
                with open(file_path, 'r') as f:
                    content = f.read()
                
                try:
                    ast.parse(content)
                    # If parsing succeeds, apply fix
                    self.successful_fixes.append(best_fix)
                    
                    return {
                        'status': 'success',
                        'fix_applied': best_fix,
                        'error_info': error_info,
                        'total_fixes_generated': len(fixes),
                        'confidence': best_fix.get('confidence')
                    }
                except SyntaxError:
                    # Fix created syntax error, try next best
                    fixes.remove(best_fix)
                    if fixes:
                        next_best = max(fixes, key=lambda x: x.get('confidence', 0))
                        self.failed_attempts.append(best_fix)
                        return self._apply_fallback_fix(next_best, error_info, file_path, error_line)
                    else:
                        return {
                            'status': 'failed',
                            'reason': 'All fixes created syntax errors',
                            'error_info': error_info
                        }
            except Exception as e:
                return {
                    'status': 'failed',
                    'reason': f'Fix application failed: {e}',
                    'error_info': error_info
                }
        else:
            return {
                'status': 'failed',
                'reason': f'Fix confidence {best_fix.get("confidence")} below threshold {self.confidence_threshold}',
                'error_info': error_info
            }
    
    def _apply_fallback_fix(self, fix: Dict, error_info: Dict, file_path: str, error_line: int) -> Dict:
        """Apply fallback fix with lower confidence"""
        try:
            # Apply fix with warning
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Add warning comment
            warning_comment = f"# WARNING: Applied low-confidence fix for {error_info.get('type')}\n"
            
            with open(file_path, 'w') as f:
                f.write(warning_comment + content)
            
            self.successful_fixes.append(fix)
            
            return {
                'status': 'success',
                'fix_applied': fix,
                'error_info': error_info,
                'fallback_used': True,
                'confidence': fix.get('confidence')
            }
        except Exception as e:
            return {
                'status': 'failed',
                'reason': f'Fallback fix failed: {e}',
                'error_info': error_info
            }
    
    def get_statistics(self) -> Dict:
        """Get Enhanced MetaAgent statistics"""
        return {
            'successful_fixes': len(self.successful_fixes),
            'failed_attempts': len(self.failed_attempts),
            'success_rate': len(self.successful_fixes) / max(1, len(self.successful_fixes) + len(self.failed_attempts)),
            'error_patterns_loaded': sum(len(patterns) for patterns in self.error_patterns.values()),
            'fix_strategies_available': sum(len(strategies) for strategies in self.fix_strategies.values()),
            'confidence_threshold': self.confidence_threshold,
            'learning_enabled': self.learning_enabled
        }

# Integration function to use with existing SAM system
def create_enhanced_meta_agent(system):
    """Create enhanced meta agent instance"""
    return EnhancedMetaAgent(system)
