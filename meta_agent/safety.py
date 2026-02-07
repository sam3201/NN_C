#!/usr/bin/env python3
"""
Safety Validator for Meta-Agent
Ensures patches are safe before application
"""

import re
import ast
import os
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def validate_diff(diff: str, max_lines: int, config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Comprehensive safety validation of a unified diff.

    Args:
        diff: The unified diff to validate
        max_lines: Maximum allowed lines in diff
        config: Safety configuration

    Returns:
        tuple: (safe: bool, reason: str)
    """
    try:
        # Basic size check
        lines = diff.splitlines()
        if len(lines) > max_lines:
            return False, f"Diff too large: {len(lines)} lines (max: {max_lines})"

        # Check for dangerous patterns
        dangerous_patterns = config.get('DANGEROUS_PATTERNS', [])
        for pattern in dangerous_patterns:
            if pattern.lower() in diff.lower():
                return False, f"Dangerous pattern detected: {pattern}"

        # Validate unified diff format
        if not _is_valid_unified_diff(diff):
            return False, "Invalid unified diff format"

        # Check for file type restrictions
        allowed_extensions = config.get('ALLOW_FILES', [])
        if allowed_extensions:
            files_affected = _extract_files_from_diff(diff)
            for file_path in files_affected:
                if not _is_allowed_file(file_path, allowed_extensions):
                    return False, f"File type not allowed: {file_path}"

        # AST-level validation for Python files
        python_files = [f for f in _extract_files_from_diff(diff) if f.endswith('.py')]
        for py_file in python_files:
            ast_valid, ast_error = _validate_python_ast(diff, py_file)
            if not ast_valid:
                return False, f"Python AST validation failed for {py_file}: {ast_error}"

        # Check for suspicious changes
        suspicious_checks = [
            _check_for_mass_deletion,
            _check_for_executable_additions,
            _check_for_config_modifications,
            _check_for_database_changes
        ]

        for check_func in suspicious_checks:
            suspicious, reason = check_func(diff)
            if suspicious:
                return False, f"Suspicious change detected: {reason}"

        return True, "Diff passed all safety checks"

    except Exception as e:
        logger.error(f"Safety validation error: {e}")
        return False, f"Safety validation failed: {e}"

def _is_valid_unified_diff(diff: str) -> bool:
    """Check if diff is a valid unified diff format."""
    lines = diff.strip().splitlines()

    if len(lines) < 3:
        return False

    # Must start with --- or +++
    if not (lines[0].startswith('--- ') or lines[0].startswith('+++ ')):
        return False

    # Must have at least one @@ hunk
    has_hunk = any(line.startswith('@@ ') for line in lines)
    if not has_hunk:
        return False

    # Must have actual changes
    has_changes = any(line.startswith('+') or line.startswith('-') for line in lines)
    if not has_changes:
        return False

    return True

def _extract_files_from_diff(diff: str) -> List[str]:
    """Extract file paths from unified diff."""
    files = []
    lines = diff.splitlines()

    for line in lines:
        if line.startswith('--- ') or line.startswith('+++ '):
            # Extract path after the marker
            path = line.split(' ', 1)[1]
            # Remove a/ or b/ prefix and timestamps
            if path.startswith(('a/', 'b/')):
                path = path[2:]
            if '\t' in path:
                path = path.split('\t')[0]
            if files and path == files[-1]:
                continue  # Skip duplicate
            files.append(path)

    return files

def _is_allowed_file(file_path: str, allowed_extensions: List[str]) -> bool:
    """Check if file has allowed extension."""
    if not allowed_extensions:
        return True

    _, ext = os.path.splitext(file_path)
    return ext in allowed_extensions or ext.lower() in [e.lower() for e in allowed_extensions]

def _validate_python_ast(diff: str, file_path: str) -> Tuple[bool, str]:
    """Validate Python syntax using AST for changed code."""
    try:
        # Extract added/removed lines for the specific file
        additions = []
        deletions = []

        lines = diff.splitlines()
        in_hunk = False
        current_file = None

        for line in lines:
            if line.startswith('+++ ') or line.startswith('--- '):
                path = line.split(' ', 1)[1]
                if path.startswith(('a/', 'b/')):
                    path = path[2:]
                if '\t' in path:
                    path = path.split('\t')[0]
                current_file = path
                continue

            if current_file != file_path:
                continue

            if line.startswith('@@ '):
                in_hunk = True
                continue

            if in_hunk and line.startswith('+'):
                additions.append(line[1:])  # Remove + prefix
            elif in_hunk and line.startswith('-'):
                deletions.append(line[1:])  # Remove - prefix

        # Try to compile the additions as Python code
        if additions:
            try:
                # Join additions and try to compile
                added_code = '\n'.join(additions)
                ast.parse(added_code)
            except SyntaxError as e:
                return False, f"Syntax error in additions: {e}"

        return True, "AST validation passed"

    except Exception as e:
        return False, f"AST validation error: {e}"

def _check_for_mass_deletion(diff: str) -> Tuple[bool, str]:
    """Check for suspicious mass deletions."""
    lines = diff.splitlines()
    deletions = sum(1 for line in lines if line.startswith('-'))

    # Flag if more than 50 lines deleted
    if deletions > 50:
        return True, f"Mass deletion detected: {deletions} lines removed"

    return False, ""

def _check_for_executable_additions(diff: str) -> Tuple[bool, str]:
    """Check for addition of executable code patterns."""
    dangerous_patterns = [
        r'subprocess\.',
        r'os\.system',
        r'eval\(',
        r'exec\(',
        r'__import__',
        r'importlib',
        r'shutil\.rmtree',
        r'os\.remove',
        r'os\.unlink'
    ]

    lines = diff.splitlines()
    for line in lines:
        if line.startswith('+'):
            code = line[1:].strip()
            for pattern in dangerous_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return True, f"Dangerous executable pattern: {pattern}"

    return False, ""

def _check_for_config_modifications(diff: str) -> Tuple[bool, str]:
    """Check for modifications to configuration files."""
    config_files = [
        'config.py', 'settings.py', '.env', 'requirements.txt',
        'setup.py', 'pyproject.toml', 'config.json', 'settings.json'
    ]

    files_affected = _extract_files_from_diff(diff)
    for file_path in files_affected:
        file_name = os.path.basename(file_path).lower()
        if file_name in config_files or file_name.endswith(('.conf', '.ini', '.cfg')):
            return True, f"Configuration file modification: {file_path}"

    return False, ""

def _check_for_database_changes(diff: str) -> Tuple[bool, str]:
    """Check for database-related changes."""
    db_patterns = [
        r'DROP\s+TABLE',
        r'DROP\s+DATABASE',
        r'DELETE\s+FROM',
        r'TRUNCATE',
        r'ALTER\s+TABLE.*DROP'
    ]

    lines = diff.splitlines()
    for line in lines:
        if line.startswith('+') or line.startswith('-'):
            code = line[1:].strip()
            for pattern in db_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return True, f"Potentially dangerous database operation: {pattern}"

    return False, ""

def create_safety_report(diff: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a comprehensive safety report for a diff."""
    report = {
        "diff_size": len(diff.splitlines()),
        "files_affected": _extract_files_from_diff(diff),
        "checks_passed": [],
        "warnings": [],
        "errors": []
    }

    # Size check
    max_lines = config.get('MAX_PATCH_LINES', 200)
    if report["diff_size"] > max_lines:
        report["errors"].append(f"Diff size exceeds limit: {report['diff_size']} > {max_lines}")
    else:
        report["checks_passed"].append("Size check")

    # Dangerous patterns
    dangerous_patterns = config.get('DANGEROUS_PATTERNS', [])
    for pattern in dangerous_patterns:
        if pattern.lower() in diff.lower():
            report["errors"].append(f"Dangerous pattern: {pattern}")
        else:
            report["checks_passed"].append(f"No {pattern}")

    # File type check
    allowed_extensions = config.get('ALLOW_FILES', [])
    for file_path in report["files_affected"]:
        if not _is_allowed_file(file_path, allowed_extensions):
            report["errors"].append(f"Disallowed file type: {file_path}")
        else:
            report["checks_passed"].append(f"File type allowed: {file_path}")

    # Overall safety
    report["safe"] = len(report["errors"]) == 0

    return report

# Test function
if __name__ == "__main__":
    # Test the safety validator
    print("🧪 Testing Safety Validator")

    from config import MAX_PATCH_LINES, DANGEROUS_PATTERNS, ALLOW_FILES

    config = {
        'MAX_PATCH_LINES': MAX_PATCH_LINES,
        'DANGEROUS_PATTERNS': DANGEROUS_PATTERNS,
        'ALLOW_FILES': ALLOW_FILES
    }

    # Test safe diff
    safe_diff = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def hello():
-    print("hello")
+    print("hello world")
"""

    safe, reason = validate_diff(safe_diff, MAX_PATCH_LINES, config)
    print(f"Safe diff: {safe} - {reason}")

    # Test dangerous diff
    dangerous_diff = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def hello():
-    print("hello")
+    subprocess.run("rm -rf /")
"""

    safe, reason = validate_diff(dangerous_diff, MAX_PATCH_LINES, config)
    print(f"Dangerous diff: {safe} - {reason}")
