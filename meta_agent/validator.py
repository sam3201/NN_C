#!/usr/bin/env python3
"""
Test Validator for Meta-Agent
Runs test suites to validate code changes
"""

import subprocess
import sys
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def run_tests(test_cmd: List[str], project_root: str, timeout: int = 60) -> Tuple[bool, Dict[str, Any]]:
    """
    Run the test suite and return results.

    Args:
        test_cmd: Command to run tests (e.g., ["pytest", "-q"])
        project_root: Root directory of the project
        timeout: Timeout in seconds

    Returns:
        tuple: (passed: bool, details: dict)
    """
    try:
        logger.info(f"Running tests: {' '.join(test_cmd)} in {project_root}")

        # Ensure we're in the right directory
        if not os.path.exists(project_root):
            return False, {"error": f"Project root does not exist: {project_root}"}

        # Run the test command
        result = subprocess.run(
            test_cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Parse results
        success = result.returncode == 0
        details = {
            "returncode": result.returncode,
            "passed": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": test_cmd,
            "timeout": timeout,
            "execution_time": getattr(result, 'elapsed', None)
        }

        # Extract test statistics if possible
        if success:
            test_stats = _extract_test_stats(result.stdout)
            details.update(test_stats)

        logger.info(f"Tests {'PASSED' if success else 'FAILED'} (exit code: {result.returncode})")

        return success, details

    except subprocess.TimeoutExpired:
        logger.error(f"Test execution timed out after {timeout} seconds")
        return False, {
            "error": f"Tests timed out after {timeout} seconds",
            "timeout": timeout,
            "passed": False
        }

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False, {
            "error": f"Test execution error: {e}",
            "passed": False
        }

def _extract_test_stats(output: str) -> Dict[str, Any]:
    """
    Extract test statistics from test output.

    Args:
        output: Test output string

    Returns:
        dict: Extracted statistics
    """
    stats = {}

    try:
        lines = output.splitlines()

        # Look for common test result patterns
        for line in lines:
            line = line.lower().strip()

            # pytest patterns
            if "passed" in line and "failed" in line:
                # Try to extract numbers
                import re
                numbers = re.findall(r'\d+', line)
                if len(numbers) >= 2:
                    stats["tests_passed"] = int(numbers[0])
                    stats["tests_failed"] = int(numbers[1])

            # unittest patterns
            elif "ran " in line and " tests" in line:
                numbers = re.findall(r'\d+', line)
                if numbers:
                    stats["tests_run"] = int(numbers[0])

            # Coverage information
            elif "coverage" in line or "%" in line:
                coverage_match = re.search(r'(\d+(?:\.\d+)?)%', line)
                if coverage_match:
                    stats["coverage"] = float(coverage_match.group(1))

    except Exception as e:
        logger.warning(f"Could not extract test stats: {e}")

    return stats

def validate_code_syntax(file_paths: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate syntax of Python files.

    Args:
        file_paths: List of file paths to validate

    Returns:
        tuple: (all_valid: bool, errors: list)
    """
    errors = []

    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                errors.append(f"File does not exist: {file_path}")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            compile(content, file_path, 'exec')
            logger.debug(f"Syntax valid: {file_path}")

        except SyntaxError as e:
            error_msg = f"Syntax error in {file_path}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

        except Exception as e:
            error_msg = f"Could not validate {file_path}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

    return len(errors) == 0, errors

def check_test_coverage(project_root: str, min_coverage: float = 80.0) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if test coverage meets minimum requirements.

    Args:
        project_root: Project root directory
        min_coverage: Minimum required coverage percentage

    Returns:
        tuple: (meets_requirement: bool, details: dict)
    """
    try:
        # Try to run coverage
        cmd = [sys.executable, "-m", "coverage", "run", "--source=.", "-m", "pytest"]
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return False, {"error": "Coverage run failed", "output": result.stderr}

        # Generate coverage report
        report_result = subprocess.run(
            ["coverage", "report", "--include=*.py"],
            cwd=project_root,
            capture_output=True,
            text=True
        )

        if report_result.returncode == 0:
            # Parse coverage percentage
            lines = report_result.stdout.splitlines()
            for line in lines:
                if "TOTAL" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            coverage_pct = float(parts[-1].rstrip('%'))
                            meets_req = coverage_pct >= min_coverage
                            return meets_req, {
                                "coverage_percentage": coverage_pct,
                                "required_minimum": min_coverage,
                                "report": report_result.stdout
                            }
                        except ValueError:
                            pass

        return False, {"error": "Could not parse coverage report"}

    except Exception as e:
        logger.error(f"Coverage check failed: {e}")
        return False, {"error": f"Coverage check error: {e}"}

def run_specific_tests(test_files: List[str], project_root: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Run tests for specific files.

    Args:
        test_files: List of test files to run
        project_root: Project root directory

    Returns:
        tuple: (passed: bool, details: dict)
    """
    if not test_files:
        return True, {"message": "No specific tests to run"}

    try:
        # Run pytest on specific files
        cmd = [sys.executable, "-m", "pytest"] + test_files
        return run_tests(cmd, project_root)

    except Exception as e:
        logger.error(f"Specific test run failed: {e}")
        return False, {"error": f"Specific test error: {e}"}

def find_affected_tests(changed_files: List[str], project_root: str) -> List[str]:
    """
    Find test files that might be affected by changes to given files.

    Args:
        changed_files: List of files that were changed
        project_root: Project root directory

    Returns:
        list: List of test files that might be affected
    """
    affected_tests = []

    try:
        # Simple heuristic: look for test files with similar names
        for changed_file in changed_files:
            if changed_file.endswith('.py'):
                # Remove .py extension and look for test_*.py files
                base_name = Path(changed_file).stem

                # Common test file patterns
                test_patterns = [
                    f"test_{base_name}.py",
                    f"tests/test_{base_name}.py",
                    f"{base_name}_test.py",
                    f"tests/{base_name}_test.py"
                ]

                for pattern in test_patterns:
                    test_path = os.path.join(project_root, pattern)
                    if os.path.exists(test_path):
                        affected_tests.append(pattern)

        # If no specific tests found, suggest running all tests
        if not affected_tests:
            affected_tests = ["."]  # Run all tests

    except Exception as e:
        logger.error(f"Could not find affected tests: {e}")
        affected_tests = ["."]  # Fallback to all tests

    return list(set(affected_tests))  # Remove duplicates

# Test function
if __name__ == "__main__":
    # Test the validator
    print("🧪 Testing Test Validator")

    from config import PROJECT_ROOT, TEST_COMMAND

    # Test basic test run
    success, details = run_tests(TEST_COMMAND, PROJECT_ROOT)
    print(f"Tests passed: {success}")
    if "tests_passed" in details:
        print(f"Tests run: {details.get('tests_passed', 0)} passed, {details.get('tests_failed', 0)} failed")

    # Test syntax validation
    test_files = ["meta_agent/config.py"]
    valid, errors = validate_code_syntax(test_files)
    print(f"Syntax valid: {valid}")
    if errors:
        print(f"Errors: {errors}")
