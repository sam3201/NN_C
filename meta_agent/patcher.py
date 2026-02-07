#!/usr/bin/env python3
"""
Patch Applier for Meta-Agent
Applies unified diffs and manages git operations
"""

import subprocess
import tempfile
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def apply_patch(diff: str, project_root: str, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Apply a unified diff patch to the project.

    Args:
        diff: Unified diff content
        project_root: Root directory of the project
        dry_run: If True, test the patch without applying it

    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Validate inputs
        if not diff or not diff.strip():
            return False, "Empty diff provided"

        if not os.path.exists(project_root):
            return False, f"Project root does not exist: {project_root}"

        # Check if we're in a git repository
        if not _is_git_repo(project_root):
            return False, "Not a git repository"

        # Check git status - should be clean or allow uncommitted changes?
        # For safety, we'll require a clean working directory
        status = _get_git_status(project_root)
        if status != "clean":
            return False, f"Working directory not clean: {status}"

        # Create a backup branch before applying
        backup_branch = f"backup_before_meta_agent_{int(__import__('time').time())}"

        try:
            _run_git_command(["checkout", "-b", backup_branch], project_root)
            logger.info(f"Created backup branch: {backup_branch}")
        except Exception as e:
            logger.warning(f"Could not create backup branch: {e}")

        # Apply the patch
        if dry_run:
            # Test the patch without applying
            result = _run_git_command(["apply", "--check", "--unsafe-paths"],
                                    project_root, input_data=diff)
        else:
            # Apply the patch
            result = _run_git_command(["apply", "--unsafe-paths"],
                                    project_root, input_data=diff)

        if result.returncode == 0:
            message = "Patch applied successfully"
            if dry_run:
                message = "Patch validation successful (dry run)"
            logger.info(message)
            return True, message
        else:
            error_msg = result.stderr.decode('utf-8', errors='ignore').strip()
            logger.error(f"Patch application failed: {error_msg}")

            # Try to restore from backup
            try:
                _run_git_command(["checkout", "-"], project_root)  # Go back to previous branch
                _run_git_command(["branch", "-D", backup_branch], project_root)  # Delete backup
            except Exception as e:
                logger.warning(f"Could not restore from backup: {e}")

            return False, f"Patch application failed: {error_msg}"

    except Exception as e:
        logger.error(f"Unexpected error during patch application: {e}")
        return False, f"Unexpected error: {e}"

def create_commit(message: str, project_root: str) -> Tuple[bool, str]:
    """
    Create a git commit with the applied changes.

    Args:
        message: Commit message
        project_root: Project root directory

    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Check if there are changes to commit
        status_result = _run_git_command(["status", "--porcelain"], project_root)
        if not status_result.stdout.strip():
            return False, "No changes to commit"

        # Add all changes
        _run_git_command(["add", "."], project_root)

        # Create commit
        result = _run_git_command(["commit", "-m", message], project_root)

        if result.returncode == 0:
            commit_hash = _run_git_command(["rev-parse", "HEAD"], project_root)
            if commit_hash.returncode == 0:
                short_hash = commit_hash.stdout.decode('utf-8').strip()[:8]
                return True, f"Committed as {short_hash}"
            else:
                return True, "Committed successfully"
        else:
            error_msg = result.stderr.decode('utf-8', errors='ignore').strip()
            return False, f"Commit failed: {error_msg}"

    except Exception as e:
        logger.error(f"Commit creation failed: {e}")
        return False, f"Commit error: {e}"

def revert_changes(project_root: str) -> Tuple[bool, str]:
    """
    Revert all uncommitted changes (hard reset).

    Args:
        project_root: Project root directory

    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Hard reset to HEAD
        result = _run_git_command(["reset", "--hard", "HEAD"], project_root)

        if result.returncode == 0:
            # Clean untracked files
            _run_git_command(["clean", "-fd"], project_root)
            return True, "Changes reverted successfully"
        else:
            error_msg = result.stderr.decode('utf-8', errors='ignore').strip()
            return False, f"Revert failed: {error_msg}"

    except Exception as e:
        logger.error(f"Revert operation failed: {e}")
        return False, f"Revert error: {e}"

def _is_git_repo(project_root: str) -> bool:
    """Check if the directory is a git repository."""
    try:
        result = _run_git_command(["rev-parse", "--git-dir"], project_root)
        return result.returncode == 0
    except:
        return False

def _get_git_status(project_root: str) -> str:
    """Get a summary of git status."""
    try:
        result = _run_git_command(["status", "--porcelain"], project_root)
        if result.returncode == 0:
            status_lines = result.stdout.decode('utf-8').strip().splitlines()
            if not status_lines:
                return "clean"
            else:
                return f"{len(status_lines)} uncommitted changes"
        else:
            return "unknown"
    except:
        return "error"

def _run_git_command(args: list, cwd: str, input_data: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a git command with proper error handling."""
    try:
        cmd = ["git"] + args

        if input_data:
            # Run with input data (for apply command)
            result = subprocess.run(
                cmd,
                cwd=cwd,
                input=input_data,
                capture_output=True,
                text=False,  # Keep as bytes for diff input
                timeout=30
            )
        else:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )

        return result

    except subprocess.TimeoutExpired:
        # Create a fake result for timeout
        fake_result = subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="Command timed out"
        )
        return fake_result
    except Exception as e:
        # Create a fake result for other errors
        fake_result = subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr=f"Command execution failed: {e}"
        )
        return fake_result

def validate_patch_format(diff: str) -> Tuple[bool, str]:
    """
    Validate that the diff is in proper unified format.

    Args:
        diff: The diff content to validate

    Returns:
        tuple: (valid: bool, message: str)
    """
    try:
        lines = diff.strip().splitlines()

        if not lines:
            return False, "Empty diff"

        # Check for unified diff markers
        if not any(line.startswith('--- ') for line in lines):
            return False, "Missing --- marker (not a unified diff)"

        if not any(line.startswith('+++ ') for line in lines):
            return False, "Missing +++ marker (not a unified diff)"

        if not any(line.startswith('@@ ') for line in lines):
            return False, "Missing @@ hunk marker (not a unified diff)"

        # Check for actual changes
        has_additions = any(line.startswith('+') for line in lines)
        has_deletions = any(line.startswith('-') for line in lines)

        if not (has_additions or has_deletions):
            return False, "No actual changes found in diff"

        return True, "Valid unified diff format"

    except Exception as e:
        return False, f"Diff validation error: {e}"

# Test function
if __name__ == "__main__":
    # Test the patcher
    print("🧪 Testing Patch Applier")

    from config import PROJECT_ROOT

    # Test git repo check
    is_git = _is_git_repo(PROJECT_ROOT)
    print(f"Git repo: {is_git}")

    if is_git:
        # Test status
        status = _get_git_status(PROJECT_ROOT)
        print(f"Git status: {status}")

        # Test validation
        test_diff = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def hello():
-    print("hello")
+    print("hello world")
"""
        valid, msg = validate_patch_format(test_diff)
        print(f"Diff validation: {valid} - {msg}")
