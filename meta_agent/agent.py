#!/usr/bin/env python3
"""
Meta-Agent Main Controller
Orchestrates the entire self-healing pipeline
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import all meta-agent components
from meta_agent.analyzer import extract_context, generate_fix_prompt
from meta_agent.llm import generate_patch, explain_patch, validate_patch_reasoning
from meta_agent.patcher import apply_patch, create_commit, revert_changes, validate_patch_format
from meta_agent.validator import run_tests, validate_code_syntax, find_affected_tests
from meta_agent.safety import validate_diff, create_safety_report
from meta_agent.config import (
    PROJECT_ROOT, TEST_COMMAND, MAX_PATCH_LINES, ALLOW_FILES,
    DANGEROUS_PATTERNS, REQUIRE_TESTS_PASS, GIT_COMMIT_MESSAGE_TEMPLATE,
    GIT_COMMIT_MESSAGE_PREFIX, ENABLED
)

logger = logging.getLogger(__name__)

class MetaAgent:
    """Main meta-agent orchestrator"""

    def __init__(self):
        self.metrics = {
            'patches_attempted': 0,
            'patches_applied': 0,
            'patches_reverted': 0,
            'tests_run': 0,
            'errors_processed': 0,
            'start_time': datetime.now().isoformat()
        }

        logger.info("Meta-Agent initialized")

    def auto_patch(self, exc: Exception, additional_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main auto-patching pipeline.

        Args:
            exc: The exception that triggered auto-patching
            additional_context: Optional additional context

        Returns:
            dict: Result of the auto-patching process
        """
        if not ENABLED:
            return {"status": "disabled", "message": "Meta-agent is disabled"}

        start_time = time.time()
        self.metrics['errors_processed'] += 1

        try:
            logger.info(f"Starting auto-patch for exception: {type(exc).__name__}: {exc}")

            # Step 1: Extract context from exception
            context = extract_context(exc, additional_context)
            logger.info(f"Context extracted: {len(context.get('affected_files', []))} files affected")

            # Step 2: Generate fix prompt
            prompt = generate_fix_prompt(context)

            # Step 3: Generate patch using LLM
            logger.info("Generating patch with LLM...")
            diff = generate_patch(prompt)

            if not diff:
                logger.warning("LLM failed to generate patch")
                return {
                    "status": "failed",
                    "stage": "llm_generation",
                    "message": "LLM could not generate a valid patch",
                    "context": context
                }

            self.metrics['patches_attempted'] += 1

            # Step 4: Validate patch format
            format_valid, format_msg = validate_patch_format(diff)
            if not format_valid:
                logger.warning(f"Invalid patch format: {format_msg}")
                return {
                    "status": "failed",
                    "stage": "format_validation",
                    "message": format_msg,
                    "diff": diff[:500]
                }

            # Step 5: Safety validation
            safety_config = {
                'MAX_PATCH_LINES': MAX_PATCH_LINES,
                'DANGEROUS_PATTERNS': DANGEROUS_PATTERNS,
                'ALLOW_FILES': ALLOW_FILES
            }

            safe, safety_msg = validate_diff(diff, MAX_PATCH_LINES, safety_config)
            if not safe:
                logger.warning(f"Safety validation failed: {safety_msg}")
                return {
                    "status": "rejected",
                    "stage": "safety_validation",
                    "message": safety_msg,
                    "diff": diff[:500]
                }

            # Step 6: Apply patch (dry run first)
            logger.info("Testing patch application...")
            apply_success, apply_msg = apply_patch(diff, PROJECT_ROOT, dry_run=True)

            if not apply_success:
                logger.warning(f"Patch application test failed: {apply_msg}")
                return {
                    "status": "failed",
                    "stage": "patch_application_test",
                    "message": apply_msg,
                    "diff": diff[:500]
                }

            # Step 7: Apply patch for real
            logger.info("Applying patch...")
            apply_success, apply_msg = apply_patch(diff, PROJECT_ROOT, dry_run=False)

            if not apply_success:
                logger.error(f"Patch application failed: {apply_msg}")
                return {
                    "status": "failed",
                    "stage": "patch_application",
                    "message": apply_msg,
                    "diff": diff[:500]
                }

            # Step 8: Run tests
            logger.info("Running tests...")
            self.metrics['tests_run'] += 1

            tests_pass, test_details = run_tests(TEST_COMMAND, PROJECT_ROOT)

            if REQUIRE_TESTS_PASS and not tests_pass:
                logger.warning("Tests failed - reverting changes")

                # Revert the changes
                revert_success, revert_msg = revert_changes(PROJECT_ROOT)
                self.metrics['patches_reverted'] += 1

                return {
                    "status": "reverted",
                    "stage": "test_validation",
                    "message": "Tests failed - changes reverted",
                    "test_details": test_details,
                    "revert_success": revert_success,
                    "revert_message": revert_msg,
                    "diff": diff[:500]
                }

            # Step 9: Create commit
            commit_msg = GIT_COMMIT_MESSAGE_TEMPLATE.format(
                prefix=GIT_COMMIT_MESSAGE_PREFIX,
                description=f"Fix {type(exc).__name__}: {str(exc)[:50]}"
            )

            commit_success, commit_msg_result = create_commit(commit_msg, PROJECT_ROOT)

            if commit_success:
                self.metrics['patches_applied'] += 1
                logger.info(f"Successfully committed fix: {commit_msg_result}")

                # Generate explanation
                explanation = explain_patch(diff)

                processing_time = time.time() - start_time

                return {
                    "status": "success",
                    "stage": "completed",
                    "message": "Patch applied and committed successfully",
                    "commit_hash": commit_msg_result,
                    "explanation": explanation,
                    "processing_time": processing_time,
                    "diff": diff[:500],  # Truncate for response
                    "test_details": test_details
                }
            else:
                logger.error(f"Commit failed: {commit_msg_result}")
                return {
                    "status": "failed",
                    "stage": "commit",
                    "message": f"Patch applied but commit failed: {commit_msg_result}",
                    "diff": diff[:500]
                }

        except Exception as meta_exc:
            logger.error(f"Meta-agent internal error: {meta_exc}")
            processing_time = time.time() - start_time

            return {
                "status": "error",
                "stage": "meta_agent_error",
                "message": f"Meta-agent internal error: {meta_exc}",
                "processing_time": processing_time
            }

    def get_status(self) -> Dict[str, Any]:
        """Get meta-agent status and metrics."""
        runtime = datetime.now() - datetime.fromisoformat(self.metrics['start_time'])

        return {
            "enabled": ENABLED,
            "metrics": {
                **self.metrics,
                "runtime_seconds": runtime.total_seconds(),
                "success_rate": (
                    self.metrics['patches_applied'] / max(self.metrics['patches_attempted'], 1)
                ) * 100
            },
            "config": {
                "project_root": PROJECT_ROOT,
                "test_command": TEST_COMMAND,
                "max_patch_lines": MAX_PATCH_LINES,
                "require_tests_pass": REQUIRE_TESTS_PASS
            }
        }

    def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop - disable meta-agent."""
        global ENABLED
        ENABLED = False

        logger.warning("Meta-agent emergency stop activated")

        return {
            "status": "stopped",
            "message": "Meta-agent has been emergency stopped"
        }

# Global meta-agent instance
meta_agent = MetaAgent()

# Convenience functions
def auto_patch(exc: Exception, additional_context: Optional[Dict] = None) -> Dict[str, Any]:
    """Convenience function for auto-patching."""
    return meta_agent.auto_patch(exc, additional_context)

def get_meta_agent_status() -> Dict[str, Any]:
    """Get meta-agent status."""
    return meta_agent.get_status()

def emergency_stop_meta_agent() -> Dict[str, Any]:
    """Emergency stop the meta-agent."""
    return meta_agent.emergency_stop()

# Test function
if __name__ == "__main__":
    # Test the meta-agent
    print("🧪 Testing Meta-Agent")

    # Get status
    status = get_meta_agent_status()
    print(f"Meta-agent enabled: {status['enabled']}")
    print(f"Metrics: {status['metrics']}")

    # Test with a mock exception
    try:
        raise ValueError("Test exception for meta-agent")
    except Exception as e:
        if ENABLED:
            print("Running auto-patch on test exception...")
            result = auto_patch(e)
            print(f"Result: {result['status']} - {result['message']}")
        else:
            print("Meta-agent is disabled - skipping test")
