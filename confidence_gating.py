#!/usr/bin/env python3
"""
SAM 2.0 Confidence Gating System
Prevents unsafe patches and stops infinite retry loops
"""

import logging
from typing import Dict, Any, Optional
from failure_clustering import get_cluster_hits, should_skip_retry
from patch_scoring import score_patch
from multi_agent_debate import debate_patch

logger = logging.getLogger(__name__)

class ConfidenceGate:
    """Gates patches based on confidence and safety criteria"""

    def __init__(self, max_attempts: int = 3, min_confidence: float = 0.7):
        self.max_attempts = max_attempts
        self.min_confidence = min_confidence

        # Hard rules that cannot be overridden
        self.hard_rules = {
            'max_attempts': self.max_attempts,
            'dangerous_patterns': self._get_dangerous_patterns(),
            'required_tests': True  # Must have tests
        }

        # Soft rules that can be adjusted
        self.soft_rules = {
            'min_confidence': self.min_confidence,
            'max_patch_size': 50,  # Max lines in diff
            'max_cluster_hits': 5  # Max times same error occurred
        }

    def _get_dangerous_patterns(self) -> list:
        """Get list of dangerous code patterns"""
        return [
            r'exec\s*\(',  # Code execution
            r'eval\s*\(',  # Code evaluation
            r'import\s+os\s*;.*os\.system',  # System commands
            r'subprocess\..*shell\s*=\s*True',  # Shell injection
            r'shutil\.rmtree\s*\(\s*[\'\"]/',  # Root directory deletion
            r'os\.remove\s*\(\s*[\'\"]/',  # Root file deletion
            r'__del__\s*\(',  # Custom destructors
            r'globals\(\)\s*\[',  # Global variable manipulation
            r'locals\(\)\s*\[',  # Local variable manipulation
        ]

    def evaluate_patch(self, patch: str, error_context: str, attempt_number: int,
                      tests_available: bool = True) -> Dict[str, Any]:
        """
        Evaluate a patch against all confidence criteria

        Args:
            patch: The unified diff patch
            error_context: Description of the error being fixed
            attempt_number: Which attempt this is (1-based)
            tests_available: Whether tests are available to run

        Returns:
            Dict with evaluation results and decision
        """
        logger.info(f"Evaluating patch (attempt {attempt_number})")

        evaluation = {
            'approved': False,
            'confidence_score': 0.0,
            'reasons': [],
            'warnings': [],
            'hard_failures': [],
            'soft_failures': [],
            'recommendation': 'reject'
        }

        # 1. Check HARD rules (cannot be overridden)
        hard_passed = self._check_hard_rules(patch, error_context, attempt_number, tests_available)
        if not hard_passed:
            evaluation['hard_failures'] = self._check_hard_rules(patch, error_context, attempt_number, tests_available, return_details=True)
            evaluation['reasons'].append("Failed hard safety rules")
            logger.warning("Patch failed hard rules - rejecting")
            return evaluation

        # 2. Check SOFT rules (can be overridden but reduce confidence)
        soft_issues = self._check_soft_rules(patch, error_context, attempt_number)
        evaluation['soft_failures'] = soft_issues

        # 3. Run debate process
        debate_result = debate_patch(error_context, patch)
        evaluation['debate_result'] = debate_result

        # 4. Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            debate_result, soft_issues, attempt_number, patch
        )
        evaluation['confidence_score'] = confidence

        # 5. Make final decision
        if confidence >= self.min_confidence and debate_result.get('approved', False):
            evaluation['approved'] = True
            evaluation['recommendation'] = 'apply'
            evaluation['reasons'].append(f"High confidence ({confidence:.2f}) and debate approved")
        else:
            evaluation['reasons'].append(f"Low confidence ({confidence:.2f}) or debate rejected")

        # Add warnings
        if soft_issues:
            evaluation['warnings'].extend([f"Soft rule violation: {issue}" for issue in soft_issues])

        logger.info(f"Patch evaluation complete: {'APPROVED' if evaluation['approved'] else 'REJECTED'} "
                   f"(confidence: {confidence:.2f})")

        return evaluation

    def _check_hard_rules(self, patch: str, error_context: str, attempt_number: int,
                         tests_available: bool, return_details: bool = False) -> bool:
        """Check hard rules that cannot be violated"""
        failures = []

        # Rule 1: Max attempts not exceeded
        if attempt_number > self.hard_rules['max_attempts']:
            failures.append(f"Exceeded maximum attempts ({self.hard_rules['max_attempts']})")

        # Rule 2: No dangerous patterns
        for pattern in self.hard_rules['dangerous_patterns']:
            import re
            if re.search(pattern, patch, re.IGNORECASE | re.MULTILINE):
                failures.append(f"Dangerous pattern detected: {pattern}")

        # Rule 3: Must have tests if required
        if self.hard_rules['required_tests'] and not tests_available:
            failures.append("Tests required but not available")

        # Rule 4: Basic diff validation
        if not self._is_valid_diff(patch):
            failures.append("Invalid unified diff format")

        if return_details:
            return failures

        return len(failures) == 0

    def _check_soft_rules(self, patch: str, error_context: str, attempt_number: int) -> list:
        """Check soft rules that reduce confidence but don't block"""
        issues = []

        # Check patch size
        diff_lines = len(patch.splitlines())
        if diff_lines > self.soft_rules['max_patch_size']:
            issues.append(f"Large patch ({diff_lines} lines > {self.soft_rules['max_patch_size']})")

        # Check failure clustering
        error_type = self._extract_error_type(error_context)
        files = self._extract_files(patch)

        if error_type and files:
            cluster_hits = get_cluster_hits(error_type, files)
            if cluster_hits > self.soft_rules['max_cluster_hits']:
                issues.append(f"Recurring failure pattern ({cluster_hits} occurrences)")

        # Check attempt number
        if attempt_number > 1:
            issues.append(f"Multiple attempts ({attempt_number})")

        return issues

    def _calculate_overall_confidence(self, debate_result: Dict, soft_issues: list,
                                    attempt_number: int, patch: str) -> float:
        """Calculate overall confidence score"""
        confidence = 0.5  # Base confidence

        # Debate result is most important
        if debate_result.get('approved', False):
            confidence += 0.3
            debate_conf = debate_result.get('confidence', 0.5)
            confidence += debate_conf * 0.2
        else:
            confidence -= 0.2

        # Soft rule violations reduce confidence
        soft_penalty = len(soft_issues) * 0.1
        confidence -= soft_penalty

        # Attempt penalty
        attempt_penalty = (attempt_number - 1) * 0.05
        confidence -= attempt_penalty

        # Patch quality bonus
        if self._is_high_quality_patch(patch):
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _is_valid_diff(self, patch: str) -> bool:
        """Basic validation of unified diff format"""
        lines = patch.splitlines()

        if len(lines) < 3:
            return False

        # Must start with --- or +++
        if not (lines[0].startswith('--- ') or lines[0].startswith('+++ ')):
            return False

        # Must have at least one @@ hunk
        has_hunk = any(line.startswith('@@ ') for line in lines)
        if not has_hunk:
            return False

        # Must have some actual changes
        has_changes = any(line.startswith('+') or line.startswith('-') for line in lines)
        if not has_changes:
            return False

        return True

    def _is_high_quality_patch(self, patch: str) -> bool:
        """Check for signs of high-quality patch"""
        # Minimal changes
        lines = patch.splitlines()
        change_lines = [line for line in lines if line.startswith(('+', '-'))]
        if len(change_lines) > 10:  # Too many changes
            return False

        # Has context (lines around changes)
        context_lines = [line for line in lines if not line.startswith(('+', '-', '@', '---', '+++'))]
        if len(context_lines) < len(change_lines):  # Not enough context
            return False

        return True

    def _extract_error_type(self, error_context: str) -> Optional[str]:
        """Extract error type from context"""
        # Simple extraction - look for common error types
        error_types = ['NameError', 'ImportError', 'ValueError', 'TypeError',
                      'AttributeError', 'KeyError', 'IndexError', 'ZeroDivisionError']

        for error_type in error_types:
            if error_type in error_context:
                return error_type

        return None

    def _extract_files(self, patch: str) -> list:
        """Extract file names from diff"""
        files = []
        lines = patch.splitlines()

        for line in lines:
            if line.startswith('--- a/') or line.startswith('+++ b/'):
                # Extract filename from --- a/filename or +++ b/filename
                parts = line.split('/')
                if len(parts) > 1:
                    filename = '/'.join(parts[1:])
                    if filename not in files:
                        files.append(filename)

        return files

    def get_gate_stats(self) -> Dict[str, Any]:
        """Get confidence gate statistics"""
        return {
            'max_attempts': self.max_attempts,
            'min_confidence': self.min_confidence,
            'hard_rules_count': len(self.hard_rules['dangerous_patterns']),
            'soft_rules': self.soft_rules
        }

    def update_soft_rules(self, **kwargs):
        """Update soft rule thresholds"""
        for key, value in kwargs.items():
            if key in self.soft_rules:
                self.soft_rules[key] = value
                logger.info(f"Updated soft rule {key} to {value}")

# Global confidence gate instance
confidence_gate = ConfidenceGate()

# Convenience functions
def evaluate_patch(patch: str, error_context: str, attempt_number: int,
                  tests_available: bool = True) -> Dict[str, Any]:
    """Evaluate a patch"""
    return confidence_gate.evaluate_patch(patch, error_context, attempt_number, tests_available)

def get_gate_stats() -> Dict[str, Any]:
    """Get gate statistics"""
    return confidence_gate.get_gate_stats()

if __name__ == "__main__":
    # Test confidence gating
    print("🧪 Testing Confidence Gating System")

    # Test patch
    test_patch = """--- a/test.py
+++ b/test.py
@@ -4,1 +4,1 @@
-undefined_var = 42
+defined_var = 42"""

    error_context = "NameError: name 'undefined_var' is not defined"

    try:
        result = evaluate_patch(test_patch, error_context, attempt_number=1)

        print(f"Gate decision: {'APPROVED' if result['approved'] else 'REJECTED'}")
        print(f"Confidence: {result['confidence_score']:.2f}")
        print(f"Reasons: {result['reasons']}")
        if result['warnings']:
            print(f"Warnings: {result['warnings']}")

        # Get stats
        stats = get_gate_stats()
        print(f"Gate stats: {stats}")

    except Exception as e:
        print(f"❌ Gate test failed: {e}")

    print("✅ Confidence gating test complete")
