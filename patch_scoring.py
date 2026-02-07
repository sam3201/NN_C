#!/usr/bin/env python3
"""
SAM 2.0 Patch Scoring System
Evaluates the quality and safety of generated patches
"""

import os
import json
import re
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PatchScorer:
    """Scores patches based on multiple criteria"""

    def __init__(self, scores_file="memory/patch_scores.json"):
        self.scores_file = scores_file
        self.patch_scores = []

        # Create memory directory
        os.makedirs("memory", exist_ok=True)

        # Load existing scores
        self._load_scores()

    def _load_scores(self):
        """Load existing patch scores"""
        try:
            if os.path.exists(self.scores_file):
                with open(self.scores_file, 'r') as f:
                    self.patch_scores = json.load(f)
                logger.info(f"✅ Loaded {len(self.patch_scores)} patch scores")
            else:
                logger.info("✅ Created new patch scoring system")
        except Exception as e:
            logger.error(f"❌ Failed to load patch scores: {e}")
            self.patch_scores = []

    def _save_scores(self):
        """Save patch scores to disk"""
        try:
            with open(self.scores_file, 'w') as f:
                json.dump(self.patch_scores, f, indent=2)
            logger.debug(f"Saved {len(self.patch_scores)} patch scores")
        except Exception as e:
            logger.error(f"❌ Failed to save patch scores: {e}")

    def score_patch(self, diff: str, tests_passed: bool, attempts: int,
                   cluster_hits: int, execution_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Score a patch based on multiple criteria

        Args:
            diff: The unified diff
            tests_passed: Whether tests pass after applying patch
            attempts: Number of attempts to generate this patch
            cluster_hits: How many times this failure pattern has occurred
            execution_time: Time taken to generate patch (optional)

        Returns:
            Dict with score and analysis
        """
        score = 0
        reasons = []

        # Base score for tests passing
        if tests_passed:
            score += 100
            reasons.append("tests_pass")
        else:
            score -= 50
            reasons.append("tests_fail")

        # Size penalty - prefer minimal changes
        diff_size = len(diff.splitlines())
        size_penalty = diff_size * 0.1
        score -= size_penalty
        reasons.append(f"size_penalty_{size_penalty:.1f}")

        # Attempt penalty - fewer attempts is better
        attempt_penalty = attempts * 5
        score -= attempt_penalty
        reasons.append(f"attempt_penalty_{attempt_penalty}")

        # Cluster penalty - recurring failures get penalty
        cluster_penalty = cluster_hits * 10
        score -= cluster_penalty
        reasons.append(f"cluster_penalty_{cluster_penalty}")

        # Bonus for clean diffs (no extra whitespace, proper formatting)
        formatting_bonus = self._evaluate_formatting(diff)
        score += formatting_bonus
        if formatting_bonus > 0:
            reasons.append(f"formatting_bonus_{formatting_bonus}")

        # Safety check - detect potentially dangerous changes
        safety_score = self._evaluate_safety(diff)
        score += safety_score
        if safety_score < 0:
            reasons.append(f"safety_penalty_{abs(safety_score)}")

        # Execution time bonus (faster is better, but not critical)
        if execution_time and execution_time < 30:  # Less than 30 seconds
            time_bonus = 5
            score += time_bonus
            reasons.append(f"time_bonus_{time_bonus}")

        # Store the score
        patch_record = {
            "diff_hash": hash(diff) % 1000000,  # Simple hash for tracking
            "score": score,
            "tests_passed": tests_passed,
            "attempts": attempts,
            "cluster_hits": cluster_hits,
            "diff_size": diff_size,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time
        }

        self.patch_scores.append(patch_record)
        self._save_scores()

        # Determine confidence level
        confidence = self._calculate_confidence(score, tests_passed, attempts)

        result = {
            "score": score,
            "confidence": confidence,
            "reasons": reasons,
            "recommendation": "apply" if confidence > 0.7 else "reject",
            "analysis": self._generate_analysis(score, reasons)
        }

        logger.info(f"Scored patch: {score:.1f} (confidence: {confidence:.2f})")
        return result

    def _evaluate_formatting(self, diff: str) -> float:
        """Evaluate diff formatting quality"""
        bonus = 0

        lines = diff.splitlines()

        # Check for proper unified diff format
        if lines and lines[0].startswith('--- ') and len(lines) > 1:
            bonus += 5

        # Check for proper @@ hunks
        has_hunks = any(line.startswith('@@ ') for line in lines)
        if has_hunks:
            bonus += 5

        # Penalty for excessive trailing whitespace
        trailing_whitespace = sum(1 for line in lines if line.rstrip() != line)
        bonus -= trailing_whitespace * 0.5

        return max(0, bonus)  # No negative bonus

    def _evaluate_safety(self, diff: str) -> float:
        """Evaluate patch safety"""
        score = 0

        # Dangerous patterns to avoid
        dangerous_patterns = [
            r'def __del__',  # Custom destructors can be dangerous
            r'import os.*system',  # System calls
            r'import subprocess',  # Subprocess calls
            r'eval\s*\(',  # Eval usage
            r'exec\s*\(',  # Exec usage
            r'rm\s+-rf',  # Dangerous file operations
            r'shutil\.rmtree',  # Dangerous directory operations
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, diff, re.IGNORECASE):
                score -= 20  # Heavy penalty for dangerous patterns

        # Bonus for safe patterns
        safe_patterns = [
            r'try:',  # Error handling
            r'except\s+\w+:',  # Specific exception handling
            r'with\s+.*:',  # Context managers
        ]

        for pattern in safe_patterns:
            if re.search(pattern, diff):
                score += 2

        return score

    def _calculate_confidence(self, score: float, tests_passed: bool, attempts: int) -> float:
        """Calculate confidence in patch quality"""
        base_confidence = 0.5

        # Tests passing is most important
        if tests_passed:
            base_confidence += 0.3
        else:
            base_confidence -= 0.2

        # Score contributes
        score_contribution = max(-0.2, min(0.2, score / 100))
        base_confidence += score_contribution

        # Attempts penalty
        attempt_penalty = (attempts - 1) * 0.1
        base_confidence -= attempt_penalty

        return max(0.0, min(1.0, base_confidence))

    def _generate_analysis(self, score: float, reasons: list) -> str:
        """Generate human-readable analysis"""
        analysis_parts = []

        if score > 80:
            analysis_parts.append("High-quality patch")
        elif score > 50:
            analysis_parts.append("Moderate quality patch")
        elif score > 0:
            analysis_parts.append("Low quality patch")
        else:
            analysis_parts.append("Poor quality patch")

        # Add specific insights
        for reason in reasons:
            if "tests_pass" in reason:
                analysis_parts.append("Tests pass ✓")
            elif "tests_fail" in reason:
                analysis_parts.append("Tests fail ✗")
            elif "size_penalty" in reason:
                analysis_parts.append("Large diff size")
            elif "safety_penalty" in reason:
                analysis_parts.append("Contains potentially unsafe code")

        return " | ".join(analysis_parts)

    def get_patch_history(self, limit: int = 10) -> list:
        """Get recent patch scoring history"""
        return self.patch_scores[-limit:] if self.patch_scores else []

    def get_scoring_stats(self) -> Dict[str, Any]:
        """Get patch scoring statistics"""
        if not self.patch_scores:
            return {"total_patches": 0}

        scores = [p["score"] for p in self.patch_scores]
        tests_passed = sum(1 for p in self.patch_scores if p["tests_passed"])

        return {
            "total_patches": len(self.patch_scores),
            "average_score": sum(scores) / len(scores),
            "tests_pass_rate": tests_passed / len(self.patch_scores),
            "highest_score": max(scores),
            "lowest_score": min(scores)
        }

# Global patch scorer instance
patch_scorer = PatchScorer()

# Convenience functions
def score_patch(diff: str, tests_passed: bool, attempts: int,
               cluster_hits: int, execution_time: Optional[float] = None) -> Dict[str, Any]:
    """Score a patch"""
    return patch_scorer.score_patch(diff, tests_passed, attempts, cluster_hits, execution_time)

def get_patch_history(limit: int = 10) -> list:
    """Get patch history"""
    return patch_scorer.get_patch_history(limit)

def get_scoring_stats() -> Dict[str, Any]:
    """Get scoring statistics"""
    return patch_scorer.get_scoring_stats()

if __name__ == "__main__":
    # Test patch scoring
    print("🧪 Testing Patch Scoring System")

    # Test patches
    test_patches = [
        ("--- a/test.py\n+++ b/test.py\n@@ -1,1 +1,2 @@\nprint('hello')\nprint('world')", True, 1, 0),
        ("--- a/test.py\n+++ b/test.py\n@@ -1,1 +1,10 @@\nprint('hello')\n# Lots of changes\n" + "\nprint('end')\n" * 8, False, 3, 2),
    ]

    for diff, tests_passed, attempts, cluster_hits in test_patches:
        result = score_patch(diff, tests_passed, attempts, cluster_hits)
        print(f"Score: {result['score']:.1f} | Confidence: {result['confidence']:.2f} | {result['analysis']}")

    # Get stats
    stats = get_scoring_stats()
    print(f"Scoring stats: {stats}")

    print("✅ Patch scoring test complete")
