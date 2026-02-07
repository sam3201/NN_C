#!/usr/bin/env python3
"""
SAM 2.0 Teacher Loop System
Enables learning from successful and failed fixes
"""

import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from memory_system import add_memory
import random

logger = logging.getLogger(__name__)

class TeacherLoop:
    """Teacher loop for learning from fix outcomes"""

    def __init__(self, training_file="memory/training_data.jsonl"):
        self.training_file = training_file
        self.examples = []

        # Create memory directory
        os.makedirs("memory", exist_ok=True)

        # Load existing training data
        self._load_training_data()

    def _load_training_data(self):
        """Load existing training examples"""
        try:
            if os.path.exists(self.training_file):
                with open(self.training_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            self.examples.append(json.loads(line.strip()))
                logger.info(f"✅ Loaded {len(self.examples)} training examples")
            else:
                logger.info("✅ Created new teacher loop system")
        except Exception as e:
            logger.error(f"❌ Failed to load training data: {e}")
            self.examples = []

    def _save_example(self, example: Dict[str, Any]):
        """Save a training example"""
        try:
            with open(self.training_file, 'a') as f:
                f.write(json.dumps(example) + '\n')
            logger.debug("Saved training example")
        except Exception as e:
            logger.error(f"❌ Failed to save training example: {e}")

    def record_fix_outcome(self, error_context: str, patch: str, success: bool,
                          tests_passed: bool, execution_time: float = None,
                          metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Record the outcome of a fix attempt

        Args:
            error_context: Description of the original error
            patch: The patch that was applied
            success: Whether the fix was successful
            tests_passed: Whether tests passed after the fix
            execution_time: How long it took to generate the fix
            metadata: Additional metadata about the fix

        Returns:
            Dict with recording results
        """
        timestamp = datetime.now().isoformat()

        example = {
            "input": error_context,
            "output": patch,
            "success": success,
            "tests_passed": tests_passed,
            "timestamp": timestamp,
            "execution_time": execution_time,
            "metadata": metadata or {}
        }

        # Save to training data
        self.examples.append(example)
        self._save_example(example)

        # Add to memory system for retrieval
        memory_text = f"{'SUCCESSFUL' if success else 'FAILED'} fix: {error_context[:100]}..."
        memory_metadata = {
            "type": "fix_outcome",
            "success": success,
            "tests_passed": tests_passed,
            "patch_size": len(patch.splitlines()),
            "timestamp": timestamp
        }

        if execution_time:
            memory_metadata["execution_time"] = execution_time

        add_memory(memory_text, memory_metadata)

        # Update learning models
        self._update_learning_models(example)

        result = {
            "recorded": True,
            "total_examples": len(self.examples),
            "success_rate": self._calculate_success_rate(),
            "learning_insights": self._extract_insights(example)
        }

        logger.info(f"Recorded fix outcome: {'SUCCESS' if success else 'FAILURE'} "
                   f"(total examples: {len(self.examples)})")

        return result

    def _update_learning_models(self, example: Dict[str, Any]):
        """Update internal learning models based on new example"""
        # This is where we could implement more sophisticated learning
        # For now, we just track patterns

        if example["success"]:
            # Learn from successful patterns
            self._learn_from_success(example)
        else:
            # Learn from failures to avoid them
            self._learn_from_failure(example)

    def _learn_from_success(self, example: Dict[str, Any]):
        """Learn patterns from successful fixes"""
        # Extract successful fix patterns
        patch = example["output"]
        error = example["input"]

        # Simple pattern learning - could be expanded
        if "NameError" in error and "undefined" in error:
            logger.debug("Learned: NameError fixes often involve variable definitions")

        if "ImportError" in error:
            logger.debug("Learned: ImportError fixes often involve import statements")

    def _learn_from_failure(self, example: Dict[str, Any]):
        """Learn from failed fixes to avoid similar mistakes"""
        # Extract failure patterns
        patch = example["output"]
        error = example["input"]

        # Track what doesn't work
        if len(patch.splitlines()) > 20:  # Large patches often fail
            logger.debug("Learned: Large patches are risky")

    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        if not self.examples:
            return 0.0

        successful = sum(1 for ex in self.examples if ex["success"])
        return successful / len(self.examples)

    def _extract_insights(self, example: Dict[str, Any]) -> List[str]:
        """Extract insights from a new example"""
        insights = []

        # Success insights
        if example["success"]:
            patch_size = len(example["output"].splitlines())
            if patch_size <= 5:
                insights.append("Small, targeted fixes are more likely to succeed")
            if example["tests_passed"]:
                insights.append("Fixes that pass tests are reliable indicators of success")

        # Failure insights
        else:
            if not example["tests_passed"]:
                insights.append("Fixes that break tests should be avoided")
            patch_size = len(example["output"].splitlines())
            if patch_size > 15:
                insights.append("Large patches are more likely to fail")

        return insights

    def get_similar_fixes(self, error_context: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get similar past fixes for the current error

        Args:
            error_context: Current error description
            limit: Max number of similar fixes to return

        Returns:
            List of similar fix examples
        """
        # Simple similarity - could be improved with embeddings
        similar = []

        for example in self.examples[-100:]:  # Look at recent examples
            similarity_score = self._calculate_similarity(error_context, example["input"])
            if similarity_score > 0.3:  # Similarity threshold
                example_with_score = example.copy()
                example_with_score["similarity_score"] = similarity_score
                similar.append(example_with_score)

        # Sort by similarity and return top matches
        similar.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similar[:limit]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts"""
        # Very basic similarity - count common words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def get_few_shot_examples(self, error_context: str, max_examples: int = 3) -> List[Dict[str, Any]]:
        """
        Get few-shot examples for prompting

        Args:
            error_context: Current error context
            max_examples: Maximum examples to return

        Returns:
            List of examples with input/output pairs
        """
        similar_fixes = self.get_similar_fixes(error_context, max_examples * 2)

        # Prioritize successful examples
        successful = [ex for ex in similar_fixes if ex["success"]]
        failed = [ex for ex in similar_fixes if not ex["success"]]

        # Return mix of successful and failed examples
        examples = successful[:max_examples//2] + failed[:max_examples//2]

        # Format for few-shot prompting
        formatted_examples = []
        for ex in examples:
            formatted_examples.append({
                "input": ex["input"],
                "output": ex["output"],
                "success": ex["success"],
                "similarity": ex.get("similarity_score", 0)
            })

        return formatted_examples

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        if not self.examples:
            return {"status": "no_training_data"}

        successful = sum(1 for ex in self.examples if ex["success"])
        tests_passed = sum(1 for ex in self.examples if ex["tests_passed"])

        # Calculate recent performance (last 10 examples)
        recent = self.examples[-10:]
        recent_success = sum(1 for ex in recent if ex["success"]) if recent else 0

        return {
            "total_examples": len(self.examples),
            "success_rate": successful / len(self.examples),
            "tests_pass_rate": tests_passed / len(self.examples),
            "recent_success_rate": recent_success / len(recent) if recent else 0,
            "average_patch_size": sum(len(ex["output"].splitlines()) for ex in self.examples) / len(self.examples),
            "learning_insights": self._get_recent_insights(5)
        }

    def _get_recent_insights(self, limit: int) -> List[str]:
        """Get recent learning insights"""
        recent = self.examples[-limit:] if self.examples else []
        insights = []

        for example in recent:
            insights.extend(self._extract_insights(example))

        # Remove duplicates and return unique insights
        return list(set(insights))

# Global teacher loop instance
teacher_loop = TeacherLoop()

# Convenience functions
def record_fix_outcome(error_context: str, patch: str, success: bool,
                      tests_passed: bool, execution_time: float = None,
                      metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Record a fix outcome"""
    return teacher_loop.record_fix_outcome(error_context, patch, success, tests_passed, execution_time, metadata)

def get_similar_fixes(error_context: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Get similar past fixes"""
    return teacher_loop.get_similar_fixes(error_context, limit)

def get_few_shot_examples(error_context: str, max_examples: int = 3) -> List[Dict[str, Any]]:
    """Get few-shot examples"""
    return teacher_loop.get_few_shot_examples(error_context, max_examples)

def get_learning_stats() -> Dict[str, Any]:
    """Get learning statistics"""
    return teacher_loop.get_learning_stats()

if __name__ == "__main__":
    # Test teacher loop
    print("🧪 Testing Teacher Loop System")

    # Record some test outcomes
    test_outcomes = [
        ("NameError: undefined_var", "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-undefined_var\n+defined_var", True, True),
        ("ImportError: no numpy", "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-# no import\n+import numpy", False, False),
        ("ValueError: invalid input", "--- a/test.py\n+++ b/test.py\n@@ -5 +5 @@\n-input_val\n+validated_input", True, True),
    ]

    for error, patch, success, tests_passed in test_outcomes:
        result = record_fix_outcome(error, patch, success, tests_passed)
        print(f"Recorded: {'SUCCESS' if success else 'FAILURE'} - {result['total_examples']} total")

    # Test retrieval
    similar = get_similar_fixes("NameError: undefined variable")
    print(f"Similar fixes found: {len(similar)}")

    # Get few-shot examples
    examples = get_few_shot_examples("ImportError: missing module")
    print(f"Few-shot examples: {len(examples)}")

    # Get stats
    stats = get_learning_stats()
    print(f"Learning stats: {stats}")

    print("✅ Teacher loop test complete")
