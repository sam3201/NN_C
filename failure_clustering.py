#!/usr/bin/env python3
"""
SAM 2.0 Failure Clustering System
Learns from recurring bugs to prevent infinite retry loops
"""

import os
import json
import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from memory_system import add_memory
import logging

logger = logging.getLogger(__name__)

class FailureClustering:
    """Clusters failures to learn from recurring bugs"""

    def __init__(self, cluster_file="memory/failure_clusters.json"):
        self.cluster_file = cluster_file
        self.failure_clusters = defaultdict(list)
        self.cluster_patterns = {}

        # Create memory directory
        os.makedirs("memory", exist_ok=True)

        # Load existing clusters
        self._load_clusters()

    def _load_clusters(self):
        """Load existing failure clusters"""
        try:
            if os.path.exists(self.cluster_file):
                with open(self.cluster_file, 'r') as f:
                    data = json.load(f)
                    self.failure_clusters = defaultdict(list, data.get('clusters', {}))
                    self.cluster_patterns = data.get('patterns', {})
                logger.info(f"✅ Loaded {len(self.failure_clusters)} failure clusters")
            else:
                logger.info("✅ Created new failure clustering system")
        except Exception as e:
            logger.error(f"❌ Failed to load failure clusters: {e}")
            self.failure_clusters = defaultdict(list)
            self.cluster_patterns = {}

    def _save_clusters(self):
        """Save failure clusters to disk"""
        try:
            data = {
                'clusters': dict(self.failure_clusters),
                'patterns': self.cluster_patterns
            }
            with open(self.cluster_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved failure clusters")
        except Exception as e:
            logger.error(f"❌ Failed to save failure clusters: {e}")

    def record_failure(self, error_type: str, traceback: str, files: List[str]) -> str:
        """
        Record a failure and return cluster ID

        Args:
            error_type: Type of error (e.g., "NameError", "ImportError")
            traceback: Full error traceback
            files: List of files involved

        Returns:
            str: Cluster ID for this failure
        """
        # Create cluster key
        sorted_files = tuple(sorted(files))
        cluster_key = (error_type, sorted_files)

        # Create cluster ID (hash of key)
        cluster_id = hashlib.md5(str(cluster_key).encode()).hexdigest()[:8]

        # Add to cluster
        failure_entry = {
            "traceback": traceback,
            "timestamp": json.dumps(None),  # Will be set by memory system
            "cluster_id": cluster_id
        }

        self.failure_clusters[cluster_key].append(failure_entry)

        # Learn pattern if this is a recurring failure
        cluster_size = len(self.failure_clusters[cluster_key])
        if cluster_size > 1:
            self._learn_pattern(cluster_key, error_type, files)

        # Add to memory system
        memory_text = f"{error_type} in {', '.join(files)}: {traceback[:200]}..."
        add_memory(memory_text, {
            "type": "failure",
            "error_type": error_type,
            "files": files,
            "cluster_id": cluster_id,
            "cluster_size": cluster_size
        })

        # Save periodically
        if len(self.failure_clusters) % 5 == 0:
            self._save_clusters()

        logger.info(f"Recorded failure: {error_type} in {files} (cluster: {cluster_id}, size: {cluster_size})")
        return cluster_id

    def _learn_pattern(self, cluster_key: Tuple, error_type: str, files: List[str]):
        """Learn patterns from recurring failures"""
        cluster_failures = self.failure_clusters[cluster_key]

        if len(cluster_failures) >= 3:  # Need multiple instances to learn pattern
            pattern_key = f"{error_type}_{'_'.join(sorted(files))}"

            # Simple pattern learning - most common error locations
            error_lines = []
            for failure in cluster_failures[-5:]:  # Look at last 5 failures
                traceback = failure["traceback"]
                # Extract line numbers from traceback
                import re
                lines = re.findall(r'line (\d+)', traceback)
                error_lines.extend([int(line) for line in lines])

            if error_lines:
                # Find most common error line
                from collections import Counter
                most_common_line = Counter(error_lines).most_common(1)[0][0]

                self.cluster_patterns[pattern_key] = {
                    "error_type": error_type,
                    "files": files,
                    "most_common_line": most_common_line,
                    "occurrences": len(cluster_failures),
                    "pattern": "recurring_error_same_line"
                }

                logger.info(f"Learned pattern: {pattern_key} (line {most_common_line})")

    def get_cluster_hits(self, error_type: str, files: List[str]) -> int:
        """
        Get how many times this failure cluster has occurred

        Args:
            error_type: Type of error
            files: Files involved

        Returns:
            int: Number of times this cluster has failed
        """
        sorted_files = tuple(sorted(files))
        cluster_key = (error_type, sorted_files)

        return len(self.failure_clusters.get(cluster_key, []))

    def get_similar_failures(self, error_type: str, files: List[str], limit: int = 3) -> List[Dict]:
        """
        Get similar past failures for context

        Args:
            error_type: Type of error
            files: Files involved
            limit: Max number of similar failures to return

        Returns:
            List[Dict]: Similar failure entries
        """
        hits = self.get_cluster_hits(error_type, files)
        if hits == 0:
            return []

        sorted_files = tuple(sorted(files))
        cluster_key = (error_type, sorted_files)

        # Return recent failures from this cluster
        cluster_failures = self.failure_clusters[cluster_key]
        return cluster_failures[-limit:] if cluster_failures else []

    def should_skip_retry(self, error_type: str, files: List[str], max_attempts: int = 3) -> bool:
        """
        Determine if we should skip retrying this failure (too many attempts)

        Args:
            error_type: Type of error
            files: Files involved
            max_attempts: Maximum attempts before giving up

        Returns:
            bool: True if should skip retry
        """
        hits = self.get_cluster_hits(error_type, files)
        return hits >= max_attempts

    def get_failure_stats(self) -> Dict:
        """Get failure clustering statistics"""
        total_failures = sum(len(cluster) for cluster in self.failure_clusters.values())
        total_clusters = len(self.failure_clusters)

        # Find most common failure types
        error_counts = defaultdict(int)
        for (error_type, _), failures in self.failure_clusters.items():
            error_counts[error_type] += len(failures)

        most_common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_failures": total_failures,
            "total_clusters": total_clusters,
            "most_common_errors": most_common_errors,
            "learned_patterns": len(self.cluster_patterns)
        }

# Global failure clustering instance
failure_clusterer = FailureClustering()

# Convenience functions
def record_failure(error_type: str, traceback: str, files: List[str]) -> str:
    """Record a failure and return cluster ID"""
    return failure_clusterer.record_failure(error_type, traceback, files)

def get_cluster_hits(error_type: str, files: List[str]) -> int:
    """Get failure cluster hit count"""
    return failure_clusterer.get_cluster_hits(error_type, files)

def get_similar_failures(error_type: str, files: List[str], limit: int = 3) -> List[Dict]:
    """Get similar past failures"""
    return failure_clusterer.get_similar_failures(error_type, files, limit)

def should_skip_retry(error_type: str, files: List[str], max_attempts: int = 3) -> bool:
    """Check if should skip retry"""
    return failure_clusterer.should_skip_retry(error_type, files, max_attempts)

def get_failure_stats() -> Dict:
    """Get failure statistics"""
    return failure_clusterer.get_failure_stats()

if __name__ == "__main__":
    # Test failure clustering
    print("🧪 Testing Failure Clustering System")

    # Simulate some failures
    test_failures = [
        ("NameError", "name 'undefined_var' is not defined", ["test.py"]),
        ("ImportError", "No module named 'missing_lib'", ["utils.py"]),
        ("NameError", "name 'undefined_var' is not defined", ["test.py"]),  # Repeat
        ("ZeroDivisionError", "division by zero", ["math.py"]),
    ]

    for error_type, traceback, files in test_failures:
        cluster_id = record_failure(error_type, traceback, files)
        print(f"Recorded: {error_type} -> cluster {cluster_id}")

    # Test queries
    hits = get_cluster_hits("NameError", ["test.py"])
    print(f"NameError in test.py hits: {hits}")

    similar = get_similar_failures("NameError", ["test.py"])
    print(f"Similar failures: {len(similar)}")

    # Test retry logic
    should_skip = should_skip_retry("NameError", ["test.py"], max_attempts=2)
    print(f"Should skip retry: {should_skip}")

    # Get stats
    stats = get_failure_stats()
    print(f"Failure stats: {stats}")

    print("✅ Failure clustering test complete")
