#!/usr/bin/env python3
"""
SAM 2.0 Memory System - SELF-RAG Integration
Provides long-term memory for the self-repairing AGI system
"""

import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SAMMemorySystem:
    """SELF-RAG memory system for SAM 2.0"""

    def __init__(self, index_file="memory/faiss_index.idx", store_file="memory/memory_store.json"):
        self.index_file = index_file
        self.store_file = store_file
        self.dimension = 384  # all-MiniLM-L6-v2 dimension

        # Create memory directory
        os.makedirs("memory", exist_ok=True)

        # Initialize sentence transformer
        try:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Memory encoder initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize memory encoder: {e}")
            self.encoder = None

        # Initialize FAISS index and store
        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing index or create new one"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.store_file):
                # Load existing index
                self.index = faiss.read_index(self.index_file)
                with open(self.store_file, 'r') as f:
                    self.store = json.load(f)
                logger.info(f"✅ Loaded existing memory: {len(self.store)} items")
            else:
                # Create new index
                self.index = faiss.IndexFlatL2(self.dimension)
                self.store = []
                logger.info("✅ Created new memory index")
        except Exception as e:
            logger.error(f"❌ Failed to load/create memory index: {e}")
            # Create fresh index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.store = []

    def add_memory(self, text: str, metadata: Dict[str, Any]):
        """Add text and metadata to memory"""
        if not self.encoder:
            logger.warning("Memory encoder not available")
            return

        try:
            # Encode text
            embedding = self.encoder.encode([text])[0]

            # Add to FAISS index
            self.index.add(np.array([embedding], dtype=np.float32))

            # Add to store with timestamp
            memory_item = {
                "text": text,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat(),
                "embedding_shape": list(embedding.shape) if hasattr(embedding, 'shape') else None
            }
            self.store.append(memory_item)

            # Save periodically (every 10 items)
            if len(self.store) % 10 == 0:
                self._save_memory()

            logger.debug(f"Added memory item: {text[:50]}...")

        except Exception as e:
            logger.error(f"❌ Failed to add memory: {e}")

    def query_memory(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Query memory for similar items"""
        if not self.encoder or len(self.store) == 0:
            return []

        try:
            # Encode query
            query_embedding = self.encoder.encode([query])[0]

            # Search index
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32), k
            )

            # Return relevant memories
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.store) and idx >= 0:
                    memory_item = self.store[idx].copy()
                    memory_item["similarity_score"] = float(distances[0][i])
                    results.append(memory_item)

            return results

        except Exception as e:
            logger.error(f"❌ Failed to query memory: {e}")
            return []

    def _save_memory(self):
        """Save memory to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)

            # Save store
            with open(self.store_file, 'w') as f:
                json.dump(self.store, f, indent=2)

            logger.debug(f"Saved memory: {len(self.store)} items")

        except Exception as e:
            logger.error(f"❌ Failed to save memory: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            "total_memories": len(self.store),
            "index_size": self.index.ntotal if hasattr(self.index, 'ntotal') else 0,
            "dimension": self.dimension,
            "encoder_available": self.encoder is not None,
            "memory_types": self._get_memory_type_counts()
        }

    def _get_memory_type_counts(self) -> Dict[str, int]:
        """Count memories by type"""
        counts = {}
        for item in self.store:
            mem_type = item.get("metadata", {}).get("type", "unknown")
            counts[mem_type] = counts.get(mem_type, 0) + 1
        return counts

    def clear_memory(self):
        """Clear all memory (for testing/reset)"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.store = []
        self._save_memory()
        logger.info("Memory cleared")

# Global memory instance
memory_system = SAMMemorySystem()

# Convenience functions
def add_memory(text: str, metadata: Dict[str, Any]):
    """Add item to memory"""
    memory_system.add_memory(text, metadata)

def query_memory(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Query memory for similar items"""
    return memory_system.query_memory(query, k)

def get_memory_stats() -> Dict[str, Any]:
    """Get memory statistics"""
    return memory_system.get_memory_stats()

if __name__ == "__main__":
    # Test memory system
    print("🧪 Testing SAM Memory System")

    # Add test memories
    test_memories = [
        ("ImportError: No module named 'numpy'", {"type": "failure", "files": ["test.py"]}),
        ("Successfully fixed division by zero", {"type": "success", "files": ["math.py"]}),
        ("Added error handling for file operations", {"type": "patch", "files": ["io.py"]})
    ]

    for text, metadata in test_memories:
        add_memory(text, metadata)

    # Query memory
    results = query_memory("import error", k=2)
    print(f"Query results: {len(results)} items")

    # Get stats
    stats = get_memory_stats()
    print(f"Memory stats: {stats}")

    print("✅ Memory system test complete")
