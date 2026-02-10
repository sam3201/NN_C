#!/usr/bin/env python3
"""
Compatibility wrapper for integrated meta-agent usage.

The unified MetaAgent now contains the enhanced capabilities; this module
keeps the old IntegratedMetaAgent interface for tooling/tests.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from meta_agent_enhanced import EnhancedMetaAgent


class IntegratedMetaAgent:
    """Thin wrapper around the unified EnhancedMetaAgent adapter."""

    def __init__(self, system: Any):
        self.system = system
        self.enhanced_meta = EnhancedMetaAgent(system)
        self.integration_mode = "unified"

    def handle_failure(
        self,
        error_message: str,
        stack_trace: str,
        file_path: Optional[str] = None,
        context: str = "runtime",
    ) -> Dict[str, Any]:
        return self.enhanced_meta.handle_failure(error_message, stack_trace, file_path, context)

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        stats = self.enhanced_meta.get_statistics()
        stats["integration_mode"] = self.integration_mode
        return stats


def create_integrated_meta_agent(system: Any) -> IntegratedMetaAgent:
    """Factory for integrated meta agent."""
    return IntegratedMetaAgent(system)


if __name__ == "__main__":
    print("IntegratedMetaAgent now proxies to the unified MetaAgent in complete_sam_unified.")
