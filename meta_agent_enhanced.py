#!/usr/bin/env python3
"""
Compatibility wrapper for the unified MetaAgent implementation.

The enhanced capabilities were merged into complete_sam_unified.MetaAgent.
This module keeps the previous EnhancedMetaAgent interface for tests/tools.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from complete_sam_unified import (
    FailureEvent,
    MetaAgent,
    ObserverAgent,
    FaultLocalizerAgent,
    PatchGeneratorAgent,
    VerifierJudgeAgent,
)
from sam_code_modifier import initialize_sam_code_modifier


class EnhancedMetaAgent:
    """Backwards-compatible adapter to the unified MetaAgent."""

    def __init__(self, system: Any):
        self.system = system
        self.project_root = getattr(system, "project_root", Path("."))
        try:
            initialize_sam_code_modifier(str(self.project_root))
        except Exception:
            # Safe to proceed; meta-agent can still reason even if modifier is off.
            pass

        observer = ObserverAgent(system)
        localizer = FaultLocalizerAgent(system)
        generator = PatchGeneratorAgent(system)
        verifier = VerifierJudgeAgent(system)
        self._meta = MetaAgent(observer, localizer, generator, verifier, system)

        self.successful_fixes = 0
        self.failed_attempts = 0
        self.confidence_threshold = getattr(self._meta, "confidence_threshold", 0.8)
        self.learning_enabled = True

    def _infer_error_type(self, error_message: str, stack_trace: str) -> str:
        for text in (stack_trace, error_message):
            if not text:
                continue
            match = re.search(r"([A-Za-z_]*Error)", text)
            if match:
                return match.group(1)
        return "RuntimeError"

    def handle_failure(
        self,
        error_message: str,
        stack_trace: str,
        file_path: Optional[str] = None,
        context: str = "runtime",
    ) -> Dict[str, Any]:
        error_type = self._infer_error_type(error_message or "", stack_trace or "")
        if file_path and file_path not in (stack_trace or ""):
            stack_trace = f'File "{file_path}", line 1\n' + (stack_trace or "")

        failure_event = FailureEvent(
            error_type=error_type,
            stack_trace=stack_trace or "",
            timestamp=datetime.now().isoformat(),
            severity="medium",
            context=context,
            message=error_message or "",
        )

        patches = []
        try:
            patches = self._meta._deterministic_patches(failure_event)
        except Exception:
            patches = []

        strategy_count = 0
        try:
            matches = self._meta._match_error_patterns(failure_event)
            for match in matches:
                strategy_count += len(self._meta.fix_strategies.get(match.get("type"), []))
        except Exception:
            strategy_count = 0

        total_fixes_generated = max(len(patches), strategy_count, 1)

        try:
            success = bool(self._meta.handle_failure(failure_event))
        except Exception:
            success = False

        if success:
            self.successful_fixes += 1
        else:
            self.failed_attempts += 1

        if success:
            confidence = max(
                [p.get("confidence", 0.0) for p in patches] or [self.confidence_threshold]
            )
        else:
            confidence = 0.0

        return {
            "status": "success" if success else "failed",
            "confidence": confidence,
            "total_fixes_generated": total_fixes_generated,
            "meta_agent_used": "unified",
        }

    def get_statistics(self) -> Dict[str, Any]:
        error_patterns = getattr(self._meta, "error_patterns", {}) or {}
        fix_strategies = getattr(self._meta, "fix_strategies", {}) or {}
        return {
            "successful_fixes": self.successful_fixes,
            "failed_attempts": self.failed_attempts,
            "success_rate": self.successful_fixes / max(1, self.successful_fixes + self.failed_attempts),
            "error_patterns_loaded": sum(len(v) for v in error_patterns.values()),
            "fix_strategies_available": sum(len(v) for v in fix_strategies.values()),
            "confidence_threshold": self.confidence_threshold,
            "learning_enabled": self.learning_enabled,
        }


def create_enhanced_meta_agent(system: Any) -> EnhancedMetaAgent:
    """Create enhanced meta agent instance (compat wrapper)."""
    return EnhancedMetaAgent(system)

