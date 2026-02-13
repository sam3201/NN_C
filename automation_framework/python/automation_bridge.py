#!/usr/bin/env python3
"""
Automation Framework - Python Bridge
High-performance Rust core with Python bindings for flexibility
Features dynamic model routing, tri-cameral governance, and concurrent subagents
"""

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Callable, Union
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

class Branch(Enum):
    CIC = "Constructive Intelligence Core"
    AEE = "Adversarial Exploration Engine"
    CSF = "Coherence Stabilization Field"

class Phase(Enum):
    PLANNING = "planning"
    ANALYSIS = "analysis"
    BUILDING = "building"
    TESTING = "testing"
    VERIFICATION = "verification"
    COMPLETE = "complete"

@dataclass
class WorkflowConfig:
    """Configuration for a workflow"""
    name: str
    description: str = ""
    high_level_plan: str = ""
    low_level_plan: str = ""
    hard_constraints: List[str] = field(default_factory=list)
    soft_constraints: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    risk_level: float = 0.5

@dataclass
class GovernanceDecision:
    """Decision from tri-cameral governance"""
    proceed: bool
    confidence: float
    cic_vote: Dict[str, Any]
    aee_vote: Dict[str, Any]
    csf_vote: Dict[str, Any]
    concerns: List[str]
    recommendations: List[str]

@dataclass
class SubagentResult:
    """Result from a subagent"""
    id: str
    success: bool
    output: str
    execution_time_ms: int
    branch: Optional[Branch] = None

class TriCameralOrchestrator:
    """Python wrapper for tri-cameral governance"""
    
    def __init__(self):
        self.decision_history = []
    
    async def evaluate_workflow(self, config: WorkflowConfig) -> GovernanceDecision:
        """Run tri-cameral evaluation on a workflow"""
        
        # Simulate tri-cameral voting
        cic_vote = self._cic_evaluate(config)
        aee_vote = self._aee_evaluate(config)
        csf_vote = self._csf_evaluate(config)
        
        # Apply decision matrix
        proceed, concerns, recommendations = self._apply_decision_matrix(
            cic_vote, aee_vote, csf_vote, config
        )
        
        confidence = (cic_vote['confidence'] + aee_vote['confidence'] + csf_vote['confidence']) / 3
        
        decision = GovernanceDecision(
            proceed=proceed,
            confidence=confidence,
            cic_vote=cic_vote,
            aee_vote=aee_vote,
            csf_vote=csf_vote,
            concerns=concerns,
            recommendations=recommendations
        )
        
        self.decision_history.append(asdict(decision))
        return decision
    
    def _cic_evaluate(self, config: WorkflowConfig) -> Dict:
        """CIC (Constructive) - Optimistic evaluation"""
        confidence = 0.8
        reasoning = f"CIC: {config.name} enables growth and expansion."
        
        if config.risk_level < 0.5:
            confidence += 0.1
            reasoning += " Low risk profile."
        
        return {
            'branch': 'CIC',
            'decision': 'Approve' if confidence > 0.7 else 'Conditional',
            'reasoning': reasoning,
            'confidence': min(confidence, 0.95)
        }
    
    def _aee_evaluate(self, config: WorkflowConfig) -> Dict:
        """AEE (Adversarial) - Pessimistic evaluation"""
        concerns = []
        confidence = 0.5
        
        if config.risk_level > 0.6:
            concerns.append(f"High risk level: {config.risk_level}")
            confidence -= 0.2
        
        if not config.invariants:
            concerns.append("No invariants specified")
            confidence -= 0.15
        
        return {
            'branch': 'AEE',
            'decision': 'Approve' if confidence > 0.7 else 'Conditional' if confidence > 0.4 else 'Reject',
            'reasoning': f"AEE: {'; '.join(concerns) if concerns else 'No major concerns'}",
            'confidence': max(confidence, 0.0)
        }
    
    def _csf_evaluate(self, config: WorkflowConfig) -> Dict:
        """CSF (Coherence) - Neutral validation"""
        violations = []
        
        for invariant in config.invariants:
            if 'violation' in invariant.lower():
                violations.append(invariant)
        
        confidence = 0.9 if not violations else max(0.0, 0.9 - len(violations) * 0.3)
        
        return {
            'branch': 'CSF',
            'decision': 'Approve' if not violations else 'Reject',
            'reasoning': f"CSF: {'All invariants preserved' if not violations else f'{len(violations)} violations'}",
            'confidence': confidence
        }
    
    def _apply_decision_matrix(self, cic, aee, csf, config) -> tuple:
        """Apply tri-cameral decision matrix"""
        concerns = []
        recommendations = []
        
        # Simplified decision logic
        approve_count = sum(1 for v in [cic, aee, csf] if v['decision'] == 'Approve')
        reject_count = sum(1 for v in [cic, aee, csf] if v['decision'] == 'Reject')
        
        if reject_count > 0:
            proceed = False
            concerns.append("One or more branches rejected")
        elif approve_count >= 2:
            proceed = True
            recommendations.append("Consensus achieved")
        else:
            proceed = True
            recommendations.append("Proceed with caution")
        
        return proceed, concerns, recommendations

class SubagentPool:
    """Manage concurrent subagents"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.active_subagents = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
    
    def spawn_parallel(self, tasks: List[Dict], handler: Callable) -> List[SubagentResult]:
        """Spawn multiple subagents in parallel"""
        futures = []
        
        for i, task in enumerate(tasks):
            future = self.executor.submit(self._execute_subagent, i, task, handler)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(SubagentResult(
                    id=f"subagent_error",
                    success=False,
                    output=str(e),
                    execution_time_ms=0
                ))
        
        return results
    
    def spawn_pipeline(self, task: Dict, reader, processor, writer) -> SubagentResult:
        """Spawn subagents in a pipeline (Reader → Processor → Writer)"""
        start_time = time.time()
        
        # Stage 1: Reader
        context = reader(task)
        
        # Stage 2: Processor
        processed = processor(context)
        
        # Stage 3: Writer
        result = writer(processed)
        
        execution_time = int((time.time() - start_time) * 1000)
        
        return SubagentResult(
            id=f"pipeline_{int(time.time())}",
            success=True,
            output=str(result),
            execution_time_ms=execution_time
        )
    
    def _execute_subagent(self, idx: int, task: Dict, handler: Callable) -> SubagentResult:
        """Execute a single subagent"""
        start_time = time.time()
        
        try:
            output = handler(task)
            success = True
        except Exception as e:
            output = str(e)
            success = False
        
        execution_time = int((time.time() - start_time) * 1000)
        
        return SubagentResult(
            id=f"subagent_{idx}_{int(time.time())}",
            success=success,
            output=str(output),
            execution_time_ms=execution_time
        )

class CyclicWorkflow:
    """Cyclic development workflow executor"""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.phases = [Phase.PLANNING, Phase.ANALYSIS, Phase.BUILDING, Phase.ANALYSIS, Phase.TESTING, Phase.VERIFICATION]
        self.current_phase_idx = 0
        self.orchestrator = TriCameralOrchestrator()
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the cyclic workflow"""
        results = []
        
        while self.current_phase_idx < len(self.phases):
            phase = self.phases[self.current_phase_idx]
            
            if phase == Phase.ANALYSIS:
                # Analysis gate - use tri-cameral governance
                decision = await self.orchestrator.evaluate_workflow(self.config)
                results.append({
                    'phase': phase.value,
                    'decision': asdict(decision)
                })
                
                if not decision.proceed:
                    # Go back to previous phase
                    if self.current_phase_idx > 0:
                        self.current_phase_idx -= 1
                    continue
            else:
                # Execute phase
                result = await self._execute_phase(phase)
                results.append({
                    'phase': phase.value,
                    'result': result
                })
            
            self.current_phase_idx += 1
        
        return {
            'success': True,
            'phases_completed': len(results),
            'results': results
        }
    
    async def _execute_phase(self, phase: Phase) -> Dict:
        """Execute a single phase"""
        print(f"Executing phase: {phase.value}")
        
        # In real implementation, this would do actual work
        await asyncio.sleep(0.1)  # Simulate work
        
        return {
            'phase': phase.value,
            'status': 'completed',
            'timestamp': time.time()
        }

class ModelRouter:
    """Dynamic model router for automatic model selection"""
    
    def __init__(self):
        self.models = {
            'claude-3-5-sonnet': {
                'name': 'Claude 3.5 Sonnet',
                'reasoning': 0.95, 'coding': 0.92, 'creativity': 0.88,
                'analysis': 0.94, 'speed': 0.75, 'cost_per_1k': 0.003,
                'context_window': 200000, 'reliability': 0.97
            },
            'claude-3-haiku': {
                'name': 'Claude 3 Haiku',
                'reasoning': 0.80, 'coding': 0.78, 'creativity': 0.75,
                'analysis': 0.82, 'speed': 0.95, 'cost_per_1k': 0.00025,
                'context_window': 200000, 'reliability': 0.90
            },
            'gpt-4': {
                'name': 'GPT-4',
                'reasoning': 0.93, 'coding': 0.90, 'creativity': 0.90,
                'analysis': 0.92, 'speed': 0.70, 'cost_per_1k': 0.03,
                'context_window': 8000, 'reliability': 0.95
            },
            'local-llm': {
                'name': 'Local LLM',
                'reasoning': 0.70, 'coding': 0.75, 'creativity': 0.65,
                'analysis': 0.72, 'speed': 0.85, 'cost_per_1k': 0.0,
                'context_window': 4000, 'reliability': 0.80
            }
        }
        self.current_model = None
        self.usage_stats = {model: {'calls': 0, 'cost': 0.0} for model in self.models}
        self.daily_budget = 100.0
        self.current_spend = 0.0
    
    def analyze_task(self, task: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Analyze task characteristics"""
        task_lower = task.lower()
        
        # Determine task type
        if any(kw in task_lower for kw in ['code', 'implement', 'function']):
            task_type = 'coding'
            complexity = 0.7
        elif any(kw in task_lower for kw in ['analyze', 'review', 'check']):
            task_type = 'analysis'
            complexity = 0.6
        elif any(kw in task_lower for kw in ['design', 'create', 'innovate']):
            task_type = 'creative'
            complexity = 0.8
        elif any(kw in task_lower for kw in ['reason', 'logic', 'deduce']):
            task_type = 'reasoning'
            complexity = 0.9
        else:
            task_type = 'general'
            complexity = 0.5
        
        # Check for safety-critical tasks
        safety_critical = any(kw in task_lower for kw in ['security', 'safety', 'invariant'])
        if safety_critical:
            complexity = min(1.0, complexity + 0.2)
        
        # Check context size
        context_size = len(context) if context else 0
        
        return {
            'type': task_type,
            'complexity': complexity,
            'context_size': context_size,
            'safety_critical': safety_critical,
            'time_sensitive': any(kw in task_lower for kw in ['quick', 'fast', 'urgent'])
        }
    
    def score_model(self, model_name: str, task_analysis: Dict) -> float:
        """Score a model for a given task"""
        model = self.models[model_name]
        task = task_analysis
        
        score = 0.0
        
        # Capability match (40%)
        if task['type'] == 'reasoning':
            score += model['reasoning'] * 0.40
        elif task['type'] == 'coding':
            score += model['coding'] * 0.40
        elif task['type'] == 'analysis':
            score += model['analysis'] * 0.40
        elif task['type'] == 'creative':
            score += model['creativity'] * 0.40
        else:
            score += 0.30
        
        # Context window fit (20%)
        if task['context_size'] <= model['context_window']:
            score += 0.20
        else:
            score += 0.05
        
        # Cost optimization (20%)
        budget_remaining = self.daily_budget - self.current_spend
        budget_ratio = budget_remaining / self.daily_budget
        
        if budget_ratio > 0.7:  # Plenty of budget
            cost_score = 1.0
        elif budget_ratio > 0.3:  # Medium budget
            cost_score = 1.0 - (model['cost_per_1k'] * 50)
        else:  # Low budget
            cost_score = 1.0 - (model['cost_per_1k'] * 200)
        
        score += max(0, cost_score) * 0.20
        
        # Speed for time-sensitive tasks (10%)
        if task['time_sensitive']:
            score += model['speed'] * 0.10
        
        # Reliability for safety-critical (10%)
        if task['safety_critical']:
            score += model['reliability'] * 0.10
        
        return score
    
    def select_model(self, task: str, context: Optional[str] = None) -> str:
        """Select best model for task"""
        analysis = self.analyze_task(task, context)
        
        # Score all models
        scores = []
        for model_name in self.models:
            score = self.score_model(model_name, analysis)
            scores.append((model_name, score))
        
        # Select best
        scores.sort(key=lambda x: x[1], reverse=True)
        best_model = scores[0][0]
        
        print(f"Selected model: {self.models[best_model]['name']} (score: {scores[best_model]:.2f})")
        return best_model
    
    def auto_switch(self, task: str, context: Optional[str] = None) -> str:
        """Automatically switch to best model"""
        new_model = self.select_model(task, context)
        
        if self.current_model != new_model:
            if self.current_model:
                print(f"Switching: {self.models[self.current_model]['name']} -> {self.models[new_model]['name']}")
            else:
                print(f"Initializing with: {self.models[new_model]['name']}")
            self.current_model = new_model
        
        return new_model
    
    def record_usage(self, model: str, tokens: int, latency_ms: int, success: bool):
        """Record model usage for tracking"""
        cost = (tokens / 1000) * self.models[model]['cost_per_1k']
        self.usage_stats[model]['calls'] += 1
        self.usage_stats[model]['cost'] += cost
        self.current_spend += cost
    
    def get_stats(self) -> Dict:
        """Get router statistics"""
        return {
            'current_model': self.current_model,
            'available_models': list(self.models.keys()),
            'current_spend': self.current_spend,
            'budget_remaining': self.daily_budget - self.current_spend,
            'usage_stats': self.usage_stats
        }

class AutomationFramework:
    """Main automation framework interface with dynamic model routing"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.orchestrator = TriCameralOrchestrator()
        self.subagent_pool = SubagentPool(
            max_concurrent=self.config.get('max_concurrent_subagents', 10)
        )
        self.model_router = ModelRouter()

    async def execute_workflow(self, workflow_config: WorkflowConfig) -> Dict[str, Any]:
        """Execute a workflow with full automation and dynamic model selection"""

        # Step 0: Dynamic model selection for this workflow
        selected_model = self.model_router.auto_switch(
            workflow_config.name,
            workflow_config.high_level_plan
        )
        print(f"Using model: {selected_model} for workflow execution")

        # Step 1: Tri-cameral governance decision
        governance_decision = await self.orchestrator.evaluate_workflow(workflow_config)

        if not governance_decision.proceed:
            return {
                'success': False,
                'reason': 'Governance rejected',
                'decision': asdict(governance_decision),
                'model_used': selected_model
            }

        # Step 2: Execute cyclic workflow
        workflow = CyclicWorkflow(workflow_config)
        workflow_result = await workflow.execute()

        # Record model usage
        self.model_router.record_usage(selected_model, 1000, 500, True)

        return {
            'success': True,
            'governance': asdict(governance_decision),
            'workflow': workflow_result,
            'model_used': selected_model,
            'model_stats': self.model_router.get_stats()
        }

    def spawn_subagents(self, tasks: List[Dict], handler: Callable) -> List[SubagentResult]:
        """Spawn parallel subagents"""
        return self.subagent_pool.spawn_parallel(tasks, handler)

    def select_model(self, task: str, context: Optional[str] = None) -> str:
        """Dynamically select best model for a task"""
        return self.model_router.select_model(task, context)

    def auto_switch_model(self, task: str, context: Optional[str] = None) -> str:
        """Automatically switch to best model"""
        return self.model_router.auto_switch(task, context)

    def get_model_stats(self) -> Dict:
        """Get model routing statistics"""
        return self.model_router.get_stats()

# Convenience functions for direct usage
def tri_cameral_cycle(task: str, high_level: str, low_level: str) -> Dict:
    """Run a tri-cameral cycle"""
    framework = AutomationFramework()
    config = WorkflowConfig(
        name=task,
        high_level_plan=high_level,
        low_level_plan=low_level
    )
    return asyncio.run(framework.execute_workflow(config))

def spawn_parallel_subagents(tasks: List[Dict], handler: Callable) -> List[SubagentResult]:
    """Spawn subagents in parallel"""
    pool = SubagentPool()
    return pool.spawn_parallel(tasks, handler)

def select_best_model(task: str, context: Optional[str] = None) -> str:
    """Select the best model for a task dynamically"""
    router = ModelRouter()
    return router.select_model(task, context)

def auto_switch_model(task: str, context: Optional[str] = None) -> str:
    """Automatically switch to best model"""
    router = ModelRouter()
    return router.auto_switch(task, context)

if __name__ == "__main__":
    # Example usage
    print("🚀 Automation Framework - Python Bridge with Dynamic Model Routing")
    print("=" * 60)

    # Example 1: Dynamic model selection for different tasks
    print("\n📊 Dynamic Model Selection Examples:")
    print("-" * 60)

    tasks = [
        ("Quick code review", "simple"),
        ("Complex architectural reasoning for distributed systems", "complex"),
        ("Safety-critical security audit", "critical"),
        ("Creative brainstorming for UI design", "creative"),
    ]

    router = ModelRouter()
    for task, complexity in tasks:
        model = router.select_model(task)
        stats = router.get_stats()
        print(f"\nTask: {task}")
        print(f"  Complexity: {complexity}")
        print(f"  Selected Model: {stats['available_models'][list(stats['available_models']).index(model)]}")

    # Example 2: Tri-cameral cycle with automatic model selection
    print("\n\n🏛️ Tri-Cameral Cycle with Dynamic Model Selection:")
    print("-" * 60)

    result = tri_cameral_cycle(
        task="Implement Phase 2",
        high_level="Add Power/Control systems",
        low_level="Create P_t and C_t classes"
    )

    print(f"\nModel Used: {result.get('model_used', 'N/A')}")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Phases Completed: {result['workflow']['phases_completed']}")

    # Example 3: Auto-switching demonstration
    print("\n\n🔄 Auto-Switching Demonstration:")
    print("-" * 60)

    sequential_tasks = [
        "Quick status check",
        "Deep analysis of system architecture",
        "Safety review of authentication code",
        "Generate creative ideas for new features"
    ]

    for i, task in enumerate(sequential_tasks, 1):
        model = auto_switch_model(task)
        print(f"{i}. Task: {task[:50]}...")
        print(f"   Model: {model}")

    print("\n✅ Automation Framework Ready!")
    print("Features:")
    print("  • Tri-cameral governance (CIC/AEE/CSF)")
    print("  • Dynamic model routing")
    print("  • Concurrent subagents")
    print("  • Resource management")
    print("  • Cyclic workflow execution")
