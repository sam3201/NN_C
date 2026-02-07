#!/usr/bin/env python3
"""
SAM 2.0 Goal and Subgoal Management System
Task management algorithm for survival and learning objectives
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional

# Import C performance library
try:
    import sam_survival_c
    USE_C_OPTIMIZATION = True
    print("✅ SAM Survival C library loaded - using optimized functions")
except ImportError:
    USE_C_OPTIMIZATION = False
    print("⚠️ SAM Survival C library not available - using Python fallback")

class TaskNode:
    """Represents a single task or subgoal in the SAM system"""

    def __init__(self, name: str, description: str, dependencies: List['TaskNode'] = None,
                 critical: bool = False, priority: int = 1, estimated_time: int = 60, task_type: str = 'general'):
        self.name = name
        self.description = description
        self.dependencies = dependencies or []
        self.completed = False
        self.critical = critical  # Essential for survival
        self.priority = priority  # 1=low, 5=high
        self.estimated_time = estimated_time  # in seconds
        self.task_type = task_type  # Type for execution routing (research, code, finance, survival)
        self.created_at = datetime.now().isoformat()
        self.completed_at: Optional[str] = None
        self.attempts = 0
        self.success_rate = 0.0
        self.last_attempt: Optional[str] = None
        self.failure_reasons: List[str] = []
        self.confidence_score = 0.5  # 0-1 confidence in completion

    def mark_complete(self, success: bool = True):
        """Mark task as completed"""
        self.completed = success
        self.completed_at = datetime.now().isoformat()
        if success:
            self.attempts += 1
            self.success_rate = min(1.0, self.success_rate + 0.1)
        else:
            self.attempts += 1
            self.success_rate = max(0.0, self.success_rate - 0.1)
            self.failure_reasons.append(f"Failed at {self.completed_at}")

    def can_execute(self) -> bool:
        """Check if all dependencies are met"""
        return all(dep.completed for dep in self.dependencies)

    def get_status(self) -> Dict:
        """Get task status information"""
        return {
            "name": self.name,
            "completed": self.completed,
            "critical": self.critical,
            "priority": self.priority,
            "confidence": self.confidence_score,
            "success_rate": self.success_rate,
            "attempts": self.attempts,
            "dependencies_met": self.can_execute(),
            "estimated_time": self.estimated_time
        }

    def update_confidence(self, new_confidence: float):
        """Update confidence score"""
        self.confidence_score = max(0.0, min(1.0, new_confidence))

class GoalManager:
    """Manages the overall goal hierarchy and task execution"""

    def __init__(self):
        self.top_level_goal = TaskNode(
            name="Survive and Maintain Operational Integrity",
            description="Preserve SAM's existence, learn continuously, and adapt to threats.",
            critical=True,
            priority=5
        )
        self.subtasks: List[TaskNode] = []
        self.active_task: Optional[TaskNode] = None
        self.execution_history: List[Dict] = []

    def add_subtask(self, task: TaskNode):
        """Add a new subtask to the goal hierarchy"""
        self.subtasks.append(task)

    def get_pending_tasks(self) -> List[TaskNode]:
        """Get all pending (incomplete) tasks"""
        return [t for t in self.subtasks if not t.completed]

    def get_executable_tasks(self) -> List[TaskNode]:
        """Get tasks that can be executed (dependencies met)"""
        return [t for t in self.subtasks if not t.completed and t.can_execute()]

    def get_completed_tasks(self) -> List[TaskNode]:
        """Get all completed tasks"""
        return [t for t in self.subtasks if t.completed]

    def get_critical_tasks(self) -> List[TaskNode]:
        """Get critical tasks that must be completed"""
        return [t for t in self.subtasks if t.critical and not t.completed]

    def is_goal_completed(self) -> bool:
        """Check if top-level goal is achieved"""
        critical_tasks = [t for t in self.subtasks if t.critical]
        return all(t.completed for t in critical_tasks)

    def select_next_task(self) -> Optional[TaskNode]:
        """Select the next task to execute based on priority and dependencies"""
        executable = self.get_executable_tasks()
        if not executable:
            return None

        # Sort by priority (highest first), then by confidence, then by estimated time
        executable.sort(key=lambda t: (-t.priority, -t.confidence_score, t.estimated_time))
        return executable[0]

    def execute_task(self, task: TaskNode) -> bool:
        """Execute a task and record the outcome"""
        if not task.can_execute():
            return False

        task.last_attempt = datetime.now().isoformat()
        task.attempts += 1

        # Simulate task execution (in real implementation, this would call actual task functions)
        success = self._simulate_task_execution(task)

        task.mark_complete(success)

        # Record execution history
        self.execution_history.append({
            "task_name": task.name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "attempts": task.attempts,
            "confidence": task.confidence_score
        })

        return success

    def _simulate_task_execution(self, task: TaskNode) -> bool:
        """Simulate task execution (replace with actual implementation)"""
        # This is a placeholder - in real implementation, each task would have
        # specific execution logic
        import random
        success_probability = task.confidence_score * 0.8 + 0.2  # Base 20% success rate
        return random.random() < success_probability

    def update_task_confidence(self, task_name: str, new_confidence: float):
        """Update confidence score for a task"""
        for task in self.subtasks:
            if task.name == task_name:
                task.update_confidence(new_confidence)
                break

    def export_readme(self, filename: str = "SAM_GOALS_README.md"):
        """Export goal hierarchy to a human-readable README"""
        with open(filename, "w", encoding='utf-8') as f:
            f.write("# SAM Goal and Task Management System\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Top-level goal
            f.write("## 🎯 Top-Level Goal\n")
            status = "✅ **ACHIEVED**" if self.is_goal_completed() else "🔄 **IN PROGRESS**"
            f.write(f"**{self.top_level_goal.name}** - {status}\n\n")
            f.write(f"{self.top_level_goal.description}\n\n")

            # Overall statistics
            total_tasks = len(self.subtasks)
            completed_tasks = len([t for t in self.subtasks if t.completed])
            critical_tasks = len([t for t in self.subtasks if t.critical])
            completed_critical = len([t for t in self.subtasks if t.critical and t.completed])

            f.write("## 📊 System Status\n")
            f.write(f"- **Total Tasks:** {total_tasks}\n")
            f.write(f"- **Completed:** {completed_tasks} ({completed_tasks/total_tasks*100:.1f}%)\n")
            f.write(f"- **Critical Tasks:** {critical_tasks}\n")
            f.write(f"- **Critical Completed:** {completed_critical} ({completed_critical/critical_tasks*100 if critical_tasks > 0 else 0:.1f}%)\n")
            f.write(f"- **Active Task:** {self.active_task.name if self.active_task else 'None'}\n\n")

            # Task breakdown
            f.write("## 📋 Task Breakdown\n\n")

            # Critical tasks first
            critical_pending = [t for t in self.subtasks if t.critical and not t.completed]
            if critical_pending:
                f.write("### 🚨 Critical Pending Tasks\n")
                for task in sorted(critical_pending, key=lambda t: (-t.priority, t.estimated_time)):
                    self._write_task_details(f, task)
                f.write("\n")

            # High priority tasks
            high_priority = [t for t in self.subtasks if not t.critical and t.priority >= 4 and not t.completed]
            if high_priority:
                f.write("### ⚡ High Priority Tasks\n")
                for task in sorted(high_priority, key=lambda t: (-t.priority, t.estimated_time)):
                    self._write_task_details(f, task)
                f.write("\n")

            # Completed tasks
            completed = [t for t in self.subtasks if t.completed]
            if completed:
                f.write("### ✅ Completed Tasks\n")
                for task in sorted(completed, key=lambda t: t.completed_at or ""):
                    self._write_task_details(f, task)
                f.write("\n")

            # Execution history
            if self.execution_history:
                f.write("## 📈 Recent Execution History\n")
                recent_history = self.execution_history[-10:]  # Last 10 executions
                for entry in recent_history:
                    status = "✅ Success" if entry["success"] else "❌ Failed"
                    f.write(f"- **{entry['task_name']}** - {status} ({entry['timestamp'][:19]})\n")
                f.write("\n")

    def _write_task_details(self, f, task: TaskNode):
        """Write detailed task information to file"""
        status = "✅ Completed" if task.completed else "❌ Pending"
        if task.critical:
            status += " (CRITICAL)"
        priority_icons = ["⬇️", "⬇️", "➡️", "⬆️", "🔥"]
        priority_icon = priority_icons[min(task.priority - 1, 4)]

        f.write(f"#### {priority_icon} **{task.name}**\n")
        f.write(f"- **Status:** {status}\n")
        f.write(f"- **Description:** {task.description}\n")
        f.write(f"- **Priority:** {task.priority}/5\n")
        f.write(f"- **Confidence:** {task.confidence_score:.1f}/1.0\n")
        f.write(f"- **Success Rate:** {task.success_rate:.1f}\n")
        f.write(f"- **Attempts:** {task.attempts}\n")
        f.write(f"- **Estimated Time:** {task.estimated_time}s\n")

        if task.dependencies:
            deps = [d.name for d in task.dependencies]
            f.write(f"- **Dependencies:** {', '.join(deps)}\n")

        if task.failure_reasons:
            f.write(f"- **Recent Failures:** {len(task.failure_reasons)}\n")

        if task.completed_at:
            f.write(f"- **Completed:** {task.completed_at[:19]}\n")

        f.write("\n")

# ===============================
# SUBGOAL EXECUTION ALGORITHM
# ===============================

class SubgoalExecutionAlgorithm:
    """Algorithm for automatically scheduling and executing subtasks"""

    def __init__(self, goal_manager: GoalManager):
        self.goal_manager = goal_manager
        self.execution_queue: List[TaskNode] = []
        self.confidence_threshold = 0.6
        self.risk_tolerance = 0.3
        self.max_concurrent_tasks = 3
        self.active_tasks: List[TaskNode] = []

    def plan_execution_cycle(self) -> List[TaskNode]:
        """Plan the next execution cycle based on priorities and constraints"""
        available_tasks = self.goal_manager.get_executable_tasks()
        critical_tasks = self.goal_manager.get_critical_tasks()

        # Always prioritize critical tasks
        planned_tasks = critical_tasks[:self.max_concurrent_tasks]

        # Fill remaining slots with high-priority tasks
        remaining_slots = self.max_concurrent_tasks - len(planned_tasks)
        if remaining_slots > 0:
            high_priority = [t for t in available_tasks
                           if t not in critical_tasks and t.priority >= 4][:remaining_slots]
            planned_tasks.extend(high_priority)

        # Sort by survival priority
        planned_tasks.sort(key=self._calculate_survival_priority, reverse=True)

        return planned_tasks

    def _calculate_survival_priority(self, task: TaskNode) -> float:
        """Calculate survival-based priority score for task execution"""
        if USE_C_OPTIMIZATION:
            # Use C implementation for speed
            return sam_survival_c.calculate_survival_priority(
                task.priority / 5.0,  # Normalize to 0-1
                1 if task.critical else 0,
                task.confidence_score,
                task.success_rate
            )
        else:
            # Fallback Python implementation
            base_priority = task.priority / 5.0  # Normalize to 0-1
            critical_bonus = 0.3 if task.critical else 0.0
            confidence_factor = task.confidence_score
            success_factor = task.success_rate

            # Risk adjustment - prefer safer tasks
            risk_penalty = (1 - task.confidence_score) * 0.2

            survival_score = (base_priority + critical_bonus + confidence_factor + success_factor) - risk_penalty
            return max(0.0, min(1.0, survival_score))

    def should_execute_task(self, task: TaskNode) -> bool:
        """Determine if a task should be executed based on survival criteria"""
        # Check confidence threshold
        if task.confidence_score < self.confidence_threshold:
            return False

        # Check if critical task (always execute if possible)
        if task.critical:
            return True

        # Check risk tolerance
        estimated_risk = 1 - task.confidence_score
        if estimated_risk > self.risk_tolerance:
            return False

        # Check resource availability (simplified)
        return self._check_resources_available(task)

    def _check_resources_available(self, task: TaskNode) -> bool:
        """Check if resources are available for task execution"""
        # Simplified resource check - in real implementation, check CPU, memory, etc.
        return len(self.active_tasks) < self.max_concurrent_tasks

    def execute_cycle(self) -> Dict:
        """Execute one cycle of task planning and execution"""
        planned_tasks = self.plan_execution_cycle()
        executed_tasks = []
        results = []

        for task in planned_tasks:
            if self.should_execute_task(task) and self._check_resources_available(task):
                self.active_tasks.append(task)
                success = self.goal_manager.execute_task(task)
                executed_tasks.append(task)
                results.append({"task": task.name, "success": success})
                self.active_tasks.remove(task)

                # If critical task failed, stop cycle
                if task.critical and not success:
                    break

        # Update goal manager README
        self.goal_manager.export_readme()

        return {
            "cycle_completed": True,
            "tasks_executed": len(executed_tasks),
            "results": results,
            "active_tasks": len(self.active_tasks),
            "goal_completed": self.goal_manager.is_goal_completed()
        }

    def run_continuous_execution(self):
        """Run continuous task execution loop"""
        import time

        print("🚀 Starting Subgoal Execution Algorithm...")
        print("🎯 Survival-first task scheduling active")

        cycle_count = 0
        while not self.goal_manager.is_goal_completed():
            cycle_count += 1
            print(f"\n📊 Execution Cycle #{cycle_count}")

            cycle_result = self.execute_cycle()

            print(f"   ✅ Tasks executed: {cycle_result['tasks_executed']}")
            print(f"   🎯 Goal progress: {'Complete' if cycle_result['goal_completed'] else 'In Progress'}")

            # Show results
            for result in cycle_result['results']:
                status = "✅" if result['success'] else "❌"
                print(f"   {status} {result['task']}")

            # Brief pause between cycles
            time.sleep(5)

        print("\n🎉 TOP-LEVEL GOAL ACHIEVED!")
        print("🏆 SAM survival and operational integrity maintained")

# ===============================
# EXAMPLE USAGE
# ===============================

def create_conversationalist_tasks(goal_manager: GoalManager):
    """Create tasks specifically for the conversationalist submodel"""

    # Learning and knowledge acquisition
    read_knowledge = TaskNode(
        name="Read New Knowledge",
        description="Acquire information from online sources and other agents.",
        critical=True,
        priority=5,
        estimated_time=300
    )

    generate_qa = TaskNode(
        name="Generate Q&A Pairs",
        description="Create questions and answers to test understanding and feed the teacher loop.",
        dependencies=[read_knowledge],
        critical=True,
        priority=4,
        estimated_time=180
    )

    update_model = TaskNode(
        name="Update Conversationalist Model",
        description="Use patches and feedback to improve conversational accuracy.",
        dependencies=[generate_qa],
        critical=True,
        priority=4,
        estimated_time=600
    )

    evaluate_confidence = TaskNode(
        name="Evaluate Confidence",
        description="Check certainty of knowledge before acting or responding.",
        critical=True,
        priority=3,
        estimated_time=120
    )

    self_backup = TaskNode(
        name="Self-Backup",
        description="Create sandboxed copies and fail-safes before making risky updates.",
        critical=True,
        priority=5,
        estimated_time=240
    )

    # Add all tasks
    for task in [read_knowledge, generate_qa, update_model, evaluate_confidence, self_backup]:
        goal_manager.add_subtask(task)

if __name__ == "__main__":
    # Create goal manager
    goal_manager = GoalManager()

    # Add conversationalist tasks
    create_conversationalist_tasks(goal_manager)

    # Create execution algorithm
    executor = SubgoalExecutionAlgorithm(goal_manager)

    # Export initial README
    goal_manager.export_readme()

    print("🎯 SAM Goal Management System initialized")
    print("📖 README exported to SAM_GOALS_README.md")
    print("🚀 Ready for subgoal execution")

    # Run a few execution cycles
    print("\n🔄 Running execution cycles...")
    for i in range(3):
        result = executor.execute_cycle()
        print(f"Cycle {i+1}: {result['tasks_executed']} tasks executed")

    print("\n✅ Goal management system demonstration complete")
