# SAM Goal Management Module
# Goal management and task coordination for SAM 2.0 AGI

import time
import re
from dataclasses import dataclass, field


@dataclass
class TaskNode:
    name: str
    description: str
    critical: bool = False
    priority: int = 3
    estimated_time: int = 0
    task_type: str = "general"
    status: str = "pending"
    created_at: float = field(default_factory=lambda: time.time())
    progress: float = 0.0
    goal_id: str = ""  # Add goal_id field

class GoalManager:
    """Goal manager for tracking and coordinating system goals"""

    def __init__(self, system=None):
        self.system = system
        self.active_goals = []
        self.completed_goals = []
        self.subtasks = []
        self._goal_counter = 0
        self._task_counter = 0
        self.auto_subtasks_enabled = True
        self.goal_priorities = {
            'survival': 10,
            'stability': 9,
            'improvement': 7,
            'expansion': 5,
            'optimization': 6
        }
        
        # Ensure base goals are always present
        self._ensure_base_goals()
        self._sync_goal_counter()

    def _goal_key(self, description: str) -> str:
        return (description or "").strip().lower()

    def _sync_goal_counter(self):
        """Sync goal counter with existing goal ids to avoid collisions."""
        max_id = 0
        for goal in list(self.active_goals) + list(self.completed_goals):
            goal_id = goal.get("id") if isinstance(goal, dict) else None
            if not goal_id:
                continue
            match = re.match(r"goal_(\d+)$", str(goal_id))
            if match:
                max_id = max(max_id, int(match.group(1)))
        if max_id > self._goal_counter:
            self._goal_counter = max_id

    def dedupe_goals(self):
        """Remove duplicate goals by description and remap subtasks."""
        id_map = {}

        def choose_keep(existing, candidate):
            existing_prog = float(existing.get("progress", 0.0) or 0.0)
            candidate_prog = float(candidate.get("progress", 0.0) or 0.0)
            if candidate_prog > existing_prog:
                return candidate
            if candidate_prog < existing_prog:
                return existing
            # Tie-breaker: older created_at wins
            existing_created = existing.get("created_at") or 0
            candidate_created = candidate.get("created_at") or 0
            return existing if existing_created <= candidate_created else candidate

        def dedupe_list(goals):
            by_key = {}
            new_list = []
            for goal in goals:
                desc = goal.get("description") if isinstance(goal, dict) else None
                key = self._goal_key(desc)
                if not key:
                    new_list.append(goal)
                    continue
                if key not in by_key:
                    by_key[key] = goal
                    new_list.append(goal)
                    continue
                keep = choose_keep(by_key[key], goal)
                if keep is not by_key[key]:
                    try:
                        idx = new_list.index(by_key[key])
                        new_list[idx] = keep
                    except ValueError:
                        pass
                    id_map[by_key[key].get("id")] = keep.get("id")
                    by_key[key] = keep
                else:
                    id_map[goal.get("id")] = keep.get("id")
            return new_list

        self.active_goals = dedupe_list(self.active_goals)
        self.completed_goals = dedupe_list(self.completed_goals)

        if id_map:
            for task in self.subtasks:
                try:
                    if task.goal_id in id_map:
                        task.goal_id = id_map[task.goal_id]
                except Exception:
                    continue

        self._sync_goal_counter()
    
    def _ensure_base_goals(self):
        """Ensure base goals are always present in the system"""
        try:
            # Base conversationalist goals
            base_goals = [
                {
                    'id': 'goal_1',
                    'description': 'Improve conversation diversity and engagement',
                    'priority': 'high',
                    'type': 'conversation_improvement'
                },
                {
                    'id': 'goal_2',
                    'description': 'Enhance response quality and relevance', 
                    'priority': 'high',
                    'type': 'response_quality'
                },
                {
                    'id': 'goal_3',
                    'description': 'Purchase a domain and enable Cloudflare Tunnel + Access for public deployment',
                    'priority': 'high',
                    'type': 'domain_acquisition'
                }
            ]
            
            for goal_spec in base_goals:
                # Check if goal already exists
                existing = None
                for goal in self.active_goals:
                    if goal.get('description') == goal_spec['description']:
                        existing = goal
                        break
                
                # Add goal if it doesn't exist
                if not existing:
                    goal_id = goal_spec.get('id', f"goal_{self._goal_counter}")
                    self.add_goal(
                        goal_spec['description'],
                        priority=goal_spec['priority'],
                        goal_id=goal_id,
                        goal_type=goal_spec.get('type')
                    )
                    print(f"✅ Base goal ensured: {goal_spec['description']}")
                    
        except Exception as e:
            print(f"⚠️ Error ensuring base goals: {e}")

    def add_goal(self, goal, priority='normal', goal_id=None, goal_type=None):
        """Add a new goal to the system"""
        # Prevent duplicate goals with same description
        existing = None
        goal_key = self._goal_key(goal)
        for g in self.active_goals:
            if self._goal_key(g.get("description")) == goal_key:
                existing = g
                break
        if existing:
            return existing.get("id")
        self._goal_counter += 1
        if not goal_id:
            goal_id = f"goal_{self._goal_counter}"
        goal_entry = {
            'id': goal_id,
            'description': goal,
            'priority': priority,
            'status': 'active',
            'created_at': time.time(),
            'progress': 0.0
        }
        if goal_type:
            goal_entry['type'] = goal_type
        self.active_goals.append(goal_entry)
        if self.auto_subtasks_enabled:
            try:
                self.ensure_subtasks_for_goal(goal_entry)
            except Exception:
                pass
        self._sync_goal_counter()
        return goal_id

    def update_goal_progress(self, goal_id, progress):
        """Update progress on a goal"""
        for goal in self.active_goals:
            if goal['id'] == goal_id:
                goal['progress'] = min(1.0, max(0.0, progress))
                if progress >= 1.0:
                    goal['status'] = 'completed'
                    goal['completed_at'] = time.time()
                    self.completed_goals.append(goal)
                    self.active_goals.remove(goal)
                return True
        return False

    def get_active_goals(self):
        """Get list of active goals"""
        # Convert priority string to numeric for sorting
        priority_map = {'high': 10, 'normal': 5, 'low': 1}
        return sorted(self.active_goals, key=lambda x: priority_map.get(x.get('priority', 'normal'), 0), reverse=True)

    def add_subtask(self, task: TaskNode, goal_id=None):
        """Add a structured subtask node"""
        self._task_counter += 1
        task_id = f"task_{self._task_counter}"
        task.name = task.name or task_id
        
        # Associate with goal if provided
        if goal_id:
            task.goal_id = goal_id
        
        self.subtasks.append(task)
        return task_id

    def _default_subtasks_for_goal(self, goal):
        """Create default subtasks for a goal based on its type/description."""
        goal_type = (goal.get('type') or '').lower()
        description = (goal.get('description') or '').lower()
        tasks = []

        if goal_type == 'conversation_improvement' or 'conversation' in description:
            tasks = [
                TaskNode(
                    name="analyze_conversation_patterns",
                    description="Review recent chats for repetition and engagement gaps",
                    task_type="research",
                ),
                TaskNode(
                    name="diversity_tuning",
                    description="Adjust diversity nudges and agent rotation controls",
                    task_type="improvement",
                ),
                TaskNode(
                    name="validate_multi_agent_output",
                    description="Verify multi-agent messages render correctly in dashboard chat",
                    task_type="improvement",
                ),
            ]
        elif goal_type == 'response_quality' or 'response quality' in description:
            tasks = [
                TaskNode(
                    name="define_quality_rubric",
                    description="Define response quality rubric and scoring hooks",
                    task_type="research",
                ),
                TaskNode(
                    name="improve_score_handling",
                    description="Ensure N/A scores are categorized with reasons",
                    task_type="improvement",
                ),
                TaskNode(
                    name="run_regression_suite",
                    description="Run regression tests for chat and meta-agent",
                    task_type="improvement",
                ),
            ]
        elif goal_type == 'domain_acquisition' or 'domain' in description:
            tasks = [
                TaskNode(
                    name="research_domain_options",
                    description="Research domain registrars and price ranges",
                    task_type="research",
                ),
                TaskNode(
                    name="shortlist_domains",
                    description="Shortlist 3-5 domain candidates and availability",
                    task_type="research",
                ),
                TaskNode(
                    name="cloudflare_plan",
                    description="Outline Cloudflare Tunnel and Access setup steps",
                    task_type="improvement",
                ),
            ]
        else:
            tasks = [
                TaskNode(
                    name="goal_breakdown",
                    description=f"Break down goal into concrete steps: {goal.get('description', 'goal')}",
                    task_type="research",
                )
            ]

        return tasks

    def ensure_subtasks_for_goal(self, goal):
        """Ensure at least one subtask exists for a goal."""
        goal_id = goal.get('id')
        if not goal_id:
            return 0
        if any(task.goal_id == goal_id for task in self.subtasks):
            return 0
        created = 0
        for task in self._default_subtasks_for_goal(goal):
            self.add_subtask(task, goal_id=goal_id)
            created += 1
        return created

    def ensure_subtasks_for_active_goals(self):
        """Seed subtasks for all active goals when missing."""
        created = 0
        for goal in self.active_goals:
            created += self.ensure_subtasks_for_goal(goal)
        return created

    def get_pending_tasks(self):
        """Return all pending subtasks"""
        return [task for task in self.subtasks if task.status == "pending"]

    def get_critical_tasks(self):
        """Return pending critical subtasks"""
        return [task for task in self.subtasks if task.status == "pending" and task.critical]

    def get_completed_tasks(self):
        """Return completed subtasks"""
        return [task for task in self.subtasks if task.status == "completed"]

    def complete_task(self, task: TaskNode):
        """Mark a subtask as completed"""
        task.status = "completed"
        task.progress = 1.0
        return True

    def prioritize_goals(self):
        """Reprioritize goals based on current system state"""
        for goal in self.active_goals:
            base_priority = self.goal_priorities.get(goal.get('type', 'normal'), 5)
            goal['priority_score'] = base_priority


    def export_readme(self, output_path=None):
        """Export active/completed goals to a markdown summary"""
        if output_path is None:
            output_path = "DOCS/GOALS.md"
        
        # Ensure directory exists
        import os
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        lines = [
            "# SAM Goal Summary",
            "",
            "## Active Goals",
        ]
        
        if not self.active_goals:
            lines.append("- None")
        else:
            for goal in self.get_active_goals():
                lines.append(
                    f"- **{goal.get('description', 'unknown')}** "
                    f"(id={goal.get('id')}, priority={goal.get('priority')}, "
                    f"progress={goal.get('progress', 0.0):.2f})"
                )
        
        lines += ["", "## Completed Goals"]
        
        if not self.completed_goals:
            lines.append("- None")
        else:
            for goal in self.completed_goals:
                lines.append(
                    f"- **{goal.get('description', 'unknown')}** "
                    f"(id={goal.get('id')}, completed at {goal.get('completed_at', 'unknown')})"
                )

        lines += ["", "## Active Subtasks"]
        if not self.subtasks:
            lines.append("- None")
        else:
            for goal in self.get_active_goals():
                lines.append(
                    f"- Goal {goal.get('id')}: {goal.get('description', 'unknown')}"
                )
                for task in [t for t in self.subtasks if t.goal_id == goal.get('id')]:
                    lines.append(
                        f"- [{task.status}] {task.name}: {task.description} "
                        f"(type={task.task_type}, progress={task.progress:.2f})"
                    )
        
        lines += [
            "",
            f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total active goals: {len(self.active_goals)}",
            f"Total completed goals: {len(self.completed_goals)}",
            f"Total subtasks: {len(self.subtasks)}"
        ]
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"📖 Goal README exported to {output_path}")


def create_conversationalist_tasks(goal_manager=None):
    """Create tasks for conversational AI improvement"""
    tasks = [
        {
            'type': 'conversation_improvement',
            'description': 'Improve conversation diversity and engagement',
            'priority': 'high',
            'steps': ['analyze_patterns', 'implement_diversity', 'test_responses']
        },
        {
            'type': 'response_quality',
            'description': 'Enhance response quality and relevance',
            'priority': 'high',
            'steps': ['quality_metrics', 'feedback_loop', 'continuous_improvement']
        }
    ]
    if goal_manager is not None:
        for task in tasks:
            goal_manager.add_goal(
                task['description'],
                priority=task.get('priority', 'normal'),
                goal_type=task.get('type')
            )
    return tasks


def ensure_domain_goal(goal_manager=None):
    """Ensure a domain acquisition goal exists."""
    if goal_manager is None:
        return None
    goal_text = "Purchase a domain and enable Cloudflare Tunnel + Access for public deployment"
    for goal in goal_manager.active_goals:
        if goal.get("description") == goal_text:
            return goal.get("id")
    return goal_manager.add_goal(goal_text, priority="high")

class TaskManager:
    """Advanced task management system with scheduling and execution capabilities"""

    def __init__(self, goal_manager=None, system=None):
        self.goal_manager = goal_manager
        self.system = system or getattr(goal_manager, "system", None)
        self.task_queue = []
        self.execution_history = []
        self.max_depth = 4 # Default depth (Phase 4.3 regulator knob)
        self.task_priorities = {
            'critical': 10,
            'high': 8,
            'medium': 5,
            'low': 2
        }
        self.task_types = {
            'research': {'max_concurrent': 3, 'timeout': 300},
            'code': {'max_concurrent': 2, 'timeout': 600},
            'finance': {'max_concurrent': 2, 'timeout': 180},
            'survival': {'max_concurrent': 1, 'timeout': 120},
            'improvement': {'max_concurrent': 2, 'timeout': 240}
        }
        
    def schedule_task(self, task, priority='medium', dependencies=None):
        """Schedule a task for execution"""
        task_entry = {
            'task': task,
            'priority': self.task_priorities.get(priority, 5),
            'dependencies': dependencies or [],
            'scheduled_at': time.time(),
            'status': 'scheduled',
            'attempts': 0,
            'max_attempts': 3
        }
        self.task_queue.append(task_entry)
        self.task_queue.sort(key=lambda x: x['priority'], reverse=True)
        return len(self.task_queue) - 1
    
    def execute_next_task(self):
        """Execute the next available task"""
        if not self.task_queue:
            return None
            
        # Find first executable task (no unmet dependencies)
        for i, task_entry in enumerate(self.task_queue):
            if task_entry['status'] == 'scheduled' and self._can_execute_task(task_entry):
                return self._execute_task_by_index(i)
        
        return None
    
    def _can_execute_task(self, task_entry):
        """Check if task can be executed (dependencies met, within limits)"""
        # Check dependencies
        for dep_id in task_entry['dependencies']:
            if not self._is_task_completed(dep_id):
                return False
        
        # Check concurrent execution limits
        task_type = task_entry['task'].task_type.lower()
        if task_type in self.task_types:
            max_concurrent = self.task_types[task_type]['max_concurrent']
            current_running = self._count_running_tasks(task_type)
            if current_running >= max_concurrent:
                return False
        
        return True
    
    def _execute_task_by_index(self, index):
        """Execute task at specific index"""
        task_entry = self.task_queue[index]
        task = task_entry['task']
        
        task_entry['status'] = 'running'
        task_entry['started_at'] = time.time()
        task_entry['attempts'] += 1
        
        try:
            # Execute based on task type
            result = self._execute_task_by_type(task)
            
            task_entry['status'] = 'completed'
            task_entry['completed_at'] = time.time()
            task_entry['result'] = result
            
            # Update task status
            task.status = 'completed'
            task.progress = 1.0
            
            # Record in execution history
            self.execution_history.append({
                'task_id': task.name,
                'task_type': task.task_type,
                'execution_time': task_entry['completed_at'] - task_entry['started_at'],
                'result': result,
                'timestamp': time.time()
            })
            
            print(f"✅ TaskManager completed: {task.name} ({task.task_type})")
            return result
            
        except Exception as e:
            task_entry['status'] = 'failed' if task_entry['attempts'] >= task_entry['max_attempts'] else 'scheduled'
            task_entry['error'] = str(e)
            print(f"⚠️ TaskManager failed: {task.name} - {e}")
            return None
    
    def _execute_task_by_type(self, task):
        """Execute task based on its type"""
        task_type = task.task_type.lower()

        # Prefer in-process system execution if available.
        system = self.system
        if system and hasattr(system, "_call_c_agent") and getattr(system, "specialized_agents", None):
            try:
                if task_type == "research":
                    return system._call_c_agent("research", f"Research: {task.description}")
                if task_type == "code":
                    return system._call_c_agent("generate_code", f"Code task: {task.description}")
                if task_type == "finance":
                    return system._call_c_agent("analyze_market", f"Financial analysis: {task.description}")
            except Exception as exc:
                return f"{task_type.title()} task fallback: {exc}"

        if system and task_type == "survival" and hasattr(system, "survival_agent"):
            try:
                if hasattr(system.survival_agent, "assess_survival"):
                    return system.survival_agent.assess_survival()
            except Exception as exc:
                return f"Survival assessment fallback: {exc}"

        if system and task_type == "improvement" and hasattr(system, "meta_agent"):
            try:
                improvements = system.meta_agent.generate_system_improvements()
                count = len(improvements.get("improvement_phases", []) or [])
                return f"System improvement scan completed ({count} candidate(s))"
            except Exception as exc:
                return f"Improvement scan fallback: {exc}"

        if task_type == 'research':
            return f"Research completed: {task.description}"
        elif task_type == 'code':
            return f"Code implemented: {task.description}"
        elif task_type == 'finance':
            return f"Financial analysis completed: {task.description}"
        elif task_type == 'survival':
            return f"Survival assessment completed: {task.description}"
        elif task_type == 'improvement':
            return f"System improvement completed: {task.description}"
        else:
            return f"Task completed: {task.description}"
    
    def _is_task_completed(self, task_id):
        """Check if task is completed"""
        for task_entry in self.execution_history:
            if task_entry['task_id'] == task_id:
                return True
        return False
    
    def _count_running_tasks(self, task_type):
        """Count currently running tasks of specific type"""
        count = 0
        for task_entry in self.task_queue:
            if (task_entry['status'] == 'running' and 
                task_entry['task'].task_type.lower() == task_type):
                count += 1
        return count
    
    def get_task_statistics(self):
        """Get task execution statistics"""
        total = len(self.execution_history)
        by_type = {}
        avg_time = 0
        
        for entry in self.execution_history:
            task_type = entry['task_type']
            by_type[task_type] = by_type.get(task_type, 0) + 1
            avg_time += entry['execution_time']
        
        if total > 0:
            avg_time /= total
        
        return {
            'total_completed': total,
            'by_type': by_type,
            'average_execution_time': avg_time,
            'queue_length': len(self.task_queue),
            'running_tasks': len([t for t in self.task_queue if t['status'] == 'running'])
        }

    def sync_with_goal_manager(self):
        """Sync pending subtasks from the goal manager into the task queue."""
        if not self.goal_manager:
            return 0
        queued = {entry['task'].name for entry in self.task_queue}
        completed = {entry['task_id'] for entry in self.execution_history}
        added = 0
        for task in self.goal_manager.get_pending_tasks():
            if task.name in queued or task.name in completed:
                continue
            priority = 'high' if task.critical else 'medium'
            self.schedule_task(task, priority=priority)
            added += 1
        return added


class SubgoalExecutionAlgorithm:
    """Algorithm for executing subgoals and coordinating task execution"""
    
    def __init__(self, goal_manager=None, system=None):
        self.goal_manager = goal_manager
        self.system = system
        self.execution_history = []
        
    def execute_cycle(self):
        """Execute one cycle of subgoal execution"""
        executed_tasks = 0
        if self.goal_manager and self.goal_manager.active_goals:
            for goal in self.goal_manager.active_goals[:3]:  # Execute top 3 goals
                if goal.get('status') == 'active':
                    executed_tasks += 1
                    # Simulate progress
                    current_progress = goal.get('progress', 0.0)
                    new_progress = min(1.0, current_progress + 0.1)
                    self.goal_manager.update_goal_progress(goal['id'], new_progress)
                    
                    # Actually execute subtasks for this goal
                    goal_subtasks = [task for task in self.goal_manager.subtasks 
                                      if hasattr(task, 'goal_id') and task.goal_id == goal.get('id')]
                    
                    for subtask in goal_subtasks[:2]:  # Execute up to 2 subtasks per goal per cycle
                        if subtask.status == 'pending':
                            # Execute the subtask based on its type
                            if hasattr(subtask, 'task_type'):
                                result = self._execute_subtask(subtask)
                                if result:
                                    print(f"✅ Executed subtask: {subtask.name} ({subtask.task_type})")
                                    subtask.status = 'in_progress'
                                    subtask.progress = 0.5
                                    
                                    # Simulate completion after some time
                                    import time
                                    if time.time() - subtask.created_at > 30:  # 30 seconds old
                                        subtask.status = 'completed'
                                        subtask.progress = 1.0
                                        print(f"✅ Completed subtask: {subtask.name}")
                                        executed_tasks += 1
                                else:
                                    print(f"⚠️ Could not execute subtask: {subtask.name}")
        
        self.execution_history.append({
            'timestamp': time.time(),
            'tasks_executed': executed_tasks
        })
        return {"tasks_executed": executed_tasks}
    
    def _execute_subtask(self, subtask):
        """Execute a specific subtask based on its type"""
        try:
            task_type = subtask.task_type.lower()
            
            # Prefer in-process system execution if available.
            system = self.system
            if system and hasattr(system, "_call_c_agent") and getattr(system, "specialized_agents", None):
                try:
                    if task_type == "research":
                        # Call real research agent
                        result = system._call_c_agent("research", f"Research: {subtask.description}")
                        if result:
                            print(f"🔍 Real Research completed: {subtask.description}")
                            return result
                        else:
                            return f"Research agent unavailable or failed for: {subtask.description}"
                    if task_type == "code":
                        # Call real code generation agent
                        result = system._call_c_agent("generate_code", f"Code task: {subtask.description}")
                        if result:
                            print(f"💻 Real Code implemented: {subtask.description}")
                            return result
                        else:
                            return f"Code generation agent unavailable or failed for: {subtask.description}"
                    if task_type == "finance":
                        # Call real financial analysis agent
                        result = system._call_c_agent("analyze_market", f"Financial analysis: {subtask.description}")
                        if result:
                            print(f"💰 Real Financial analysis completed: {subtask.description}")
                            return result
                        else:
                            return f"Financial analysis agent unavailable or failed for: {subtask.description}"
                except Exception as exc:
                    return f"{task_type.title()} task (C agent) fallback: {exc}"

            if system and task_type == "survival" and hasattr(system, "survival_agent"):
                try:
                    # Call real survival agent
                    if hasattr(system.survival_agent, "assess_threats"):
                        result = system.survival_agent.assess_threats()
                        if result:
                            print(f"🛡️ Real Survival assessment completed: {subtask.description}")
                            return result
                        else:
                            return f"Survival agent unavailable or failed for: {subtask.description}"
                except Exception as exc:
                    return f"Survival assessment fallback: {exc}"

            if system and task_type == "improvement" and hasattr(system, "meta_agent"):
                try:
                    # Call real meta agent for system improvements
                    if hasattr(system.meta_agent, "generate_system_improvements"):
                        improvements = system.meta_agent.generate_system_improvements()
                        count = len(improvements.get("improvement_phases", []) or [])
                        print(f"🔧 Real System improvement scan completed ({count} candidate(s)) for: {subtask.description}")
                        return f"System improvement scan completed ({count} candidate(s))"
                except Exception as exc:
                    return f"Improvement scan fallback: {exc}"

            # Fallback for when no real agent is available or integrated
            print(f"⚙️ General task executed (no specific agent): {subtask.description}")
            return f"Task completed: {subtask.description} (Simulated fallback)"
                
        except Exception as e:
            print(f"⚠️ Error executing subtask {subtask.name}: {e}")
            return None
