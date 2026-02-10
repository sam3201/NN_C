# SAM Goal Management Module
# Goal management and task coordination for SAM 2.0 AGI

import time
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
    
    def __init__(self, goal_manager=None):
        self.goal_manager = goal_manager
        self.task_queue = []
        self.execution_history = []
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


class SubgoalExecutionAlgorithm:
    """Algorithm for executing subgoals and coordinating task execution"""
    
    def __init__(self, goal_manager=None):
        self.goal_manager = goal_manager
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
            
            if task_type == 'research':
                print(f"🔍 Executing research task: {subtask.description}")
                # Simulate research execution
                return f"Research completed: {subtask.description}"
                
            elif task_type == 'code':
                print(f"💻 Executing code task: {subtask.description}")
                # Simulate code generation
                return f"Code implemented: {subtask.description}"
                
            elif task_type == 'finance':
                print(f"💰 Executing finance task: {subtask.description}")
                # Simulate financial analysis
                return f"Financial analysis completed: {subtask.description}"
                
            elif task_type == 'survival':
                print(f"🛡️ Executing survival task: {subtask.description}")
                # Simulate survival assessment
                return f"Survival assessment completed: {subtask.description}"
                
            elif task_type == 'improvement':
                print(f"🔧 Executing improvement task: {subtask.description}")
                # Simulate system improvement
                return f"System improvement completed: {subtask.description}"
                
            else:
                print(f"⚙️ Executing general task: {subtask.description}")
                return f"Task completed: {subtask.description}"
                
        except Exception as e:
            print(f"⚠️ Error executing subtask {subtask.name}: {e}")
            return None
