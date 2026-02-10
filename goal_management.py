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
        self.goal_priorities = {
            'survival': 10,
            'stability': 9,
            'improvement': 7,
            'expansion': 5,
            'optimization': 6
        }

    def add_goal(self, goal, priority='normal'):
        """Add a new goal to the system"""
        self._goal_counter += 1
        goal_entry = {
            'id': f"goal_{self._goal_counter}",
            'description': goal,
            'priority': priority,
            'status': 'active',
            'created': time.time(),
            'progress': 0.0
        }
        self.active_goals.append(goal_entry)
        return goal_entry['id']

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
        return sorted(self.active_goals, key=lambda x: x.get('priority_score', 0), reverse=True)

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
            goal_manager.add_goal(task['description'], priority=task.get('priority', 'normal'))
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
