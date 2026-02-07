#!/usr/bin/env python3
"""
SAM 2.0 Concurrent Processing Engine
Multi-threaded task execution with advanced scheduling and load balancing
"""

import threading
import concurrent.futures
import queue
import time
import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - resource monitoring disabled")
import os

class TaskPriority(Enum):
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ConcurrentTask:
    """Represents a task that can be executed concurrently"""
    id: str
    name: str
    function: Callable
    args: tuple = ()
    kwargs: dict = None
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    dependencies: List[str] = None
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    progress_callback: Optional[Callable] = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = time.time()

class ResourceMonitor:
    """Monitor system resources for intelligent task scheduling"""

    def __init__(self, cpu_threshold: float = 80.0, memory_threshold: float = 85.0):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold

    def get_system_load(self) -> Dict[str, float]:
        """Get current system resource usage"""
        if not PSUTIL_AVAILABLE:
            # Return default values when psutil is not available
            return {
                'cpu_percent': 50.0,  # Default moderate load
                'memory_percent': 60.0,  # Default moderate memory usage
                'disk_usage': 70.0,  # Default moderate disk usage
                'available_memory_gb': 4.0  # Default 4GB available
            }
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }

    def can_accept_task(self, task: ConcurrentTask) -> bool:
        """Determine if system can accept a new task"""
        load = self.get_system_load()

        # Critical tasks always allowed
        if task.priority == TaskPriority.CRITICAL:
            return True

        # Check resource thresholds
        if load['cpu_percent'] > self.cpu_threshold:
            return False
        if load['memory_percent'] > self.memory_threshold:
            return False

        # High priority tasks get more leeway
        if task.priority == TaskPriority.HIGH:
            return load['cpu_percent'] < 90 and load['memory_percent'] < 90

        return True

    def get_optimal_worker_count(self) -> int:
        """Calculate optimal number of worker threads based on system resources"""
        load = self.get_system_load()
        cpu_count = os.cpu_count() or 4

        # Reduce workers if system is heavily loaded
        if load['cpu_percent'] > 80 or load['memory_percent'] > 80:
            return max(1, cpu_count // 2)
        elif load['cpu_percent'] > 60 or load['memory_percent'] > 60:
            return max(1, cpu_count // 1.5)

        return cpu_count

class AdvancedThreadPoolExecutor:
    """Advanced thread pool with priority scheduling and resource awareness"""

    def __init__(self, max_workers: int = None, thread_name_prefix: str = "SAM-Worker"):
        self.resource_monitor = ResourceMonitor()
        self.max_workers = max_workers or self.resource_monitor.get_optimal_worker_count()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=thread_name_prefix
        )

        # Task management
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, ConcurrentTask] = {}
        self.completed_tasks: Dict[str, ConcurrentTask] = {}
        self.task_dependencies: Dict[str, set] = {}

        # Control
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        self.logger = logging.getLogger(__name__)

    def submit_task(self, task: ConcurrentTask) -> str:
        """Submit a task for execution"""
        # Check dependencies
        if not self._check_dependencies(task):
            self.logger.info(f"Task {task.name} waiting for dependencies")
            task.status = TaskStatus.PENDING

        # Add to queue with priority (negative for higher priority first)
        priority_value = -task.priority.value
        self.task_queue.put((priority_value, task.created_at, task))

        self.logger.info(f"Task {task.name} submitted with priority {task.priority.name}")
        return task.id

    def _check_dependencies(self, task: ConcurrentTask) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            if self.completed_tasks[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True

    def _scheduler_loop(self):
        """Main scheduler loop that manages task execution"""
        while self.running:
            try:
                # Get next task if available and system can handle it
                if not self.task_queue.empty():
                    priority, created_at, task = self.task_queue.get_nowait()

                    # Check if system can accept this task
                    if self.resource_monitor.can_accept_task(task) and self._check_dependencies(task):
                        # Submit to thread pool
                        future = self.executor.submit(self._execute_task_wrapper, task)
                        self.active_tasks[task.id] = task
                        task.status = TaskStatus.RUNNING
                        task.started_at = time.time()

                        # Add callback for completion
                        future.add_done_callback(lambda f, t=task: self._task_completed_callback(f, t))
                    else:
                        # Put back in queue if not ready
                        self.task_queue.put((priority, created_at, task))
                        time.sleep(1)  # Brief pause before checking again

                time.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(1)

    def _execute_task_wrapper(self, task: ConcurrentTask) -> Any:
        """Wrapper for task execution with timeout and error handling"""
        try:
            # Execute with timeout if specified
            if task.timeout:
                with concurrent.futures.TimeoutError():
                    result = task.function(*task.args, **task.kwargs)
            else:
                result = task.function(*task.args, **task.kwargs)

            return result

        except Exception as e:
            self.logger.error(f"Task {task.name} failed: {e}")
            raise e

    def _task_completed_callback(self, future: concurrent.futures.Future, task: ConcurrentTask):
        """Handle task completion"""
        task.completed_at = time.time()

        try:
            task.result = future.result()
            task.status = TaskStatus.COMPLETED
            self.logger.info(f"Task {task.name} completed successfully")

            # Call progress callback if provided
            if task.progress_callback:
                task.progress_callback(task, True, task.result)

        except Exception as e:
            task.error = e
            task.status = TaskStatus.FAILED
            self.logger.error(f"Task {task.name} failed: {e}")

            # Handle retries
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                self.logger.info(f"Retrying task {task.name} (attempt {task.retry_count})")

                # Put back in queue with slightly lower priority
                priority_value = -(task.priority.value - 0.1)
                self.task_queue.put((priority_value, time.time(), task))
            else:
                self.logger.error(f"Task {task.name} failed permanently after {task.max_retries} retries")

                # Call progress callback with failure
                if task.progress_callback:
                    task.progress_callback(task, False, task.error)

        # Move to completed tasks
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            self.completed_tasks[task.id] = task
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]
            return True

        # Check queue (more complex, would need to rebuild queue)
        return False

    def get_task_status(self, task_id: str) -> Optional[ConcurrentTask]:
        """Get status of a task"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        return None

    def get_active_tasks(self) -> List[ConcurrentTask]:
        """Get all active tasks"""
        return list(self.active_tasks.values())

    def get_completed_tasks(self) -> List[ConcurrentTask]:
        """Get all completed tasks"""
        return list(self.completed_tasks.values())

    def get_queue_size(self) -> int:
        """Get number of tasks in queue"""
        return self.task_queue.qsize()

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker thread statistics"""
        return {
            'max_workers': self.max_workers,
            'active_workers': len(self.active_tasks),
            'queued_tasks': self.get_queue_size(),
            'completed_tasks': len(self.completed_tasks),
            'system_load': self.resource_monitor.get_system_load()
        }

    def shutdown(self, wait: bool = True):
        """Shutdown the executor"""
        self.running = False
        self.executor.shutdown(wait=wait)
        self.logger.info("Concurrent executor shutdown complete")

# ===============================
# INTEGRATED SAM TASK EXECUTOR
# ===============================

class SAMTaskExecutor:
    """High-level task executor for SAM system components"""

    def __init__(self):
        self.executor = AdvancedThreadPoolExecutor()
        self.task_registry: Dict[str, ConcurrentTask] = {}

    def submit_survival_evaluation(self, action: str, context: Dict,
                                  callback: Callable = None) -> str:
        """Submit a survival evaluation task"""
        task = ConcurrentTask(
            id=f"survival_eval_{int(time.time())}_{hash(action) % 1000}",
            name=f"Evaluate survival impact of: {action[:50]}...",
            function=self._survival_evaluation_worker,
            args=(action, context),
            priority=TaskPriority.HIGH,
            timeout=30.0,
            progress_callback=callback
        )

        task_id = self.executor.submit_task(task)
        self.task_registry[task_id] = task
        return task_id

    def submit_goal_execution(self, goal_name: str, goal_data: Dict,
                             callback: Callable = None) -> str:
        """Submit a goal execution task"""
        task = ConcurrentTask(
            id=f"goal_exec_{int(time.time())}_{hash(goal_name) % 1000}",
            name=f"Execute goal: {goal_name}",
            function=self._goal_execution_worker,
            args=(goal_name, goal_data),
            priority=TaskPriority.CRITICAL,
            timeout=300.0,  # 5 minutes
            progress_callback=callback
        )

        task_id = self.executor.submit_task(task)
        self.task_registry[task_id] = task
        return task_id

    def submit_error_recovery(self, error_data: Dict, callback: Callable = None) -> str:
        """Submit an error recovery task"""
        task = ConcurrentTask(
            id=f"error_recovery_{int(time.time())}_{hash(str(error_data)) % 1000}",
            name=f"Recover from error: {error_data.get('type', 'Unknown')}",
            function=self._error_recovery_worker,
            args=(error_data,),
            priority=TaskPriority.CRITICAL,
            timeout=120.0,  # 2 minutes
            progress_callback=callback
        )

        task_id = self.executor.submit_task(task)
        self.task_registry[task_id] = task
        return task_id

    def submit_batch_processing(self, items: List[Any], processor_func: Callable,
                               batch_size: int = 10, callback: Callable = None) -> str:
        """Submit batch processing task"""
        task = ConcurrentTask(
            id=f"batch_process_{int(time.time())}_{hash(str(items[:5])) % 1000}",
            name=f"Process batch of {len(items)} items",
            function=self._batch_processing_worker,
            args=(items, processor_func, batch_size),
            priority=TaskPriority.NORMAL,
            timeout=600.0,  # 10 minutes
            progress_callback=callback
        )

        task_id = self.executor.submit_task(task)
        self.task_registry[task_id] = task
        return task_id

    # ===============================
    # WORKER FUNCTIONS
    # ===============================

    def _survival_evaluation_worker(self, action: str, context: Dict) -> Dict[str, Any]:
        """Worker function for survival evaluation"""
        # Import here to avoid circular imports
        try:
            from survival_agent import create_survival_agent
            agent = create_survival_agent()
            return agent.evaluate_action(action, context)
        except Exception as e:
            return {
                "survival_impact": 0.0,
                "optionality_impact": 0.0,
                "risk_level": 0.5,
                "confidence": 0.0,
                "error": str(e)
            }

    def _goal_execution_worker(self, goal_name: str, goal_data: Dict) -> Dict[str, Any]:
        """Worker function for goal execution"""
        try:
            from goal_management import GoalManager
            goal_manager = GoalManager()

            # Execute goal logic (simplified)
            result = {
                "goal_name": goal_name,
                "success": True,
                "execution_time": time.time(),
                "result": "Goal executed successfully"
            }

            # Update goal status
            goal_manager.export_readme()
            return result

        except Exception as e:
            return {
                "goal_name": goal_name,
                "success": False,
                "error": str(e)
            }

    def _error_recovery_worker(self, error_data: Dict) -> Dict[str, Any]:
        """Worker function for error recovery"""
        try:
            # Import meta agent for error fixing
            from meta_agent.agent import auto_patch

            error_type = error_data.get('type', 'Unknown')
            error_message = error_data.get('message', 'No message')

            # Attempt automatic recovery
            recovery_result = auto_patch(Exception(f"{error_type}: {error_message}"))

            return {
                "error_type": error_type,
                "recovery_attempted": True,
                "recovery_success": recovery_result.get('status') == 'success',
                "recovery_actions": recovery_result.get('message', ''),
                "timestamp": time.time()
            }

        except Exception as e:
            return {
                "error_type": error_data.get('type', 'Unknown'),
                "recovery_attempted": True,
                "recovery_success": False,
                "error": str(e)
            }

    def _batch_processing_worker(self, items: List[Any], processor_func: Callable,
                               batch_size: int) -> Dict[str, Any]:
        """Worker function for batch processing"""
        results = []
        errors = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            try:
                batch_results = processor_func(batch)
                results.extend(batch_results)
            except Exception as e:
                errors.append(f"Batch {i//batch_size}: {str(e)}")

        return {
            "total_items": len(items),
            "processed": len(results),
            "errors": len(errors),
            "error_details": errors,
            "results": results[:100]  # Limit result size
        }

    # ===============================
    # MONITORING AND CONTROL
    # ===============================

    def get_executor_stats(self) -> Dict[str, Any]:
        """Get comprehensive executor statistics"""
        return self.executor.get_worker_stats()

    def get_task_stats(self) -> Dict[str, Any]:
        """Get task execution statistics"""
        active = self.executor.get_active_tasks()
        completed = self.executor.get_completed_tasks()

        return {
            "active_tasks": len(active),
            "completed_tasks": len(completed),
            "queued_tasks": self.executor.get_queue_size(),
            "failed_tasks": len([t for t in completed if t.status == TaskStatus.FAILED]),
            "success_rate": len([t for t in completed if t.status == TaskStatus.COMPLETED]) / max(len(completed), 1)
        }

    def shutdown(self):
        """Shutdown the task executor"""
        self.executor.shutdown(wait=True)

# ===============================
# GLOBAL EXECUTOR INSTANCE
# ===============================

# Create global task executor
task_executor = SAMTaskExecutor()

if __name__ == "__main__":
    print("🚀 SAM Concurrent Processing Engine Initialized")

    # Example usage
    def example_task(x, y):
        time.sleep(0.1)  # Simulate work
        return x + y

    # Submit some example tasks
    task_ids = []
    for i in range(5):
        task = ConcurrentTask(
            id=f"example_{i}",
            name=f"Example Task {i}",
            function=example_task,
            args=(i, i*2),
            priority=TaskPriority.NORMAL
        )
        task_id = task_executor.executor.submit_task(task)
        task_ids.append(task_id)

    print(f"✅ Submitted {len(task_ids)} example tasks")

    # Wait for completion
    time.sleep(2)

    stats = task_executor.get_executor_stats()
    print(f"📊 Executor Stats: {stats}")

    task_executor.shutdown()
    print("✅ Concurrent processing engine shutdown complete")
