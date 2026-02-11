# SAM Concurrent Executor Module
# Concurrent task execution for SAM 2.0 AGI

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Any

class TaskExecutor:
    """Concurrent task executor with thread pool management"""
    
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks = {}
        self.completed_tasks = []
        self.task_counter = 0
        
    def submit_task(self, func: Callable, *args, **kwargs):
        """Submit a task for concurrent execution"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        future = self.executor.submit(func, *args, **kwargs)
        self.active_tasks[task_id] = {
            'future': future,
            'function': func.__name__,
            'args': args,
            'kwargs': kwargs,
            'submitted_at': time.time()
        }
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            future = task['future']
            
            if future.done():
                try:
                    result = future.result()
                    status = 'completed'
                except Exception as e:
                    result = str(e)
                    status = 'failed'
                
                return {
                    'task_id': task_id,
                    'status': status,
                    'result': result,
                    'completed_at': time.time()
                }
            else:
                return {
                    'task_id': task_id,
                    'status': 'running',
                    'submitted_at': task['submitted_at']
                }
        else:
            return {'task_id': task_id, 'status': 'not_found'}
    
    def wait_for_completion(self, task_ids: List[str] = None, timeout: float = 30.0):
        """Wait for tasks to complete"""
        if task_ids is None:
            task_ids = list(self.active_tasks.keys())
        
        futures = [self.active_tasks[tid]['future'] for tid in task_ids if tid in self.active_tasks]
        
        try:
            for future in as_completed(futures, timeout=timeout):
                # Task completed
                pass
        except Exception as e:
            print(f"Task execution timeout or error: {e}")
    
    def cleanup_completed_tasks(self):
        """Clean up completed tasks"""
        completed = []
        for task_id, task in self.active_tasks.items():
            if task['future'].done():
                completed.append(task_id)
                self.completed_tasks.append({
                    'task_id': task_id,
                    'completed_at': time.time()
                })
        
        for task_id in completed:
            del self.active_tasks[task_id]
        
        return len(completed)
    
    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=True)

# Global task executor instance
task_executor = TaskExecutor(max_workers=4)

print("✅ Concurrent executor initialized")
