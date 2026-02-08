# SAM Circuit Breaker Module
# Circuit breaker pattern implementation for SAM 2.0 AGI

import time
import threading
from typing import Dict, Any, Callable, Optional
from enum import Enum

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, blocking calls
    HALF_OPEN = "half_open"  # Testing if service has recovered

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker"""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    print("🔄 Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN - call blocked")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset"""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            print("✅ Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"⚠️ Circuit breaker OPENED after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'threshold': self.failure_threshold
        }

class ResilienceManager:
    """Resilience manager for system-wide fault tolerance"""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.fallback_handlers = {}
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'circuit_breaker_activations': 0
        }
        
    def create_circuit_breaker(self, 
                           name: str,
                           failure_threshold: int = 5,
                           recovery_timeout: float = 60.0) -> CircuitBreaker:
        """Create a named circuit breaker"""
        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def register_fallback(self, name: str, handler: Callable):
        """Register a fallback handler for when circuit breaker opens"""
        self.fallback_handlers[name] = handler
    
    def execute_with_resilience(self, 
                              name: str,
                              func: Callable,
                              *args, **kwargs) -> Any:
        """Execute function with resilience protection"""
        self.metrics['total_calls'] += 1
        
        try:
            if name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[name]
                result = circuit_breaker.call(func, *args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self.metrics['successful_calls'] += 1
            return result
            
        except Exception as e:
            self.metrics['failed_calls'] += 1
            
            # Try fallback if available
            if name in self.fallback_handlers:
                print(f"🔄 Using fallback for {name}")
                return self.fallback_handlers[name](*args, **kwargs)
            else:
                raise e
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get resilience metrics"""
        return self.metrics.copy()
    
    def get_circuit_breaker_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of specific circuit breaker"""
        if name in self.circuit_breakers:
            return self.circuit_breakers[name].get_state()
        return None

# Global resilience manager instance
resilience_manager = ResilienceManager()

print("✅ Circuit breaker initialized")
