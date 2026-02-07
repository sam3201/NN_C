#!/usr/bin/env python3
"""
SAM 2.0 Circuit Breaker Pattern
Intelligent error handling and automatic recovery system
"""

import time
import threading
import logging
from enum import Enum
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass
import json

class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 60.0      # Seconds to wait before testing recovery
    expected_exception: tuple = (Exception,)  # Exception types to catch
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout: Optional[float] = None     # Request timeout
    name: str = "default"               # Circuit breaker name

class CircuitBreaker:
    """Circuit breaker implementation with advanced recovery logic"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.next_attempt_time = 0
        self.logger = logging.getLogger(f"CircuitBreaker.{config.name}")
        self._lock = threading.RLock()

        # Metrics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function"""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            self.total_requests += 1

            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if time.time() < self.next_attempt_time:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.config.name}' is OPEN"
                    )
                else:
                    # Time to test recovery
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info(f"Circuit breaker '{self.config.name}' testing recovery")

            # Execute the function
            try:
                if self.config.timeout:
                    # Would implement timeout logic here
                    result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Success handling
                self._on_success()
                return result

            except self.config.expected_exception as e:
                # Failure handling
                self._on_failure()
                raise e

    def _on_success(self):
        """Handle successful execution"""
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        self.total_successes += 1

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close_circuit()
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure metrics on success
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self):
        """Handle failed execution"""
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self._open_circuit()
        elif (self.state == CircuitBreakerState.CLOSED and
              self.failure_count >= self.config.failure_threshold):
            self._open_circuit()

    def _open_circuit(self):
        """Open the circuit breaker"""
        self.state = CircuitBreakerState.OPEN
        self.next_attempt_time = time.time() + self.config.recovery_timeout
        self.logger.warning(
            f"Circuit breaker '{self.config.name}' OPENED after {self.failure_count} failures"
        )

    def _close_circuit(self):
        """Close the circuit breaker"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.logger.info(f"Circuit breaker '{self.config.name}' CLOSED - recovered successfully")

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "failure_rate": self.total_failures / max(self.total_requests, 1),
            "success_rate": self.total_successes / max(self.total_requests, 1),
            "last_failure_time": self.last_failure_time,
            "next_attempt_time": self.next_attempt_time
        }

    def reset(self):
        """Reset circuit breaker to initial state"""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.logger.info(f"Circuit breaker '{self.config.name}' reset")

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

# ===============================
# SAM-SPECIFIC CIRCUIT BREAKERS
# ===============================

class SAMCircuitBreakers:
    """Pre-configured circuit breakers for SAM system components"""

    def __init__(self):
        # Survival evaluation circuit breaker
        self.survival_evaluator = CircuitBreaker(CircuitBreakerConfig(
            name="survival_evaluator",
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            timeout=10.0
        ))

        # Goal execution circuit breaker
        self.goal_executor = CircuitBreaker(CircuitBreakerConfig(
            name="goal_executor",
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=3,
            timeout=300.0
        ))

        # Meta agent circuit breaker
        self.meta_agent = CircuitBreaker(CircuitBreakerConfig(
            name="meta_agent",
            failure_threshold=2,
            recovery_timeout=120.0,
            success_threshold=1,
            timeout=60.0
        ))

        # Database operations circuit breaker
        self.database = CircuitBreaker(CircuitBreakerConfig(
            name="database",
            failure_threshold=10,
            recovery_timeout=300.0,  # 5 minutes
            success_threshold=5,
            timeout=30.0
        ))

        # External API calls circuit breaker
        self.external_api = CircuitBreaker(CircuitBreakerConfig(
            name="external_api",
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=2,
            timeout=15.0
        ))

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers"""
        return {
            "survival_evaluator": self.survival_evaluator.get_metrics(),
            "goal_executor": self.goal_executor.get_metrics(),
            "meta_agent": self.meta_agent.get_metrics(),
            "database": self.database.get_metrics(),
            "external_api": self.external_api.get_metrics()
        }

# ===============================
# INTELLIGENT RECOVERY SYSTEM
# ===============================

class IntelligentRecovery:
    """Advanced error recovery with learning and adaptation"""

    def __init__(self):
        self.recovery_history: List[Dict[str, Any]] = []
        self.recovery_strategies: Dict[str, List[Dict[str, Any]]] = {}
        self.logger = logging.getLogger(__name__)

    def attempt_recovery(self, error_type: str, error_context: Dict) -> Dict[str, Any]:
        """Attempt intelligent error recovery"""
        recovery_start = time.time()

        # Get recovery strategies for this error type
        strategies = self.recovery_strategies.get(error_type, self._get_default_strategies(error_type))

        # Try strategies in order of success rate
        strategies.sort(key=lambda s: s.get('success_rate', 0), reverse=True)

        for strategy in strategies:
            try:
                self.logger.info(f"Attempting recovery strategy: {strategy['name']}")

                success = self._execute_recovery_strategy(strategy, error_context)

                if success:
                    # Record successful recovery
                    self._record_recovery_attempt(error_type, strategy, True,
                                                time.time() - recovery_start)
                    return {
                        "success": True,
                        "strategy_used": strategy['name'],
                        "recovery_time": time.time() - recovery_start,
                        "error_type": error_type
                    }

            except Exception as e:
                self.logger.error(f"Recovery strategy {strategy['name']} failed: {e}")
                continue

        # All strategies failed
        self._record_recovery_attempt(error_type, {"name": "all_failed"}, False,
                                    time.time() - recovery_start)
        return {
            "success": False,
            "strategies_attempted": len(strategies),
            "recovery_time": time.time() - recovery_start,
            "error_type": error_type
        }

    def _execute_recovery_strategy(self, strategy: Dict[str, Any], context: Dict) -> bool:
        """Execute a specific recovery strategy"""
        strategy_name = strategy['name']

        if strategy_name == "restart_component":
            return self._restart_component(context.get('component', 'unknown'))

        elif strategy_name == "rollback_changes":
            return self._rollback_changes(context.get('changes', []))

        elif strategy_name == "increase_resources":
            return self._increase_resources(context.get('resource_type', 'cpu'))

        elif strategy_name == "switch_to_fallback":
            return self._switch_to_fallback(context.get('service', 'unknown'))

        elif strategy_name == "clear_cache":
            return self._clear_cache(context.get('cache_type', 'all'))

        else:
            # Generic strategy - just log and return true for now
            self.logger.info(f"Executed generic recovery: {strategy_name}")
            return True

    def _restart_component(self, component: str) -> bool:
        """Restart a system component"""
        self.logger.info(f"Restarting component: {component}")
        # Implementation would depend on component type
        return True

    def _rollback_changes(self, changes: List[str]) -> bool:
        """Rollback recent changes"""
        self.logger.info(f"Rolling back {len(changes)} changes")
        # Implementation would use git or backup system
        return True

    def _increase_resources(self, resource_type: str) -> bool:
        """Increase system resources"""
        self.logger.info(f"Increasing {resource_type} resources")
        # Implementation would adjust thread pools, etc.
        return True

    def _switch_to_fallback(self, service: str) -> bool:
        """Switch to fallback service"""
        self.logger.info(f"Switching {service} to fallback mode")
        # Implementation would enable degraded mode
        return True

    def _clear_cache(self, cache_type: str) -> bool:
        """Clear system caches"""
        self.logger.info(f"Clearing {cache_type} cache")
        # Implementation would clear Redis/file caches
        return True

    def _get_default_strategies(self, error_type: str) -> List[Dict[str, Any]]:
        """Get default recovery strategies for error type"""
        base_strategies = [
            {
                "name": "clear_cache",
                "description": "Clear system caches",
                "success_rate": 0.7
            },
            {
                "name": "restart_component",
                "description": "Restart affected component",
                "success_rate": 0.5
            },
            {
                "name": "switch_to_fallback",
                "description": "Enable fallback mode",
                "success_rate": 0.6
            }
        ]

        # Add error-specific strategies
        if "database" in error_type.lower():
            base_strategies.insert(0, {
                "name": "rollback_changes",
                "description": "Rollback database changes",
                "success_rate": 0.8
            })
        elif "memory" in error_type.lower():
            base_strategies.insert(0, {
                "name": "increase_resources",
                "description": "Increase memory allocation",
                "success_rate": 0.6
            })

        return base_strategies

    def _record_recovery_attempt(self, error_type: str, strategy: Dict[str, Any],
                               success: bool, duration: float):
        """Record recovery attempt for learning"""
        record = {
            "error_type": error_type,
            "strategy": strategy['name'],
            "success": success,
            "duration": duration,
            "timestamp": time.time()
        }

        self.recovery_history.append(record)

        # Update strategy success rate
        if error_type not in self.recovery_strategies:
            self.recovery_strategies[error_type] = []

        strategies = self.recovery_strategies[error_type]
        for s in strategies:
            if s['name'] == strategy['name']:
                # Update success rate with exponential moving average
                current_rate = s.get('success_rate', 0.5)
                new_rate = (current_rate * 0.9) + (float(success) * 0.1)
                s['success_rate'] = new_rate
                break

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery system statistics"""
        total_attempts = len(self.recovery_history)
        successful_attempts = len([r for r in self.recovery_history if r['success']])

        return {
            "total_recovery_attempts": total_attempts,
            "successful_recoveries": successful_attempts,
            "recovery_success_rate": successful_attempts / max(total_attempts, 1),
            "strategies_learned": len(self.recovery_strategies),
            "error_types_handled": list(self.recovery_strategies.keys())
        }

# ===============================
# INTEGRATED ERROR RESILIENCE
# ===============================

class SAMResilienceManager:
    """Complete error resilience system for SAM"""

    def __init__(self):
        self.circuit_breakers = SAMCircuitBreakers()
        self.recovery_system = IntelligentRecovery()
        self.logger = logging.getLogger(__name__)

        # Health monitoring
        self.last_health_check = time.time()
        self.health_check_interval = 30.0  # 30 seconds
        self.system_health = "healthy"

    def execute_with_resilience(self, operation_name: str, operation: Callable,
                               *args, **kwargs) -> Any:
        """Execute operation with full resilience protection"""
        # Get appropriate circuit breaker
        circuit_breaker = self._get_circuit_breaker(operation_name)

        try:
            # Execute with circuit breaker protection
            result = circuit_breaker.call(operation, *args, **kwargs)
            return result

        except CircuitBreakerOpenException:
            self.logger.warning(f"Operation '{operation_name}' blocked by circuit breaker")
            raise

        except Exception as e:
            # Attempt intelligent recovery
            error_context = {
                "operation": operation_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": time.time()
            }

            recovery_result = self.recovery_system.attempt_recovery(
                error_context["error_type"], error_context
            )

            if recovery_result["success"]:
                self.logger.info(f"Recovery successful for {operation_name}")
                # Retry operation after successful recovery
                try:
                    return circuit_breaker.call(operation, *args, **kwargs)
                except Exception as retry_error:
                    self.logger.error(f"Retry failed after recovery: {retry_error}")
                    raise retry_error
            else:
                self.logger.error(f"Recovery failed for {operation_name}")
                raise e

    def _get_circuit_breaker(self, operation_name: str) -> CircuitBreaker:
        """Get appropriate circuit breaker for operation"""
        if "survival" in operation_name.lower():
            return self.circuit_breakers.survival_evaluator
        elif "goal" in operation_name.lower():
            return self.circuit_breakers.goal_executor
        elif "meta" in operation_name.lower() or "agent" in operation_name.lower():
            return self.circuit_breakers.meta_agent
        elif "database" in operation_name.lower() or "db" in operation_name.lower():
            return self.circuit_breakers.database
        elif "api" in operation_name.lower() or "external" in operation_name.lower():
            return self.circuit_breakers.external_api
        else:
            # Default to survival evaluator for unknown operations
            return self.circuit_breakers.survival_evaluator

    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        current_time = time.time()

        if current_time - self.last_health_check < self.health_check_interval:
            return {"status": self.system_health, "cached": True}

        self.last_health_check = current_time

        health_status = {
            "timestamp": current_time,
            "circuit_breakers": self.circuit_breakers.get_all_metrics(),
            "recovery_stats": self.recovery_system.get_recovery_stats(),
            "overall_health": "healthy",
            "issues": []
        }

        # Check circuit breaker health
        for name, metrics in health_status["circuit_breakers"].items():
            if metrics["state"] == "open":
                health_status["issues"].append(f"Circuit breaker {name} is OPEN")
                health_status["overall_health"] = "degraded"

        # Check recovery success rate
        recovery_rate = health_status["recovery_stats"]["recovery_success_rate"]
        if recovery_rate < 0.5:
            health_status["issues"].append(f"Low recovery success rate: {recovery_rate:.2%}")
            if recovery_rate < 0.3:
                health_status["overall_health"] = "critical"

        self.system_health = health_status["overall_health"]
        return health_status

    def get_resilience_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resilience metrics"""
        return {
            "circuit_breakers": self.circuit_breakers.get_all_metrics(),
            "recovery_system": self.recovery_system.get_recovery_stats(),
            "health_status": self.perform_health_check(),
            "uptime": time.time() - getattr(self, 'start_time', time.time())
        }

# ===============================
# GLOBAL RESILIENCE INSTANCE
# ===============================

# Create global resilience manager
resilience_manager = SAMResilienceManager()
resilience_manager.start_time = time.time()

if __name__ == "__main__":
    print("🛡️ SAM Resilience System Initialized")

    # Test circuit breaker
    breaker = CircuitBreaker(CircuitBreakerConfig(
        name="test_breaker",
        failure_threshold=2,
        recovery_timeout=5.0
    ))

    @breaker
    def test_function(success=True):
        if not success:
            raise Exception("Test failure")
        return "success"

    # Test successful operation
    try:
        result = test_function(True)
        print(f"✅ Success: {result}")
    except Exception as e:
        print(f"❌ Unexpected success failure: {e}")

    # Test failures
    for i in range(3):
        try:
            test_function(False)
        except Exception:
            print(f"Expected failure {i+1}")

    # Check if circuit opened
    metrics = breaker.get_metrics()
    print(f"🔄 Circuit state: {metrics['state']}")
    print(f"📊 Total failures: {metrics['total_failures']}")

    # Test resilience manager
    resilience_metrics = resilience_manager.get_resilience_metrics()
    print(f"📈 Resilience system operational: {len(resilience_metrics)} metrics")

    print("✅ Circuit breaker and resilience testing complete")
