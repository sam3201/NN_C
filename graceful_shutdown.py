#!/usr/bin/env python3
"""
Graceful Shutdown Manager for SAM 2.0
Handles graceful shutdown on SIGINT, SIGTERM, and other signals
"""

import signal
import sys
import time
import json
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Callable, Optional

class GracefulShutdownManager:
    """Manages graceful shutdown of SAM 2.0 components"""
    
    def __init__(self):
        self.shutdown_handlers = []
        self.shutdown_flag = False
        self.shutdown_start_time = None
        self.max_shutdown_time = 30  # seconds
        self.components_status = {}
        
        # Register signal handlers
        self._register_signal_handlers()
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Windows-specific signals
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        if self.shutdown_flag:
            print("\n🚨 Force shutdown detected. Exiting immediately...")
            sys.exit(1)
        
        print(f"\n🛑 Shutdown signal received ({signal.Signals(signum).name})")
        print("🔄 Starting graceful shutdown...")
        
        self.shutdown_flag = True
        self.shutdown_start_time = time.time()
        
        # Start shutdown process
        self._initiate_graceful_shutdown()
    
    def register_component(self, name: str, shutdown_func: Callable, priority: int = 0):
        """Register a component for graceful shutdown"""
        self.shutdown_handlers.append({
            'name': name,
            'shutdown_func': shutdown_func,
            'priority': priority,
            'status': 'running'
        })
        
        # Sort by priority (higher priority = shuts down first)
        self.shutdown_handlers.sort(key=lambda x: x['priority'], reverse=True)
        
        print(f"✅ Registered component for shutdown: {name} (priority: {priority})")
    
    def _initiate_graceful_shutdown(self):
        """Initiate graceful shutdown of all components"""
        print(f"📋 Shutting down {len(self.shutdown_handlers)} components...")
        
        # Shutdown components in priority order
        for handler in self.shutdown_handlers:
            if self.shutdown_flag:
                try:
                    print(f"🔄 Shutting down: {handler['name']}...")
                    handler['status'] = 'shutting_down'
                    
                    # Call shutdown function with timeout
                    start_time = time.time()
                    handler['shutdown_func']()
                    end_time = time.time()
                    
                    handler['status'] = 'shutdown'
                    handler['shutdown_time'] = end_time - start_time
                    
                    print(f"✅ {handler['name']} shutdown complete ({handler['shutdown_time']:.2f}s)")
                    
                except Exception as e:
                    handler['status'] = 'error'
                    handler['error'] = str(e)
                    print(f"❌ Error shutting down {handler['name']}: {e}")
        
        # Final cleanup
        self._final_cleanup()
    
    def _final_cleanup(self):
        """Perform final cleanup operations"""
        print("🧹 Performing final cleanup...")
        
        # Save shutdown report
        self._save_shutdown_report()
        
        # Clear any remaining resources
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Close any remaining file handles
            import sys
            if hasattr(sys, 'stdout'):
                sys.stdout.flush()
            if hasattr(sys, 'stderr'):
                sys.stderr.flush()
            
        except Exception as e:
            print(f"⚠️ Error during final cleanup: {e}")
        
        shutdown_duration = time.time() - self.shutdown_start_time
        print(f"✅ Graceful shutdown complete ({shutdown_duration:.2f}s)")
    
    def _save_shutdown_report(self):
        """Save shutdown report to file"""
        report = {
            'shutdown_time': datetime.now().isoformat(),
            'shutdown_duration': time.time() - self.shutdown_start_time if self.shutdown_start_time else 0,
            'signal_received': self.shutdown_flag,
            'components': self.shutdown_handlers,
            'total_components': len(self.shutdown_handlers),
            'successful_shutdowns': sum(1 for h in self.shutdown_handlers if h['status'] == 'shutdown'),
            'failed_shutdowns': sum(1 for h in self.shutdown_handlers if h['status'] == 'error')
        }
        
        try:
            with open('shutdown_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            print("📄 Shutdown report saved to shutdown_report.json")
        except Exception as e:
            print(f"⚠️ Could not save shutdown report: {e}")
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress"""
        return self.shutdown_flag
    
    def get_component_status(self) -> Dict:
        """Get status of all registered components"""
        return {h['name']: h['status'] for h in self.shutdown_handlers}

# Global shutdown manager
shutdown_manager = GracefulShutdownManager()

def register_shutdown_handler(name: str, shutdown_func: Callable, priority: int = 0):
    """Register a component for graceful shutdown"""
    shutdown_manager.register_component(name, shutdown_func, priority)

def is_shutting_down() -> bool:
    """Check if shutdown is in progress"""
    return shutdown_manager.is_shutting_down()

def get_component_status() -> Dict:
    """Get status of all components"""
    return shutdown_manager.get_component_status()

# Decorator for graceful shutdown handling
def graceful_shutdown_handler(name: str, priority: int = 0):
    """Decorator to register function for graceful shutdown"""
    def decorator(func):
        register_shutdown_handler(name, func, priority)
        return func
    return decorator

# Context manager for shutdown-aware operations
class ShutdownAwareContext:
    """Context manager that checks for shutdown during operations"""
    
    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        if is_shutting_down():
            raise InterruptedError(f"Cannot start {self.operation_name} - shutdown in progress")
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if is_shutting_down():
            print(f"⚠️ {self.operation_name} interrupted by shutdown")
        return False

def shutdown_aware_operation(operation_name: str = "operation"):
    """Create shutdown-aware context manager"""
    return ShutdownAwareContext(operation_name)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Graceful Shutdown Manager")
    print("=" * 50)
    
    # Test component shutdown functions
    def shutdown_flask():
        print("  🔄 Closing Flask connections...")
        time.sleep(1)
        print("  ✅ Flask shutdown complete")
    
    def shutdown_database():
        print("  🔄 Closing database connections...")
        time.sleep(0.5)
        print("  ✅ Database shutdown complete")
    
    def shutdown_monitoring():
        print("  🔄 Stopping monitoring...")
        time.sleep(0.3)
        print("  ✅ Monitoring shutdown complete")
    
    # Register components
    register_shutdown_handler("Flask Server", shutdown_flask, priority=3)
    register_shutdown_handler("Database", shutdown_database, priority=2)
    register_shutdown_handler("Monitoring", shutdown_monitoring, priority=1)
    
    print("✅ Components registered for graceful shutdown")
    print("📋 Press Ctrl+C to test graceful shutdown")
    print("🔄 System running...")
    
    try:
        # Simulate system running
        while not is_shutting_down():
            time.sleep(1)
            print(f"  ⏰ System active... (Press Ctrl+C to shutdown)")
            
    except KeyboardInterrupt:
        # This will be caught by the signal handler
        pass
    
    print("🏁 Test complete")
