#!/usr/bin/env python3
"""
Flask App Optimizer with Async Support
Optimized version of correct_sam_hub.py with timeout handling
"""

import asyncio
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Try to import aiohttp, fallback to mock if not available
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("⚠️ aiohttp not available, using mock async implementation")

class MockSession:
    """Mock session for testing without aiohttp"""
    async def get(self, url, **kwargs):
        return MockResponse(f"Mock GET response for {url}")
    
    async def post(self, url, **kwargs):
        return MockResponse(f"Mock POST response for {url}")
    
    async def close(self):
        pass

class MockResponse:
    """Mock response for testing"""
    def __init__(self, text):
        self._text = text
    
    async def text(self):
        return self._text

class AsyncOptimizer:
    """Optimize Flask app with async operations and better error handling"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.session = None
        
    async def __aenter__(self):
        if AIOHTTP_AVAILABLE:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=10)
            )
        else:
            # Mock session for testing
            self.session = MockSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and AIOHTTP_AVAILABLE:
            await self.session.close()
        self.executor.shutdown(wait=False)
    
    def async_timeout_handler(self, timeout_seconds=30):
        """Decorator to add timeout handling to functions"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    loop = asyncio.get_event_loop()
                    future = loop.run_in_executor(self.executor, func, *args, **kwargs)
                    return await asyncio.wait_for(future, timeout=timeout_seconds)
                except FutureTimeoutError:
                    print(f"⚠️ Function {func.__name__} timed out after {timeout_seconds}s")
                    return None
                except Exception as e:
                    print(f"❌ Error in {func.__name__}: {e}")
                    return None
            return async_wrapper
        return decorator
    
    async def async_web_request(self, url: str, method: str = 'GET', **kwargs):
        """Async web request with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if method.upper() == 'GET':
                    async with self.session.get(url, **kwargs) as response:
                        return await response.text()
                elif method.upper() == 'POST':
                    async with self.session.post(url, **kwargs) as response:
                        return await response.text()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ Web request failed after {max_retries} attempts: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return None
    
    def optimize_ollama_call(self, original_func):
        """Optimize Ollama calls with better timeout handling"""
        @wraps(original_func)
        def wrapper(*args, **kwargs):
            try:
                # Add timeout to subprocess calls
                import subprocess
                if 'timeout' not in kwargs:
                    kwargs['timeout'] = 25  # Default timeout
                
                result = original_func(*args, **kwargs)
                
                # Check if result is a subprocess.Popen
                if hasattr(result, 'communicate'):
                    try:
                        stdout, stderr = result.communicate(timeout=kwargs['timeout'])
                        return stdout.decode('utf-8') if stdout else ""
                    except subprocess.TimeoutExpired:
                        result.kill()
                        return None
                
                return result
                
            except Exception as e:
                print(f"⚠️ Optimized Ollama call failed: {e}")
                return None
        
        return wrapper

# Example usage in Flask app
class OptimizedFlaskApp:
    """Optimized Flask app with async support"""
    
    def __init__(self):
        self.optimizer = AsyncOptimizer()
        
    def apply_optimizations(self, app):
        """Apply optimizations to Flask app"""
        
        # Optimize common timeout-prone functions
        if hasattr(app, '_augment_with_neural_net'):
            app._augment_with_neural_net = self.optimizer.optimize_ollama_call(
                app._augment_with_neural_net
            )
        
        if hasattr(app, '_ollama_teacher_improve'):
            app._ollama_teacher_improve = self.optimizer.optimize_ollama_call(
                app._ollama_teacher_improve
            )
        
        # Add async route handlers
        @app.route('/api/async-search', methods=['POST'])
        async def async_search():
            """Async search endpoint"""
            data = await request.get_json()
            query = data.get('query', '')
            
            async with self.optimizer as opt:
                results = await opt.async_web_request(
                    f"https://duckduckgo.com/html/?q={query}"
                )
            
            return jsonify({'results': results})
        
        return app

# Memory optimization utilities
class MemoryOptimizer:
    """Optimize memory usage in Flask app"""
    
    @staticmethod
    def cleanup_after_request(response):
        """Clean up memory after each request"""
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear any cached data if needed
        return response
    
    @staticmethod
    def implement_connection_pooling():
        """Implement database/connection pooling"""
        # This would integrate with your existing connection management
        pass

# Caching utilities
class CacheManager:
    """Simple caching for frequently accessed data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_times = {}
        self.cache_ttl = 300  # 5 minutes
    
    def get(self, key):
        """Get cached value"""
        if key in self.cache:
            if time.time() - self.cache_times[key] < self.cache_ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.cache_times[key]
        return None
    
    def set(self, key, value):
        """Set cached value"""
        self.cache[key] = value
        self.cache_times[key] = time.time()
    
    def clear_expired(self):
        """Clear expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.cache_times.items()
            if current_time - timestamp > self.cache_ttl
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_times.pop(key, None)

# Global instances
cache_manager = CacheManager()
memory_optimizer = MemoryOptimizer()

def apply_all_optimizations(flask_app):
    """Apply all optimizations to Flask app"""
    optimizer = OptimizedFlaskApp()
    optimized_app = optimizer.apply_optimizations(flask_app)
    
    # Add memory cleanup
    optimized_app.after_request(memory_optimizer.cleanup_after_request)
    
    return optimized_app
