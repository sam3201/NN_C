#!/usr/bin/env python3
"""
SAM 2.0 Production Web Server
Optimized Flask + Gunicorn deployment with performance enhancements
"""

import os
import sys
import logging
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import time
from functools import wraps
import threading
import json
from datetime import datetime

# Import SAM components
from sam_config import config
from correct_sam_hub import CorrectSAMHub
from survival_agent import create_survival_agent
from goal_management import GoalManager, SubgoalExecutionAlgorithm, create_conversationalist_tasks

# ===============================
# PERFORMANCE OPTIMIZATIONS
# ===============================

def rate_limit(max_requests: int, window_seconds: int = 60):
    """Rate limiting decorator"""
    requests = {}

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()

            if client_ip not in requests:
                requests[client_ip] = []

            # Clean old requests
            requests[client_ip] = [t for t in requests[client_ip]
                                 if current_time - t < window_seconds]

            if len(requests[client_ip]) >= max_requests:
                return jsonify({"error": "Rate limit exceeded"}), 429

            requests[client_ip].append(current_time)
            return f(*args, **kwargs)
        return wrapper
    return decorator

def timing_middleware(app):
    """Performance timing middleware"""
    @app.before_request
    def start_timer():
        request.start_time = time.time()

    @app.after_request
    def log_request(response):
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            app.logger.info('.4f'
                          f'[{request.method}] {request.path} -> {response.status_code}')
        return response

# ===============================
# OPTIMIZED FLASK APPLICATION
# ===============================

class OptimizedSAMWebServer:
    """Production-ready web server with optimizations"""

    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = config.SECRET_KEY
        self.app.wsgi_app = ProxyFix(self.app.wsgi_app, x_proto=1, x_host=1)

        # Configure logging
        self.setup_logging()

        # Apply optimizations
        timing_middleware(self.app)
        CORS(self.app, origins=config.ALLOWED_ORIGINS, supports_credentials=True)

        # Initialize SAM components
        self.sam_hub = None
        self.survival_agent = None
        self.goal_manager = None
        self.goal_executor = None

        # Performance metrics
        self.request_count = 0
        self.avg_response_time = 0.0
        self.error_count = 0

    def setup_logging(self):
        """Configure production logging"""
        log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

        # File handler
        if config.LOG_FILE:
            file_handler = logging.FileHandler(config.LOG_FILE)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # Configure app logger
        self.app.logger.handlers = []
        if config.CONSOLE_LOGGING:
            self.app.logger.addHandler(console_handler)
        if config.LOG_FILE:
            self.app.logger.addHandler(file_handler)
        self.app.logger.setLevel(log_level)

    def initialize_sam_components(self):
        """Initialize all SAM components with error handling"""
        try:
            self.app.logger.info("Initializing SAM components...")

            # Initialize SAM Hub
            if config.ENABLE_GOAL_MANAGEMENT:
                self.sam_hub = CorrectSAMHub()
                self.app.logger.info("✅ SAM Hub initialized")

            # Initialize Survival Agent
            if config.ENABLE_SURVIVAL_AGENT:
                self.survival_agent = create_survival_agent()
                self.app.logger.info("✅ Survival Agent initialized")

            # Initialize Goal Management
            if config.ENABLE_GOAL_MANAGEMENT:
                self.goal_manager = GoalManager()
                create_conversationalist_tasks(self.goal_manager)
                self.goal_executor = SubgoalExecutionAlgorithm(self.goal_manager)
                self.app.logger.info("✅ Goal Management initialized")

            self.app.logger.info("✅ All SAM components initialized successfully")

        except Exception as e:
            self.app.logger.error(f"❌ SAM initialization failed: {e}")
            raise

    def setup_routes(self):
        """Setup optimized API routes"""

        @self.app.route('/')
        def index():
            """SAM Dashboard"""
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>SAM 2.0 - Survival-First AGI</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .metric { background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }
                    .status-good { color: green; }
                    .status-warning { color: orange; }
                    .status-error { color: red; }
                </style>
            </head>
            <body>
                <h1>🧠 SAM 2.0 - Survival-First AGI System</h1>
                <div class="metric">
                    <h3>System Status</h3>
                    <p>Status: <span class="status-good">✅ Operational</span></p>
                    <p>Uptime: <span id="uptime">Loading...</span></p>
                </div>
                <div class="metric">
                    <h3>Survival Metrics</h3>
                    <p>Survival Score: <span id="survival">Loading...</span></p>
                    <p>Active Goals: <span id="goals">Loading...</span></p>
                </div>
                <div class="metric">
                    <h3>API Endpoints</h3>
                    <ul>
                        <li><a href="/api/system/status">System Status</a></li>
                        <li><a href="/api/survival/status">Survival Status</a></li>
                        <li><a href="/api/goals/status">Goal Status</a></li>
                        <li><a href="/api/performance/metrics">Performance Metrics</a></li>
                    </ul>
                </div>
                <script>
                    // Auto-refresh metrics
                    setInterval(() => {
                        fetch('/api/system/status')
                            .then(r => r.json())
                            .then(data => {
                                document.getElementById('uptime').textContent = Math.floor(data.uptime / 60) + 'm ' + (data.uptime % 60) + 's';
                            });
                        fetch('/api/survival/status')
                            .then(r => r.json())
                            .then(data => {
                                document.getElementById('survival').textContent = (data.survival_score * 100).toFixed(1) + '%';
                                document.getElementById('goals').textContent = data.active_goals;
                            });
                    }, 5000);
                </script>
            </body>
            </html>
            """)

        @self.app.route('/api/system/status')
        @rate_limit(config.RATE_LIMIT_REQUESTS)
        def system_status():
            """Optimized system status endpoint"""
            try:
                status = {
                    "status": "operational",
                    "timestamp": datetime.now().isoformat(),
                    "version": "2.0.0",
                    "uptime": time.time() - getattr(self, 'start_time', time.time()),
                    "config": config.to_dict(),
                    "performance": {
                        "request_count": self.request_count,
                        "avg_response_time": self.avg_response_time,
                        "error_count": self.error_count
                    }
                }

                if self.sam_hub:
                    status["sam_hub"] = "active"
                if self.survival_agent:
                    status["survival_score"] = self.survival_agent.survival_score
                if self.goal_manager:
                    status["goals_completed"] = len([t for t in self.goal_manager.subtasks if t.completed])
                    status["total_goals"] = len(self.goal_manager.subtasks)

                return jsonify(status)
            except Exception as e:
                self.error_count += 1
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/survival/status')
        @rate_limit(config.RATE_LIMIT_REQUESTS)
        def survival_status():
            """Survival agent status"""
            try:
                if not self.survival_agent:
                    return jsonify({"error": "Survival agent not enabled"}), 503

                # Get current survival assessment
                context = {
                    "threats": {"high": False, "medium": False, "low": False},
                    "resources": {"adequate": True},
                    "timestamp": datetime.now().isoformat()
                }

                return jsonify({
                    "survival_score": self.survival_agent.survival_score,
                    "confidence_threshold": config.CONFIDENCE_THRESHOLD,
                    "risk_tolerance": config.RISK_TOLERANCE,
                    "active_context": context,
                    "last_updated": datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/goals/status')
        @rate_limit(config.RATE_LIMIT_REQUESTS)
        def goals_status():
            """Goal management status"""
            try:
                if not self.goal_manager:
                    return jsonify({"error": "Goal management not enabled"}), 503

                return jsonify({
                    "goal_completed": self.goal_manager.is_goal_completed(),
                    "total_tasks": len(self.goal_manager.subtasks),
                    "completed_tasks": len([t for t in self.goal_manager.subtasks if t.completed]),
                    "pending_tasks": len(self.goal_manager.get_pending_tasks()),
                    "critical_tasks": len([t for t in self.goal_manager.subtasks if t.critical]),
                    "active_task": self.goal_manager.active_task.name if self.goal_manager.active_task else None,
                    "readme_available": os.path.exists("SAM_GOALS_README.md")
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/performance/metrics')
        @rate_limit(config.RATE_LIMIT_REQUESTS)
        def performance_metrics():
            """Performance metrics endpoint"""
            try:
                return jsonify({
                    "c_optimization_enabled": config.ENABLE_C_OPTIMIZATION,
                    "monitoring_interval": config.MONITORING_INTERVAL,
                    "max_concurrent_tasks": config.MAX_CONCURRENT_TASKS,
                    "rate_limit": config.RATE_LIMIT_REQUESTS,
                    "workers": config.WORKERS,
                    "system_metrics": {
                        "request_count": self.request_count,
                        "avg_response_time": ".4f",
                        "error_rate": (self.error_count / max(self.request_count, 1)) * 100,
                        "uptime": time.time() - getattr(self, 'start_time', time.time())
                    }
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/goals/readme')
        def goals_readme():
            """Serve the goals README file"""
            try:
                if os.path.exists("SAM_GOALS_README.md"):
                    with open("SAM_GOALS_README.md", "r", encoding='utf-8') as f:
                        content = f.read()
                    return jsonify({"readme": content})
                else:
                    return jsonify({"error": "README not generated yet"}), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def start_background_monitoring(self):
        """Start optimized background monitoring"""
        def monitoring_loop():
            while True:
                try:
                    # Survival evaluation
                    if self.survival_agent and self.goal_manager:
                        threats = {"high": False, "medium": False, "low": False}
                        pending_critical = len(self.goal_manager.get_critical_tasks())

                        if pending_critical > 2:
                            threats["medium"] = True

                        context = {
                            "threats": threats,
                            "resources": {"adequate": True},
                            "timestamp": datetime.now().isoformat()
                        }

                        # Update survival score based on context
                        if threats["high"]:
                            self.survival_agent.survival_score = max(0.0, self.survival_agent.survival_score - 0.05)
                        elif threats["medium"]:
                            self.survival_agent.survival_score = max(0.0, self.survival_agent.survival_score - 0.02)

                    # Goal execution cycle
                    if self.goal_executor:
                        cycle_result = self.goal_executor.execute_cycle()
                        if cycle_result["tasks_executed"] > 0:
                            self.app.logger.info(f"🎯 Goal cycle: {cycle_result['tasks_executed']} tasks executed")

                    # Update README periodically
                    if self.goal_manager and hasattr(self.goal_manager, 'export_readme'):
                        self.goal_manager.export_readme()

                    time.sleep(config.MONITORING_INTERVAL)

                except Exception as e:
                    self.app.logger.error(f"Monitoring error: {e}")
                    self.error_count += 1
                    time.sleep(5)

        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        self.app.logger.info("✅ Background monitoring started with optimizations")

    def run(self, host: str = None, port: int = None, debug: bool = None):
        """Run the optimized web server"""
        host = host or config.HOST
        port = port or config.PORT
        debug = debug if debug is not None else config.DEBUG

        self.start_time = time.time()

        self.app.logger.info("🚀 Starting Optimized SAM Web Server")
        self.app.logger.info(f"🌐 Host: {host}:{port}")
        self.app.logger.info(f"🔧 Workers: {config.WORKERS}")
        self.app.logger.info(f"⚡ C Optimization: {'Enabled' if config.ENABLE_C_OPTIMIZATION else 'Disabled'}")
        self.app.logger.info(f"🛡️ Survival Agent: {'Enabled' if config.ENABLE_SURVIVAL_AGENT else 'Disabled'}")
        self.app.logger.info(f"🎯 Goal Management: {'Enabled' if config.ENABLE_GOAL_MANAGEMENT else 'Disabled'}")

        # In production, use a WSGI server like Gunicorn
        if not debug and config.WORKERS > 1:
            self.app.logger.info("💡 Production mode: Consider using Gunicorn for better performance")
            self.app.logger.info(f"   gunicorn -w {config.WORKERS} -b {host}:{port} sam_web_server:app")

        self.app.run(host=host, port=port, debug=debug, threaded=True)

# ===============================
# PRODUCTION DEPLOYMENT
# ===============================

def create_production_app():
    """Create production-ready SAM application"""
    server = OptimizedSAMWebServer()
    server.initialize_sam_components()
    server.setup_routes()
    server.start_background_monitoring()
    return server.app

def create_development_app():
    """Create development SAM application"""
    server = OptimizedSAMWebServer()
    server.initialize_sam_components()
    server.setup_routes()
    server.start_background_monitoring()
    return server

# For Gunicorn deployment
app = create_production_app()

if __name__ == "__main__":
    server = create_development_app()
    server.run()
