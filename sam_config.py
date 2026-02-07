#!/usr/bin/env python3
"""
SAM 2.0 Production Configuration
Environment variables and performance tuning
"""

import os
from typing import Dict, Any

# ===============================
# PRODUCTION CONFIGURATION
# ===============================

class SAMConfig:
    """Centralized configuration management for SAM system"""

    def __init__(self):
        # Web Server Configuration
        self.HOST = os.getenv('SAM_HOST', '127.0.0.1')
        self.PORT = int(os.getenv('SAM_PORT', '8080'))
        self.DEBUG = os.getenv('SAM_DEBUG', 'false').lower() == 'true'
        self.WORKERS = int(os.getenv('SAM_WORKERS', '4'))  # For Gunicorn

        # Performance Tuning
        self.MAX_CONCURRENT_TASKS = int(os.getenv('SAM_MAX_TASKS', '10'))
        self.MONITORING_INTERVAL = int(os.getenv('SAM_MONITOR_INTERVAL', '30'))
        self.CONFIDENCE_THRESHOLD = float(os.getenv('SAM_CONFIDENCE_THRESHOLD', '0.6'))
        self.RISK_TOLERANCE = float(os.getenv('SAM_RISK_TOLERANCE', '0.3'))

        # Storage Configuration
        self.DATA_DIR = os.getenv('SAM_DATA_DIR', './sam_data')
        self.BACKUP_INTERVAL = int(os.getenv('SAM_BACKUP_INTERVAL', '3600'))  # 1 hour
        self.MAX_LOG_SIZE = int(os.getenv('SAM_MAX_LOG_SIZE', '10000000'))  # 10MB

        # Security Configuration
        self.SECRET_KEY = os.getenv('SAM_SECRET_KEY', 'dev-secret-key-change-in-production')
        self.ALLOWED_ORIGINS = os.getenv('SAM_ALLOWED_ORIGINS', 'http://localhost:8080,http://127.0.0.1:8080').split(',')
        self.RATE_LIMIT_REQUESTS = int(os.getenv('SAM_RATE_LIMIT', '100'))  # per minute

        # AI/ML Configuration
        self.LLM_MODEL = os.getenv('SAM_LLM_MODEL', 'qwen2.5-coder:7b')
        self.OLLAMA_HOST = os.getenv('SAM_OLLAMA_HOST', 'http://localhost:11434')
        self.EMBEDDING_MODEL = os.getenv('SAM_EMBEDDING_MODEL', 'sentence-transformers')

        # Survival Configuration
        self.SURVIVAL_UPDATE_INTERVAL = int(os.getenv('SAM_SURVIVAL_UPDATE', '300'))  # 5 minutes
        self.CRITICAL_THREAT_THRESHOLD = float(os.getenv('SAM_CRITICAL_THREAT', '0.8'))
        self.BACKUP_ON_THREAT = os.getenv('SAM_BACKUP_ON_THREAT', 'true').lower() == 'true'

        # Logging Configuration
        self.LOG_LEVEL = os.getenv('SAM_LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.getenv('SAM_LOG_FILE', 'sam_system.log')
        self.CONSOLE_LOGGING = os.getenv('SAM_CONSOLE_LOG', 'true').lower() == 'true'

        # Feature Flags
        self.ENABLE_C_OPTIMIZATION = os.getenv('SAM_ENABLE_C_OPT', 'true').lower() == 'true'
        self.ENABLE_SURVIVAL_AGENT = os.getenv('SAM_ENABLE_SURVIVAL', 'true').lower() == 'true'
        self.ENABLE_GOAL_MANAGEMENT = os.getenv('SAM_ENABLE_GOALS', 'true').lower() == 'true'
        self.ENABLE_AUTO_BACKUP = os.getenv('SAM_ENABLE_BACKUP', 'true').lower() == 'true'

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'web_server': {
                'host': self.HOST,
                'port': self.PORT,
                'debug': self.DEBUG,
                'workers': self.WORKERS
            },
            'performance': {
                'max_concurrent_tasks': self.MAX_CONCURRENT_TASKS,
                'monitoring_interval': self.MONITORING_INTERVAL,
                'confidence_threshold': self.CONFIDENCE_THRESHOLD,
                'risk_tolerance': self.RISK_TOLERANCE
            },
            'storage': {
                'data_dir': self.DATA_DIR,
                'backup_interval': self.BACKUP_INTERVAL,
                'max_log_size': self.MAX_LOG_SIZE
            },
            'security': {
                'rate_limit_requests': self.RATE_LIMIT_REQUESTS,
                'allowed_origins': self.ALLOWED_ORIGINS
            },
            'ai_config': {
                'llm_model': self.LLM_MODEL,
                'ollama_host': self.OLLAMA_HOST,
                'embedding_model': self.EMBEDDING_MODEL
            },
            'survival': {
                'update_interval': self.SURVIVAL_UPDATE_INTERVAL,
                'critical_threat_threshold': self.CRITICAL_THREAT_THRESHOLD,
                'backup_on_threat': self.BACKUP_ON_THREAT
            },
            'logging': {
                'level': self.LOG_LEVEL,
                'file': self.LOG_FILE,
                'console': self.CONSOLE_LOGGING
            },
            'features': {
                'c_optimization': self.ENABLE_C_OPTIMIZATION,
                'survival_agent': self.ENABLE_SURVIVAL_AGENT,
                'goal_management': self.ENABLE_GOAL_MANAGEMENT,
                'auto_backup': self.ENABLE_AUTO_BACKUP
            }
        }

    def validate(self) -> bool:
        """Validate configuration settings"""
        issues = []

        # Check required directories
        if not os.path.exists(self.DATA_DIR):
            try:
                os.makedirs(self.DATA_DIR, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create data directory: {e}")

        # Check port availability
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((self.HOST, self.PORT)) == 0:
                issues.append(f"Port {self.PORT} on {self.HOST} is already in use")

        # Check numeric ranges
        if not (0.0 <= self.CONFIDENCE_THRESHOLD <= 1.0):
            issues.append("Confidence threshold must be between 0.0 and 1.0")

        if not (0.0 <= self.RISK_TOLERANCE <= 1.0):
            issues.append("Risk tolerance must be between 0.0 and 1.0")

        if issues:
            print("❌ Configuration validation failed:")
            for issue in issues:
                print(f"   - {issue}")
            return False

        print("✅ Configuration validation passed")
        return True

# Global configuration instance
config = SAMConfig()

if __name__ == "__main__":
    print("🔧 SAM 2.0 Configuration Manager")
    print("=" * 50)

    # Validate configuration
    if config.validate():
        print("\n📋 Current Configuration:")
        import json
        print(json.dumps(config.to_dict(), indent=2))

        print("\n💡 To modify settings, set environment variables:")
        print("   export SAM_PORT=9090")
        print("   export SAM_DEBUG=true")
        print("   export SAM_WORKERS=8")
        print("   export SAM_MAX_TASKS=20")
    else:
        print("\n❌ Please fix configuration issues before running SAM")
