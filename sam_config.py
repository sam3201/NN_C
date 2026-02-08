# SAM Configuration Module
# System configuration and settings for SAM 2.0 AGI

config = {
    # System Identification
    'system_name': 'SAM 2.0 AGI',
    'system_version': '2.0.0',
    'system_description': 'Self-healing AGI system with multi-agent orchestration',

    # Core Settings
    'max_agents': 20,
    'default_agent_timeout': 30,
    'system_health_check_interval': 60,

    # RAM-Aware Settings
    'ram_warning_threshold': 0.8,  # 80% RAM usage
    'ram_critical_threshold': 0.9, # 90% RAM usage
    'ram_emergency_threshold': 0.95, # 95% RAM usage

    # Model Preferences
    'preferred_models': {
        'lightweight': ['phi:latest', 'codellama:latest'],
        'medium': ['mistral:latest', 'llama3.1:latest'],
        'heavy': ['deepseek-coder:latest', 'qwen2.5-coder:latest']
    },

    # Safety Settings
    'confidence_threshold': 0.8,
    'auto_resolution_enabled': False,  # Disabled during bootstrap
    'self_modification_enabled': False,  # Disabled during bootstrap
    'max_retries': 3,

    # Integration Settings
    'integrations': {
        'ollama': {'enabled': True, 'endpoint': 'http://localhost:11434'},
        'gmail': {'enabled': False},  # Requires API key
        'github': {'enabled': True},  # Read-only without token
        'google_drive': {'enabled': False}  # Requires credentials
    },

    # Logging Settings
    'log_level': 'INFO',
    'log_max_size': 10 * 1024 * 1024,  # 10MB
    'log_backup_count': 5,

    # Web Interface Settings
    'web_port': 5004,
    'web_host': '0.0.0.0',
    'web_debug': False,

    # Performance Settings
    'max_concurrent_tasks': 5,
    'task_timeout': 300,  # 5 minutes
    'cleanup_interval': 3600,  # 1 hour

    # Experimental Features
    'experimental_features': {
        'auto_code_modification': False,
        'dynamic_agent_creation': False,
        'advanced_self_healing': False
    }
}

# Runtime configuration (can be modified at runtime)
runtime_config = {
    'bootstrap_complete': False,
    'system_ready': False,
    'emergency_mode': False,
    'maintenance_mode': False
}

def get_config(key=None, default=None):
    """Get configuration value"""
    if key is None:
        return config

    keys = key.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value

def set_config(key, value):
    """Set configuration value"""
    keys = key.split('.')
    target = config
    for k in keys[:-1]:
        if k not in target:
            target[k] = {}
        target = target[k]
    target[keys[-1]] = value

def update_runtime_config(key, value):
    """Update runtime configuration"""
    runtime_config[key] = value

def get_runtime_config(key=None, default=None):
    """Get runtime configuration value"""
    if key is None:
        return runtime_config
    return runtime_config.get(key, default)
