# SAM Configuration Module
# System configuration and settings for SAM 2.0 AGI

import os

config = {
    # System Identification
    'system_name': 'SAM-D',
    'system_version': '5.0.0 (ΨΔ•Ω-Core Recursive)',
    'system_description': 'Self-healing AGI system with multi-agent orchestration and ΨΔ-Core morphogenesis',

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
        'google_drive': {
            'enabled': False,  # Requires credentials
            'client_id': None,
            'client_secret': None
        }
    },

    # Security Settings
    'security': {
        'admin_emails_allowlist': [],  # List of email addresses allowed to perform admin actions
        'ip_allowlist': [],            # List of IP addresses allowed to access sensitive endpoints
        'restrict_admin_panel': False  # Whether to restrict admin panel access to allowed IPs/Emails
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

def load_env_config():
    """Load configuration from environment variables."""
    admin_emails = os.getenv("SAM_ADMIN_EMAILS")
    if admin_emails:
        config['security']['admin_emails_allowlist'] = [e.strip() for e in admin_emails.split(',')]

    ip_allowlist = os.getenv("SAM_IP_ALLOWLIST")
    allowed_ips = os.getenv("SAM_ALLOWED_IPS")
    combined_ips = []
    if ip_allowlist:
        combined_ips.extend([ip.strip() for ip in ip_allowlist.split(',')])
    if allowed_ips:
        combined_ips.extend([ip.strip() for ip in allowed_ips.split(',')])
    
    if combined_ips:
        config['security']['ip_allowlist'] = list(set(combined_ips)) # De-duplicate
    
    restrict_admin = os.getenv("SAM_RESTRICT_ADMIN_PANEL")
    if restrict_admin:
        config['security']['restrict_admin_panel'] = restrict_admin.lower() in ('true', '1', 'yes')

    # Google Drive Integration
    google_client_id = os.getenv("SAM_GOOGLE_CLIENT_ID")
    google_client_secret = os.getenv("SAM_GOOGLE_CLIENT_SECRET")
    if google_client_id and google_client_secret:
        config['integrations']['google_drive']['enabled'] = True
        config['integrations']['google_drive']['client_id'] = google_client_id
        config['integrations']['google_drive']['client_secret'] = google_client_secret


# Load environment configuration on import
load_env_config()
