# Secrets Loader - Auto-loads API keys from .secrets/
# DO NOT COMMIT THIS FILE TO GIT

import os
from pathlib import Path

SECRETS_DIR = Path(__file__).parent / ".secrets"

def load_secrets():
    """Load API keys from secrets directory"""
    
    # Load Kimi K2.5
    kimi_file = SECRETS_DIR / "KIMI_K_2.5.py"
    if kimi_file.exists():
        content = kimi_file.read_text()
        # Extract API key from the file
        for line in content.split('\n'):
            if 'Authorization' in line and 'Bearer' in line:
                key = line.split('Bearer')[1].strip().strip('",')
                os.environ.setdefault('KIMI_API_KEY', key)
                print(f"✅ Loaded Kimi API key")
                break
    
    # Load other secrets as needed
    # Add more secrets loaders here

# Auto-load on import
load_secrets()
