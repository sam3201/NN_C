#!/usr/bin/env python3
"""
SAM Admin Account Manager
Creates and manages admin accounts for SAM system access
"""

import os
import json
import hashlib
from pathlib import Path

# Admin credentials file
ADMIN_CREDS_FILE = Path("secrets/admin_credentials.json")

def create_admin_account():
    """Create admin account for SAM access"""
    print("🔐 SAM Admin Account Creator")
    print("=" * 40)
    
    # Ensure secrets directory exists
    ADMIN_CREDS_FILE.parent.mkdir(exist_ok=True)
    
    # Default admin credentials
    admin_accounts = {
        "sam_admin@localhost.local": {
            "password": "sam_admin_2024",
            "role": "admin",
            "created": "2024-02-09",
            "purpose": "Local system administration"
        },
        "meta_agent@sam.system": {
            "password": "meta_test_2024", 
            "role": "meta_agent",
            "created": "2024-02-09",
            "purpose": "MetaAgent testing and validation"
        }
    }
    
    # Save credentials
    try:
        with open(ADMIN_CREDS_FILE, 'w') as f:
            json.dump(admin_accounts, f, indent=2)
        
        print(f"✅ Admin credentials saved to: {ADMIN_CREDS_FILE}")
        print("\n📋 Admin Accounts Created:")
        print("-" * 30)
        
        for email, data in admin_accounts.items():
            print(f"📧 Email: {email}")
            print(f"🔑 Password: {data['password']}")
            print(f"👤 Role: {data['role']}")
            print(f"📝 Purpose: {data['purpose']}")
            print("-" * 30)
        
        print("\n🌐 Login URLs:")
        print(f"   Local: http://localhost:5004/login")
        print(f"   Dashboard: http://localhost:5004/")
        
        print("\n🔧 Usage:")
        print("   1. Use these credentials to login to SAM")
        print("   2. Access MetaAgent validation at /api/meta/test")
        print("   3. Full admin privileges granted")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating admin accounts: {e}")
        return False

def load_admin_credentials():
    """Load existing admin credentials"""
    if ADMIN_CREDS_FILE.exists():
        try:
            with open(ADMIN_CREDS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading credentials: {e}")
            return None
    return None

if __name__ == "__main__":
    # Check if accounts already exist
    existing = load_admin_credentials()
    if existing:
        print("📋 Existing Admin Accounts:")
        print("-" * 30)
        for email, data in existing.items():
            print(f"📧 {email}")
            print(f"🔑 Password: {data['password']}")
            print(f"👤 Role: {data['role']}")
            print("-" * 30)
        
        response = input("\n🔄 Recreate accounts? (y/N): ").strip().lower()
        if response != 'y':
            print("✅ Keeping existing accounts")
            exit(0)
    
    # Create new accounts
    if create_admin_account():
        print("\n🎉 Admin accounts created successfully!")
        print("💡 Store these credentials securely for future access")
    else:
        print("\n❌ Failed to create admin accounts")
