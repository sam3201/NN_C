# SAM Auth Manager - Secure Version
# Handles token generation, role-based verification, and encrypted persistence

import os
import json
import secrets
import string
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

class AuthManager:
    """
    Manages access tokens and RBAC for SAM-D.
    - Owner: Persistent master token.
    - Admin: Persistent privileged tokens.
    - User: Transient session tokens.
    Secrets are encrypted at rest using Fernet.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.auth_path = project_root / "sam_data" / "auth.enc"
        self.key_path = project_root / "secrets" / "master.key"
        self.audit_log_path = project_root / "logs" / "audit.jsonl"
        self.tokens = {} # token -> {user_id, role, expiry}
        self.user_data = {} # user_id -> {role, name, permanent_token}
        
        self._setup_encryption()
        self._load_auth()
        self._ensure_owner_token()

    def _setup_encryption(self):
        """Initialize or load the encryption key."""
        if not self.key_path.exists():
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            key = Fernet.generate_key()
            self.key_path.write_bytes(key)
            # Restrict permissions
            try:
                os.chmod(self.key_path, 0o600)
            except:
                pass
        
        self.fernet = Fernet(self.key_path.read_bytes()) if CRYPTO_AVAILABLE else None

    def _load_auth(self):
        if self.auth_path.exists():
            try:
                data_bytes = self.auth_path.read_bytes()
                if self.fernet:
                    decrypted = self.fernet.decrypt(data_bytes)
                    data = json.loads(decrypted)
                else:
                    data = json.loads(data_bytes)
                
                self.user_data = data.get("users", {})
                # Load permanent tokens into active map
                for uid, uinfo in self.user_data.items():
                    if "token" in uinfo:
                        self.tokens[uinfo["token"]] = {"user_id": uid, "role": uinfo["role"]}
            except Exception as e:
                print(f"⚠️ Auth load failed (likely decryption error): {e}")

    def _save_auth(self):
        try:
            self.auth_path.parent.mkdir(parents=True, exist_ok=True)
            data_str = json.dumps({"users": self.user_data}, indent=2)
            if self.fernet:
                encrypted = self.fernet.encrypt(data_str.encode())
                self.auth_path.write_bytes(encrypted)
            else:
                self.auth_path.write_text(data_str)
            
            # Restrict permissions
            try:
                os.chmod(self.auth_path, 0o600)
            except:
                pass
        except Exception as e:
            print(f"⚠️ Auth save failed: {e}")

    def _ensure_owner_token(self):
        """Generate and protect the master owner token."""
        owner_id = os.getenv("SAM_OWNER_ID", "owner_1")
        parent_identity = os.getenv("SAM_PARENT_IDENTITY", "UNSET")
        
        if parent_identity == "UNSET":
            print("⚠️ WARNING: SAM_PARENT_IDENTITY is not set. System is in UNPARENTED mode.")
        
        # Check if owner already has a token
        if owner_id in self.user_data and "token" in self.user_data[owner_id]:
            return

        # Generate a high-entropy master token
        master_token = "sam_owner_" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(48))
        
        self.user_data[owner_id] = {
            "role": "owner",
            "name": "System Owner",
            "token": master_token
        }
        self.tokens[master_token] = {"user_id": owner_id, "role": "owner"}
        self._save_auth()
        
        print(f"👑 MASTER TOKEN GENERATED for {owner_id}")
        self._export_to_shell(master_token)

    def _export_to_shell(self, token: str):
        """Echo the owner token into .zshrc for easy access."""
        zshrc = Path.home() / ".zshrc"
        export_line = f'export SAM_OWNER_TOKEN="{token}"'
        
        try:
            if zshrc.exists():
                content = zshrc.read_text()
                if "SAM_OWNER_TOKEN" not in content:
                    with open(zshrc, "a") as f:
                        f.write(f"\n# SAM-D AGI Master Token\n{export_line}\n")
                    print(f"📝 Master token exported to {zshrc}")
            else:
                zshrc.write_text(f"{export_line}\n")
                print(f"📝 Created {zshrc} and exported master token")
        except Exception as e:
            print(f"⚠️ Failed to export token to shell: {e}")

    def log_access(self, user_id: str, role: str, endpoint: str, success: bool, message: str = ""):
        """Audit logging for all access attempts."""
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "role": role,
            "endpoint": endpoint,
            "success": success,
            "msg": message
        }
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except:
            pass

    def validate_token(self, token: str) -> Tuple[Optional[str], str]:
        """Verify token and return (user_id, role)"""
        if not token:
            return None, "user"
            
        # Sanitize token input
        token = "".join(c for f, c in enumerate(token) if c in (string.ascii_letters + string.digits + "_"))
        
        if token in self.tokens:
            info = self.tokens[token]
            return info["user_id"], info["role"]
            
        return None, "user"

    def can_access_finance(self, user_id: str) -> bool:
        """Strict check for financial data access (Owner only)."""
        return self.get_role(user_id) == "owner"

    def set_role(self, caller_id: str, target_id: str, role: str) -> bool:
        """Update a user's role with strict hierarchy enforcement."""
        # Sanitize identifiers
        caller_id = "".join(c for c in caller_id if c.isalnum() or c == "_")
        target_id = "".join(c for c in target_id if c.isalnum() or c == "_")
        
        caller_role = self.get_role(caller_id)
        target_role = self.get_role(target_id)
        
        # 1. Owner is immutable - cannot be changed by anyone (including themselves via this API)
        if target_id == os.getenv("SAM_OWNER_ID", "owner_1") or target_role == "owner":
            return False
            
        # 2. Only owner can set 'admin' role
        if role == "admin":
            if caller_role != "owner":
                self.log_access(caller_id, caller_role, "set_role_admin_attempt", False, f"Denied elevation of {target_id} to admin")
                return False
            
        # 3. Only owner or admin can set roles
        if caller_role not in ("owner", "admin"):
            return False
            
        # 4. Admins can only manage 'user' roles (they can whitelist/add users)
        if caller_role == "admin" and role != "user":
            self.log_access(caller_id, caller_role, "set_role_non_user_attempt", False, f"Admin tried to set role {role} for {target_id}")
            return False
            
        # 5. Admins cannot demote/modify other admins or the owner
        if caller_role == "admin" and target_role in ("owner", "admin"):
            return False

        if target_id not in self.user_data:
            self.user_data[target_id] = {"name": f"User {target_id}"}
            
        self.user_data[target_id]["role"] = role
        self._save_auth()
        
        # Update active tokens role
        for tinfo in self.tokens.values():
            if tinfo["user_id"] == target_id:
                tinfo["role"] = role
        
        self.log_access(caller_id, caller_role, "set_role_success", True, f"Set {target_id} to {role}")
        return True

    def get_role(self, user_id: str) -> str:
        """Return the role of a given user ID."""
        if user_id == os.getenv("SAM_OWNER_ID", "owner_1"):
            return "owner"
        if user_id in self.user_data:
            return self.user_data[user_id].get("role", "user")
        return "user"

    def create_admin_token(self, caller_id: str, admin_id: str) -> Optional[str]:
        """Generate a persistent admin token. Only owner can do this."""
        if self.get_role(caller_id) != "owner":
            return None
            
        admin_id = "".join(c for c in admin_id if c.isalnum() or c == "_")
        token = "sam_admin_" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
        
        self.user_data[admin_id] = {
            "role": "admin",
            "name": f"Admin {admin_id}",
            "token": token
        }
        self.tokens[token] = {"user_id": admin_id, "role": "admin"}
        self._save_auth()
        return token

    def create_session_token(self, user_id: str) -> str:
        """Generate a temporary user token with memory capping."""
        user_id = "".join(c for c in user_id if c.isalnum() or c == "_")
        token = "sam_user_" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(24))
        
        # Memory capping for transient tokens (limit to 1000 active sessions)
        if len([t for t in self.tokens.values() if t["role"] == "user"]) > 1000:
            # Find and remove oldest user token
            user_tokens = [t for t in self.tokens if self.tokens[t]["role"] == "user"]
            if user_tokens:
                del self.tokens[user_tokens[0]]
                
        self.tokens[token] = {"user_id": user_id, "role": "user"}
        return token

def create_auth_manager(project_root: Path):
    return AuthManager(project_root)
