# SAM Auth Manager
# Handles token generation, role-based verification, and persistence

import os
import json
import secrets
import string
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

class AuthManager:
    """
    Manages access tokens and RBAC for SAM-D.
    - Owner: Persistent master token (saved to env).
    - Admin: Persistent privileged tokens.
    - User: Transient session tokens.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.auth_path = project_root / "sam_data" / "auth.json"
        self.tokens = {} # token -> {user_id, role, expiry}
        self.user_data = {} # user_id -> {role, name, permanent_token}
        
        self._load_auth()
        self._ensure_owner_token()

    def _load_auth(self):
        if self.auth_path.exists():
            try:
                data = json.loads(self.auth_path.read_text())
                self.user_data = data.get("users", {})
                # Load permanent tokens into active map
                for uid, uinfo in self.user_data.items():
                    if "token" in uinfo:
                        self.tokens[uinfo["token"]] = {"user_id": uid, "role": uinfo["role"]}
            except Exception as e:
                print(f"⚠️ Auth load failed: {e}")

    def _save_auth(self):
        try:
            self.auth_path.parent.mkdir(parents=True, exist_ok=True)
            self.auth_path.write_text(json.dumps({"users": self.user_data}, indent=2))
        except Exception as e:
            print(f"⚠️ Auth save failed: {e}")

    def _ensure_owner_token(self):
        """Generate and protect the master owner token."""
        owner_id = os.getenv("SAM_OWNER_ID", "owner_1")
        
        # Check if owner already has a token
        if owner_id in self.user_data and "token" in self.user_data[owner_id]:
            return

        # Generate a high-entropy master token
        master_token = "sam_owner_" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
        
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

    def validate_token(self, token: str) -> Tuple[Optional[str], str]:
        """Verify token and return (user_id, role)"""
        if not token:
            return None, "user"
            
        if token in self.tokens:
            info = self.tokens[token]
            return info["user_id"], info["role"]
            
        return None, "user"

    def create_admin_token(self, admin_id: str) -> str:
        """Generate a persistent admin token."""
        token = "sam_admin_" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(24))
        self.user_data[admin_id] = {
            "role": "admin",
            "name": f"Admin {admin_id}",
            "token": token
        }
        self.tokens[token] = {"user_id": admin_id, "role": "admin"}
        self._save_auth()
        return token

    def create_session_token(self, user_id: str) -> str:
        """Generate a temporary user token."""
        token = "sam_user_" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16))
        self.tokens[token] = {"user_id": user_id, "role": "user"}
        # Note: Session tokens are not persisted to auth.json
        return token

    def set_role(self, target_id: str, role: str) -> bool:
        """Update a user's role and persist."""
        if role not in ("owner", "admin", "user"):
            return False
        if target_id not in self.user_data:
            self.user_data[target_id] = {"name": f"User {target_id}"}
        self.user_data[target_id]["role"] = role
        self._save_auth()
        # Update active tokens role
        for tinfo in self.tokens.values():
            if tinfo["user_id"] == target_id:
                tinfo["role"] = role
        return True

    def create_admin(self, admin_id: str) -> Tuple[str, str]:
        """Create a new admin and return (admin_id, token)."""
        token = self.create_admin_token(admin_id)
        return admin_id, token

def create_auth_manager(project_root: Path):
    return AuthManager(project_root)
