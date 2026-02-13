# SAM-D (ΨΔ•Ω-Core v5.0.0 Recursive) Deployment Checklist

Follow this checklist to ensure a secure, robust production deployment.

## 1. Environment Preparation
- [ ] **Python Version**: Ensure Python 3.11+ is installed.
- [ ] **C Compiler**: Ensure `gcc` or `clang` is available for C-Core compilation.
- [ ] **Disk Space**: At least 10GB free for logs, distilled models, and local databases.
- [ ] **Network**: Ensure Port 5005 is available.

## 2. Security Setup (Critical)
- [ ] **Owner Identity**: Set `export SAM_OWNER_ID="your_secure_id"` in your environment.
- [ ] **Admin Token**: Set `export SAM_ADMIN_TOKEN="your_complex_token"` (used as a fallback/initial gate).
- [ ] **Master Key**: Verify that `secrets/master.key` is generated on first boot and NOT committed to Git.
- [ ] **Encryption**: Confirm that `sam_data/auth.enc` is being created (verifies Fernet encryption is active).

## 3. Launch & Validation
- [ ] **Initial Build**: Run `./run_production.sh` and monitor the output for successful C compilation.
- [ ] **Master Token**: After boot, check your `~/.zshrc` for the `SAM_OWNER_TOKEN`.
- [ ] **Dashboard Access**: Navigate to `http://localhost:5005` and enter your Master Token in the "Admin Token" field.
- [ ] **Audit Trail**: Open the "Security Audit Logs" card and verify that your own access was recorded.

## 4. Remote Access (Optional but Recommended)
- [ ] **Cloudflare Tunnel**: Install `cloudflared` and set up a tunnel to `localhost:5005` for secure access from your phone without exposing ports.
- [ ] **Mobile Test**: Log in from your mobile browser using the Master Token.

## 5. Ongoing Monitoring
- [ ] **Health Intelligence**: Monitor the "Sensory Map" for anomalies.
- [ ] **Hot-Reload**: Verify that changing a file (e.g., a comment in `complete_sam_unified.py`) triggers a Git push and system restart.

**The system is now self-securing. Any unauthorized attempt to access administrative modules will be logged and blocked.**
