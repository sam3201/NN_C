# SAM-D Deployment Guide

This document provides instructions for deploying SAM-D securely using Cloudflare Tunnel and Cloudflare Access.

## I. Public Secure Deployment (Cloudflare Tunnel + Access)

### 1. Prerequisites
- A domain under your control.
- A Cloudflare account.
- `cloudflared` installed on the host machine.

### 2. Infrastructure Setup
1. **Create a Tunnel**: In the Cloudflare Zero Trust dashboard, go to **Networks → Tunnels**. Create a new tunnel and save the **Tunnel Token**.
2. **Add Public Hostname**: Add a public hostname (e.g., `sam.yourdomain.com`) pointing to `http://localhost:5004` (or your configured port).
3. **Enable Cloudflare Access**: Go to **Access → Applications**. Create a self-hosted application for `sam.yourdomain.com`.
   - **Policy**: Add a policy to allow specific emails or identity providers.
   - **CORS**: Ensure CORS is configured if you plan to access the API from other domains.

### 3. Environment Configuration
Add the following to your `.env` or profile file (e.g., `profiles/full.env`):

```dotenv
# Deployment Basics
SAM_OAUTH_REDIRECT_BASE=https://sam.yourdomain.com
SAM_TRUST_PROXY=1

# Cloudflare Tunnel Token
CF_TUNNEL_TOKEN=your_cloudflare_tunnel_token_here

# Security Allowlists
SAM_OWNER_EMAIL=your@email.com
SAM_ADMIN_EMAILS=admin1@email.com,admin2@email.com
SAM_ALLOWED_EMAILS=user1@email.com,user2@email.com

# IP Restrictions (Optional when using Access)
# SAM_ALLOWED_IPS=127.0.0.1,your_office_ip
```

### 4. Running the System
Start SAM using the universal runner:
```bash
python3 run.py --profile full
```

In a separate terminal (or as a background process), start the tunnel:
```bash
./tools/run_cloudflare_tunnel.sh
```

## II. Health Checks & Validation

### 1. Quick Health Check
Verify the system is reachable via the tunnel:
```bash
curl -I https://sam.yourdomain.com/api/health
```

### 2. Verify Authentication
Try to access an admin endpoint without a session:
```bash
curl -X GET https://sam.yourdomain.com/api/auth/status
# Should return 401 or redirect to /login
```

## III. Recommended Access Policy (JSON Example)
In Cloudflare Access, you should ideally have a policy that matches:
- **Action**: Allow
- **Include**: 
  - Emails: `your@email.com`, `admins@email.com`
  - (Optional) IP Ranges: Your trusted networks.

---
*Note: SAM-D automatically detects the `Cf-Access-Authenticated-User-Email` header when `SAM_TRUST_PROXY=1` is set, allowing for seamless login for users already authenticated by Cloudflare.*
