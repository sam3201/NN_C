# Cloudflare Tunnel + Access (Public Secure Deployment)

This is the **security‑first** path for public access once you own a domain.

## Prerequisites
- A domain under your control
- Cloudflare account
- `cloudflared` installed on the host machine

## 1) Create a Tunnel in Cloudflare
1. Go to **Zero Trust → Access → Tunnels**.
2. Create a new tunnel and note the **Tunnel Token**.

## 2) Configure DNS
Add a DNS record in Cloudflare:
- `sam.yourdomain.com` → Cloudflare Tunnel

## 3) Set SAM Environment
Add to `.env.local`:
```
SAM_OAUTH_REDIRECT_BASE=https://sam.yourdomain.com
SAM_TRUST_PROXY=1
SAM_ALLOWED_IPS=  # optional, can be empty when using Access
```

If you want OAuth:
```
SAM_GOOGLE_CLIENT_ID=...
SAM_GOOGLE_CLIENT_SECRET=...
SAM_GITHUB_CLIENT_ID=...
SAM_GITHUB_CLIENT_SECRET=...
```

## 4) Run the Tunnel
```
CF_TUNNEL_TOKEN=... ./tools/run_cloudflare_tunnel.sh
```

## 5) Enable Cloudflare Access
In **Zero Trust → Access → Applications**:
- Protect `sam.yourdomain.com`
- Require email allowlist or identity provider

## Notes
- Keep the admin tools behind login + admin email list.
- If using Access, you can loosen IP allowlist.
