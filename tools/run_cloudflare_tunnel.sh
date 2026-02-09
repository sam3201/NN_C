#!/bin/bash
set -euo pipefail

if [ -z "${CF_TUNNEL_TOKEN:-}" ]; then
  echo "Missing CF_TUNNEL_TOKEN"
  exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared not installed. See DOCS/DEPLOYMENT_CLOUDFLARE.md"
  exit 1
fi

echo "Starting Cloudflare tunnel..."
cloudflared tunnel --no-autoupdate run --token "${CF_TUNNEL_TOKEN}"
