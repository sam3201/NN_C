#!/usr/bin/env bash
set -euo pipefail

echo "==> Checking prerequisites"
command -v git >/dev/null || { echo "git is required"; exit 1; }
command -v node >/dev/null || echo "NOTE: node is recommended for Gemini CLI"
command -v python >/dev/null || { echo "python is required for aider"; exit 1; }

echo "==> Installing Aider (pipx)"
python -m pip install --user -q pipx || true
python -m pipx ensurepath || true
pipx install -q aider-chat || pipx upgrade -q aider-chat

echo "==> Installing Gemini CLI (npm) if available"
if command -v npm >/dev/null; then
  npm install -g @google/gemini-cli || true
else
  echo "NOTE: npm not found, skipping Gemini CLI"
fi

echo "==> Repo scaffolding"
mkdir -p ai
test -f ai/brief.md || cat > ai/brief.md <<'EOF'
# Brief
Describe the feature/bug/refactor you want.

## Constraints
- language/framework:
- must keep API compatibility:
- performance / security notes:
- tests to run:
EOF

test -f ai/notes.md || echo "# Notes" > ai/notes.md
echo "Done."

