#!/bin/sh
# Universal runner for SAM / NN_C
# Works without just, bash, or platform-specific tools

set -e

PROFILE="${1:-full}"

echo "=== SAM Universal Runner ==="
echo "Profile: $PROFILE"

# Detect OS
OS="$(uname 2>/dev/null || echo unknown)"

# Prefer just if available
if command -v just >/dev/null 2>&1; then
  echo "✔ just detected — using Justfile"
  exec just run "$PROFILE"
fi

# Windows environments (Git Bash, MSYS, Cygwin)
case "$OS" in
  MINGW*|MSYS*|CYGWIN*)
    if command -v powershell >/dev/null 2>&1; then
      echo "✔ Windows shell detected — using bootstrap.ps1"
      exec powershell -NoProfile -ExecutionPolicy Bypass \
        -File "./bootstrap.ps1" "$PROFILE"
    fi
    ;;
esac

# POSIX systems (macOS / Linux)
if [ -x "./bootstrap.sh" ]; then
  echo "✔ Using bootstrap.sh"
  exec ./bootstrap.sh "$PROFILE"
fi

echo "❌ No supported runner found."
echo "Expected one of:"
echo "  - just"
echo "  - ./bootstrap.sh"
echo "  - bootstrap.ps1 (Windows)"
exit 1

