#!/bin/bash
# SAM-D Git Synchronizer
# Pushes the current main branch to both the parent repository and the official SAM AGI repo.

set -e

echo "📡 Syncing SAM-D Codebase..."

# 1. Parent Origin
echo "   - Pushing to origin/main..."
git push origin main

# 2. Official SAM AGI
echo "   - Pushing to sam_agi_official/main..."
git push sam_agi_official main

echo "✅ Sync complete. All remotes up to date."
