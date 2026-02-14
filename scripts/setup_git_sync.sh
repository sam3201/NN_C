#!/bin/bash
# SAM-D Git Sync Setup
# Configures git to push to both remotes by default.

set -e

echo "🔧 Configuring Dual-Remote Sync..."

# Check if remotes exist
if ! git remote | grep -q "^sam_agi_official$"; then
    echo "❌ Error: sam_agi_official remote not found."
    echo "Please run: git remote add sam_agi_official git@github.com:samaisystemagi/SAM_AGI.git"
    exit 1
fi

# Configure 'all' remote
echo "   - Adding sync urls to 'origin'..."
git remote set-url --add --push origin https://github.com/sam3201/NN_C.git
git remote set-url --add --push origin git@github.com:samaisystemagi/SAM_AGI.git

echo "✅ Dual-remote sync configured. 'git push' will now push to BOTH repositories."
