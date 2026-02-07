#!/bin/bash
# Simple build script for SAM C library

cd "$(dirname "$0")"

echo "Building SAM Neural Core..."

# Build with just the essential files
gcc -Wall -Wextra -O3 -fPIC -std=c99 \
    -I ORGANIZED/UTILS/SAM/SAM \
    -I ORGANIZED/UTILS/models/MLP \
    -shared -lm \
    -dynamiclib \
    -o libsam_core.dylib \
    ORGANIZED/UTILS/SAM/SAM/sam_morphogenesis.c \
    ORGANIZED/UTILS/SAM/SAM/sam_full_context.c \
    ORGANIZED/UTILS/models/MLP/NN.c \
    2>&1

if [ $? -eq 0 ]; then
    echo "✅ Build successful: libsam_core.dylib"
    ls -lh libsam_core.dylib
else
    echo "❌ Build failed - using Python fallback"
fi
