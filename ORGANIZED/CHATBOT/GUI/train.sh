#!/bin/bash

# Build script for chatbot training

echo "Building chatbot trainer..."

# Source files
SRC_FILES="train_chatbot.c ../SAM/SAM.c ../utils/NN/transformer.c ../utils/NN/NEAT.c ../utils/NN/NN.c"

# Compile
gcc -o train_chatbot $SRC_FILES \
    -I. \
    -I../SAM \
    -I../utils/NN \
    -lm \
    -O2 \
    -Wall

if [ $? -eq 0 ]; then
    echo "✓ Build successful! Run ./train_chatbot to start training."
else
    echo "✗ Build failed!"
    exit 1
fi

