#!/bin/bash

# Build script for Hugging Face integration

echo "Building Hugging Face integration..."

# Source files for trainer
SRC_FILES_TRAINER="hf_trainer.c ../SAM/SAM.c ../utils/NN/transformer.c ../utils/NN/NEAT.c ../utils/NN/NN.c"

# Source files for communicator
SRC_FILES_COMM="sam_hf_bridge.c ../SAM/SAM.c ../utils/NN/transformer.c ../utils/NN/NEAT.c ../utils/NN/NN.c"

# Compile trainer
echo "Building hf_trainer..."
gcc -o hf_trainer $SRC_FILES_TRAINER \
    -I. \
    -I../SAM \
    -I../utils/NN \
    -lm \
    -O2 \
    -Wall

if [ $? -ne 0 ]; then
    echo "✗ Trainer build failed!"
    exit 1
fi

# Compile communicator
echo "Building sam_hf_bridge..."
gcc -o sam_hf_bridge $SRC_FILES_COMM \
    -I. \
    -I../SAM \
    -I../utils/NN \
    -lm \
    -O2 \
    -Wall

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "Built programs:"
    echo "  - hf_trainer: Train SAM using HF models"
    echo "  - sam_hf_bridge: Communicate with HF models"
    echo ""
    echo "Next steps:"
    echo "1. Install Python dependencies: pip install -r requirements.txt"
    echo "2. Train: ./hf_trainer [model_name] [epochs] [data_file]"
    echo "3. Communicate: ./sam_hf_bridge [model_name] [interactive]"
    echo ""
    echo "Examples:"
    echo "  ./hf_trainer bert-base-uncased 10 ../utils/DATASETS/RomeoAndJuliet.txt"
    echo "  ./sam_hf_bridge"
    echo "  ./sam_hf_bridge bert-base-uncased interactive"
else
    echo "✗ Communicator build failed!"
    exit 1
fi

