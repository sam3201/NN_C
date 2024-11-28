#!/bin/bash

# Compile the sam_agi.c file with all dependencies
gcc sam_agi.c SAM/SAM.c utils/NN/TRANSFORMER.c utils/NN/NN.c utils/NN/NEAT.c utils/DATASETS/dataset.c utils/NN/TOKENIZER.c -w -o sam_agi -lm -I./utils/NN -I./utils

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful! Starting training..."
    lldb ./sam_agi
else
    echo "Compilation failed!"
    exit 1
fi

rm sam_agi
