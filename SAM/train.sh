#!/bin/bash

# Compile the SAM AGI training program
gcc -o sam_agi sam_agi.c SAM.c ../utils/NN/NEAT.c ../utils/NN/NN.c ../utils/NN/TRANSFORMER.c -lm -Wall -Wextra

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    
    # Check if task_id argument is provided
    if [ "$#" -ne 1 ]; then
        echo "Usage: $0 <task_id>"
        exit 1
    fi
    
    # Run the training with the provided task_id
    ./sam_agi "$1"
else
    echo "Compilation failed!"
    exit 1
fi
