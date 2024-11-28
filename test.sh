#!/bin/bash

# Compile the program
gcc -o test sam_agi.c SAM/SAM.c utils/NN/TRANSFORMER.c utils/NN/NEAT.c utils/NN/NN.c -lm

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    # Run the program in test mode with the test dataset
    ./test test utils/DATASETS/test-v1.1.json
else
    echo "Compilation failed!"
    exit 1
fi

rm test
