#!/bin/bash

# Compile with debugging symbols
gcc -w -o sim sim.c ../utils/Raylib/raylib.c ../utils/NN/NN.c ../utils/NN/NEAT.c 

if [ $? -eq 0 ]; then
    echo "Compilation successful! Running the game..."
    ./sim
else
    echo "Compilation failed!"
    exit 1
fi

echo "Ran Successfully"
rm sim 
