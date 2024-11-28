#!/bin/bash

# Compile with debugging symbols
gcc -w -o sim sim.c ../utils/NN/NN.c ../utils/NN/NEAT.c -I. -I/opt/homebrew/include -L/opt/homebrew/lib -lraylib -lpthread -lm -framework OpenGL -framework Cocoa -framework IOKit -O0

if [ $? -eq 0 ]; then
    echo "Compilation successful! Running the game..."
    ./sim
else
    echo "Compilation failed!"
    exit 1
fi

echo "Ran Successfully"
rm sim 
