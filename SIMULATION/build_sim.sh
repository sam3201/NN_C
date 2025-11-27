#!/bin/bash

if [ ! -d "../utils/Raylib" ]; then
  echo "Raylib not found. Installing..."
  git clone https://github.com/raysan5/raylib.git ../utils/Raylib/
else
  echo "Raylib found. Skipping installation."
fi

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
