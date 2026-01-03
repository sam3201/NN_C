#!/bin/bash

if [ ! -d "../utils/Raylib" ]; then
  echo "Raylib not found. Installing..."
  git clone https://github.com/raysan5/raylib.git ../utils/Raylib/
else
  echo "Raylib found. Skipping installation."
fi

if [ ! -f "../utils/Raylib/src/libraylib.a" ]; then
    echo "Building raylib..."
    cd ../utils/Raylib/src
    make PLATFORM=PLATFORM_DESKTOP
    cd ../../../SIMULATION
fi


gcc -w sim.c \
    ../utils/NN/NN.c \
    ../utils/NN/NEAT.c \
    ../utils/NN/MEMORY/MEMORY.c \
    -I../utils/NN \
    -I../utils/NN/MEMORY \
    -I../utils/Raylib/src \
    -L../utils/Raylib/src \
    -lraylib \
    -framework OpenGL \
    -framework Cocoa \
    -framework IOKit \
    -framework CoreVideo \
    -o sim

if [ $? -eq 0 ]; then
    echo "Compilation successful! Running the game..."
    ./sim
else
    echo "Compilation failed!"
    exit 1
fi

echo "Ran Successfully"
rm sim

