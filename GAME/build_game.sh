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

gcc -w game.c ../utils/NN/NN.c ../utils/NN/MUZE/all.h \
    -I../utils/Raylib/src \
    -L../utils/Raylib/src \
    -lraylib \
    -framework OpenGL \
    -framework Cocoa \
    -framework IOKit \
    -framework CoreVideo \
    -o game 

if [ $? -eq 0 ]; then
    echo "Compilation successful! Running the game..."
    ./game
else
    echo "Compilation failed!"
    exit 1
fi

echo "Ran Successfully"
rm game 

