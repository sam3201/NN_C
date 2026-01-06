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

MUZE_SRC=$(find ../utils/NN/MUZE -name "*.c" | tr '\n' ' ')
NN_SRC=$(find ../utils/NN -name "*.c" | tr '\n' ' ')
SAM_SRC=$(find ../SAM -name "*.c" | tr '\n' ' ')

gcc -w game.c $NN_SRC $MUZE_SRC $SAM_SRC \
  -I../utils/NN -I../utils/NN/MUZE -I../SAM -I../utils/Raylib/src \
  -L../utils/Raylib/src -lraylib \
  -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
  -arch arm64 \
  -o game

if [ $? -eq 0 ]; then
    echo "Compilation successful! Running the game..."
    ./sim
else
    echo "Compilation failed!"
    exit 1
fi

echo "Ran Successfully"
rm sim

