#!/bin/bash

if [ ! -d "../utils/Raylib" ]; then
  echo "Raylib not found. Installing..."
  git clone https://github.com/raysan5/raylib.git ../utils/Raylib/
else
  echo "Raylib found. Skipping installation."
fi

if [ ! -f "../utils/Raylib/src/libraylib.a" ]; then
  echo "Building Raylib..."
  cd ../utils/Raylib/src || exit
  make PLATFORM=PLATFORM_DESKTOP
  cd ../../GAME || exit
fi

echo "Compiling game..."

MUZE_SRC=$(find ../utils/NN/MUZE -name "*.c" | tr '\n' ' ')
SAM_SRC=$(find ../SAM -name "*.c" | tr '\n' ' ')

gcc -w game.c $MUZE_SRC $SAM_SRC ../SAM/SAM.c \
    -I../utils/Raylib/src \
    -I../utils/NN/MUZE \
    -I../SAM/SAM.c \
    -I../utils/NN/TRANSFORMER.c \
    -L../utils/Raylib/src -lraylib \
    -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
    -o game

if [ $? -eq 0 ]; then
    echo "Compilation successful! Running the game..."
    ./game
else
    echo "Compilation failed!"
    exit 1
fi

echo "Cleaning up..."
rm game

