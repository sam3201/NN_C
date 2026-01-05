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

# --- Collect all source files ---
MUZE_SRC=$(find ../utils/NN/MUZE -name "*.c" | tr '\n' ' ')
NN_SRC="../utils/NN/NN.c ../utils/NN/TRANSFORMER.c ../utils/NN/NEAT.c"
SAM_SRC=$(find ../SAM -name "*.c" | tr '\n' ' ')

# --- Compile and link ---
gcc -w game.c $MUZE_SRC $NN_SRC $SAM_SRC \
    -I../utils/Raylib/src \
    -I../utils/NN \
    -I../SAM \
    -L../utils/Raylib/src -lraylib \
    -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
    -arch arm64 \
    -o game

# --- Run game if compilation succeeded ---
if [ $? -eq 0 ]; then
    echo "Compilation successful! Running the game..."
    ./game
else
    echo "Compilation failed!"
    exit 1
fi

# --- Optional cleanup ---
rm game

