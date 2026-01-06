#!/bin/bash
set -euo pipefail

RAYLIB_DIR="../utils/Raylib"
RAYLIB_LIB="$RAYLIB_DIR/src/libraylib.a"

if [ ! -d "$RAYLIB_DIR" ]; then
  echo "Raylib not found. Installing..."
  git clone https://github.com/raysan5/raylib.git "$RAYLIB_DIR"
else
  echo "Raylib found. Skipping installation."
fi

if [ ! -f "$RAYLIB_LIB" ]; then
  echo "Building Raylib..."
  (cd "$RAYLIB_DIR/src" && make PLATFORM=PLATFORM_DESKTOP)
fi

echo "Collecting sources..."

# Build arrays (no whitespace bugs)
mapfile -t MUZE_FILES < <(find ../utils/NN/MUZE -name "*.c" -print)
mapfile -t SAM_FILES  < <(find ../SAM -name "*.c" -print)

NN_FILES=(../utils/NN/NN.c ../utils/NN/TRANSFORMER.c ../utils/NN/NEAT.c)

# Sanity checks so you get a useful error
if [ ${#MUZE_FILES[@]} -eq 0 ]; then
  echo "ERROR: No MUZE .c files found under ../utils/NN/MUZE"
  exit 1
fi

if [ ${#SAM_FILES[@]} -eq 0 ]; then
  echo "ERROR: No SAM .c files found under ../SAM"
  exit 1
fi

echo "Compiling game..."

gcc -w \
  game.c \
  "${NN_FILES[@]}" \
  "${MUZE_FILES[@]}" \
  "${SAM_FILES[@]}" \
  -I../utils/NN \
  -I../utils/NN/MUZE \
  -I../SAM \
  -I../utils/Raylib/src \
  -L../utils/Raylib/src \
  -lraylib \
  -framework OpenGL \
  -framework Cocoa \
  -framework IOKit \
  -framework CoreVideo \
  -arch arm64 \
  -o game

echo "Compilation successful! Running the game..."
./game

rm -f game

