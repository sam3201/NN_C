#!/bin/sh
set -eu

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

MUZE_SRC=$(find ../utils/NN/MUZE -name "*.c" -print | tr '\n' ' ')
SAM_SRC=$(find ../SAM -name "*.c" -print | tr '\n' ' ')
NN_SRC="../utils/NN/NN.c ../utils/NN/TRANSFORMER.c ../utils/NN/NEAT.c"

# Trim leading/trailing spaces (prevents the ' ' phantom arg)
MUZE_SRC=$(printf "%s" "$MUZE_SRC" | sed 's/^ *//; s/ *$//')
SAM_SRC=$(printf "%s" "$SAM_SRC" | sed 's/^ *//; s/ *$//')

if [ -z "$MUZE_SRC" ]; then
  echo "ERROR: No MUZE .c files found under ../utils/NN/MUZE"
  exit 1
fi

if [ -z "$SAM_SRC" ]; then
  echo "ERROR: No SAM .c files found under ../SAM"
  exit 1
fi

echo "Compiling game..."

gcc -w game.c $NN_SRC $MUZE_SRC $SAM_SRC \
  -I../utils/NN -I../utils/NN/MUZE -I../SAM -I../utils/Raylib/src \
  -L../utils/Raylib/src -lraylib \
  -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
  -arch arm64 \
  -o game

echo "Compilation successful! Running the game..."
./game


if [ $? -eq 0 ]; then
  echo "Game exited successfully."
  rm game
else
  echo "Game exited with an error."
fi
