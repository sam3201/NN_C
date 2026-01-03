#!/bin/bash

# ---------------- Raylib check ----------------
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

# ---------------- Compile Game ----------------
echo "Compiling game..."

# Collect all MuZero source files
MUZE_SRC="../utils/NN/MUZE/ewc.c \
../utils/NN/MUZE/growth.c \
../utils/NN/MUZE/mcts.c \
../utils/NN/MUZE/muzero_model.c \
../utils/NN/MUZE/replay_buffer.c \
../utils/NN/MUZE/selfplay.c \
../utils/NN/MUZE/toy_env.c \
../utils/NN/MUZE/trainer.c \
../utils/NN/MUZE/util.c"

gcc -w game.c $MUZE_SRC \
    -I../utils/Raylib/src \
    -I../utils/NN/MUZE \
    -L../utils/Raylib/src -lraylib \
    -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
    -o game

# ---------------- Run ----------------
if [ $? -eq 0 ]; then
    echo "Compilation successful! Running the game..."
    ./game
else
    echo "Compilation failed!"
    exit 1
fi

# Optional: cleanup
# echo "Cleaning up..."
# rm game

