#!/bin/sh
set -eu

# -----------------------
# Paths
# -----------------------
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# -----------------------
# SDL3 deps
# -----------------------
SDL_CFLAGS="$(pkg-config --cflags sdl3 sdl3-ttf)"
SDL_LIBS="$(pkg-config --libs sdl3 sdl3-ttf)"

# -----------------------
# Collect sources (dedupe)
# -----------------------
echo "Collecting sources..."

# Core NN sources (explicit, stable)
NN_SRC="../utils/NN/NN/NN.c ../utils/NN/TRANSFORMER/TRANSFORMER.c ../utils/NN/NEAT/NEAT.c ../utils/NN/CONVOLUTION/CONVOLUTION.c"
SDL_SRC_DIR="../utils/SDL/"
SDL_BUILD_DIR="../utils/SDL/build"

# RL_AGENT sources (including all subdirectories, excluding test files)
RL_AGENT_SRC="$(find ../RL_AGENT -type f -name '*.c' -print | grep -v test | sort -u | tr '\n' ' ')"

# MUZE + SAM sources (deduped)
MUZE_SRC="$(find ../utils/NN/MUZE -type f -name '*.c' -print | sort -u | tr '\n' ' ')"
SAM_SRC="$(find ../SAM -type f -name '*.c' -print | sort -u | tr '\n' ' ')"

# Trim leading/trailing spaces (prevents phantom args)
MUZE_SRC="$(printf "%s" "$MUZE_SRC" | sed 's/^ *//; s/ *$//')"
SAM_SRC="$(printf "%s" "$SAM_SRC" | sed 's/^ *//; s/ *$//')"
RL_AGENT_SRC="$(printf "%s" "$RL_AGENT_SRC" | sed 's/^ *//; s/ *$//')"

if [ -z "$MUZE_SRC" ]; then
  echo "ERROR: No MUZE .c files found under ../utils/NN/MUZE"
  exit 1
fi

if [ -z "$SAM_SRC" ]; then
  echo "ERROR: No SAM .c files found under ../SAM"
  exit 1
fi

echo "MUZE files:"
printf "  %s\n" $MUZE_SRC
echo "SAM files:"
printf "  %s\n" $SAM_SRC
echo "RL_AGENT files:"
printf "  %s\n" $RL_AGENT_SRC

# -----------------------
# Compile
# -----------------------
echo "Compiling game..."

# Use clang on macOS if you want; gcc often maps to clang anyway.
CC="${CC:-gcc}"
FLAGS="${FLAGS:-} -g"

# Note: -w hides warnings. Consider removing once you're stable.
"$CC" $FLAGS \
  game.c \
  $SDL_COMPAT_SRC \
  $NN_SRC \
  $RL_AGENT_SRC \
  $MUZE_SRC \
  $SAM_SRC \
  -I../utils/NN/NN \
  -I../utils/NN/CONVOLUTION \
  -I../utils/NN/MUZE \
  -I../SAM \
  -I../RL_AGENT \
  -I../utils/NN/TRANSFORMER \
  -I../utils/NN/NEAT \
  -I../utils/SDL/build/include \
  $SDL_CFLAGS \
  $SDL_LIBS \
  -pthread -lm \
  -framework OpenGL \
  -arch arm64 \
  -o game 

echo "Compilation successful! Running the game..."
lldb ./game
status=$?

echo "Game exited with status $status"
rm -f ./game
rm -rf game.dSYM
exit "$status"
