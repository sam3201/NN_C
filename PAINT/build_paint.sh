#!/bin/sh
set -eu

# -----------------------
# Paths
# -----------------------
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# -----------------------
# Raylib deps 
# -----------------------
RAYLIB_CFLAGS="$(pkg-config --cflags ttf)"
RAYLIB_LIBS="$(pkg-config --libs sdl3 sdl3-ttf)"

# -----------------------
# Collect sources (dedupe)
# -----------------------
echo "Collecting sources..."

# Core NN sources (explicit, stable)
NN_SRC="../utils/NN/NN.c ../utils/NN/TRANSFORMER.c ../utils/NN/NEAT.c"
SDL_COMPAT_SRC="../utils/SDL3/SDL3_compat.c"

# MUZE + SAM sources (deduped)
MUZE_SRC="$(find ../utils/NN/MUZE -type f -name '*.c' -print | sort -u | tr '\n' ' ')"
SAM_SRC="$(find ../SAM -type f -name '*.c' -print | sort -u | tr '\n' ' ')"

# Trim leading/trailing spaces (prevents phantom args)
MUZE_SRC="$(printf "%s" "$MUZE_SRC" | sed 's/^ *//; s/ *$//')"
SAM_SRC="$(printf "%s" "$SAM_SRC" | sed 's/^ *//; s/ *$//')"

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

# -----------------------
# Compile
# -----------------------
echo "Compiling paint..."

# Use clang on macOS if you want; gcc often maps to clang anyway.
CC="${CC:-gcc}"
FLAGS="${FLAGS:-} -w"

# Note: -w hides warnings. Consider removing once you're stable.
"$CC" $FLAGS \
  paint.c \
  $SDL_COMPAT_SRC \
  $NN_SRC \
  $MUZE_SRC \
  $SAM_SRC \
  -I../utils/NN \
  -I../utils/NN/MUZE \
  -I../SAM \
  -I../utils/SDL3 \
  $SDL_CFLAGS \
  $SDL_LIBS \
  -pthread -lm \
  -framework OpenGL \
  -arch arm64 \
  -o paint 

echo "Compilation successful! Running the paint..."
./paint
status=$?

echo "Game exited with status $status"
rm -f ./paint
rm -rf paint.dSYM
exit "$status"
