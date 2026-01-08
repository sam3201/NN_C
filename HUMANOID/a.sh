#!/bin/sh
set -eu

# -----------------------
# Paths
# -----------------------
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

RAYLIB_DIR="../utils/Raylib"
RAYLIB_SRC="$RAYLIB_DIR/src"
RAYLIB_LIB="$RAYLIB_SRC/libraylib.a"

# -----------------------
# Raylib install/build
# -----------------------
if [ ! -d "$RAYLIB_DIR" ]; then
  echo "Raylib not found. Installing..."
  git clone https://github.com/raysan5/raylib.git "$RAYLIB_DIR"
else
  echo "Raylib found. Skipping installation."
fi

if [ ! -f "$RAYLIB_LIB" ]; then
  echo "Building Raylib..."
  (cd "$RAYLIB_SRC" && make PLATFORM=PLATFORM_DESKTOP)
fi

# -----------------------
# Collect sources (dedupe)
# -----------------------
echo "Collecting sources..."

# Core NN sources (explicit, stable)
NN_SRC="../utils/NN/NN.c ../utils/NN/TRANSFORMER.c ../utils/NN/NEAT.c"

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
echo "Compiling sim..."

# Use clang on macOS if you want; gcc often maps to clang anyway.
CC="${CC:-clang}"

# Note: -w hides warnings. Consider removing once you're stable.
"$CC" -g -w \
  sim.c \
  $NN_SRC \
  $MUZE_SRC \
  $SAM_SRC \
  -I../utils/NN \
  -I../utils/NN/MUZE \
  -I../SAM \
  -I"$RAYLIB_SRC" \
  -L"$RAYLIB_SRC" -lraylib -pthread \
  -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
  -arch arm64 \
  -o sim

echo "Compilation successful! Running the sim..."
lldb ./sim
status=$?

echo "Game exited with status $status"
echo "Cleaning up..."
rm -rf sim.dSYM
rm sim
exit "$status"


