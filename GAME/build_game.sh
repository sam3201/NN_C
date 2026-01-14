#!/bin/sh
set -eu

# -----------------------
# Paths
# -----------------------
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# -----------------------
# Toolchain
# -----------------------
CC="${CC:-cc}"
CFLAGS="${CFLAGS:-} -g -O0"
LDFLAGS="${LDFLAGS:-}"

# -----------------------
# SDL3 (local build)
# -----------------------
SDL_SRC_DIR="../utils/SDL"
SDL_BUILD_DIR="../utils/SDL/build"

SDL_TTF_SRC_DIR="../utils/SDL_ttf"
SDL_TTF_BUILD_DIR="../utils/SDL_ttf/build"

SDL_INCLUDES="-I${SDL_SRC_DIR}/include -I${SDL_TTF_SRC_DIR}/include"
SDL_LIBS="-F${SDL_BUILD_DIR} -framework SDL3 -L${SDL_TTF_BUILD_DIR} -lSDL3_ttf"
SDL_RPATH="-Wl,-rpath,${SDL_BUILD_DIR} -Wl,-rpath,${SDL_TTF_BUILD_DIR}"

# -----------------------
# Raylib (local build)
# -----------------------
#
RAYLIB_DIR="../utils/Raylib"
RAYLIB_LIB="${RAYLIB_DIR}/src/libraylib.a"
RAYLIB_INCLUDES="-I${RAYLIB_DIR}/src"
RAYLIB_LINK="${RAYLIB_LIB} -framework Cocoa -framework IOKit -framework CoreVideo"

# -----------------------
# Collect sources
# -----------------------
echo "Collecting sources..."

NN_SRC="../utils/NN/NN/NN.c ../utils/NN/TRANSFORMER/TRANSFORMER.c ../utils/NN/NEAT/NEAT.c ../utils/NN/CONVOLUTION/CONVOLUTION.c"

RL_AGENT_SRC="$(find ../RL_AGENT -type f -name '*.c' -print | grep -v test | sort -u | tr '\n' ' ')"
MUZE_SRC="$(find ../utils/NN/MUZE -type f -name '*.c' -print | sort -u | tr '\n' ' ')"
SAM_SRC="$(find ../SAM -type f -name '*.c' -print | sort -u | tr '\n' ' ')"

# Trim leading/trailing spaces
RL_AGENT_SRC="$(printf "%s" "$RL_AGENT_SRC" | sed 's/^ *//; s/ *$//')"
MUZE_SRC="$(printf "%s" "$MUZE_SRC" | sed 's/^ *//; s/ *$//')"
SAM_SRC="$(printf "%s" "$SAM_SRC" | sed 's/^ *//; s/ *$//')"

[ -n "$MUZE_SRC" ] || { echo "ERROR: No MUZE .c files found under ../utils/NN/MUZE"; exit 1; }
[ -n "$SAM_SRC" ]  || { echo "ERROR: No SAM .c files found under ../SAM"; exit 1; }

echo "MUZE files:";     printf "  %s\n" $MUZE_SRC
echo "SAM files:";      printf "  %s\n" $SAM_SRC
echo "RL_AGENT files:"; printf "  %s\n" $RL_AGENT_SRC

# -----------------------
# Includes
# -----------------------
INCLUDES="
  -I../utils/NN/NN
  -I../utils/NN/CONVOLUTION
  -I../utils/NN/MUZE
  -I../SAM
  -I../RL_AGENT
  -I../utils/NN/TRANSFORMER
  -I../utils/NN/NEAT
  ${RAYLIB_INCLUDES}
  ${SDL_INCLUDES}
"

# -----------------------
# Link flags (macOS)
# -----------------------
LINK="
  ${SDL_LIBS}
  ${SDL_RPATH}
  ${RAYLIB_LINK}
  -pthread -lm
  -framework OpenGL
  -arch arm64
"

# -----------------------
# Build
# -----------------------
echo "Compiling game..."

# shellcheck disable=SC2086
$CC $CFLAGS \
  game.c \
  $NN_SRC \
  $RL_AGENT_SRC \
  $MUZE_SRC \
  $SAM_SRC \
  $INCLUDES \
  $LDFLAGS \
  $LINK \
  -o game

echo "Compilation successful!"

# -----------------------
# Run
# -----------------------
# if [ "${RUN_UNDER_LLDB:-1}" -eq 1 ]; then
  #lldb ./game
#else
 # ./game
#fi

#status=$?

#echo "Game exited with status $status"

#rm -f ./game
#rm -rf ./game.dSYM
#exit "$status"

