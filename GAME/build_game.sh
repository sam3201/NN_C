#!/bin/bash

# -----------------------
# Configuration
# -----------------------
set -eu

# Paths
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Default mode
MODE="${1:-compile}"
LLDB_MODE="${2:-0}"

# -----------------------
# Help
# -----------------------
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [MODE] [LLDB_MODE]"
    echo ""
    echo "Modes:"
    echo "  compile     - Just compile the game (default)"
    echo "  build       - Compile and run the game"
    echo "  lldb        - Compile and run under lldb debugger"
    echo "  clean       - Clean build artifacts"
    echo ""
    echo "LLDB Mode (optional):"
    echo "  0           - Don't run under lldb (default for build mode)"
    echo "  1           - Run under lldb (default for lldb mode)"
    echo ""
    echo "Examples:"
    echo "  $0                # Compile only"
    echo "  $0 compile        # Compile only"
    echo "  $0 build          # Compile and run"
    echo "  $0 lldb           # Compile and run with lldb"
    echo "  $0 build 1        # Compile and run with lldb"
    echo ""
    exit 0
fi

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
RAYLIB_DIR="../utils/Raylib"
RAYLIB_LIB="${RAYLIB_DIR}/src/libraylib.a"
RAYLIB_INCLUDES="-I${RAYLIB_DIR}/src"
RAYLIB_LINK="${RAYLIB_LIB} -framework Cocoa -framework IOKit -framework CoreVideo"

# -----------------------
# Logging
# -----------------------
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/build_${TIMESTAMP}.log"
DEBUG_LOG="logs/debug_${TIMESTAMP}.log"

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
  -Igenerated
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
# Clean mode
# -----------------------
if [ "$MODE" = "clean" ]; then
    echo "Cleaning build artifacts..."
    rm -f game game.exe
    rm -rf game.dSYM
    rm -f *.o
    echo "Clean completed!"
    exit 0
fi

# -----------------------
# Compile
# -----------------------
echo "Building game... Logging to $LOG_FILE"

# shellcheck disable=SC2086
$CC $CFLAGS \
  generated/impl.c \
  game.c \
  $NN_SRC \
  $RL_AGENT_SRC \
  $MUZE_SRC \
  $SAM_SRC \
  $INCLUDES \
  $LDFLAGS \
  $LINK \
    -o game 2>&1 | tee "$LOG_FILE"

# Check if compilation was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Build successful!"
    
    # -----------------------
    # Execute based on mode
    # -----------------------
    case "$MODE" in
        "compile")
            echo "Compilation completed. Log saved to $LOG_FILE"
            ;;
        "build")
            echo "Running game..."
            if [ "${LLDB_MODE:-0}" -eq 1 ]; then
                echo "Running under lldb..."
                timeout 30s lldb --batch -o run -o bt -- ./game 2>&1 | tee -a "$LOG_FILE"
            else
                ./game
            fi
            ;;
        "lldb")
            echo "Running game under lldb..."
            timeout 30s lldb --batch -o run -o bt -- ./game 2>&1 | tee "$DEBUG_LOG"
            echo "Debug session completed. Log saved to $DEBUG_LOG"
            ;;
        *)
            echo "Unknown mode: $MODE"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
else
    echo "Build failed! Check $LOG_FILE for details."
    exit 1
fi

# -----------------------
# Cleanup
# -----------------------
echo "Cleaning up log directory..."
# Keep only the last log file
rm -f logs/* | head -n 1
echo "Done."
