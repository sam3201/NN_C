#!/usr/bin/env bash
set -euo pipefail

CC=clang
CFLAGS="-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra -lcurses"

# ---- defaults ----
DEFAULT_TEST="sam_muze_toy_test.c"
DEFAULT_OUT="a.sh"

# ---- user input (safe with -u) ----
TEST_SRC="${1:-$DEFAULT_TEST}"
OUT="${2:-$DEFAULT_OUT}"

# ---- sanity checks ----
if [ ! -f "$TEST_SRC" ]; then
  echo "Error: test source '$TEST_SRC' not found"
  echo "Usage: $0 [test.c] [output]"
  exit 1
fi

# ---- include dirs ----
INCLUDES=(
  -I../NN
  -I../NN/MUZE
  -I../NN/MEMORY
  -I../NN/TOKENIZER
  -I../NN/TRANSFORMER
  -I../../sAM
)

# ---- auto-discover sources ----
MUZE_SRC=$(find ../NN/MUZE -type f -name "*.c" -print)
NN_SRC=$(find ../NN -maxdepth 1 -type f -name "*.c" -print)
SAM_SRC=$(find ../../sAM -type f -name "*.c" -print)

# ---- build ----
echo "======================================"
echo "Building test"
echo "  Test:   $TEST_SRC"
echo "  Output: $OUT"
echo "======================================"

$CC $CFLAGS \
  "${INCLUDES[@]}" \
  "$TEST_SRC" \
  $MUZE_SRC \
  $NN_SRC \
  $SAM_SRC \
  -lm -o "$OUT"

echo
echo "Build OK"
echo "Run with: ./$OUT"

if [ "$?" -eq 0 ]; then
  exit 0
else
  exit 1
fi
  

