#!/usr/bin/env bash
set -euo pipefail

CC=clang
CFLAGS="-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra"

# ---- include dirs ----
INCLUDES=(
  -I../NN
  -I../NN/MUZE
  -I../NN/MEMORY
  -I../NN/TOKENIZER
  -I../NN/TRANSFORMER
  -I../../sAM
)

# ---- user input test entrypoint ---
TEST_SRC=$1
OUT=TEST_SRC.o

# ---- auto-discover sources ----
MUZE_SRC=$(find ../NN/MUZE -type f -name "*.c" -print)
NN_SRC=$(find ../NN -maxdepth 1 -type f -name "*.c" -print)

# If your SAM directory is literally "sAM" (as shown), use that:
SAM_SRC=$(find ../../sAM -type f -name "*.c" -print)

# If you sometimes use ../../SAM instead, swap to this:
# SAM_SRC=$(find ../../SAM -type f -name "*.c" -print)

# ---- build ----
echo "Building $OUT..."
echo "Test: $TEST_SRC"
echo "MUZE_SRC:"
echo "$MUZE_SRC" | sed 's/^/  /'
echo "NN_SRC:"
echo "$NN_SRC" | sed 's/^/  /'
echo "SAM_SRC:"
echo "$SAM_SRC" | sed 's/^/  /'

$CC $CFLAGS \
  "${INCLUDES[@]}" \
  "$TEST_SRC" \
  $MUZE_SRC \
  $NN_SRC \
  $SAM_SRC \
  -lm -o "$OUT"

echo "OK: ./$OUT"

