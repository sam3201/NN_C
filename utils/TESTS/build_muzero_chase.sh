#!/bin/bash
set -euo pipefail

gcc -o muzero_chase \
  utils/TESTS/muzero_chase_curses_test.c \
  utils/NN/MUZE/*.c \
  utils/NN/NN.c \
  -lm -lncurses

echo "Built muzero_chase"
