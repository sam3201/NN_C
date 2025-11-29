#ifndef ENV_H
#define ENV_H

#include <stdio.h>
#include <stdlib.h>
#include "../utils/Raylib/src/raylib.h"

typedef struct Env {
  static size_t width;
  static size_t height;

  Rectangle board[width][height];
} Env;

#endif
