#ifndef ENV_H
#define ENV_H

#include <stdio.h>
#include <stdlib.h>
#include "../utils/Raylib/src/raylib.h"

typedef struct Env {
  size_t width;
  size_t height;

  Rectangle **board;
} Env;

#endif
