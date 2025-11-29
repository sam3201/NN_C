#ifndef ENV_H
#define ENV_H

#include <stdio.h>
#include <stdlib.h>
#include <../utils/Raylib/raylib.h>
typedef struct Env {
  size_t width;
  size_t height;

  char **board;
} Env;

#endif
