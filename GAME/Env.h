#ifndef ENV_H
#define ENV_H

#include <stdio.h>
#include <stdlib.h>

typedef struct Env {
  size_t width;
  size_t height;

  char **board;
} Env;

#endif
