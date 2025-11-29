#ifndef ENV_H
#define ENV_H

#include <stdio.h>
#include <stdlib.h>
#include "../utils/Raylib/src/raylib.h"
#inclide "../utils/NEAT/NEAT.h"

typedef struct {
  enum EntityType {
    PLAYER,
    ENEMY,
  } type;

  NEAT_t *neat; 

} entity; 

typedef struct Env {
  const static size_t width;
  const static size_t height;

  Rectangle env[width][height];
  Entity *entities;
} Env;

#endif
