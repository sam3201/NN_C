#ifndef ENV_H
#define ENV_H

#include <stdio.h>
#include <stdlib.h>
#include "../utils/Raylib/src/raylib.h"
#include "../utils/NN/NEAT.h"
#include "../utils/NN/NN.h"

typedef struct {
  enum EntityType {
    PLAYER,
    ENEMY,
  } type;

  typedef struct {
  NEAT_t *neat;
  int exploration;

} Agent;

} entity; 

typedef struct Env {
  const static size_t width;
  const static size_t height;

  Rectangle env[width][height];
  Entity *entities;
} Env;

#endif
