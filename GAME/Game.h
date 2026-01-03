#ifndef GAME_H
#define GAME_H

#include "../utils/NN/NEAT.h"
#include "../utils/NN/NN.h"
#include "../utils/Raylib/src/raylib.h"
#include <stdio.h>
#include <stdlib.h>

void *movePlayer(int dir);
void *moveEnemy(int dir);

// Fixed enum
typedef enum { PLAYER, ENEMY } EntityType;

// Agent struct
typedef struct {
  NEAT_t *neat;
  int exploration;
  int exploration_reward;
  int battle_reward;
} Agent;

// Entity struct
typedef struct {
  EntityType type;
  Agent *agent; // pointer to agent if needed
} Entity;

// Environment constants
#define ENV_WIDTH 100
#define ENV_HEIGHT 100

// Env struct
typedef struct {
  Rectangle env[ENV_WIDTH][ENV_HEIGHT];
  Entity *entities;
  size_t entity_count; // how many entities
} Env;

#endif
