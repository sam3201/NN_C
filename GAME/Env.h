#ifndef ENV_H
#define ENV_H

#include <stdio.h>
#include <stdlib.h>
#include "../utils/Raylib/src/raylib.h"
#include "../utils/NN/NEAT.h"
#include "../utils/NN/NN.h"

// Fixed enum
typedef enum {
    PLAYER,
    ENEMY
} EntityType;

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
    Agent *agent;  // pointer to agent if needed
} Entity;

// Environment constants
#define ENV_WIDTH  100
#define ENV_HEIGHT 100

// Env struct
typedef struct {
    Rectangle env[ENV_WIDTH][ENV_HEIGHT];
    Entity *entities;
} Env;

#endif

