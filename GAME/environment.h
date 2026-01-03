#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "../utils/Raylib/src/raylib.h"
#include <stdbool.h>

#define CHUNK_SIZE 32
#define TILE_SIZE 8
#define MAX_RESOURCES 10
#define MAX_MOBS 10

typedef struct {
  Vector2 position;
  float value;
  int type;
  bool visited;
} Resource;

typedef struct {
  Vector2 position;
  float value;
  int type;
  bool visited;
} Mob;

typedef struct {
  int biome_type; // 0=grassland,1=forest,2=desert etc
  int terrain[CHUNK_SIZE][CHUNK_SIZE];
  Resource resources[MAX_RESOURCES];
  Mob mobs[MAX_MOBS];
  bool generated;
} Chunk;

typedef struct {
  float health;
  float stamina;
  float attack;
  float vision_radius;
  float move_speed;
  Vector2 position;
  int agent_id;
  Color color;
} Agent;

typedef struct {
  Vector2 position;
  float health;
  float stamina;
  float max_health;
  float max_stamina;
  float move_speed;
  float attack_damage;
  float attack_range;

  int wood;
  int stone;
  int food;

  bool alive;
} Player;

void init_world();
Chunk *get_chunk(int cx, int cy);
void generate_chunk(Chunk *chunk, int cx, int cy);
void draw_world(Vector2 camera);

extern Player player;

#endif
