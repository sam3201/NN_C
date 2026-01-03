#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "../utils/Raylib/src/raylib.h"
#include <stdbool.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

#define CHUNK_SIZE 32
#define TILE_SIZE 8
#define MAX_RESOURCES 512
#define MAX_MOBS 10

typedef enum {
  RES_TREE = 0,
  RES_ROCK,
  RES_FOOD,
  RES_NONE,
  RES_COUNT
} ResourceType;

typedef struct {
  Vector2 position;
  ResourceType type;
  int health;
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

void init_player(void);
void update_player(void);
void draw_player(Vector2 camera);
void draw_ui(void);

void spawn_resource(Vector2 pos, ResourceType type) {
  if (resource_count >= MAX_RESOURCES)
    return;

  Resource *r = &resources[resource_count++];
  r->position = pos;
  r->type = type;
  r->health = (type == RES_TREE) ? 5 : 8;
  r->visited = false;
}

void draw_resources(Vector2 camera) {
  for (int i = 0; i < resource_count; i++) {
    Resource *r = &resources[i];
    if (!r->health <= 0)
      continue;

    Vector2 s = {r->position.x - camera.x, r->position.y - camera.y};

    if (r->type == RES_TREE) {
      DrawRectangle(s.x - 4, s.y - 12, 8, 16, DARKGREEN);
      DrawCircle(s.x, s.y - 16, 10, GREEN);
    } else {
      DrawCircle(s.x, s.y, 8, GRAY);
    }
  }
}

extern Player player;
extern Resource resources[MAX_RESOURCES];
extern int resource_count = 0;

Resource *find_nearest_resource(Vector2 pos, float max_dist);

#endif
