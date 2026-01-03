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
  TOOL_HAND = 0,
  TOOL_AXE,
  TOOL_PICKAXE,
  TOOL_NONE,
  TOOL_COUNT
} ToolType;

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

extern Player player;
extern Resource resources[MAX_RESOURCES];
extern int resource_count;

extern ToolType tool;
player.tool = TOOL_HAND;

Resource *find_nearest_resource(Vector2 pos, float max_dist);
void draw_resources(Vector2 camera);
void spawn_resource(Vector2 pos, ResourceType type);

#endif
