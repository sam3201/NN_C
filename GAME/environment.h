#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "../utils/Raylib/src/raylib.h"
#include <stdbool.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

#define WORLD_SIZE 128
#define CHUNK_SIZE 32
#define TILE_SIZE 8
#define MAX_RESOURCES 512
#define MAX_MOBS 10

#define BASE_RADIUS 8
#define MAX_AGENTS 8

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
  Vector2 position;
  float health;
  float stamina;
  int agent_id;
  Color tribe_color;
  bool alive;
  float flash_timer;
} Agent;

typedef struct {
  int biome_type;
  int terrain[CHUNK_SIZE][CHUNK_SIZE];
  Resource resources[MAX_RESOURCES];
  int resource_count;
  Mob mobs[MAX_MOBS];
  Agent agents[MAX_AGENTS];
  bool generated;
} Chunk;

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

  ToolType tool;
} Player;

typedef struct {
  Vector2 position;
  float radius;
} Base;

void init_world();
Chunk *get_chunk(int cx, int cy);
void generate_chunk(Chunk *chunk, int cx, int cy);
void draw_world(Vector2 camera);

void init_player(void);
void update_player(void);
void draw_player(Vector2 camera);
void draw_ui(void);

void init_base(void);

void update_agent(Agent *a);

extern Player player;
extern Base agent_base;
extern Resource resources[MAX_RESOURCES];
extern int resource_count;

extern ToolType tool;

Resource *find_nearest_resource(Vector2 pos, float max_dist);
void draw_resources(Vector2 camera);
void spawn_resource(Vector2 pos, ResourceType type);

#endif
