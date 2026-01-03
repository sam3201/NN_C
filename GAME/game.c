#include "../utils/NN/MUZE/all.h"
#include "../utils/Raylib/src/raylib.h"
#include "../utils/Raylib/src/raymath.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

/* =======================
   GLOBAL CONFIG
======================= */

int SCREEN_WIDTH;
int SCREEN_HEIGHT;
float TILE_SIZE;

#define WORLD_SIZE 128
#define CHUNK_SIZE 32

#define MAX_RESOURCES 512
#define MAX_MOBS 10
#define MAX_AGENTS 8
#define BASE_RADIUS 8
#define MAX_BASE_PARTICLES 32

/* =======================
   ENUMS
======================= */

typedef enum { TOOL_HAND = 0, TOOL_AXE, TOOL_PICKAXE, TOOL_NONE } ToolType;
typedef enum { RES_TREE = 0, RES_ROCK, RES_FOOD, RES_NONE } ResourceType;

typedef enum {
  ACTION_UP = 0,
  ACTION_DOWN,
  ACTION_LEFT,
  ACTION_RIGHT,
  ACTION_ATTACK,
  ACTION_HARVEST,
  ACTION_COUNT
} ActionType;

/* =======================
   STRUCTS
======================= */

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
  MuModel *brain;
  size_t input_size;
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
  float health, stamina;
  float max_health, max_stamina;
  float move_speed;
  int wood, stone, food;
  bool alive;
  ToolType tool;
} Player;

typedef struct {
  Vector2 position;
  float radius;
} Base;

typedef struct {
  Vector2 pos;
  float lifetime;
} BaseParticle;

/* =======================
   GLOBAL STATE
======================= */

Chunk world[WORLD_SIZE][WORLD_SIZE];
Player player;
Base agent_base;
BaseParticle base_particles[MAX_BASE_PARTICLES];

/* =======================
   HELPERS
======================= */

static inline int wrap(int v) { return (v + WORLD_SIZE) % WORLD_SIZE; }

static inline float randf(float min, float max) {
  return min + (float)rand() / RAND_MAX * (max - min);
}

Color biome_color(int t) {
  switch (t) {
  case 0:
    return (Color){120, 200, 120, 255};
  case 1:
    return (Color){40, 160, 40, 255};
  case 2:
    return (Color){130, 130, 130, 255};
  default:
    return RAYWHITE;
  }
}

/* =======================
   WORLD
======================= */

Chunk *get_chunk(int cx, int cy) {
  cx = wrap(cx);
  cy = wrap(cy);

  Chunk *c = &world[cx][cy];
  if (c->generated)
    return c;

  c->generated = true;
  c->biome_type = (abs(cx) + abs(cy)) % 3;

  for (int i = 0; i < CHUNK_SIZE; i++)
    for (int j = 0; j < CHUNK_SIZE; j++)
      c->terrain[i][j] = c->biome_type;

  c->resource_count = (c->biome_type == 1) ? 12 : 6;

  for (int i = 0; i < c->resource_count; i++) {
    c->resources[i].type = rand() % 3;
    c->resources[i].position =
        (Vector2){rand() % CHUNK_SIZE, rand() % CHUNK_SIZE};
    c->resources[i].health = 100;
  }

  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &c->agents[i];
    a->alive = true;
    a->health = a->stamina = 100;
    a->flash_timer = 0;
    a->agent_id = i;

    a->tribe_color = (i % 4 == 0)   ? RED
                     : (i % 4 == 1) ? BLUE
                     : (i % 4 == 2) ? GREEN
                                    : YELLOW;

    if (cx == WORLD_SIZE / 2 && cy == WORLD_SIZE / 2) {
      float ang = (float)i / MAX_AGENTS * 2 * PI;
      float d = randf(2, BASE_RADIUS - 1);
      a->position = (Vector2){agent_base.position.x + cosf(ang) * d,
                              agent_base.position.y + sinf(ang) * d};
    } else {
      a->position = (Vector2){rand() % CHUNK_SIZE, rand() % CHUNK_SIZE};
    }

    MuConfig cfg = {
        .obs_dim = 10, .latent_dim = 32, .action_count = ACTION_COUNT};
    a->brain = mu_model_create(&cfg);
    a->input_size = cfg.obs_dim;
  }

  return c;
}

/* =======================
   PLAYER
======================= */

void init_player(void) {
  player.position = (Vector2){0, 0};
  player.max_health = player.health = 100;
  player.max_stamina = player.stamina = 100;
  player.move_speed = 2.0f;
  player.alive = true;
  player.tool = TOOL_HAND;
}

void update_player(void) {
  Vector2 m = {0};
  if (IsKeyDown(KEY_W))
    m.y -= 1;
  if (IsKeyDown(KEY_S))
    m.y += 1;
  if (IsKeyDown(KEY_A))
    m.x -= 1;
  if (IsKeyDown(KEY_D))
    m.x += 1;

  if (m.x && m.y)
    m = Vector2Scale(m, 0.707f);
  player.position =
      Vector2Add(player.position, Vector2Scale(m, player.move_speed));
}

/* =======================
   AGENTS
======================= */

MCTSParams mcts_params = {.num_simulations = 40,
                          .c_puct = 1.2f,
                          .discount = 0.95f,
                          .temperature = 1.0f};

int decide_action(Agent *a, float *obs) {
  if (!a || !a->brain)
    return rand() % ACTION_COUNT;

  MCTSResult r = mcts_run(a->brain, obs, &mcts_params);
  int act = r.chosen_action;
  mcts_result_free(&r);

  return (act >= 0 && act < ACTION_COUNT) ? act : rand() % ACTION_COUNT;
}

void update_agent(Agent *a) {
  if (!a->alive)
    return;

  float obs[a->input_size];
  for (size_t i = 0; i < a->input_size; i++)
    obs[i] = randf(0, 1);

  int act = decide_action(a, obs);

  if (act == ACTION_UP)
    a->position.y -= 0.4f;
  if (act == ACTION_DOWN)
    a->position.y += 0.4f;
  if (act == ACTION_LEFT)
    a->position.x -= 0.4f;
  if (act == ACTION_RIGHT)
    a->position.x += 0.4f;

  float d = Vector2Distance(a->position, agent_base.position);
  if (d < BASE_RADIUS) {
    a->health = fminf(a->health + 0.3f, 100);
    a->stamina = fminf(a->stamina + 0.3f, 100);
    a->flash_timer = 0.2f;
  } else {
    a->flash_timer = fmaxf(0, a->flash_timer - 0.01f);
  }
}

/* =======================
   BASE
======================= */

void init_base(void) {
  agent_base.position = (Vector2){WORLD_SIZE / 2, WORLD_SIZE / 2};
  agent_base.radius = BASE_RADIUS;

  for (int i = 0; i < MAX_BASE_PARTICLES; i++) {
    base_particles[i].pos = agent_base.position;
    base_particles[i].lifetime = randf(0, 1);
  }
}

/* =======================
   MAIN
======================= */

int main(void) {
  srand(time(NULL));

  InitWindow(1280, 800, "MUZE Game");

  SCREEN_WIDTH = GetScreenWidth();
  SCREEN_HEIGHT = GetScreenHeight();
  TILE_SIZE = SCREEN_HEIGHT / 18.0f;

  SetTargetFPS(60);

  init_base();
  init_player();

  Camera2D cam = {0};
  cam.zoom = 1.0f;
  cam.offset = (Vector2){SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2};

  while (!WindowShouldClose()) {
    update_player();

    int cx = player.position.x / (CHUNK_SIZE * TILE_SIZE);
    int cy = player.position.y / (CHUNK_SIZE * TILE_SIZE);
    Chunk *c = get_chunk(cx, cy);

    for (int i = 0; i < MAX_AGENTS; i++)
      update_agent(&c->agents[i]);

    cam.target = player.position;

    BeginDrawing();
    ClearBackground(SKYBLUE);
    BeginMode2D(cam);

    for (int dx = -1; dx <= 1; dx++)
      for (int dy = -1; dy <= 1; dy++) {
        Chunk *ch = get_chunk(cx + dx, cy + dy);

        for (int i = 0; i < CHUNK_SIZE; i++)
          for (int j = 0; j < CHUNK_SIZE; j++) {
            DrawRectangle((cx + dx) * CHUNK_SIZE * TILE_SIZE + i * TILE_SIZE,
                          (cy + dy) * CHUNK_SIZE * TILE_SIZE + j * TILE_SIZE,
                          TILE_SIZE, TILE_SIZE, biome_color(ch->terrain[i][j]));
          }

        for (int i = 0; i < MAX_AGENTS; i++) {
          Agent *a = &ch->agents[i];
          if (!a->alive)
            continue;

          Vector2 p = {
              (cx + dx) * CHUNK_SIZE * TILE_SIZE + a->position.x * TILE_SIZE,
              (cy + dy) * CHUNK_SIZE * TILE_SIZE + a->position.y * TILE_SIZE};

          DrawCircleV(p, TILE_SIZE * 0.35f, a->tribe_color);
          if (a->flash_timer > 0)
            DrawCircleV(p, TILE_SIZE * 0.25f, Fade(WHITE, 0.6f));
        }
      }

    DrawCircle(agent_base.position.x * TILE_SIZE,
               agent_base.position.y * TILE_SIZE, agent_base.radius * TILE_SIZE,
               DARKGRAY);

    DrawCircleV(player.position, TILE_SIZE * 0.4f, RED);

    EndMode2D();
    EndDrawing();
  }

  CloseWindow();
  return 0;
}
