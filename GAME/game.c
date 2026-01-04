#include "../utils/NN/MUZE/all.h"
#include "../utils/Raylib/src/raylib.h"
#include "../utils/Raylib/src/raymath.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* =======================
   GLOBAL CONFIG
======================= */
#define WORLD_SIZE 128
#define CHUNK_SIZE 32

#define MAX_RESOURCES 512
#define MAX_MOBS 16

#define TRIBE_COUNT 2
#define AGENT_PER_TRIBE 8
#define MAX_AGENTS (TRIBE_COUNT * AGENT_PER_TRIBE)

#define BASE_RADIUS 8

#define HARVEST_DISTANCE 1.0f
#define HARVEST_AMOUNT 1
#define ATTACK_DISTANCE 1.0f
#define ATTACK_DAMAGE 1

#define TRAIN_INTERVAL 1

Vector2 camera_pos;
float WORLD_SCALE = 12.0f;

/* =======================
   ENUMS
======================= */
typedef enum {
  RES_TREE = 0,
  RES_ROCK,
  RES_GOLD,
  RES_FOOD,
  RES_NONE
} ResourceType;

typedef enum { MOB_PIG = 0, MOB_SHEEP, MOB_SKELETON, MOB_ZOMBIE } MobType;

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
  MobType type;
  int health;
  bool visited;
} Mob;

typedef struct {
  int biome_type;
  int terrain[CHUNK_SIZE][CHUNK_SIZE];
  Resource resources[MAX_RESOURCES];
  int resource_count;
  Mob mobs[MAX_MOBS];
  bool generated;
} Chunk;

typedef struct {
  Vector2 position;
  float radius;
} Base;

typedef struct {
  int tribe_id;
  Color color;
  Base base;
  int agent_start;
  int agent_count;
} Tribe;

typedef struct {
  Vector2 position;
  float health, stamina;
  int agent_id;
  bool alive;
  float flash_timer;
  MuModel *brain;
  int age;
} Agent;

/* =======================
   GLOBAL STATE
======================= */
Chunk world[WORLD_SIZE][WORLD_SIZE];
Tribe tribes[TRIBE_COUNT];
Agent agents[MAX_AGENTS];

int SCREEN_WIDTH, SCREEN_HEIGHT;
float TILE_SIZE;

/* =======================
   HELPERS
======================= */
static inline int wrap(int v) { return (v + WORLD_SIZE) % WORLD_SIZE; }
static inline float randf(float a, float b) {
  return a + (float)rand() / RAND_MAX * (b - a);
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

  c->resource_count = 8;
  for (int i = 0; i < c->resource_count; i++) {
    c->resources[i].type = rand() % 4;
    c->resources[i].position =
        (Vector2){rand() % CHUNK_SIZE, rand() % CHUNK_SIZE};
    c->resources[i].health = 100;
    c->resources[i].visited = false;
  }

  for (int i = 0; i < MAX_MOBS; i++) {
    c->mobs[i].type = rand() % 4;
    c->mobs[i].position = (Vector2){rand() % CHUNK_SIZE, rand() % CHUNK_SIZE};
    c->mobs[i].health = 100;
    c->mobs[i].visited = false;
  }

  return c;
}

/* =======================
   TRIBES & AGENTS
======================= */
void init_tribes(void) {
  Color colors[] = {RED, BLUE, GREEN, ORANGE};
  float spacing = 24.0f;

  for (int t = 0; t < TRIBE_COUNT; t++) {
    Tribe *tr = &tribes[t];
    tr->tribe_id = t;
    tr->color = colors[t % 4];
    tr->agent_start = t * AGENT_PER_TRIBE;
    tr->agent_count = AGENT_PER_TRIBE;

    tr->base.position =
        (Vector2){WORLD_SIZE / 2 + cosf(t * 2 * PI / TRIBE_COUNT) * spacing,
                  WORLD_SIZE / 2 + sinf(t * 2 * PI / TRIBE_COUNT) * spacing};
    tr->base.radius = BASE_RADIUS;
  }
}

void init_agents(void) {
  MuConfig cfg = {
      .obs_dim = 10, .latent_dim = 32, .action_count = ACTION_COUNT};

  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &agents[i];
    a->agent_id = i;
    a->alive = true;
    a->health = a->stamina = 100;
    a->flash_timer = 0;
    a->age = 0;
    a->brain = mu_model_create(&cfg);

    Tribe *tr = &tribes[i / AGENT_PER_TRIBE];
    float ang = randf(0, 2 * PI);
    float d = randf(2, tr->base.radius - 1);
    a->position = (Vector2){tr->base.position.x + cosf(ang) * d,
                            tr->base.position.y + sinf(ang) * d};
  }
}

/* =======================
   OBSERVATION & RL
======================= */
void encode_observation(Agent *a, Chunk *c, float *obs) {
  Tribe *tr = &tribes[a->agent_id / AGENT_PER_TRIBE];

  obs[0] = a->health / 100.0f;
  obs[1] = a->stamina / 100.0f;

  Vector2 to_base = Vector2Subtract(tr->base.position, a->position);
  float d = Vector2Length(to_base);

  obs[2] = fminf(d / (BASE_RADIUS * 4.0f), 1.0f);
  obs[3] = (d > 0) ? to_base.x / d : 0;
  obs[4] = (d > 0) ? to_base.y / d : 0;

  obs[5] = 1.0f;
  obs[6] = obs[7] = 0.0f;

  float nearest = 9999.0f;
  for (int i = 0; i < c->resource_count; i++) {
    Vector2 dv = Vector2Subtract(c->resources[i].position, a->position);
    float nd = Vector2Length(dv);
    if (nd < nearest) {
      nearest = nd;
      obs[5] = fminf(nd / CHUNK_SIZE, 1.0f);
      obs[6] = dv.x / (nd + 1e-4f);
      obs[7] = dv.y / (nd + 1e-4f);
    }
  }

  obs[8] = (d < BASE_RADIUS) ? 1.0f : 0.0f;
  obs[9] = 1.0f;
}

int decide_action(Agent *a, float *obs) { return rand() % ACTION_COUNT; }

/* =======================
   AGENT UPDATE
======================= */
void update_agent(Agent *a) {
  if (!a->alive)
    return;

  int cx = (int)(a->position.x / CHUNK_SIZE);
  int cy = (int)(a->position.y / CHUNK_SIZE);
  Chunk *c = get_chunk(cx, cy);

  float obs[10];
  encode_observation(a, c, obs);

  int action = decide_action(a, obs);

  switch (action) {
  case ACTION_UP:
    a->position.y -= 0.5f;
    break;
  case ACTION_DOWN:
    a->position.y += 0.5f;
    break;
  case ACTION_LEFT:
    a->position.x -= 0.5f;
    break;
  case ACTION_RIGHT:
    a->position.x += 0.5f;
    break;
  default:
    break;
  }

  Tribe *tr = &tribes[a->agent_id / AGENT_PER_TRIBE];
  float d = Vector2Distance(a->position, tr->base.position);
  if (d < BASE_RADIUS) {
    a->health = fminf(a->health + 0.5f, 100);
    a->stamina = fminf(a->stamina + 0.5f, 100);
  } else {
    a->stamina -= 0.05f;
  }

  if (a->health <= 0 || a->stamina <= 0) {
    a->alive = false;
    mu_model_end_episode(a->brain, -1.0f);
  } else {
    mu_model_step(a->brain, obs, action, 0.01f);
  }

  a->age++;
}

/* =======================
   MAIN
======================= */
int main(void) {
  srand(time(NULL));

  InitWindow(1280, 800, "MUZE Tribal Simulation");
  SCREEN_WIDTH = GetScreenWidth();
  SCREEN_HEIGHT = GetScreenHeight();
  TILE_SIZE = SCREEN_HEIGHT / 18.0f;
  SetTargetFPS(60);

  camera_pos = (Vector2){WORLD_SIZE / 2, WORLD_SIZE / 2};

  init_tribes();
  init_agents();

  int train_timer = 0;

  while (!WindowShouldClose()) {
    for (int i = 0; i < MAX_AGENTS; i++)
      update_agent(&agents[i]);

    train_timer++;
    if (train_timer >= TRAIN_INTERVAL) {
      for (int i = 0; i < MAX_AGENTS; i++)
        mu_model_train(agents[i].brain);
      train_timer = 0;
    }

    BeginDrawing();
    ClearBackground((Color){20, 20, 20, 255});

    /* Draw bases */
    for (int t = 0; t < TRIBE_COUNT; t++) {
      Vector2 bp = Vector2Subtract(tribes[t].base.position, camera_pos);
      bp = Vector2Scale(bp, WORLD_SCALE);
      bp.x += SCREEN_WIDTH / 2;
      bp.y += SCREEN_HEIGHT / 2;

      DrawCircleLines(bp.x, bp.y, tribes[t].base.radius * WORLD_SCALE,
                      tribes[t].color);
    }

    /* Draw agents */
    for (int i = 0; i < MAX_AGENTS; i++) {
      if (!agents[i].alive)
        continue;

      Vector2 p = Vector2Subtract(agents[i].position, camera_pos);
      p = Vector2Scale(p, WORLD_SCALE);
      p.x += SCREEN_WIDTH / 2;
      p.y += SCREEN_HEIGHT / 2;

      DrawCircleV(p, WORLD_SCALE * 0.3f,
                  tribes[agents[i].agent_id / AGENT_PER_TRIBE].color);
    }

    DrawText("MUZE Tribal Simulation", 20, 20, 20, RAYWHITE);
    EndDrawing();
  }

  CloseWindow();
  return 0;
}
