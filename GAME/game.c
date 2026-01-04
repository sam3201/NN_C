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

Color biome_colors[] = {
    (Color){40, 120, 40, 255},   // grass
    (Color){140, 140, 140, 255}, // stone
    (Color){200, 180, 80, 255},  // desert
};

Color mob_colors[] = {
    PINK,     // pig
    RAYWHITE, // sheep
    DARKGRAY, // skeleton
    GREEN     // zombie
};

Color resource_colors[] = {
    DARKGREEN, // tree
    GRAY,      // rock
    GOLD,      // gold
    RED        // food
};

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
  float *data;
  int size;
  int capacity;
} ObsBuffer;

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
  MuModel *brain;
  float reward_accumulator;

} Tribe;

typedef struct {
  Vector2 position;
  float health, stamina;
  int agent_id;
  bool alive;
  float flash_timer;
  int age;
} Agent;

typedef struct {
  Vector2 position;
  float health;
  float stamina;
} Player;

/* =======================
   GLOBAL STATE
======================= */
Chunk world[WORLD_SIZE][WORLD_SIZE];
Tribe tribes[TRIBE_COUNT];
Agent agents[MAX_AGENTS];
Player player;

int SCREEN_WIDTH, SCREEN_HEIGHT;
float TILE_SIZE;

/* =======================
   OBS BUFFER
 * ====================== */
static inline void obs_init(ObsBuffer *o) {
  o->capacity = 32;
  o->size = 0;
  o->data = malloc(sizeof(float) * o->capacity);
}

static inline void obs_push(ObsBuffer *o, float v) {
  if (o->size >= o->capacity) {
    o->capacity *= 2;
    o->data = realloc(o->data, sizeof(float) * o->capacity);
  }
  o->data[o->size++] = v;
}

static inline void obs_free(ObsBuffer *o) {
  free(o->data);
  o->data = NULL;
  o->size = o->capacity = 0;
}
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

void draw_chunks(void) {
  int view_radius = 6;

  int pcx = (int)(player.position.x / CHUNK_SIZE);
  int pcy = (int)(player.position.y / CHUNK_SIZE);

  for (int dx = -view_radius; dx <= view_radius; dx++) {
    for (int dy = -view_radius; dy <= view_radius; dy++) {
      int cx = pcx + dx;
      int cy = pcy + dy;
      Chunk *c = get_chunk(cx, cy);

      Vector2 world_pos = {cx * CHUNK_SIZE, cy * CHUNK_SIZE};

      Vector2 screen = Vector2Subtract(world_pos, camera_pos);
      screen = Vector2Scale(screen, WORLD_SCALE);
      screen.x += SCREEN_WIDTH / 2;
      screen.y += SCREEN_HEIGHT / 2;

      DrawRectangle(screen.x, screen.y, CHUNK_SIZE * WORLD_SCALE,
                    CHUNK_SIZE * WORLD_SCALE,
                    Fade(biome_colors[c->biome_type], 0.9f));
    }
  }
}

void draw_resources(void) {
  int pcx = (int)(player.position.x / CHUNK_SIZE);
  int pcy = (int)(player.position.y / CHUNK_SIZE);

  for (int dx = -6; dx <= 6; dx++) {
    for (int dy = -6; dy <= 6; dy++) {
      Chunk *c = get_chunk(pcx + dx, pcy + dy);

      for (int i = 0; i < c->resource_count; i++) {
        Resource *r = &c->resources[i];
        if (r->health <= 0)
          continue;

        Vector2 wp = {(pcx + dx) * CHUNK_SIZE + r->position.x,
                      (pcy + dy) * CHUNK_SIZE + r->position.y};

        Vector2 sp = Vector2Subtract(wp, camera_pos);
        sp = Vector2Scale(sp, WORLD_SCALE);
        sp.x += SCREEN_WIDTH / 2;
        sp.y += SCREEN_HEIGHT / 2;

        DrawCircleV(sp, WORLD_SCALE * 0.25f, resource_colors[r->type]);
      }
    }
  }
}

void draw_mobs(void) {
  int pcx = (int)(player.position.x / CHUNK_SIZE);
  int pcy = (int)(player.position.y / CHUNK_SIZE);

  for (int dx = -6; dx <= 6; dx++) {
    for (int dy = -6; dy <= 6; dy++) {
      Chunk *c = get_chunk(pcx + dx, pcy + dy);

      for (int i = 0; i < MAX_MOBS; i++) {
        Mob *m = &c->mobs[i];
        if (m->health <= 0)
          continue;

        Vector2 wp = {(pcx + dx) * CHUNK_SIZE + m->position.x,
                      (pcy + dy) * CHUNK_SIZE + m->position.y};

        Vector2 sp = Vector2Subtract(wp, camera_pos);
        sp = Vector2Scale(sp, WORLD_SCALE);
        sp.x += SCREEN_WIDTH / 2;
        sp.y += SCREEN_HEIGHT / 2;

        DrawCircleV(sp, WORLD_SCALE * 0.35f, mob_colors[m->type]);
      }
    }
  }
}

/* =======================
   TRIBES & AGENTS
======================= */
void init_tribes(void) {
  Color colors[] = {RED, BLUE, GREEN, ORANGE};
  float spacing = 24.0f;

  MuConfig cfg = {.obs_dim = 64, // expandable, not fixed memory
                  .latent_dim = 64,
                  .action_count = ACTION_COUNT};

  for (int t = 0; t < TRIBE_COUNT; t++) {
    Tribe *tr = &tribes[t];
    tr->tribe_id = t;
    tr->color = colors[t % 4];
    tr->agent_start = t * AGENT_PER_TRIBE;
    tr->agent_count = AGENT_PER_TRIBE;
    tr->reward_accumulator = 0.0f;

    tr->brain = mu_model_create(&cfg);

    tr->base.position =
        (Vector2){WORLD_SIZE / 2 + cosf(t * 2 * PI / TRIBE_COUNT) * spacing,
                  WORLD_SIZE / 2 + sinf(t * 2 * PI / TRIBE_COUNT) * spacing};
    tr->base.radius = BASE_RADIUS;
  }
}

void init_agents(void) {
  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &agents[i];
    a->agent_id = i;
    a->alive = true;
    a->health = a->stamina = 100;
    a->flash_timer = 0;
    a->age = 0;

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
void encode_observation(Agent *a, Chunk *c, ObsBuffer *obs) {
  Tribe *tr = &tribes[a->agent_id / AGENT_PER_TRIBE];

  Vector2 to_base = Vector2Subtract(tr->base.position, a->position);
  float dbase = Vector2Length(to_base);

  obs_push(obs, a->health / 100.0f);
  obs_push(obs, a->stamina / 100.0f);
  obs_push(obs, fminf(dbase / 64.0f, 1.0f));
  obs_push(obs, to_base.x / (dbase + 1e-4f));
  obs_push(obs, to_base.y / (dbase + 1e-4f));

  /* Resource density */
  obs_push(obs, (float)c->resource_count / MAX_RESOURCES);

  /* Mean resource direction */
  Vector2 mean = {0};
  for (int i = 0; i < c->resource_count; i++) {
    mean = Vector2Add(mean,
                      Vector2Subtract(c->resources[i].position, a->position));
  }

  if (c->resource_count > 0)
    mean = Vector2Scale(mean, 1.0f / c->resource_count);

  float md = Vector2Length(mean);
  obs_push(obs, mean.x / (md + 1e-4f));
  obs_push(obs, mean.y / (md + 1e-4f));

  obs_push(obs, dbase < BASE_RADIUS ? 1.0f : 0.0f); // in base
  obs_push(obs, 1.0f);                              // bias
}

int decide_action(Agent *a, ObsBuffer *obs) {
  Tribe *tr = &tribes[a->agent_id / AGENT_PER_TRIBE];

  MCTSParams mp = {.num_simulations = 32,
                   .c_puct = 1.5f,
                   .max_depth = 16,
                   .dirichlet_alpha = 0.3f,
                   .dirichlet_eps = 0.25f,
                   .temperature = 0.8f,
                   .discount = 0.95f};

  MCTSResult r = mcts_run(tr->brain, obs->data, &mp);
  int action = r.chosen_action;
  mcts_result_free(&r);
  return action;
}

/* =======================
   AGENT UPDATE
======================= */
void update_agent(Agent *a) {
  if (!a->alive)
    return;

  int cx = (int)(a->position.x / CHUNK_SIZE);
  int cy = (int)(a->position.y / CHUNK_SIZE);
  Chunk *c = get_chunk(cx, cy);

  float reward = 0.0f;

  Tribe *tr = &tribes[a->agent_id / AGENT_PER_TRIBE];
  if (Vector2Length(Vector2Subtract(a->position, tr->base.position)) <
      BASE_RADIUS) {
    a->health = fminf(a->health + 0.5f, 100);
    a->stamina = fminf(a->stamina + 0.5f, 100);
    reward += 0.01f;
  } else {
    a->stamina -= 0.05f;
    reward -= 0.001f;
  }

  if (!a->alive) {
    reward -= 1.0f;
  }

  tribes[a->agent_id / AGENT_PER_TRIBE].reward_accumulator += reward;

  ObsBuffer obs;
  obs_init(&obs);

  encode_observation(a, c, &obs);
  MuCortex *cortex = &tribes[a->agent_id / AGENT_PER_TRIBE].cortex;

  int action = muze_plan(cortex, obs.data, obs.size, ACTION_COUNT);

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

  float d = Vector2Distance(a->position, tr->base.position);
  if (d < BASE_RADIUS) {
    a->health = fminf(a->health + 0.5f, 100);
    a->stamina = fminf(a->stamina + 0.5f, 100);
  } else {
    a->stamina -= 0.05f;
  }

  if (a->health <= 0 || a->stamina <= 0) {
    a->alive = false;
    mu_model_end_episode(tr->brain, -1.0f);
  } else {
    mu_model_step(tr->brain, obs.data, action, reward);
    obs_free(&obs);
  }

  a->age++;
}

/* =======================
   PLAYER
======================= */
void init_player(void) {
  player.position = (Vector2){WORLD_SIZE / 2, WORLD_SIZE / 2};
  player.health = 100;
  player.stamina = 100;
}

void update_player(void) {
  float speed = 0.6f;

  if (IsKeyDown(KEY_W))
    player.position.y -= speed;
  if (IsKeyDown(KEY_S))
    player.position.y += speed;
  if (IsKeyDown(KEY_A))
    player.position.x -= speed;
  if (IsKeyDown(KEY_D))
    player.position.x += speed;

  if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
    int cx = (int)(player.position.x / CHUNK_SIZE);
    int cy = (int)(player.position.y / CHUNK_SIZE);
    Chunk *c = get_chunk(cx, cy);

    for (int i = 0; i < c->resource_count; i++) {
      Resource *r = &c->resources[i];
      Vector2 rp = {cx * CHUNK_SIZE + r->position.x,
                    cy * CHUNK_SIZE + r->position.y};

      if (Vector2Distance(player.position, rp) < HARVEST_DISTANCE) {
        r->health -= 25;
        player.stamina -= 2;
        break;
      }
    }
  }

  player.stamina = fmaxf(0, player.stamina - 0.02f);
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

  init_tribes();
  init_agents();
  init_player();

  int train_timer = 0;

  while (!WindowShouldClose()) {
    camera_pos.x += (player.position.x - camera_pos.x) * 0.1f;
    camera_pos.y += (player.position.y - camera_pos.y) * 0.1f;

    update_player();
    for (int i = 0; i < MAX_AGENTS; i++)
      update_agent(&agents[i]);

    train_timer++;
    if (train_timer >= TRAIN_INTERVAL) {
      for (int t = 0; t < TRIBE_COUNT; t++) {
        mu_model_end_episode(tribes[t].brain, tribes[t].reward_accumulator);

        mu_model_train(tribes[t].brain);
        tribes[t].reward_accumulator = 0.0f;
      }
      train_timer = 0;
    }

    BeginDrawing();
    ClearBackground((Color){20, 20, 20, 255});

    draw_chunks();
    draw_resources();
    draw_mobs();

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

    /* Draw player */
    Vector2 pp = Vector2Subtract(player.position, camera_pos);
    pp = Vector2Scale(pp, WORLD_SCALE);
    pp.x += SCREEN_WIDTH / 2;
    pp.y += SCREEN_HEIGHT / 2;

    DrawCircleV(pp, WORLD_SCALE * 0.45f, YELLOW);

    DrawText("MUZE Tribal Simulation", 20, 20, 20, RAYWHITE);
    EndDrawing();
  }

  CloseWindow();
  return 0;
}
