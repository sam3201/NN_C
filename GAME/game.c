#include "../SAM/SAM.h"
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
#define AGENT_PER_TRIBE 2
#define MAX_AGENTS (TRIBE_COUNT * AGENT_PER_TRIBE)

#define BASE_RADIUS 8

#define HARVEST_DISTANCE 1.0f
#define HARVEST_AMOUNT 1
#define ATTACK_DISTANCE 1.0f
#define ATTACK_DAMAGE 1

#define OBS_DIM 64
#define TRAIN_INTERVAL 1

Vector2 camera_pos;
float WORLD_SCALE = 12.0f;
// Visual scaling: at least +50%
static float scale_size = 1.5f; // 1.5 = +50%
// optional: you can tweak per-category
static float RESOURCE_SCALE = 1.0f;
static float MOB_SCALE = 1.0f;

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
} Tribe;

typedef struct {
  Vector2 position;
  float health, stamina;
  int agent_id;
  bool alive;
  float flash_timer;
  int agent_start;
  SAM_t *sam;
  MuCortex *cortex;
  float reward_accumulator;
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

static inline float clamp01(float x) {
  if (x < 0.0f)
    return 0.0f;
  if (x > 1.0f)
    return 1.0f;
  return x;
}

static inline float safe_norm(float v, float denom) {
  return v / (denom + 1e-6f);
}

/* Ensure obs is exactly OBS_DIM:
   - if fewer -> pad zeros
   - if more  -> truncate
*/
static inline void obs_finalize_fixed(ObsBuffer *o, int target_dim) {
  if (!o)
    return;
  if (o->size > target_dim) {
    o->size = target_dim;
    return;
  }
  while (o->size < target_dim)
    obs_push(o, 0.0f);
}

static inline Vector2 world_to_screen(Vector2 wp) {
  Vector2 sp = Vector2Subtract(wp, camera_pos);
  sp = Vector2Scale(sp, WORLD_SCALE);
  sp.x += SCREEN_WIDTH / 2;
  sp.y += SCREEN_HEIGHT / 2;
  return sp;
}

static inline float px(float base) { return base * WORLD_SCALE * scale_size; }

static void draw_health_bar(Vector2 sp, float w, float h, float t01,
                            Color fill) {
  // background
  DrawRectangle((int)(sp.x - w * 0.5f), (int)(sp.y - h), (int)w, (int)h,
                (Color){0, 0, 0, 160});
  // fill
  float fw = w * (t01 < 0 ? 0 : (t01 > 1 ? 1 : t01));
  DrawRectangle((int)(sp.x - w * 0.5f), (int)(sp.y - h), (int)fw, (int)h, fill);
  // outline
  DrawRectangleLines((int)(sp.x - w * 0.5f), (int)(sp.y - h), (int)w, (int)h,
                     (Color){0, 0, 0, 220});
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
      int cx = pcx + dx;
      int cy = pcy + dy;
      Chunk *c = get_chunk(cx, cy);

      for (int i = 0; i < c->resource_count; i++) {
        Resource *r = &c->resources[i];
        if (r->health <= 0)
          continue;

        Vector2 wp = {(float)(cx * CHUNK_SIZE) + r->position.x,
                      (float)(cy * CHUNK_SIZE) + r->position.y};
        Vector2 sp = world_to_screen(wp);

        float s = px(0.20f) * RESOURCE_SCALE; // base size
        float s2 = s * 1.2f;

        // health tint
        float hp01 = (float)r->health / 100.0f;

        switch (r->type) {
        case RES_TREE: {
          // shadow
          DrawEllipse((int)sp.x, (int)(sp.y + s * 0.65f), (int)(s * 0.9f),
                      (int)(s * 0.35f), (Color){0, 0, 0, 70});

          // trunk
          DrawRectangle((int)(sp.x - s * 0.18f), (int)(sp.y - s * 0.20f),
                        (int)(s * 0.36f), (int)(s * 0.65f),
                        (Color){120, 80, 40, 255});
          DrawRectangleLines((int)(sp.x - s * 0.18f), (int)(sp.y - s * 0.20f),
                             (int)(s * 0.36f), (int)(s * 0.65f),
                             (Color){0, 0, 0, 150});

          // canopy (3 blobs)
          Color leaf = (Color){30, (unsigned char)(140 + 60 * hp01), 30, 255};
          DrawCircleV((Vector2){sp.x, sp.y - s * 0.55f}, s * 0.55f, leaf);
          DrawCircleV((Vector2){sp.x - s * 0.45f, sp.y - s * 0.35f}, s * 0.45f,
                      leaf);
          DrawCircleV((Vector2){sp.x + s * 0.45f, sp.y - s * 0.35f}, s * 0.45f,
                      leaf);
          DrawCircleLines((int)sp.x, (int)(sp.y - s * 0.55f), s * 0.55f,
                          (Color){0, 0, 0, 110});

          // small fruit dots
          DrawCircleV((Vector2){sp.x - s * 0.18f, sp.y - s * 0.55f}, s * 0.06f,
                      RED);
          DrawCircleV((Vector2){sp.x + s * 0.22f, sp.y - s * 0.40f}, s * 0.06f,
                      RED);

          // health bar
          draw_health_bar((Vector2){sp.x, sp.y - s2 * 1.2f}, s * 1.4f,
                          s * 0.18f, hp01, (Color){60, 220, 60, 255});
        } break;

        case RES_ROCK: {
          DrawEllipse((int)sp.x, (int)(sp.y + s * 0.55f), (int)(s * 0.9f),
                      (int)(s * 0.32f), (Color){0, 0, 0, 70});

          // rock: polygon-ish using circles + outlines
          Color rock = (Color){120, 120, 120, 255};
          DrawCircleV((Vector2){sp.x, sp.y}, s * 0.55f, rock);
          DrawCircleV((Vector2){sp.x - s * 0.35f, sp.y + s * 0.05f}, s * 0.40f,
                      rock);
          DrawCircleV((Vector2){sp.x + s * 0.35f, sp.y + s * 0.10f}, s * 0.45f,
                      rock);

          // highlight
          DrawCircleV((Vector2){sp.x - s * 0.15f, sp.y - s * 0.18f}, s * 0.18f,
                      (Color){200, 200, 200, 160});

          // outline
          DrawCircleLines((int)sp.x, (int)sp.y, s * 0.55f,
                          (Color){0, 0, 0, 120});

          draw_health_bar((Vector2){sp.x, sp.y - s2 * 1.1f}, s * 1.4f,
                          s * 0.18f, hp01, (Color){180, 180, 180, 255});
        } break;

        case RES_GOLD: {
          DrawEllipse((int)sp.x, (int)(sp.y + s * 0.55f), (int)(s * 0.9f),
                      (int)(s * 0.32f), (Color){0, 0, 0, 70});

          // nugget: bright + rim + sparkle
          Color gold = (Color){240, 210, 70, 255};
          DrawCircleV((Vector2){sp.x, sp.y}, s * 0.55f, gold);
          DrawCircleV((Vector2){sp.x - s * 0.28f, sp.y + s * 0.08f}, s * 0.38f,
                      gold);
          DrawCircleV((Vector2){sp.x + s * 0.30f, sp.y + s * 0.10f}, s * 0.40f,
                      gold);

          DrawCircleV((Vector2){sp.x - s * 0.12f, sp.y - s * 0.18f}, s * 0.14f,
                      (Color){255, 255, 255, 160});
          DrawCircleLines((int)sp.x, (int)sp.y, s * 0.55f,
                          (Color){0, 0, 0, 140});

          // sparkle
          DrawLine((int)(sp.x + s * 0.55f), (int)(sp.y - s * 0.55f),
                   (int)(sp.x + s * 0.75f), (int)(sp.y - s * 0.75f), RAYWHITE);
          DrawLine((int)(sp.x + s * 0.75f), (int)(sp.y - s * 0.55f),
                   (int)(sp.x + s * 0.55f), (int)(sp.y - s * 0.75f), RAYWHITE);

          draw_health_bar((Vector2){sp.x, sp.y - s2 * 1.1f}, s * 1.4f,
                          s * 0.18f, hp01, (Color){240, 210, 70, 255});
        } break;

        case RES_FOOD: {
          DrawEllipse((int)sp.x, (int)(sp.y + s * 0.55f), (int)(s * 0.9f),
                      (int)(s * 0.32f), (Color){0, 0, 0, 70});

          // berry/fruit: red body + green leaf
          Color fruit = (Color){220, 60, 60, 255};
          DrawCircleV((Vector2){sp.x, sp.y}, s * 0.55f, fruit);
          DrawCircleV((Vector2){sp.x - s * 0.15f, sp.y - s * 0.15f}, s * 0.18f,
                      (Color){255, 255, 255, 120});
          DrawCircleLines((int)sp.x, (int)sp.y, s * 0.55f,
                          (Color){0, 0, 0, 130});

          // leaf
          DrawTriangle((Vector2){sp.x, sp.y - s * 0.55f},
                       (Vector2){sp.x - s * 0.25f, sp.y - s * 0.75f},
                       (Vector2){sp.x + s * 0.25f, sp.y - s * 0.75f},
                       (Color){40, 180, 60, 255});

          draw_health_bar((Vector2){sp.x, sp.y - s2 * 1.1f}, s * 1.4f,
                          s * 0.18f, hp01, (Color){220, 60, 60, 255});
        } break;

        default:
          // fallback
          DrawCircleV(sp, s * 0.5f, resource_colors[r->type]);
          break;
        }
      }
    }
  }
}

void draw_mobs(void) {
  int pcx = (int)(player.position.x / CHUNK_SIZE);
  int pcy = (int)(player.position.y / CHUNK_SIZE);

  for (int dx = -6; dx <= 6; dx++) {
    for (int dy = -6; dy <= 6; dy++) {
      int cx = pcx + dx;
      int cy = pcy + dy;
      Chunk *c = get_chunk(cx, cy);

      for (int i = 0; i < MAX_MOBS; i++) {
        Mob *m = &c->mobs[i];
        if (m->health <= 0)
          continue;

        Vector2 wp = {(float)(cx * CHUNK_SIZE) + m->position.x,
                      (float)(cy * CHUNK_SIZE) + m->position.y};
        Vector2 sp = world_to_screen(wp);

        float s = px(0.26f) * MOB_SCALE; // base mob radius
        float hp01 = (float)m->health / 100.0f;

        // shadow
        DrawEllipse((int)sp.x, (int)(sp.y + s * 0.85f), (int)(s * 1.2f),
                    (int)(s * 0.40f), (Color){0, 0, 0, 80});

        switch (m->type) {
        case MOB_PIG: {
          // body
          DrawCircleV((Vector2){sp.x, sp.y}, s * 0.85f,
                      (Color){255, 160, 190, 255});
          DrawCircleLines((int)sp.x, (int)sp.y, s * 0.85f,
                          (Color){0, 0, 0, 120});

          // snout
          DrawCircleV((Vector2){sp.x + s * 0.55f, sp.y + s * 0.10f}, s * 0.28f,
                      (Color){255, 120, 160, 255});
          DrawCircleV((Vector2){sp.x + s * 0.62f, sp.y + s * 0.05f}, s * 0.06f,
                      (Color){120, 60, 80, 255});
          DrawCircleV((Vector2){sp.x + s * 0.50f, sp.y + s * 0.05f}, s * 0.06f,
                      (Color){120, 60, 80, 255});

          // ears
          DrawTriangle((Vector2){sp.x - s * 0.30f, sp.y - s * 0.70f},
                       (Vector2){sp.x - s * 0.55f, sp.y - s * 0.95f},
                       (Vector2){sp.x - s * 0.10f, sp.y - s * 0.90f},
                       (Color){255, 140, 175, 255});
          DrawTriangle((Vector2){sp.x + s * 0.10f, sp.y - s * 0.70f},
                       (Vector2){sp.x + s * 0.35f, sp.y - s * 0.95f},
                       (Vector2){sp.x + s * 0.55f, sp.y - s * 0.85f},
                       (Color){255, 140, 175, 255});

          // eyes
          DrawCircleV((Vector2){sp.x + s * 0.15f, sp.y - s * 0.15f}, s * 0.08f,
                      BLACK);
          DrawCircleV((Vector2){sp.x - s * 0.10f, sp.y - s * 0.15f}, s * 0.08f,
                      BLACK);

          draw_health_bar((Vector2){sp.x, sp.y - s * 1.35f}, s * 1.6f,
                          s * 0.18f, hp01, (Color){80, 220, 80, 255});
        } break;

        case MOB_SHEEP: {
          // wool (cloudy)
          Color wool = (Color){245, 245, 245, 255};
          DrawCircleV((Vector2){sp.x, sp.y}, s * 0.80f, wool);
          DrawCircleV((Vector2){sp.x - s * 0.45f, sp.y + s * 0.10f}, s * 0.55f,
                      wool);
          DrawCircleV((Vector2){sp.x + s * 0.45f, sp.y + s * 0.10f}, s * 0.55f,
                      wool);
          DrawCircleLines((int)sp.x, (int)sp.y, s * 0.80f,
                          (Color){0, 0, 0, 90});

          // face
          DrawCircleV((Vector2){sp.x + s * 0.55f, sp.y + s * 0.20f}, s * 0.35f,
                      (Color){70, 70, 70, 255});
          // eyes
          DrawCircleV((Vector2){sp.x + s * 0.62f, sp.y + s * 0.10f}, s * 0.06f,
                      RAYWHITE);
          DrawCircleV((Vector2){sp.x + s * 0.62f, sp.y + s * 0.10f}, s * 0.03f,
                      BLACK);

          draw_health_bar((Vector2){sp.x, sp.y - s * 1.35f}, s * 1.6f,
                          s * 0.18f, hp01, (Color){80, 220, 80, 255});
        } break;

        case MOB_SKELETON: {
          // skull
          DrawCircleV((Vector2){sp.x, sp.y - s * 0.10f}, s * 0.65f,
                      (Color){230, 230, 230, 255});
          DrawCircleLines((int)sp.x, (int)(sp.y - s * 0.10f), s * 0.65f,
                          (Color){0, 0, 0, 140});

          // jaw
          DrawRectangle((int)(sp.x - s * 0.40f), (int)(sp.y + s * 0.25f),
                        (int)(s * 0.80f), (int)(s * 0.25f),
                        (Color){210, 210, 210, 255});
          DrawRectangleLines((int)(sp.x - s * 0.40f), (int)(sp.y + s * 0.25f),
                             (int)(s * 0.80f), (int)(s * 0.25f),
                             (Color){0, 0, 0, 140});

          // eyes
          DrawCircleV((Vector2){sp.x - s * 0.20f, sp.y - s * 0.20f}, s * 0.12f,
                      BLACK);
          DrawCircleV((Vector2){sp.x + s * 0.20f, sp.y - s * 0.20f}, s * 0.12f,
                      BLACK);

          // ribs
          for (int k = 0; k < 4; k++) {
            float yy = sp.y + s * (0.10f + 0.12f * k);
            DrawLine((int)(sp.x - s * 0.35f), (int)yy, (int)(sp.x + s * 0.35f),
                     (int)yy, (Color){0, 0, 0, 110});
          }

          draw_health_bar((Vector2){sp.x, sp.y - s * 1.35f}, s * 1.6f,
                          s * 0.18f, hp01, (Color){220, 90, 90, 255});
        } break;

        case MOB_ZOMBIE: {
          // body
          Color body = (Color){80, 180, 80, 255};
          DrawCircleV((Vector2){sp.x, sp.y}, s * 0.85f, body);
          DrawCircleLines((int)sp.x, (int)sp.y, s * 0.85f,
                          (Color){0, 0, 0, 140});

          // mouth
          DrawRectangle((int)(sp.x - s * 0.25f), (int)(sp.y + s * 0.15f),
                        (int)(s * 0.50f), (int)(s * 0.18f),
                        (Color){120, 50, 50, 255});

          // eyes
          DrawCircleV((Vector2){sp.x - s * 0.20f, sp.y - s * 0.15f}, s * 0.10f,
                      RAYWHITE);
          DrawCircleV((Vector2){sp.x + s * 0.20f, sp.y - s * 0.15f}, s * 0.10f,
                      RAYWHITE);
          DrawCircleV((Vector2){sp.x - s * 0.20f, sp.y - s * 0.15f}, s * 0.05f,
                      BLACK);
          DrawCircleV((Vector2){sp.x + s * 0.20f, sp.y - s * 0.15f}, s * 0.05f,
                      BLACK);

          // little "scar"
          DrawLine((int)(sp.x + s * 0.40f), (int)(sp.y - s * 0.10f),
                   (int)(sp.x + s * 0.55f), (int)(sp.y + s * 0.05f),
                   (Color){30, 90, 30, 255});

          draw_health_bar((Vector2){sp.x, sp.y - s * 1.35f}, s * 1.6f,
                          s * 0.18f, hp01, (Color){220, 90, 90, 255});
        } break;
        }
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

  for (int t = 0; t < TRIBE_COUNT; t++) {
    Tribe *tr = &tribes[t];
    tr->tribe_id = t;
    tr->color = colors[t % 4];
    tr->base.position =
        (Vector2){WORLD_SIZE / 2 + cosf(t * 2 * PI / TRIBE_COUNT) * spacing,
                  WORLD_SIZE / 2 + sinf(t * 2 * PI / TRIBE_COUNT) * spacing};
    tr->base.radius = BASE_RADIUS;
  }
}

void init_agents(void) {
  MuConfig cfg = {.obs_dim = 64, // expandable, not fixed memory
                  .latent_dim = 64,
                  .action_count = ACTION_COUNT};

  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &agents[i];
    a->agent_id = i;
    a->alive = true;
    a->health = a->stamina = 100;
    a->flash_timer = 0;
    a->age = 0;
    a->agent_start = i;
    a->reward_accumulator = 0.0f;

    a->sam = SAM_init(cfg.obs_dim, cfg.action_count, 4, 0);
    a->cortex = SAM_as_MUZE(a->sam);

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

  // --- self status ---
  obs_push(obs, clamp01(a->health / 100.0f));
  obs_push(obs, clamp01(a->stamina / 100.0f));

  // --- base features ---
  Vector2 to_base = Vector2Subtract(tr->base.position, a->position);
  float dbase = Vector2Length(to_base);
  float base_dir_x = safe_norm(to_base.x, dbase);
  float base_dir_y = safe_norm(to_base.y, dbase);

  obs_push(obs, clamp01(dbase / 64.0f));
  obs_push(obs, base_dir_x);
  obs_push(obs, base_dir_y);
  obs_push(obs, (dbase < tr->base.radius) ? 1.0f : 0.0f); // in base

  // --- chunk density ---
  obs_push(obs, clamp01((float)c->resource_count / (float)MAX_RESOURCES));

  // --- nearest resource (by type) in VIEW CHUNK ONLY (fast) ---
  // Features: for each resource type [tree, rock, gold, food]:
  //   - nearest distance (normalized)
  //   - nearest direction (x,y) normalized
  //   - availability bit (1 if found)
  const int R_TYPES = 4;
  float best_d[R_TYPES];
  Vector2 best_dir[R_TYPES];
  int found[R_TYPES];

  for (int t = 0; t < R_TYPES; t++) {
    best_d[t] = 1e9f;
    best_dir[t] = (Vector2){0, 0};
    found[t] = 0;
  }

  // We need this chunk's world origin
  // NOTE: c is the chunk agent is currently in, so:
  int cx = (int)(a->position.x / CHUNK_SIZE);
  int cy = (int)(a->position.y / CHUNK_SIZE);
  Vector2 chunk_origin =
      (Vector2){(float)(cx * CHUNK_SIZE), (float)(cy * CHUNK_SIZE)};

  for (int i = 0; i < c->resource_count; i++) {
    Resource *r = &c->resources[i];
    if (r->health <= 0)
      continue;
    int t = (int)r->type;
    if (t < 0 || t >= R_TYPES)
      continue;

    // Convert resource local -> world
    Vector2 r_world = Vector2Add(chunk_origin, r->position);

    Vector2 dvec = Vector2Subtract(r_world, a->position);
    float d = Vector2Length(dvec);

    if (d < best_d[t]) {
      best_d[t] = d;
      best_dir[t] = dvec;
      found[t] = 1;
    }
  }

  for (int t = 0; t < R_TYPES; t++) {
    if (!found[t]) {
      obs_push(obs, 1.0f); // distance "far"
      obs_push(obs, 0.0f); // dir x
      obs_push(obs, 0.0f); // dir y
      obs_push(obs, 0.0f); // found bit
    } else {
      float d = best_d[t];
      Vector2 v = best_dir[t];
      obs_push(obs, clamp01(d / 32.0f)); // normalize to something reasonable
      obs_push(obs, safe_norm(v.x, d));
      obs_push(obs, safe_norm(v.y, d));
      obs_push(obs, 1.0f);
    }
  }

  // --- nearest mob (any type) in this chunk ---
  float best_mob_d = 1e9f;
  Vector2 best_mob_dir = (Vector2){0, 0};
  int mob_found = 0;
  int mob_type = 0;

  for (int i = 0; i < MAX_MOBS; i++) {
    Mob *m = &c->mobs[i];
    if (m->health <= 0)
      continue;

    Vector2 m_world = Vector2Add(chunk_origin, m->position);
    Vector2 dvec = Vector2Subtract(m_world, a->position);
    float d = Vector2Length(dvec);

    if (d < best_mob_d) {
      best_mob_d = d;
      best_mob_dir = dvec;
      mob_found = 1;
      mob_type = (int)m->type;
    }
  }

  if (!mob_found) {
    obs_push(obs, 1.0f);
    obs_push(obs, 0.0f);
    obs_push(obs, 0.0f);
    obs_push(obs, 0.0f);
  } else {
    obs_push(obs, clamp01(best_mob_d / 32.0f));
    obs_push(obs, safe_norm(best_mob_dir.x, best_mob_d));
    obs_push(obs, safe_norm(best_mob_dir.y, best_mob_d));
    obs_push(obs, (float)mob_type / 3.0f); // 0..3 -> 0..1
  }

  // bias
  obs_push(obs, 1.0f);

  // final: enforce fixed-size vector
  obs_finalize_fixed(obs, OBS_DIM);
}

int decide_action(Agent *a, ObsBuffer *obs) {
  return muze_plan(a->cortex, obs->data, (size_t)obs->size,
                   (size_t)ACTION_COUNT);
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

  Tribe *tr = &tribes[a->agent_id / AGENT_PER_TRIBE];

  float reward = 0.0f;

  // living cost / base bonus
  float dist_to_base =
      Vector2Length(Vector2Subtract(a->position, tr->base.position));
  if (dist_to_base < BASE_RADIUS) {
    a->health = fminf(a->health + 0.5f, 100.0f);
    a->stamina = fminf(a->stamina + 0.5f, 100.0f);
    reward += 0.01f;
  } else {
    a->stamina -= 0.05f;
    reward -= 0.001f;
  }

  a->reward_accumulator += reward;

  ObsBuffer obs;
  obs_init(&obs);
  encode_observation(a, c, &obs);

  MuCortex *cortex = a->cortex;
  int action = muze_plan(cortex, obs.data, obs.size, ACTION_COUNT);

  // apply action
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

  // post-move stamina drain / regen (optional, keep if you like it)
  dist_to_base = Vector2Distance(a->position, tr->base.position);
  if (dist_to_base < BASE_RADIUS) {
    a->health = fminf(a->health + 0.5f, 100.0f);
    a->stamina = fminf(a->stamina + 0.5f, 100.0f);
  } else {
    a->stamina -= 0.05f;
  }

  a->age++;

  int terminal = 0;
  if (a->health <= 0.0f || a->stamina <= 0.0f) {
    a->alive = false;
    terminal = 1;
    reward -= 1.0f; // death penalty (now it actually happens)
  }

  // ONLY learn through the cortex adapter (SAM brain)
  cortex->learn(cortex->brain, obs.data, obs.size, action, reward, terminal);

  obs_free(&obs);
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

  while (!WindowShouldClose()) {
    camera_pos.x += (player.position.x - camera_pos.x) * 0.1f;
    camera_pos.y += (player.position.y - camera_pos.y) * 0.1f;

    update_player();
    for (int i = 0; i < MAX_AGENTS; i++)
      update_agent(&agents[i]);

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

      DrawCircleLinesV(bp, tribes[t].base.radius * WORLD_SCALE,
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
    /* Display FPS */
    DrawText(TextFormat("FPS: %d", GetFPS()), 20, 40, 20, RAYWHITE);
    EndDrawing();
  }

  CloseWindow();
  return 0;
}
