#include "../SAM/SAM.h"
#include "../utils/NN/MUZE/all.h"
#include "../utils/Raylib/src/raylib.h"
#include "../utils/Raylib/src/raymath.h"
#include <math.h>
#include <pthread.h>
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
#define MAX_MOBS 64

#define TRIBE_COUNT 2
#define AGENT_PER_TRIBE 8
#define MAX_AGENTS (TRIBE_COUNT * AGENT_PER_TRIBE)

#define BASE_RADIUS 8

#define HARVEST_DISTANCE 5.0f
#define HARVEST_AMOUNT 1
#define ATTACK_DISTANCE 3.0f
#define ATTACK_DAMAGE 1

#define PLAYER_HARVEST_DAMAGE 10
#define PLAYER_ATTACK_DAMAGE 5

#define PLAYER_HARVEST_COOLDOWN 0.18f
#define PLAYER_ATTACK_COOLDOWN 0.22f

#define PLAYER_MINE_DAMAGE 18
#define PLAYER_MINE_COOLDOWN 0.26f
#define PLAYER_MINE_STAMINA_COST 3.0f

#define STAMINA_REGEN_RATE 5.0f
#define STAMINA_DRAIN_RATE 0.5f

#define MAX_PICKUPS 256

#define MOB_AGGRO_RANGE 10.0f
#define MOB_ATTACK_RANGE 1.25f
#define MOB_SPEED_PASSIVE 0.55f
#define MOB_SPEED_HOSTILE 0.85f

#define MAX_PROJECTILES 64

#define OBS_DIM 64
#define TRAIN_INTERVAL 1

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

typedef enum { PICK_FOOD = 0, PICK_SHARD, PICK_ARROW } PickupType;

// ----- prototypes (so C doesn't need ordering) -----
typedef struct Chunk Chunk; // forward decl (since Chunk is defined later)

static inline float res_radius_world(ResourceType t);
static inline float mob_radius_world(MobType t);
static int world_pos_blocked_nearby(int cx, int cy, Vector2 worldPos,
                                    float radius, int self_cx, int self_cy);
Chunk *get_chunk(int cx, int cy);
// raids / day-night mob maintenance
static void despawn_hostiles_if_day(Chunk *c);
static void spawn_raid_wave(void);

// you already define this later, but spawn_raid_wave will use it:
static void spawn_mob_at_world(MobType type, Vector2 world_pos);

/* =======================
   STRUCTS
======================= */

typedef struct {
  float *data;
  int size;
  int capacity;
} ObsBuffer;

typedef struct {
  Vector2 position; // LOCAL pos inside chunk
  ResourceType type;
  int health;
  bool visited;

  // animation
  float hit_timer;   // shake
  float break_flash; // white flash
} Resource;

typedef struct {
  Vector2 position; // LOCAL pos inside chunk
  MobType type;
  int health;
  bool visited;

  // AI / motion (chunk-local)
  Vector2 vel;
  float ai_timer;    // when to pick new wander dir
  float aggro_timer; // stays angry after being hit
  float attack_cd;
  float hurt_timer;  // flash
  float lunge_timer; // attack anim
} Mob;

typedef struct {
  bool alive;
  Vector2 pos; // WORLD pos
  Vector2 vel;
  float ttl;
  int damage;
} Projectile;

typedef struct Chunk {
  int biome_type;
  int terrain[CHUNK_SIZE][CHUNK_SIZE];
  Resource resources[MAX_RESOURCES];
  int resource_count;
  Mob mobs[MAX_MOBS];
  bool generated;
  float mob_spawn_timer;

  pthread_rwlock_t lock;
} Chunk;

typedef struct {
  Vector2 position;
  float radius;
} Base;

typedef struct {
  int tribe_id;
  Color color;
  Base base;
  float integrity; // 0..base_health
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
  int last_action;
} Agent;

typedef struct {
  Vector2 position;
  float health;
  float stamina;
} Player;

typedef struct {
  bool alive;
  Vector2 pos; // WORLD pos
  PickupType type;
  int amount;
  float ttl;
  float bob_t;
} Pickup;

typedef struct {
  const char *name;
  int wood, stone, gold, food;
  int *unlock_flag; // set to 1 on craft
} Recipe;

/* =======================
   GLOBAL STATE
======================= */
#define WORKER_COUNT 8

static pthread_t workers[WORKER_COUNT];
static pthread_mutex_t job_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t job_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t done_cv = PTHREAD_COND_INITIALIZER;

static int job_next_agent = 0;
static int job_done_workers = 0;
static int job_active = 0;
static int job_quit = 0;

Chunk world[WORLD_SIZE][WORLD_SIZE];
Tribe tribes[TRIBE_COUNT];
Agent agents[MAX_AGENTS];
Player player;

int SCREEN_WIDTH, SCREEN_HEIGHT;
float TILE_SIZE;

Vector2 camera_pos;
float WORLD_SCALE = 50.0f;
// Visual scaling: at least +50%
static float RESOURCE_SCALE = 10.0f;
// global “everything bigger” knob
static float scale_size = 1.65f; // bigger overall (was 1.5)

// per-resource tuning
static float TREE_SCALE = 2.35f; // MUCH larger
static float ROCK_SCALE = 1.35f;
static float GOLD_SCALE = 1.25f;
static float FOOD_SCALE = 1.20f;

// mobs can also be bigger if you want
static float MOB_SCALE = 3.15f;

// ------------------- Day/Night -------------------
static float time_of_day = 0.25f;         // 0..1 (0 = midnight, 0.5 = noon)
static float day_length_seconds = 240.0f; // 4 minutes per full day
static int is_night_cached = 0;

static int inv_wood = 0, inv_stone = 0, inv_gold = 0, inv_food = 0;

static float player_harvest_cd = 0.0f;
static float player_attack_cd = 0.0f;
static float player_hurt_timer = 0.0f;

bool crafting_open = false;

bool has_axe = false;
bool has_pickaxe = false;
bool has_sword = false;

static Pickup pickups[MAX_PICKUPS];

static int inv_shards = 0;
static int inv_arrows = 0;

// smooth zoom
static float target_world_scale = 50.0f;

// camera shake
static float cam_shake = 0.0f;

// store last hand positions (screen space) for cooldown rings
static Vector2 g_handL = {0}, g_handR = {0};

// raid
static float raid_timer = 0.0f;
static float raid_interval = 4.5f; // seconds between mini-waves at night
static int was_night = 0;

static Projectile projectiles[MAX_PROJECTILES];

static Recipe recipes[] = {
    {"Axe (Wood+Stone)", 3, 2, 0, 0, &has_axe},
    {"Pickaxe (Wood+Stone)", 3, 3, 0, 0, &has_pickaxe},
    {"Sword (Stone+Gold)", 0, 4, 2, 0, &has_sword},
    {"Armor (Stone+Gold)", 0, 5, 2, 0, NULL},

};
static int recipe_count = sizeof(recipes) / sizeof(recipes[0]);

static int can_afford(const Recipe *r) {
  return inv_wood >= r->wood && inv_stone >= r->stone && inv_gold >= r->gold &&
         inv_food >= r->food;
}

static void craft(const Recipe *r) {
  if (!can_afford(r))
    return;
  inv_wood -= r->wood;
  inv_stone -= r->stone;
  inv_gold -= r->gold;
  inv_food -= r->food;
  if (r->unlock_flag)
    *r->unlock_flag = 1;
}

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

static inline float clampf(float v, float a, float b) {
  return (v < a) ? a : (v > b) ? b : v;
}

static inline Vector2 clamp_local_to_chunk(Vector2 lp) {
  lp.x = clampf(lp.x, 0.25f, (float)CHUNK_SIZE - 0.25f);
  lp.y = clampf(lp.y, 0.25f, (float)CHUNK_SIZE - 0.25f);
  return lp;
}

static inline int is_night(void) {
  // night from ~0.75 -> 1.0 and 0.0 -> 0.25 (tweak)
  return (time_of_day >= 0.75f || time_of_day <= 0.25f);
}

static void update_daynight(float dt) {
  time_of_day += dt / day_length_seconds;
  while (time_of_day >= 1.0f)
    time_of_day -= 1.0f;
  is_night_cached = is_night();
}

// a simple overlay (dark at night)
static void draw_daynight_overlay(void) {
  float t = time_of_day;

  // “how dark is it” curve: darkest at midnight, brightest at noon
  // (0 at noon, 1 at midnight)
  float midnight_dist = fabsf(t - 0.0f);
  midnight_dist = fminf(midnight_dist, fabsf(t - 1.0f));
  float noon_dist = fabsf(t - 0.5f);

  // 0..1 where 1 is midnight-ish
  float night01 = clamp01((0.5f - noon_dist) * 2.0f);
  night01 = 1.0f - night01;

  unsigned char a = (unsigned char)(150 * night01); // max darkness alpha
  DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, (Color){10, 20, 40, a});

  DrawText(is_night_cached ? "Night" : "Day", 20, 210, 20, RAYWHITE);
  DrawText(TextFormat("Time: %0.2f", time_of_day), 20, 235, 18, RAYWHITE);
}

static inline float px(float base) { return base * WORLD_SCALE * scale_size; }

static inline float player_attack_range(void) {
  // sword gives a bit more reach
  return ATTACK_DISTANCE + (has_sword ? 0.55f : 0.0f);
}

static inline int player_attack_damage(void) {
  return PLAYER_ATTACK_DAMAGE + (has_sword ? 10 : 0);
}

static inline float player_attack_cooldown(void) {
  return PLAYER_ATTACK_COOLDOWN * (has_sword ? 0.88f : 1.0f);
}

static inline int player_resource_damage(ResourceType t) {
  // tree prefers axe
  if (t == RES_TREE)
    return PLAYER_HARVEST_DAMAGE + (has_axe ? 18 : 0);

  // rocks/gold prefer pickaxe
  if (t == RES_ROCK || t == RES_GOLD)
    return PLAYER_MINE_DAMAGE + (has_pickaxe ? 16 : 0);

  // food is “harvestable”
  if (t == RES_FOOD)
    return PLAYER_HARVEST_DAMAGE;

  return PLAYER_HARVEST_DAMAGE;
}

static inline float player_resource_cooldown(ResourceType t) {
  if (t == RES_TREE)
    return PLAYER_HARVEST_COOLDOWN * (has_axe ? 0.75f : 1.0f);
  if (t == RES_ROCK || t == RES_GOLD)
    return PLAYER_MINE_COOLDOWN * (has_pickaxe ? 0.78f : 1.0f);
  if (t == RES_FOOD)
    return PLAYER_HARVEST_COOLDOWN;
  return PLAYER_HARVEST_COOLDOWN;
}

static inline float player_resource_stamina_cost(ResourceType t) {
  if (t == RES_ROCK || t == RES_GOLD)
    return PLAYER_MINE_STAMINA_COST * (has_pickaxe ? 0.80f : 1.0f);
  if (t == RES_TREE)
    return 2.0f * (has_axe ? 0.85f : 1.0f);
  if (t == RES_FOOD)
    return 1.5f;
  return 2.0f;
}

static void spawn_pickup(PickupType type, Vector2 pos, int amount) {
  for (int i = 0; i < MAX_PICKUPS; i++) {
    if (!pickups[i].alive) {
      pickups[i].alive = true;
      pickups[i].pos = pos;
      pickups[i].type = type;
      pickups[i].amount = amount;
      pickups[i].ttl = 30.0f;
      pickups[i].bob_t = randf(0.0f, 10.0f);
      return;
    }
  }
}

static void give_pickup(PickupType t, int amount) {
  if (amount <= 0)
    return;
  switch (t) {
  case PICK_FOOD:
    inv_food += amount;
    break;
  case PICK_SHARD:
    inv_shards += amount;
    break;
  case PICK_ARROW:
    inv_arrows += amount;
    break;
  default:
    break;
  }
}

static void update_pickups(float dt) {
  for (int i = 0; i < MAX_PICKUPS; i++) {
    Pickup *p = &pickups[i];
    if (!p->alive)
      continue;
    p->ttl -= dt;
    p->bob_t += dt;
    if (p->ttl <= 0.0f)
      p->alive = false;
  }
}

static void collect_nearby_pickups(void) {
  for (int i = 0; i < MAX_PICKUPS; i++) {
    Pickup *p = &pickups[i];
    if (!p->alive)
      continue;
    if (Vector2Distance(p->pos, player.position) < 1.15f) {
      give_pickup(p->type, p->amount);
      p->alive = false;
    }
  }
}

static Vector2 nearest_base_pos(Vector2 wp) {
  float bestD = 1e9f;
  Vector2 best = tribes[0].base.position;
  for (int t = 0; t < TRIBE_COUNT; t++) {
    float d = Vector2Distance(wp, tribes[t].base.position);
    if (d < bestD) {
      bestD = d;
      best = tribes[t].base.position;
    }
  }
  return best;
}

static inline Vector2 world_to_screen(Vector2 wp) {
  Vector2 sp = Vector2Subtract(wp, camera_pos);
  sp = Vector2Scale(sp, WORLD_SCALE);
  sp.x += SCREEN_WIDTH / 2;
  sp.y += SCREEN_HEIGHT / 2;
  return sp;
}

static void draw_pickups(void) {
  for (int i = 0; i < MAX_PICKUPS; i++) {
    Pickup *p = &pickups[i];
    if (!p->alive)
      continue;

    Vector2 sp = world_to_screen(p->pos);
    float s = px(0.12f);

    float bob = sinf(p->bob_t * 5.0f) * (s * 0.35f);
    sp.y += bob;

    // shadow
    DrawEllipse((int)sp.x, (int)(sp.y + s * 1.2f), (int)(s * 1.4f),
                (int)(s * 0.55f), (Color){0, 0, 0, 70});

    switch (p->type) {
    case PICK_FOOD: {
      DrawCircleV(sp, s * 0.75f, (Color){220, 60, 60, 255});
      DrawCircleLines((int)sp.x, (int)sp.y, s * 0.75f, (Color){0, 0, 0, 160});
      DrawTriangle((Vector2){sp.x, sp.y - s * 0.9f},
                   (Vector2){sp.x - s * 0.35f, sp.y - s * 1.25f},
                   (Vector2){sp.x + s * 0.35f, sp.y - s * 1.25f},
                   (Color){40, 180, 60, 255});
    } break;

    case PICK_SHARD: {
      DrawPoly(sp, 4, s * 0.9f, 45.0f, (Color){170, 210, 255, 255});
      DrawPolyLines(sp, 4, s * 0.9f, 45.0f, (Color){0, 0, 0, 160});
    } break;

    case PICK_ARROW: {
      // tiny arrow icon
      Vector2 a = sp;
      Vector2 b = (Vector2){sp.x + s * 1.6f, sp.y};
      DrawLineEx(a, b, s * 0.25f, (Color){230, 230, 230, 255});
      DrawTriangle((Vector2){b.x, b.y},
                   (Vector2){b.x - s * 0.5f, b.y - s * 0.35f},
                   (Vector2){b.x - s * 0.5f, b.y + s * 0.35f},
                   (Color){230, 230, 230, 255});
      DrawLineEx((Vector2){sp.x - s * 0.6f, sp.y - s * 0.3f},
                 (Vector2){sp.x - s * 0.2f, sp.y}, s * 0.18f,
                 (Color){180, 140, 90, 255});
      DrawLineEx((Vector2){sp.x - s * 0.6f, sp.y + s * 0.3f},
                 (Vector2){sp.x - s * 0.2f, sp.y}, s * 0.18f,
                 (Color){180, 140, 90, 255});
    } break;
    }
  }
}

static void on_mob_killed(MobType t, Vector2 mob_world_pos) {
  // small drop scatter
  Vector2 p = mob_world_pos;
  p.x += randf(-0.35f, 0.35f);
  p.y += randf(-0.35f, 0.35f);

  if (t == MOB_PIG || t == MOB_SHEEP) {
    spawn_pickup(PICK_FOOD, p, 1 + (rand() % 2));
  } else if (t == MOB_ZOMBIE) {
    spawn_pickup(PICK_SHARD, p, 1 + (rand() % 3));
  } else if (t == MOB_SKELETON) {
    spawn_pickup(PICK_ARROW, p, 2 + (rand() % 3));
    spawn_pickup(PICK_SHARD, p, 1);
  }
}

static void give_drop(ResourceType t) {
  switch (t) {
  case RES_TREE:
    inv_wood += 1;
    break;
  case RES_ROCK:
    inv_stone += 1;
    break;
  case RES_GOLD:
    inv_gold += 1;
    break;
  case RES_FOOD:
    inv_food += 1;
    break;
  default:
    break;
  }
}

static const char *res_name(ResourceType t) {
  switch (t) {
  case RES_TREE:
    return "Tree";
  case RES_ROCK:
    return "Rock";
  case RES_GOLD:
    return "Gold";
  case RES_FOOD:
    return "Food";
  default:
    return "None";
  }
}

static const char *mob_name(MobType t) {
  switch (t) {
  case MOB_PIG:
    return "Pig";
  case MOB_SHEEP:
    return "Sheep";
  case MOB_SKELETON:
    return "Skeleton";
  case MOB_ZOMBIE:
    return "Zombie";
  default:
    return "Mob";
  }
}

static inline float safe_norm(float v, float denom) {
  return v / (denom + 1e-6f);
}

/* Ensure obs is exactly OBS_DIM:
   - if fewer -> pad zeros
   - if more  -> truncate
*/

static inline float wrap_angle(float a) {
  while (a > PI)
    a -= 2.0f * PI;
  while (a < -PI)
    a += 2.0f * PI;
  return a;
}

static inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

static inline float lerp_angle(float a, float b, float t) {
  float d = wrap_angle(b - a);
  return a + d * t;
}

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
static inline int mob_is_alive(const Mob *m) { return m->health > 0; }

static int mob_too_close_local(const Chunk *c, Vector2 p, float minD) {
  for (int i = 0; i < MAX_MOBS; i++) {
    const Mob *m = &c->mobs[i];
    if (!mob_is_alive(m))
      continue;
    if (Vector2Distance(m->position, p) < minD)
      return 1;
  }
  return 0;
}

static int chunk_alive_mobs(Chunk *c) {
  int n = 0;
  for (int i = 0; i < MAX_MOBS; i++)
    if (c->mobs[i].health > 0)
      n++;
  return n;
}

static int find_free_mob_slot(Chunk *c) {
  for (int i = 0; i < MAX_MOBS; i++)
    if (c->mobs[i].health <= 0)
      return i;
  return -1;
}

static MobType pick_spawn_type(int night, int biome) {
  // Simple rule: day = mostly passive, night = mostly hostile
  if (!night) {
    return (rand() % 2 == 0) ? MOB_PIG : MOB_SHEEP;
  } else {
    // night hostiles; biome can bias
    if (biome == 2)
      return MOB_SKELETON; // desert -> more skeletons
    return (rand() % 2 == 0) ? MOB_ZOMBIE : MOB_SKELETON;
  }
}

static void init_mob(Mob *m, MobType type, Vector2 local_pos, int make_angry) {
  m->type = type;
  m->position = clamp_local_to_chunk(local_pos);

  m->health = 100;
  m->visited = false;
  m->vel = (Vector2){0, 0};
  m->ai_timer = randf(0.2f, 1.2f);
  m->aggro_timer = make_angry ? 2.0f : 0.0f;
  m->attack_cd = randf(0.2f, 1.0f);
  m->hurt_timer = 0.0f;
  m->lunge_timer = 0.0f;
}

static void try_spawn_mobs_in_chunk(Chunk *c, int cx, int cy, float dt) {
  c->mob_spawn_timer -= dt;
  if (c->mob_spawn_timer > 0.0f)
    return;

  c->mob_spawn_timer = randf(1.5f, 4.0f);

  int night = is_night_cached;

  // cap populations differently for day/night
  int cap = night ? 10 : 6;
  if (chunk_alive_mobs(c) >= cap)
    return;

  // spawn 1 mob
  int slot = find_free_mob_slot(c);
  if (slot < 0)
    return;

  Mob *m = &c->mobs[slot];
  MobType mt = pick_spawn_type(night, c->biome_type);

  // spacing: larger mobs -> larger min distance
  Vector2 p = {0};
  int placed = 0;

  float rad = mob_radius_world(mt);

  for (int tries = 0; tries < 35; tries++) {
    p = (Vector2){randf(0.8f, CHUNK_SIZE - 0.8f),
                  randf(0.8f, CHUNK_SIZE - 0.8f)};

    Vector2 worldPos = (Vector2){(float)(cx * CHUNK_SIZE) + p.x,
                                 (float)(cy * CHUNK_SIZE) + p.y};

    if (!world_pos_blocked_nearby(cx, cy, worldPos, rad, cx, cy)) {
      placed = 1;
      break;
    }
  }

  if (!placed) {
    // fallback (still float, but no spacing guarantee)
    p = (Vector2){randf(0.8f, CHUNK_SIZE - 0.8f),
                  randf(0.8f, CHUNK_SIZE - 0.8f)};
  }

  init_mob(m, mt, p, /*make_angry=*/0);
}

static void player_try_attack_mob_in_chunk(Chunk *c, int cx, int cy) {
  if (player_attack_cd > 0.0f)
    return;

  float range = player_attack_range();
  int dmg = player_attack_damage();

  // Find nearest mob in this chunk
  Mob *best = NULL;
  float bestD = 1e9f;

  for (int i = 0; i < MAX_MOBS; i++) {
    Mob *m = &c->mobs[i];
    if (m->health <= 0)
      continue;

    Vector2 mw = (Vector2){cx * CHUNK_SIZE + m->position.x,
                           cy * CHUNK_SIZE + m->position.y};

    float d = Vector2Distance(player.position, mw);
    if (d < range && d < bestD) {
      bestD = d;
      best = m;
    }
  }

  if (!best)
    return;

  // Attack!
  player_attack_cd = player_attack_cooldown();
  best->health -= dmg;

  best->hurt_timer = 0.18f;
  best->aggro_timer = 3.0f;  // make it angry
  best->lunge_timer = 0.10f; // small visual kick
  cam_shake = fmaxf(cam_shake, 0.10f);

  if (best->health <= 0) {
    Vector2 mob_world_pos = (Vector2){cx * CHUNK_SIZE + best->position.x,
                                      cy * CHUNK_SIZE + best->position.y};
    on_mob_killed(best->type, mob_world_pos);
    best->health = 0;
  }
}

static void player_try_harvest_resource_in_chunk(Chunk *c, int cx, int cy) {
  if (player_harvest_cd > 0.0f)
    return;

  Resource *best = NULL;
  float bestD = 1e9f;

  for (int i = 0; i < c->resource_count; i++) {
    Resource *r = &c->resources[i];
    if (r->health <= 0)
      continue;

    Vector2 rw = (Vector2){cx * CHUNK_SIZE + r->position.x,
                           cy * CHUNK_SIZE + r->position.y};

    float d = Vector2Distance(player.position, rw);
    if (d < HARVEST_DISTANCE && d < bestD) {
      bestD = d;
      best = r;
    }
  }

  if (!best)
    return;

  float cd = player_resource_cooldown(best->type);
  float cost = player_resource_stamina_cost(best->type);
  int dmg = player_resource_damage(best->type);

  if (player.stamina <= cost)
    return;

  player_harvest_cd = cd;
  player.stamina -= cost;

  best->health -= dmg;
  best->hit_timer = 0.14f;
  best->break_flash = 0.06f;

  if (best->type == RES_ROCK || best->type == RES_GOLD) {
    cam_shake = fmaxf(cam_shake, 0.08f);
  }

  if (best->health <= 0) {
    give_drop(best->type);
    best->health = 0;
  }
}

Chunk *get_chunk(int cx, int cy) {
  cx = wrap(cx);
  cy = wrap(cy);

  Chunk *c = &world[cx][cy];

  // Fast path (already generated)
  if (c->generated)
    return c;

  // Thread-safe generation: only ONE thread generates a chunk.
  pthread_rwlock_wrlock(&c->lock);

  // Another thread may have generated it while we waited.
  if (c->generated) {
    pthread_rwlock_unlock(&c->lock);
    return c;
  }

  // --------- GENERATE ONCE ----------
  c->generated = true;

  // biome
  c->biome_type = (abs(cx) + abs(cy)) % 3;

  // terrain fill
  for (int i = 0; i < CHUNK_SIZE; i++) {
    for (int j = 0; j < CHUNK_SIZE; j++) {
      c->terrain[i][j] = c->biome_type;
    }
  }

  // resources
  c->resource_count = 0;
  {
    const int desired = 12; // density
    for (int k = 0; k < desired && c->resource_count < MAX_RESOURCES; k++) {
      ResourceType rt = (ResourceType)(rand() % 4);
      float rad = res_radius_world(rt);

      int placed = 0;
      Vector2 local = {0};

      for (int tries = 0; tries < 80; tries++) {
        local = (Vector2){
            randf(0.9f, (float)CHUNK_SIZE - 0.9f),
            randf(0.9f, (float)CHUNK_SIZE - 0.9f),
        };

        Vector2 worldPos = (Vector2){
            (float)(cx * CHUNK_SIZE) + local.x,
            (float)(cy * CHUNK_SIZE) + local.y,
        };

        if (!world_pos_blocked_nearby(cx, cy, worldPos, rad, cx, cy)) {
          placed = 1;
          break;
        }
      }

      if (!placed)
        continue;

      Resource *r = &c->resources[c->resource_count++];
      r->type = rt;
      r->position = clamp_local_to_chunk(local);
      r->health = 100;
      r->visited = false;
      r->hit_timer = 0.0f;
      r->break_flash = 0.0f;
    }
  }

  // mobs init (ONLY ONCE)
  for (int i = 0; i < MAX_MOBS; i++) {
    c->mobs[i].health = 0;
    c->mobs[i].visited = false;
    c->mobs[i].vel = (Vector2){0, 0};
    c->mobs[i].ai_timer = 0.0f;
    c->mobs[i].aggro_timer = 0.0f;
    c->mobs[i].attack_cd = 0.0f;
    c->mobs[i].hurt_timer = 0.0f;
    c->mobs[i].lunge_timer = 0.0f;
  }

  // spawn timer
  c->mob_spawn_timer = randf(1.0f, 3.0f);

  // seed mobs (ONE loop, spacing-aware)
  {
    const int seed_count = 6; // tweak 4..8
    int night = is_night_cached;

    for (int k = 0; k < seed_count; k++) {
      int slot = find_free_mob_slot(c);
      if (slot < 0)
        break;

      MobType mt = pick_spawn_type(night, c->biome_type);
      float rad = mob_radius_world(mt);

      int placed = 0;
      Vector2 p = {0};

      for (int tries = 0; tries < 55; tries++) {
        p = (Vector2){
            randf(0.9f, (float)CHUNK_SIZE - 0.9f),
            randf(0.9f, (float)CHUNK_SIZE - 0.9f),
        };

        Vector2 worldPos = (Vector2){
            (float)(cx * CHUNK_SIZE) + p.x,
            (float)(cy * CHUNK_SIZE) + p.y,
        };

        if (!world_pos_blocked_nearby(cx, cy, worldPos, rad, cx, cy)) {
          placed = 1;
          break;
        }
      }

      if (!placed)
        continue;

      init_mob(&c->mobs[slot], mt, p, /*make_angry=*/0);
    }
  }

  pthread_rwlock_unlock(&c->lock);
  return c;
}

static inline float res_radius_world(ResourceType t) {
  switch (t) {
  case RES_TREE:
    return 1.35f;
  case RES_ROCK:
    return 1.05f;
  case RES_GOLD:
    return 1.00f;
  case RES_FOOD:
    return 0.90f;
  default:
    return 1.00f;
  }
}

// ---- Mob radius in WORLD units (used for spacing / collisions during spawn)
// ----
static inline float mob_radius_world(MobType t) {
  switch (t) {
  case MOB_PIG:
    return 0.95f;
  case MOB_SHEEP:
    return 0.95f;
  case MOB_SKELETON:
    return 0.90f;
  case MOB_ZOMBIE:
    return 0.98f;
  default:
    return 0.95f;
  }
}

// ---- Spacing test: is "worldPos" too close to ANY resource/mob nearby? ----
// IMPORTANT: we do NOT call get_chunk() here (avoids recursive generation).
// We only check chunks that are already generated.
// ---- Spacing test: is "worldPos" too close to ANY resource/mob nearby? ----
// IMPORTANT:
// - Does NOT call get_chunk() (no recursive generation)
// - Safely reads generated chunks by taking a read lock
// - If the caller already holds the lock for (self_cx,self_cy), pass those so
//   we DON'T try to lock it again (avoids deadlock).
static int world_pos_blocked_nearby(int cx, int cy, Vector2 worldPos,
                                    float radius, int self_cx, int self_cy) {
  const float padding = 0.25f; // extra spacing so things don't touch

  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {

      int ncx = wrap(cx + dx);
      int ncy = wrap(cy + dy);

      Chunk *c = &world[ncx][ncy];
      if (!c->generated)
        continue;

      // Avoid deadlock: if caller already holds this chunk's lock, don't lock
      // it again.
      bool is_self = (ncx == wrap(self_cx) && ncy == wrap(self_cy));

      if (!is_self) {
        pthread_rwlock_rdlock(&c->lock);
        // chunk could have been toggled while we waited (rare, but safe)
        if (!c->generated) {
          pthread_rwlock_unlock(&c->lock);
          continue;
        }
      }

      Vector2 origin =
          (Vector2){(float)(ncx * CHUNK_SIZE), (float)(ncy * CHUNK_SIZE)};

      // check resources
      for (int i = 0; i < c->resource_count; i++) {
        Resource *r = &c->resources[i];
        if (r->health <= 0)
          continue;

        Vector2 r_world = Vector2Add(origin, r->position);
        float rr = res_radius_world(r->type);

        if (Vector2Distance(worldPos, r_world) < (radius + rr + padding)) {
          if (!is_self)
            pthread_rwlock_unlock(&c->lock);
          return 1;
        }
      }

      // check mobs
      for (int i = 0; i < MAX_MOBS; i++) {
        Mob *m = &c->mobs[i];
        if (m->health <= 0)
          continue;

        Vector2 m_world = Vector2Add(origin, m->position);
        float mr = mob_radius_world(m->type);

        if (Vector2Distance(worldPos, m_world) < (radius + mr + padding)) {
          if (!is_self)
            pthread_rwlock_unlock(&c->lock);
          return 1;
        }
      }

      if (!is_self) {
        pthread_rwlock_unlock(&c->lock);
      }
    }
  }

  return 0;
}

void draw_chunks(void) {
  int view_radius = 6;

  int pcx = (int)(player.position.x / CHUNK_SIZE);
  int pcy = (int)(player.position.y / CHUNK_SIZE);

  float chunk_px = (float)CHUNK_SIZE * WORLD_SCALE;

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

      // IMPORTANT: snap to pixel grid to avoid seams
      int sx = (int)floorf(screen.x);
      int sy = (int)floorf(screen.y);

      // IMPORTANT: ceil size (and +1) so we never leave a 1px gap
      int sw = (int)ceilf(chunk_px) + 1;
      int sh = (int)ceilf(chunk_px) + 1;

      DrawRectangle(sx, sy, sw, sh, Fade(biome_colors[c->biome_type], 0.9f));
    }
  }
}

static void spawn_mob_at_world(MobType type, Vector2 world_pos) {
  float rad = mob_radius_world(type);

  Vector2 chosen = world_pos;
  int found = 0;

  for (int tries = 0; tries < 20; tries++) {
    Vector2 cand = world_pos;
    cand.x += randf(-1.2f, 1.2f);
    cand.y += randf(-1.2f, 1.2f);

    int ccx = (int)(cand.x / CHUNK_SIZE);
    int ccy = (int)(cand.y / CHUNK_SIZE);

    if (!world_pos_blocked_nearby(ccx, ccy, cand, rad, -999999, -999999)) {
      chosen = cand;
      found = 1;
      break;
    }
  }

  // if we didn't find a clean spot, just use the original (fallback)
  if (!found)
    chosen = world_pos;

  int cx = (int)(chosen.x / CHUNK_SIZE);
  int cy = (int)(chosen.y / CHUNK_SIZE);

  Chunk *c = get_chunk(cx, cy);
  int slot = find_free_mob_slot(c);
  if (slot < 0)
    return;

  Vector2 origin =
      (Vector2){(float)(cx * CHUNK_SIZE), (float)(cy * CHUNK_SIZE)};
  Vector2 local = Vector2Subtract(chosen, origin);

  init_mob(&c->mobs[slot], type, local, /*make_angry=*/1);
}

static void despawn_hostiles_if_day(Chunk *c) {
  // If it's night, do nothing.
  if (is_night_cached)
    return;

  // Daytime: remove hostile mobs in this chunk so day feels safer.
  // You can tune this: either hard-despawn all hostiles, or probabilistic.
  for (int i = 0; i < MAX_MOBS; i++) {
    Mob *m = &c->mobs[i];
    if (m->health <= 0)
      continue;

    bool hostile = (m->type == MOB_ZOMBIE || m->type == MOB_SKELETON);
    if (!hostile)
      continue;

    // Option A: despawn all hostiles immediately:
    m->health = 0;

    // Option B (comment A out, uncomment B) for softer cleanup:
    // if ((rand() % 100) < 35) m->health = 0;
  }
}

static void spawn_raid_wave(void) {
  // Called only at night in your main loop.
  // Spawn a small wave near each base.
  for (int t = 0; t < TRIBE_COUNT; t++) {
    Tribe *tr = &tribes[t];

    // How many attackers per mini-wave per base:
    int count = 1 + (rand() % 3); // 1..3

    for (int k = 0; k < count; k++) {
      // pick hostile type
      MobType mt = (rand() % 2 == 0) ? MOB_ZOMBIE : MOB_SKELETON;

      // spawn in a ring around the base
      float ang = randf(0.0f, 2.0f * PI);
      float r = tr->base.radius + randf(6.0f, 14.0f); // distance from base edge

      Vector2 pos = (Vector2){
          tr->base.position.x + cosf(ang) * r,
          tr->base.position.y + sinf(ang) * r,
      };

      spawn_mob_at_world(mt, pos);
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
        // shake animation when hit
        float mul = 1.0f;
        switch (r->type) {
        case RES_TREE:
          mul = TREE_SCALE;
          break;
        case RES_ROCK:
          mul = ROCK_SCALE;
          break;
        case RES_GOLD:
          mul = GOLD_SCALE;
          break;
        case RES_FOOD:
          mul = FOOD_SCALE;
          break;
        default:
          mul = 1.0f;
          break;
        }
        float s = px(0.20f) * RESOURCE_SCALE * mul;
        float s2 = s * 1.2f;

        // shake animation when hit
        if (r->hit_timer > 0.0f) {
          float k = r->hit_timer * 18.0f;
          sp.x += sinf(GetTime() * 70.0f) * (s * 0.02f) * k;
          sp.y += cosf(GetTime() * 55.0f) * (s * 0.02f) * k;
        }

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

          // cracks (more cracks when low hp)
          if (r->health > 0) {
            float hp01 = (float)r->health / 100.0f;
            int cracks = (hp01 > 0.66f) ? 0 : (hp01 > 0.33f) ? 2 : 4;
            for (int k = 0; k < cracks; k++) {
              float a = (float)k / (float)(cracks + 1) * PI;
              DrawLine((int)(sp.x - cosf(a) * s * 0.30f),
                       (int)(sp.y - sinf(a) * s * 0.25f),
                       (int)(sp.x + cosf(a) * s * 0.18f),
                       (int)(sp.y + sinf(a) * s * 0.22f),
                       (Color){0, 0, 0, 120});
            }
          }
          // break flash overlay
          if (r->break_flash > 0.0f) {
            DrawCircleV((Vector2){sp.x, sp.y}, s * 0.35f,
                        (Color){255, 255, 255,
                                (unsigned char)(120 * clamp01(r->break_flash *
                                                              20.0f))});
          }

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
          // cracks (more cracks when low hp)
          if (r->health > 0) {
            float hp01 = (float)r->health / 100.0f;
            int cracks = (hp01 > 0.66f) ? 0 : (hp01 > 0.33f) ? 2 : 4;
            for (int k = 0; k < cracks; k++) {
              float a = (float)k / (float)(cracks + 1) * PI;
              DrawLine((int)(sp.x - cosf(a) * s * 0.30f),
                       (int)(sp.y - sinf(a) * s * 0.25f),
                       (int)(sp.x + cosf(a) * s * 0.18f),
                       (int)(sp.y + sinf(a) * s * 0.22f),
                       (Color){0, 0, 0, 120});
            }
          }
          // break flash overlay
          if (r->break_flash > 0.0f) {
            DrawCircleV((Vector2){sp.x, sp.y}, s * 0.35f,
                        (Color){255, 255, 255,
                                (unsigned char)(120 * clamp01(r->break_flash *
                                                              20.0f))});
          }

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
          // cracks (more cracks when low hp)
          if (r->health > 0) {
            float hp01 = (float)r->health / 100.0f;
            int cracks = (hp01 > 0.66f) ? 0 : (hp01 > 0.33f) ? 2 : 4;
            for (int k = 0; k < cracks; k++) {
              float a = (float)k / (float)(cracks + 1) * PI;
              DrawLine((int)(sp.x - cosf(a) * s * 0.30f),
                       (int)(sp.y - sinf(a) * s * 0.25f),
                       (int)(sp.x + cosf(a) * s * 0.18f),
                       (int)(sp.y + sinf(a) * s * 0.22f),
                       (Color){0, 0, 0, 120});
            }
          }
          // break flash overlay
          if (r->break_flash > 0.0f) {
            DrawCircleV((Vector2){sp.x, sp.y}, s * 0.35f,
                        (Color){255, 255, 255,
                                (unsigned char)(120 * clamp01(r->break_flash *
                                                              20.0f))});
          }

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
          // cracks (more cracks when low hp)
          if (r->health > 0) {
            float hp01 = (float)r->health / 100.0f;
            int cracks = (hp01 > 0.66f) ? 0 : (hp01 > 0.33f) ? 2 : 4;
            for (int k = 0; k < cracks; k++) {
              float a = (float)k / (float)(cracks + 1) * PI;
              DrawLine((int)(sp.x - cosf(a) * s * 0.30f),
                       (int)(sp.y - sinf(a) * s * 0.25f),
                       (int)(sp.x + cosf(a) * s * 0.18f),
                       (int)(sp.y + sinf(a) * s * 0.22f),
                       (Color){0, 0, 0, 120});
            }
          }
          // break flash overlay
          if (r->break_flash > 0.0f) {
            DrawCircleV((Vector2){sp.x, sp.y}, s * 0.35f,
                        (Color){255, 255, 255,
                                (unsigned char)(120 * clamp01(r->break_flash *
                                                              20.0f))});
          }

        } break;

        default:
          // fallback
          fprintf(stderr, "Unknown resource type: %d\n", r->type);
          exit(1);
          break;
        }
      }
    }
  }
}

// small helpers for drawing
static inline Color lerp_color(Color a, Color b, float t) {
  t = clamp01(t);
  Color out;
  out.r = (unsigned char)(a.r + (b.r - a.r) * t);
  out.g = (unsigned char)(a.g + (b.g - a.g) * t);
  out.b = (unsigned char)(a.b + (b.b - a.b) * t);
  out.a = (unsigned char)(a.a + (b.a - a.a) * t);
  return out;
}

static inline Color mul_color(Color a, Color b, float t) {
  // multiply blend toward (a*b) by amount t
  t = clamp01(t);
  Color m;
  m.r = (unsigned char)((a.r * b.r) / 255);
  m.g = (unsigned char)((a.g * b.g) / 255);
  m.b = (unsigned char)((a.b * b.b) / 255);
  m.a = a.a;
  return lerp_color(a, m, t);
}

void draw_mobs(void) {

  int pcx = (int)(player.position.x / CHUNK_SIZE);
  int pcy = (int)(player.position.y / CHUNK_SIZE);

  float tnow = (float)GetTime();

  for (int dx = -6; dx <= 6; dx++) {
    for (int dy = -6; dy <= 6; dy++) {
      int cx = pcx + dx;
      int cy = pcy + dy;
      Chunk *c = get_chunk(cx, cy);

      // use this chunk's biome as a subtle tint for everything inside it
      Color biome = biome_colors[c->biome_type];

      for (int i = 0; i < MAX_MOBS; i++) {
        Mob *m = &c->mobs[i];
        if (m->health <= 0)
          continue;

        Vector2 wp = {(float)(cx * CHUNK_SIZE) + m->position.x,
                      (float)(cy * CHUNK_SIZE) + m->position.y};
        Vector2 sp = world_to_screen(wp);

        float s = px(0.26f) * MOB_SCALE;
        float hp01 = (float)m->health / 100.0f;

        // idle bob
        float bob =
            sinf(tnow * 6.0f + (float)(i * 13 + m->type * 7)) * (s * 0.05f);
        sp.y += bob;

        // base mob color by type
        Color base = mob_colors[m->type];

        // biome tint (subtle)
        base = mul_color(base, biome, 0.18f);

        // aggro tint (blend toward red while aggro_timer is active)
        if (m->aggro_timer > 0.0f) {
          float a01 =
              clamp01(m->aggro_timer / 3.0f); // 3.0f matches your set value
          base = lerp_color(base, (Color){220, 60, 60, 255}, 0.35f * a01);
        }

        // hurt flash (blend toward white while hurt_timer is active)
        if (m->hurt_timer > 0.0f) {
          float h01 = clamp01(m->hurt_timer * 6.0f); // quick strong flash
          base = lerp_color(base, RAYWHITE, h01);
        }

        // lunge toward player (visual only)
        if (m->lunge_timer > 0.0f) {
          Vector2 toP = Vector2Subtract(player.position, wp);
          float d = Vector2Length(toP);
          Vector2 dir =
              (d > 1e-3f) ? Vector2Scale(toP, 1.0f / d) : (Vector2){0, 0};
          float push = (0.10f + 0.20f * clamp01(m->lunge_timer * 8.0f));
          sp.x += dir.x * (s * push);
          sp.y += dir.y * (s * push);
        }

        // tiny hurt shake
        if (m->hurt_timer > 0.0f) {
          float k = clamp01(m->hurt_timer * 10.0f);
          sp.x += sinf(tnow * 80.0f + (float)i) * (s * 0.02f) * k;
          sp.y += cosf(tnow * 65.0f + (float)i) * (s * 0.02f) * k;
        }

        // shadow
        DrawEllipse((int)sp.x, (int)(sp.y + s * 0.85f), (int)(s * 1.2f),
                    (int)(s * 0.40f), (Color){0, 0, 0, 80});

        // draw per-type, but using the computed "base" tint everywhere
        switch (m->type) {
        case MOB_PIG: {
          // body
          DrawCircleV((Vector2){sp.x, sp.y}, s * 0.85f, base);
          DrawCircleLines((int)sp.x, (int)sp.y, s * 0.85f,
                          (Color){0, 0, 0, 120});

          // snout (slightly darker/lighter from base)
          Color snout = lerp_color(base, (Color){255, 120, 160, 255}, 0.55f);
          DrawCircleV((Vector2){sp.x + s * 0.55f, sp.y + s * 0.10f}, s * 0.28f,
                      snout);

          DrawCircleV((Vector2){sp.x + s * 0.62f, sp.y + s * 0.05f}, s * 0.06f,
                      (Color){120, 60, 80, 255});
          DrawCircleV((Vector2){sp.x + s * 0.50f, sp.y + s * 0.05f}, s * 0.06f,
                      (Color){120, 60, 80, 255});

          // ears
          Color ear = lerp_color(base, (Color){255, 140, 175, 255}, 0.45f);
          DrawTriangle((Vector2){sp.x - s * 0.30f, sp.y - s * 0.70f},
                       (Vector2){sp.x - s * 0.55f, sp.y - s * 0.95f},
                       (Vector2){sp.x - s * 0.10f, sp.y - s * 0.90f}, ear);
          DrawTriangle((Vector2){sp.x + s * 0.10f, sp.y - s * 0.70f},
                       (Vector2){sp.x + s * 0.35f, sp.y - s * 0.95f},
                       (Vector2){sp.x + s * 0.55f, sp.y - s * 0.85f}, ear);

          // eyes
          DrawCircleV((Vector2){sp.x + s * 0.15f, sp.y - s * 0.15f}, s * 0.08f,
                      BLACK);
          DrawCircleV((Vector2){sp.x - s * 0.10f, sp.y - s * 0.15f}, s * 0.08f,
                      BLACK);

          draw_health_bar((Vector2){sp.x, sp.y - s * 1.35f}, s * 1.6f,
                          s * 0.18f, hp01, (Color){80, 220, 80, 255});
        } break;

        case MOB_SHEEP: {
          // wool: keep mostly white but still accept biome tint a bit
          Color wool = lerp_color((Color){245, 245, 245, 255}, base, 0.18f);
          DrawCircleV((Vector2){sp.x, sp.y}, s * 0.80f, wool);
          DrawCircleV((Vector2){sp.x - s * 0.45f, sp.y + s * 0.10f}, s * 0.55f,
                      wool);
          DrawCircleV((Vector2){sp.x + s * 0.45f, sp.y + s * 0.10f}, s * 0.55f,
                      wool);
          DrawCircleLines((int)sp.x, (int)sp.y, s * 0.80f,
                          (Color){0, 0, 0, 90});

          // face
          Color face = (Color){70, 70, 70, 255};
          face = mul_color(face, biome, 0.10f);
          if (m->hurt_timer > 0.0f)
            face = lerp_color(face, RAYWHITE, clamp01(m->hurt_timer * 6.0f));

          DrawCircleV((Vector2){sp.x + s * 0.55f, sp.y + s * 0.20f}, s * 0.35f,
                      face);

          // eye
          DrawCircleV((Vector2){sp.x + s * 0.62f, sp.y + s * 0.10f}, s * 0.06f,
                      RAYWHITE);
          DrawCircleV((Vector2){sp.x + s * 0.62f, sp.y + s * 0.10f}, s * 0.03f,
                      BLACK);

          draw_health_bar((Vector2){sp.x, sp.y - s * 1.35f}, s * 1.6f,
                          s * 0.18f, hp01, (Color){80, 220, 80, 255});
        } break;

        case MOB_SKELETON: {
          // bone: slightly tinted by biome + hurt flash already applied via
          // base
          Color bone = lerp_color((Color){230, 230, 230, 255}, base, 0.55f);

          // skull
          DrawCircleV((Vector2){sp.x, sp.y - s * 0.10f}, s * 0.65f, bone);
          DrawCircleLines((int)sp.x, (int)(sp.y - s * 0.10f), s * 0.65f,
                          (Color){0, 0, 0, 140});

          // jaw
          DrawRectangle((int)(sp.x - s * 0.40f), (int)(sp.y + s * 0.25f),
                        (int)(s * 0.80f), (int)(s * 0.25f), bone);
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
          DrawCircleV((Vector2){sp.x, sp.y}, s * 0.85f, base);
          DrawCircleLines((int)sp.x, (int)sp.y, s * 0.85f,
                          (Color){0, 0, 0, 140});

          // mouth
          Color mouth = (Color){120, 50, 50, 255};
          if (m->hurt_timer > 0.0f)
            mouth = lerp_color(mouth, RAYWHITE, clamp01(m->hurt_timer * 6.0f));
          DrawRectangle((int)(sp.x - s * 0.25f), (int)(sp.y + s * 0.15f),
                        (int)(s * 0.50f), (int)(s * 0.18f), mouth);

          // eyes
          DrawCircleV((Vector2){sp.x - s * 0.20f, sp.y - s * 0.15f}, s * 0.10f,
                      RAYWHITE);
          DrawCircleV((Vector2){sp.x + s * 0.20f, sp.y - s * 0.15f}, s * 0.10f,
                      RAYWHITE);
          DrawCircleV((Vector2){sp.x - s * 0.20f, sp.y - s * 0.15f}, s * 0.05f,
                      BLACK);
          DrawCircleV((Vector2){sp.x + s * 0.20f, sp.y - s * 0.15f}, s * 0.05f,
                      BLACK);

          // scar
          DrawLine((int)(sp.x + s * 0.40f), (int)(sp.y - s * 0.10f),
                   (int)(sp.x + s * 0.55f), (int)(sp.y + s * 0.05f),
                   (Color){30, 90, 30, 255});

          draw_health_bar((Vector2){sp.x, sp.y - s * 1.35f}, s * 1.6f,
                          s * 0.18f, hp01, (Color){220, 90, 90, 255});
        } break;

        default:
          break;
        }
      }
    }
  }
}

static void draw_player(Vector2 pp_screen) {
  // Body sizing
  float bodyR = WORLD_SCALE * 0.60f * scale_size;
  float outlineR = bodyR * 1.02f;

  Color outline = (Color){20, 20, 20, 180};
  Color body = (Color){255, 220, 120, 255};
  Color blush = (Color){255, 140, 160, 170};
  Color eyeW = RAYWHITE;
  Color eyeB = (Color){35, 35, 35, 255};
  Color shadow = (Color){0, 0, 0, 70};

  // Shadow
  DrawEllipse((int)pp_screen.x, (int)(pp_screen.y + bodyR * 0.85f),
              (int)(bodyR * 1.35f), (int)(bodyR * 0.45f), shadow);

  // Outline + body
  DrawCircleV(pp_screen, outlineR, outline);
  DrawCircleV(pp_screen, bodyR, body);

  // Highlight
  DrawCircleV(
      (Vector2){pp_screen.x - bodyR * 0.25f, pp_screen.y - bodyR * 0.25f},
      bodyR * 0.18f, (Color){255, 255, 255, 120});

  // --- Mouse-aim direction (screen space) ---
  Vector2 mouse = GetMousePosition();
  Vector2 aim = Vector2Subtract(mouse, pp_screen);
  float aimLen = Vector2Length(aim);
  if (aimLen < 1e-3f)
    aimLen = 1e-3f;
  float aimAng = atan2f(aim.y, aim.x);

  // --- Animated hands that orbit and “point” toward mouse ---
  float t = (float)GetTime();

  float handR = bodyR * 0.28f;
  float handArm = bodyR * 0.90f; // distance from center

  // slight breathing wiggle
  float wiggle = sinf(t * 7.0f) * (bodyR * 0.04f);

  // Perpendicular offset so hands sit "around" the aim line
  Vector2 aimDir = (Vector2){cosf(aimAng), sinf(aimAng)};
  Vector2 perp = (Vector2){-aimDir.y, aimDir.x};

  // LEFT HAND: EXACTLY on aim direction (this fixes your “left hand angle”)
  Vector2 hl = Vector2Add(pp_screen, Vector2Scale(aimDir, handArm + wiggle));

  // RIGHT HAND: slightly behind + offset sideways so you can see both hands
  Vector2 hr = pp_screen;
  hr = Vector2Add(hr, Vector2Scale(aimDir, (handArm - bodyR * 0.12f) - wiggle));
  hr = Vector2Add(hr, Vector2Scale(perp, bodyR * 0.22f));

  // little “finger nub” pointing toward mouse from each hand
  float nubR = handR * 0.28f;
  Vector2 toMouseL = Vector2Subtract(mouse, hl);
  Vector2 toMouseR = Vector2Subtract(mouse, hr);

  Vector2 dirL = Vector2Normalize(toMouseL);
  Vector2 dirR = Vector2Normalize(toMouseR);

  if (Vector2Length(toMouseL) < 1e-3f)
    dirL = aimDir;
  if (Vector2Length(toMouseR) < 1e-3f)
    dirR = aimDir;

  Vector2 hl_nub = Vector2Add(hl, Vector2Scale(dirL, handR * 0.65f));
  Vector2 hr_nub = Vector2Add(hr, Vector2Scale(dirR, handR * 0.65f));

  // draw hands (outline + fill)
  Color handFill = (Color){255, 210, 110, 255};

  DrawCircleV(hl, handR * 1.02f, outline);
  DrawCircleV(hl, handR, handFill);
  DrawCircleV(hl_nub, nubR * 1.02f, outline);
  DrawCircleV(hl_nub, nubR, handFill);

  DrawCircleV(hr, handR * 1.02f, outline);
  DrawCircleV(hr, handR, handFill);
  DrawCircleV(hr_nub, nubR * 1.02f, outline);
  DrawCircleV(hr_nub, nubR, handFill);

  // Feet
  float footR = bodyR * 0.26f;
  Vector2 fl = {pp_screen.x - bodyR * 0.25f, pp_screen.y + bodyR * 0.70f};
  Vector2 fr = {pp_screen.x + bodyR * 0.25f, pp_screen.y + bodyR * 0.70f};

  DrawCircleV(fl, footR * 1.02f, outline);
  DrawCircleV(fr, footR * 1.02f, outline);
  DrawCircleV(fl, footR, (Color){255, 160, 120, 255});
  DrawCircleV(fr, footR, (Color){255, 160, 120, 255});

  // Face
  float eyeOffX = bodyR * 0.22f;
  float eyeOffY = bodyR * 0.12f;
  float eyeR = bodyR * 0.13f;

  Vector2 eL = {pp_screen.x - eyeOffX, pp_screen.y - eyeOffY};
  Vector2 eR = {pp_screen.x + eyeOffX, pp_screen.y - eyeOffY};

  DrawCircleV(eL, eyeR * 1.05f, outline);
  DrawCircleV(eR, eyeR * 1.05f, outline);
  DrawCircleV(eL, eyeR, eyeW);
  DrawCircleV(eR, eyeR, eyeW);

  float blink = (sinf(t * 2.5f) > 0.97f) ? 0.35f : 1.0f;
  float pupR = eyeR * 0.45f;
  DrawEllipse((int)eL.x, (int)eL.y, (int)(pupR * 1.1f), (int)(pupR * blink),
              eyeB);
  DrawEllipse((int)eR.x, (int)eR.y, (int)(pupR * 1.1f), (int)(pupR * blink),
              eyeB);

  Vector2 mouth = {pp_screen.x, pp_screen.y + bodyR * 0.18f};
  DrawCircleV(mouth, bodyR * 0.06f, (Color){120, 60, 60, 255});

  Vector2 bl = {pp_screen.x - bodyR * 0.38f, pp_screen.y + bodyR * 0.05f};
  Vector2 br = {pp_screen.x + bodyR * 0.38f, pp_screen.y + bodyR * 0.05f};
  DrawCircleV(bl, bodyR * 0.10f, blush);
  DrawCircleV(br, bodyR * 0.10f, blush);

  g_handL = hl;
  g_handR = hr;
}

static void draw_agent_detailed(const Agent *a, Vector2 sp, Color tribeColor) {
  float r = WORLD_SCALE * 0.22f * scale_size; // base agent size
  float t = (float)GetTime();

  Color outline = (Color){0, 0, 0, 140};
  Color skin = (Color){235, 210, 190, 255};
  Color cloth = tribeColor;

  // tiny bob
  float bob = sinf(t * 6.0f + (float)a->agent_id) * (r * 0.08f);
  sp.y += bob;

  // shadow
  DrawEllipse((int)sp.x, (int)(sp.y + r * 1.15f), (int)(r * 1.5f),
              (int)(r * 0.45f), (Color){0, 0, 0, 70});

  // body (tunic)
  DrawCircleV((Vector2){sp.x, sp.y + r * 0.40f}, r * 0.95f, cloth);
  DrawCircleLines((int)sp.x, (int)(sp.y + r * 0.40f), r * 0.95f, outline);

  // head
  DrawCircleV((Vector2){sp.x, sp.y - r * 0.25f}, r * 0.85f, skin);
  DrawCircleLines((int)sp.x, (int)(sp.y - r * 0.25f), r * 0.85f, outline);

  // highlight
  DrawCircleV((Vector2){sp.x - r * 0.25f, sp.y - r * 0.55f}, r * 0.20f,
              (Color){255, 255, 255, 120});

  // eyes
  DrawCircleV((Vector2){sp.x - r * 0.18f, sp.y - r * 0.30f}, r * 0.10f,
              RAYWHITE);
  DrawCircleV((Vector2){sp.x + r * 0.18f, sp.y - r * 0.30f}, r * 0.10f,
              RAYWHITE);
  DrawCircleV((Vector2){sp.x - r * 0.18f, sp.y - r * 0.30f}, r * 0.05f, BLACK);
  DrawCircleV((Vector2){sp.x + r * 0.18f, sp.y - r * 0.30f}, r * 0.05f, BLACK);

  // headband (tribe marker)
  DrawRectangle((int)(sp.x - r * 0.70f), (int)(sp.y - r * 0.55f),
                (int)(r * 1.40f), (int)(r * 0.22f), tribeColor);
  DrawRectangleLines((int)(sp.x - r * 0.70f), (int)(sp.y - r * 0.55f),
                     (int)(r * 1.40f), (int)(r * 0.22f), outline);

  // tiny “tool” (points based on a subtle idle direction)
  // If you later store last action direction, plug it here.
  float ang = sinf(t * 1.5f + a->agent_id) * 0.8f;
  Vector2 toolDir = {cosf(ang), sinf(ang)};
  Vector2 toolPos = {sp.x + toolDir.x * (r * 0.95f),
                     sp.y + toolDir.y * (r * 0.95f)};

  // handle
  DrawLineEx(sp, toolPos, r * 0.14f, (Color){120, 80, 40, 255});
  // head (axe/spear-ish)
  DrawCircleV(toolPos, r * 0.20f, (Color){180, 180, 180, 255});
  DrawCircleLines((int)toolPos.x, (int)toolPos.y, r * 0.20f, outline);

  // small health/stamina pips
  float hp01 = clamp01(a->health / 100.0f);
  float st01 = clamp01(a->stamina / 100.0f);
  draw_health_bar((Vector2){sp.x, sp.y - r * 1.35f}, r * 1.7f, r * 0.18f, hp01,
                  (Color){80, 220, 80, 255});
  draw_health_bar((Vector2){sp.x, sp.y - r * 1.10f}, r * 1.7f, r * 0.18f, st01,
                  (Color){80, 160, 255, 255});

  // --- intent icon above head ---
  Vector2 icon = (Vector2){sp.x, sp.y - r * 1.55f};
  Color ic = (Color){0, 0, 0, 180};
  DrawCircleV(icon, r * 0.35f, (Color){255, 255, 255, 200});
  DrawCircleLines((int)icon.x, (int)icon.y, r * 0.35f, ic);

  switch (a->last_action) {
  case ACTION_ATTACK: {
    // sword-ish line
    DrawLineEx((Vector2){icon.x - r * 0.10f, icon.y + r * 0.14f},
               (Vector2){icon.x + r * 0.14f, icon.y - r * 0.14f}, r * 0.10f,
               (Color){120, 120, 120, 255});
    DrawCircleV((Vector2){icon.x + r * 0.16f, icon.y - r * 0.16f}, r * 0.10f,
                (Color){200, 200, 200, 255});
  } break;

  case ACTION_HARVEST: {
    // pickaxe-ish
    DrawLineEx((Vector2){icon.x - r * 0.16f, icon.y - r * 0.05f},
               (Vector2){icon.x + r * 0.16f, icon.y + r * 0.12f}, r * 0.10f,
               (Color){120, 80, 40, 255});
    DrawCircleV((Vector2){icon.x + r * 0.18f, icon.y + r * 0.14f}, r * 0.10f,
                (Color){180, 180, 180, 255});
  } break;

  default: {
    // move: little arrow
    DrawTriangle((Vector2){icon.x, icon.y - r * 0.16f},
                 (Vector2){icon.x - r * 0.14f, icon.y + r * 0.12f},
                 (Vector2){icon.x + r * 0.14f, icon.y + r * 0.12f},
                 (Color){60, 60, 60, 255});
  } break;
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
    tr->integrity = 100.0f;
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

    a->last_action = ACTION_COUNT;

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
      obs_push(obs,
               clamp01(d / 32.0f)); // normalize to something reasonable
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
  pthread_rwlock_rdlock(&c->lock);
  encode_observation(a, c, &obs);
  pthread_rwlock_unlock(&c->lock);

  MuCortex *cortex = a->cortex;
  int action = muze_plan(cortex, obs.data, obs.size, ACTION_COUNT);
  a->last_action = action;

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

static void spawn_projectile(Vector2 pos, Vector2 dir, float speed, float ttl,
                             int dmg) {
  for (int i = 0; i < MAX_PROJECTILES; i++) {
    if (!projectiles[i].alive) {
      projectiles[i].alive = true;
      projectiles[i].pos = pos;
      projectiles[i].vel = Vector2Scale(Vector2Normalize(dir), speed);
      projectiles[i].ttl = ttl;
      projectiles[i].damage = dmg;
      return;
    }
  }
}

static void update_projectiles(float dt) {
  for (int i = 0; i < MAX_PROJECTILES; i++) {
    Projectile *p = &projectiles[i];
    if (!p->alive)
      continue;

    p->ttl -= dt;
    if (p->ttl <= 0.0f) {
      p->alive = false;
      continue;
    }

    p->pos = Vector2Add(p->pos, Vector2Scale(p->vel, dt));

    // hit player (simple circle hit)
    float d = Vector2Distance(p->pos, player.position);
    if (d < 0.55f) {
      player.health -= (float)p->damage;
      player_hurt_timer = 0.18f;
      p->alive = false;
      cam_shake = fmaxf(cam_shake, 0.10f);
    }
  }
}

static void draw_projectiles(void) {
  for (int i = 0; i < MAX_PROJECTILES; i++) {
    Projectile *p = &projectiles[i];
    if (!p->alive)
      continue;

    Vector2 sp = world_to_screen(p->pos);
    float r = WORLD_SCALE * 0.07f * scale_size;
    DrawCircleV(sp, r, (Color){220, 220, 220, 255});
    DrawCircleLines((int)sp.x, (int)sp.y, r, (Color){0, 0, 0, 160});
  }
}

static void update_mob_ai(Mob *m, Vector2 chunk_origin, float dt) {
  // timers
  if (m->ai_timer > 0)
    m->ai_timer -= dt;
  if (m->aggro_timer > 0)
    m->aggro_timer -= dt;
  if (m->attack_cd > 0)
    m->attack_cd -= dt;
  if (m->hurt_timer > 0)
    m->hurt_timer -= dt;
  if (m->lunge_timer > 0)
    m->lunge_timer -= dt;

  // compute WORLD pos for decisions
  Vector2 mw = Vector2Add(chunk_origin, m->position);

  Vector2 basePos = nearest_base_pos(mw);
  Vector2 toB = Vector2Subtract(basePos, mw);
  float dB = Vector2Length(toB);
  Vector2 dirB = (dB > 1e-3f) ? Vector2Scale(toB, 1.0f / dB) : (Vector2){0, 0};

  // only fully “aware” if player is in same chunk (keeps it simple & cheap)
  int pcx = (int)(player.position.x / CHUNK_SIZE);
  int pcy = (int)(player.position.y / CHUNK_SIZE);
  int mcx = (int)(mw.x / CHUNK_SIZE);
  int mcy = (int)(mw.y / CHUNK_SIZE);
  bool player_same_chunk = (pcx == mcx && pcy == mcy);

  Vector2 toP = Vector2Subtract(player.position, mw);
  float dP = Vector2Length(toP);
  Vector2 dirP = (dP > 1e-3f) ? Vector2Scale(toP, 1.0f / dP) : (Vector2){0, 0};

  bool hostile = (m->type == MOB_ZOMBIE || m->type == MOB_SKELETON);

  // pick wander direction sometimes
  if (m->ai_timer <= 0.0f) {
    m->ai_timer = randf(0.35f, 1.25f);
    float ang = randf(0, 2 * PI);
    m->vel = (Vector2){cosf(ang), sinf(ang)};
  }

  float speed = MOB_SPEED_PASSIVE;

  // PASSIVE: flee if close or angry
  if (!hostile) {
    bool scared = player_same_chunk && (dP < 4.0f || m->aggro_timer > 0.0f);
    if (scared) {
      speed = 0.95f;
      m->vel = Vector2Scale(dirP, -1.0f); // run away
    }
  }

  // HOSTILE: chase / skeleton kite + shoot
  if (hostile) {
    speed = MOB_SPEED_HOSTILE;

    bool aggroP = player_same_chunk &&
                  ((dP < MOB_AGGRO_RANGE) || (m->aggro_timer > 0.0f));

    if (aggroP) {
      // your existing zombie/skeleton vs player logic (keep as-is)
      // ...
    } else if (is_night_cached) {
      // raid behavior: march toward nearest base even if player not around
      m->vel = dirB;

      // if inside base -> damage it
      for (int t = 0; t < TRIBE_COUNT; t++) {
        Tribe *tr = &tribes[t];
        float db = Vector2Distance(mw, tr->base.position);
        if (db < tr->base.radius + 0.8f) {
          if (m->attack_cd <= 0.0f) {
            m->attack_cd = 0.85f;
            m->lunge_timer = 0.12f;
            tr->integrity = fmaxf(0.0f, tr->integrity - 5.0f);
            cam_shake = fmaxf(cam_shake, 0.12f);
          }
        }
      }
    }
    if (player_same_chunk) {
      bool aggro = (dP < MOB_AGGRO_RANGE) || (m->aggro_timer > 0.0f);
      if (aggro) {
        speed = MOB_SPEED_HOSTILE;

        if (m->type == MOB_ZOMBIE) {
          // chase
          m->vel = dirP;

          // melee attack
          if (dP < MOB_ATTACK_RANGE && m->attack_cd <= 0.0f) {
            m->attack_cd = 0.8f;
            m->lunge_timer = 0.18f;
            player.health -= 6.0f;
            player_hurt_timer = 0.18f;
          }
        } else if (m->type == MOB_SKELETON) {
          // keep some distance
          float desired = 6.0f;
          if (dP < desired - 0.8f)
            m->vel = Vector2Scale(dirP, -1.0f); // back up
          else if (dP > desired + 1.2f)
            m->vel = dirP; // approach
          else
            m->vel = Vector2Scale(m->vel, 0.5f); // drift

          // shoot
          if (dP < 12.0f && m->attack_cd <= 0.0f) {
            m->attack_cd = 1.1f;
            m->lunge_timer = 0.12f;
            spawn_projectile(mw, dirP, 9.0f, 2.0f, 8);
          }
        }
      }
    }
  }
  // move (chunk-local)
  Vector2 delta = Vector2Scale(m->vel, speed * dt);
  m->position = Vector2Add(m->position, delta);
  m->position = clamp_local_to_chunk(m->position);
}

static void draw_crafting_ui(void) {
  if (!crafting_open)
    return;

  int x = 14, y = 260, w = 360, h = 28 + recipe_count * 22;
  DrawRectangle(x, y, w, h, (Color){0, 0, 0, 120});
  DrawRectangleLines(x, y, w, h, (Color){0, 0, 0, 220});
  DrawText("Crafting (TAB)", x + 10, y + 6, 18, RAYWHITE);

  for (int i = 0; i < recipe_count; i++) {
    Recipe *r = &recipes[i];
    Color c = can_afford(r) ? RAYWHITE : (Color){180, 180, 180, 255};

    DrawText(TextFormat("%d) %s  [W%d S%d G%d F%d]%s", i + 1, r->name, r->wood,
                        r->stone, r->gold, r->food,
                        (r->unlock_flag && *r->unlock_flag) ? " (OWNED)" : ""),
             x + 10, y + 30 + i * 20, 16, c);
  }
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
  float dt = GetFrameTime();

  // --- cooldown timers ---
  if (player_harvest_cd > 0.0f)
    player_harvest_cd -= dt;
  if (player_attack_cd > 0.0f)
    player_attack_cd -= dt;
  if (player_hurt_timer > 0.0f)
    player_hurt_timer -= dt;

  // --- movement (time-based feels better) ---
  float speed = 0.6f; // world units per frame-ish
  // If you want true time-based movement, use: float move = speed * (dt
  // * 60.0f);
  float move = speed;

  if (IsKeyDown(KEY_W))
    player.position.y -= move;
  if (IsKeyDown(KEY_S))
    player.position.y += move;
  if (IsKeyDown(KEY_A))
    player.position.x -= move;
  if (IsKeyDown(KEY_D))
    player.position.x += move;

  // --- crafting toggle ---
  if (IsKeyPressed(KEY_TAB)) {
    crafting_open = !crafting_open;
  }

  // --- crafting input (1..9) only when crafting menu is open ---
  if (crafting_open) {
    for (int i = 0; i < recipe_count && i < 9; i++) {
      // top-row number keys
      bool pressed = IsKeyPressed((KeyboardKey)(KEY_ONE + i));

      // keypad number keys (optional but nice)
      pressed = pressed || IsKeyPressed((KeyboardKey)(KEY_KP_1 + i));

      if (pressed) {
        craft(&recipes[i]);
      }
    }
  }

  // --- shoot arrow (F) if you have ammo ---
  if (IsKeyPressed(KEY_F) && inv_arrows > 0) {
    Vector2 mouse = GetMousePosition();
    Vector2 pp = world_to_screen(player.position);
    Vector2 aim = Vector2Subtract(mouse, pp);

    Vector2 dir = Vector2Normalize(aim);
    inv_arrows--;

    spawn_projectile(player.position, dir, 14.0f, 1.8f,
                     12 + (has_sword ? 4 : 0));
  }

  // --- zoom controls ---
  if (IsKeyDown(KEY_EQUAL))
    target_world_scale += 60.0f * dt;
  if (IsKeyDown(KEY_MINUS))
    target_world_scale -= 60.0f * dt;
  target_world_scale = clampf(target_world_scale, 0.0f, 100.0f);

  // --- current chunk ---
  int cx = (int)(player.position.x / CHUNK_SIZE);
  int cy = (int)(player.position.y / CHUNK_SIZE);
  Chunk *c = get_chunk(cx, cy);

  // --- Interactions ---
  // IMPORTANT: only regen stamina when NOT spending it this frame
  bool spent_stamina_this_frame = false;

  // LMB = attack mobs
  if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON) && player_harvest_cd <= 0.0f) {
    pthread_rwlock_wrlock(&c->lock);
    player_try_attack_mob_in_chunk(c, cx, cy);
    pthread_rwlock_unlock(&c->lock);
    // (attacks currently don't cost stamina in your code)
  }

  // RMB = harvest/mine resources (this spends stamina inside
  // player_try_harvest...)
  if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && player_attack_cd <= 0.0f) {
    float before = player.stamina;

    pthread_rwlock_wrlock(&c->lock);
    player_try_harvest_resource_in_chunk(c, cx, cy);
    pthread_rwlock_unlock(&c->lock);

    // if harvest actually happened, stamina decreased
    if (player.stamina < before - 0.0001f) {
      spent_stamina_this_frame = true;
    }
  }

  // --- stamina regen (time-based, only when not spending this frame) ---
  if (!spent_stamina_this_frame && player.stamina < 100.0f) {
    player.stamina = fminf(100.0f, player.stamina + STAMINA_REGEN_RATE * dt);
  }

  // clamp health
  if (player.health < 0.0f)
    player.health = 0.0f;
}

static void update_visible_world(float dt) {
  int pcx = (int)(player.position.x / CHUNK_SIZE);
  int pcy = (int)(player.position.y / CHUNK_SIZE);

  for (int dx = -6; dx <= 6; dx++) {
    for (int dy = -6; dy <= 6; dy++) {
      int cx = pcx + dx;
      int cy = pcy + dy;
      Chunk *c = get_chunk(cx, cy);

      pthread_rwlock_wrlock(&c->lock);

      Vector2 chunk_origin =
          (Vector2){(float)(cx * CHUNK_SIZE), (float)(cy * CHUNK_SIZE)};

      // resources animation decay
      for (int i = 0; i < c->resource_count; i++) {
        Resource *r = &c->resources[i];
        if (r->hit_timer > 0)
          r->hit_timer -= dt;
        if (r->break_flash > 0)
          r->break_flash -= dt;
      }

      // mobs
      despawn_hostiles_if_day(c);
      try_spawn_mobs_in_chunk(c, cx, cy, dt);

      // mobs AI
      for (int i = 0; i < MAX_MOBS; i++) {
        Mob *m = &c->mobs[i];
        if (m->health <= 0)
          continue;
        update_mob_ai(m, chunk_origin, dt);
      }
      pthread_rwlock_unlock(&c->lock);
    }
  }
}

static void draw_ui(void) {
  // panel
  DrawRectangle(14, 14, 280, 128, (Color){0, 0, 0, 110});
  DrawRectangleLines(14, 14, 280, 128, (Color){0, 0, 0, 200});

  // bars
  float hp01 = clamp01(player.health / 100.0f);
  float st01 = clamp01(player.stamina / 100.0f);

  DrawText("Player", 24, 20, 18, RAYWHITE);
  DrawText(TextFormat("HP: %d", (int)player.health), 24, 44, 16, RAYWHITE);
  DrawText(TextFormat("ST: %d", (int)player.stamina), 24, 64, 16, RAYWHITE);

  // bar visuals
  DrawRectangle(120, 46, 160, 12, (Color){0, 0, 0, 140});
  DrawRectangle(120, 46, (int)(160 * hp01), 12, (Color){80, 220, 80, 255});
  DrawRectangleLines(120, 46, 160, 12, (Color){0, 0, 0, 200});

  DrawRectangle(120, 66, 160, 12, (Color){0, 0, 0, 140});
  DrawRectangle(120, 66, (int)(160 * st01), 12, (Color){80, 160, 255, 255});
  DrawRectangleLines(120, 66, 160, 12, (Color){0, 0, 0, 200});

  // inventory
  DrawText(TextFormat("Wood: %d  Stone: %d", inv_wood, inv_stone), 24, 90, 16,
           RAYWHITE);
  DrawText(TextFormat("Gold: %d  Food: %d", inv_gold, inv_food), 24, 110, 16,
           RAYWHITE);

  // crosshair
  Vector2 m = GetMousePosition();
  DrawCircleLines((int)m.x, (int)m.y, 10, (Color){0, 0, 0, 200});
  DrawLine((int)m.x - 14, (int)m.y, (int)m.x + 14, (int)m.y,
           (Color){0, 0, 0, 200});
  DrawLine((int)m.x, (int)m.y - 14, (int)m.x, (int)m.y + 14,
           (Color){0, 0, 0, 200});

  // cooldown rings around hands
  float hFrac = 1.0f - clamp01(player_harvest_cd / PLAYER_HARVEST_COOLDOWN);
  float aFrac = 1.0f - clamp01(player_attack_cd / PLAYER_ATTACK_COOLDOWN);

  float rr = 12.0f;
  DrawRing(g_handL, rr - 3, rr, -90, -90 + 360.0f * aFrac, 24,
           (Color){255, 140, 80, 220});
  DrawRing(g_handR, rr - 3, rr, -90, -90 + 360.0f * hFrac, 24,
           (Color){80, 160, 255, 220});

  DrawText(TextFormat("Shards: %d  Arrows: %d", inv_shards, inv_arrows), 24,
           130, 16, RAYWHITE);

  // base integrity
  int y0 = 150;
  for (int t = 0; t < TRIBE_COUNT; t++) {
    float v = clamp01(tribes[t].integrity / 100.0f);
    DrawText(TextFormat("Base %d", t), 24, y0 + t * 22, 16, tribes[t].color);
    DrawRectangle(90, y0 + 4 + t * 22, 140, 10, (Color){0, 0, 0, 140});
    DrawRectangle(90, y0 + 4 + t * 22, (int)(140 * v), 10, tribes[t].color);
    DrawRectangleLines(90, y0 + 4 + t * 22, 140, 10, (Color){0, 0, 0, 200});
  }
}

static void draw_hover_label(void) {
  int hp = -1;

  // find nearest in current chunk within a small radius
  int cx = (int)(player.position.x / CHUNK_SIZE);
  int cy = (int)(player.position.y / CHUNK_SIZE);
  Chunk *c = get_chunk(cx, cy);

  const char *label = NULL;
  float bestD = 1e9f;

  // resources
  for (int i = 0; i < c->resource_count; i++) {
    Resource *r = &c->resources[i];
    if (r->health <= 0)
      continue;
    Vector2 rw = (Vector2){cx * CHUNK_SIZE + r->position.x,
                           cy * CHUNK_SIZE + r->position.y};
    float d = Vector2Distance(player.position, rw);
    if (d < 2.2f && d < bestD) {
      bestD = d;
      label = res_name(r->type);
      hp = r->health;
    }
  }

  // mobs (prefer mobs if close)
  for (int i = 0; i < MAX_MOBS; i++) {
    Mob *m = &c->mobs[i];
    if (m->health <= 0)
      continue;
    Vector2 mw = (Vector2){cx * CHUNK_SIZE + m->position.x,
                           cy * CHUNK_SIZE + m->position.y};
    float d = Vector2Distance(player.position, mw);
    if (d < 2.6f && d < bestD) {
      bestD = d;
      label = mob_name(m->type);
      hp = m->health;
    }
  }

  if (label) {
    Vector2 mp = GetMousePosition();
    DrawRectangle((int)mp.x + 14, (int)mp.y + 10, 160, 22,
                  (Color){0, 0, 0, 140});
    DrawRectangleLines((int)mp.x + 14, (int)mp.y + 10, 160, 22,
                       (Color){0, 0, 0, 220});
    if (hp >= 0) {
      DrawText(TextFormat("%s (%d)", label, hp), (int)mp.x + 22, (int)mp.y + 13,
               16, RAYWHITE);
    } else {
      DrawText(label, (int)mp.x + 22, (int)mp.y + 13, 16, RAYWHITE);
    }
  }
}

static void draw_minimap(void) {
  int x = 310, y = 14; // top row, to the right of your panel
  int size = 160;

  DrawRectangle(x, y, size, size, (Color){0, 0, 0, 110});
  DrawRectangleLines(x, y, size, size, (Color){0, 0, 0, 220});

  // sample area around player (world units)
  float radius = 28.0f;
  int cells = 40; // 40x40 grid
  float cell = (float)size / (float)cells;

  for (int gy = 0; gy < cells; gy++) {
    for (int gx = 0; gx < cells; gx++) {
      float nx = ((float)gx / (float)(cells - 1)) * 2.0f - 1.0f;
      float ny = ((float)gy / (float)(cells - 1)) * 2.0f - 1.0f;

      Vector2 wp = (Vector2){player.position.x + nx * radius,
                             player.position.y + ny * radius};

      int cx = (int)(wp.x / CHUNK_SIZE);
      int cy = (int)(wp.y / CHUNK_SIZE);
      Chunk *c = get_chunk(cx, cy);

      Color bc = Fade(biome_colors[c->biome_type], 0.85f);
      DrawRectangle((int)(x + gx * cell), (int)(y + gy * cell),
                    (int)ceilf(cell), (int)ceilf(cell), bc);
    }
  }

  // bases
  for (int t = 0; t < TRIBE_COUNT; t++) {
    Vector2 d = Vector2Subtract(tribes[t].base.position, player.position);
    if (fabsf(d.x) > radius || fabsf(d.y) > radius)
      continue;
    float pxm = (d.x / (radius * 2.0f) + 0.5f) * size;
    float pym = (d.y / (radius * 2.0f) + 0.5f) * size;
    DrawCircle((int)(x + pxm), (int)(y + pym), 3, tribes[t].color);
  }

  // mobs (nearby)
  int pcx = (int)(player.position.x / CHUNK_SIZE);
  int pcy = (int)(player.position.y / CHUNK_SIZE);
  for (int dx = -2; dx <= 2; dx++) {
    for (int dy = -2; dy <= 2; dy++) {
      int cx = pcx + dx;
      int cy = pcy + dy;
      Chunk *c = get_chunk(cx, cy);
      Vector2 origin =
          (Vector2){(float)(cx * CHUNK_SIZE), (float)(cy * CHUNK_SIZE)};
      for (int i = 0; i < MAX_MOBS; i++) {
        Mob *m = &c->mobs[i];
        if (m->health <= 0)
          continue;
        Vector2 mw = Vector2Add(origin, m->position);
        Vector2 d = Vector2Subtract(mw, player.position);
        if (fabsf(d.x) > radius || fabsf(d.y) > radius)
          continue;

        float pxm = (d.x / (radius * 2.0f) + 0.5f) * size;
        float pym = (d.y / (radius * 2.0f) + 0.5f) * size;
        DrawPixel((int)(x + pxm), (int)(y + pym), (Color){240, 80, 80, 255});
      }
    }
  }

  // player dot
  DrawCircle(x + size / 2, y + size / 2, 3, RAYWHITE);
  DrawCircleLines(x + size / 2, y + size / 2, 3, (Color){0, 0, 0, 200});
}

static void draw_hurt_vignette(void) {
  if (player_hurt_timer <= 0.0f)
    return;
  float t = clamp01(player_hurt_timer / 0.18f);
  unsigned char a = (unsigned char)(120 * t);
  DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, (Color){120, 0, 0, a});
}

/* =======================
   THREAD

 * ======================= */

static void run_agent_jobs(void) {
  pthread_mutex_lock(&job_mtx);
  job_next_agent = 0;
  job_done_workers = 0;
  job_active = 1;
  pthread_cond_broadcast(&job_cv);

  while (job_active) {
    pthread_cond_wait(&done_cv, &job_mtx);
  }
  pthread_mutex_unlock(&job_mtx);
}

static void *agent_worker(void *arg) {
  (void)arg;

  for (;;) {
    // wait for a job
    pthread_mutex_lock(&job_mtx);
    while (!job_active && !job_quit) {
      pthread_cond_wait(&job_cv, &job_mtx);
    }
    if (job_quit) {
      pthread_mutex_unlock(&job_mtx);
      break;
    }
    pthread_mutex_unlock(&job_mtx);

    // do work: pull agents until none left
    for (;;) {
      pthread_mutex_lock(&job_mtx);
      int idx = job_next_agent++;
      pthread_mutex_unlock(&job_mtx);

      if (idx >= MAX_AGENTS)
        break;
      update_agent(&agents[idx]);
    }

    // notify done
    pthread_mutex_lock(&job_mtx);
    job_done_workers++;
    if (job_done_workers == WORKER_COUNT) {
      job_active = 0;
      pthread_cond_signal(&done_cv);
    }
    pthread_mutex_unlock(&job_mtx);
  }

  return NULL;
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

  for (int x = 0; x < WORLD_SIZE; x++) {
    for (int y = 0; y < WORLD_SIZE; y++) {
      pthread_rwlock_init(&world[x][y].lock, NULL);
      world[x][y].generated = false;
      world[x][y].resource_count = 0;
      world[x][y].mob_spawn_timer = 0.0f;
    }
  }

  for (int i = 0; i < WORKER_COUNT; i++) {
    pthread_create(&workers[i], NULL, agent_worker, NULL);
  }

  for (int i = 0; i < MAX_PROJECTILES; i++)
    projectiles[i].alive = false;

  while (!WindowShouldClose()) {
    float dt = GetFrameTime();

    camera_pos.x += (player.position.x - camera_pos.x) * 0.1f;
    camera_pos.y += (player.position.y - camera_pos.y) * 0.1f;

    if (cam_shake > 0.0f) {
      cam_shake -= dt;
      float mag = cam_shake * 0.65f;
      camera_pos.x += randf(-mag, mag);
      camera_pos.y += randf(-mag, mag);
    }
    WORLD_SCALE = lerp(WORLD_SCALE, target_world_scale, 0.12f);

    update_player();
    update_visible_world(dt);
    update_projectiles(dt);
    update_daynight(dt);

    run_agent_jobs(); // update agents

    // detect transition night->day for reward
    int now_night = is_night_cached;
    if (was_night && !now_night) {
      // dawn reward: shards + small base repair
      inv_shards += 5;
      for (int t = 0; t < TRIBE_COUNT; t++) {
        tribes[t].integrity = fminf(100.0f, tribes[t].integrity + 15.0f);
      }
    }
    was_night = now_night;

    // raid spawner
    if (is_night_cached) {
      raid_timer -= dt;
      if (raid_timer <= 0.0f) {
        raid_timer = raid_interval;
        spawn_raid_wave();
      }
    } else {
      raid_timer = 1.5f;
    }

    for (int i = 0; i < MAX_AGENTS; i++) {
      if (!agents[i].alive)
        continue;
      int acx = (int)(agents[i].position.x / CHUNK_SIZE);
      int acy = (int)(agents[i].position.y / CHUNK_SIZE);
      (void)get_chunk(acx, acy);
    }

    update_pickups(dt);
    collect_nearby_pickups();

    BeginDrawing();
    ClearBackground((Color){20, 20, 20, 255});

    draw_chunks();
    draw_resources();
    draw_mobs();
    draw_pickups();
    draw_projectiles();

    // bases
    for (int t = 0; t < TRIBE_COUNT; t++) {
      Vector2 bp = world_to_screen(tribes[t].base.position);
      DrawCircleLinesV(bp, tribes[t].base.radius * WORLD_SCALE,
                       tribes[t].color);
    }

    // agents
    for (int i = 0; i < MAX_AGENTS; i++) {
      if (!agents[i].alive)
        continue;
      Vector2 ap = world_to_screen(agents[i].position);
      Color tc = tribes[agents[i].agent_id / AGENT_PER_TRIBE].color;
      draw_agent_detailed(&agents[i], ap, tc);
    }

    // player
    Vector2 pp = world_to_screen(player.position);
    draw_player(pp);

    // UI + debug
    draw_ui();
    draw_minimap();
    draw_crafting_ui();
    draw_hover_label();
    draw_daynight_overlay(); // AFTER world draw, before EndDrawing
    draw_hurt_vignette();

    DrawText("MUZE Tribal Simulation", 20, 160, 20, RAYWHITE);
    DrawText(TextFormat("FPS: %d", GetFPS()), 20, 185, 20, RAYWHITE);

    EndDrawing();
  }

  pthread_mutex_lock(&job_mtx);
  job_quit = 1;
  pthread_cond_broadcast(&job_cv);
  pthread_mutex_unlock(&job_mtx);

  for (int i = 0; i < WORKER_COUNT; i++) {
    pthread_join(workers[i], NULL);
  }

  CloseWindow();
  return 0;
}
