#include "../SAM/SAM.h"
#include "../utils/NN/MUZE/all.h"
#include "../utils/Raylib/src/raylib.h"
#include "../utils/Raylib/src/raymath.h"
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// ------------------- Compile helpers / forward decls -------------------
#ifndef SAVE_ROOT
#define SAVE_ROOT "saves"
#endif

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

#define PLAYER_HARVEST_DAMAGE 3
#define PLAYER_ATTACK_DAMAGE 3
#define PLAYER_TAKEN_DAMAGE 1
#define AGENT_TAKEN_DAMAGE 1

#define PLAYER_HARVEST_COOLDOWN 0.18f
#define PLAYER_ATTACK_COOLDOWN 0.22f

#define PLAYER_MINE_DAMAGE 3
#define PLAYER_MINE_COOLDOWN 0.26f
#define PLAYER_MINE_STAMINA_COST 3.0f

#define STAMINA_REGEN_RATE 5.0f
#define STAMINA_DRAIN_RATE 0.5f

#define MAX_PICKUPS 256

#define MOB_AGGRO_RANGE 10.0f
#define MOB_ATTACK_RANGE 1.25f
#define MOB_SPEED_PASSIVE 2.55f
#define MOB_SPEED_SCARED (MOB_SPEED_PASSIVE * 2.0f)
#define MOB_SPEED_HOSTILE 3.85f

#define MAX_PROJECTILES 64
// ---- Player bow charge ----
#define BOW_CHARGE_TIME 0.75f // seconds to full charge
#define BOW_CHARGE_MIN01                                                       \
  0.12f // minimum charge required to fire (prevents tap-fizzle)
#define PLAYER_FIRE_COOLDOWN 0.18f

// agent continuous fire: one FIRE action keeps firing for a bit
#define AGENT_FIRE_LATCH_TIME 0.85f // seconds of "auto-fire" after ACTION_FIRE
#define AGENT_FIRE_CANCEL_ON_MOVE 1 // set 0 if you want strafing auto-fire

#define BOW_SPEED_MIN 10.0f
#define BOW_SPEED_MAX 22.0f
#define BOW_TTL_MIN 1.10f
#define BOW_TTL_MAX 2.20f
#define BOW_DMG_MIN 8
#define BOW_DMG_MAX 20

// =======================
// REWARD SHAPING
// =======================
#define R_SURVIVE_PER_TICK (0.0005f)

#define R_ATTACK_HIT (0.015f)
#define R_ATTACK_KILL (0.180f)
#define R_ATTACK_WASTE (-0.020f) // attack when nothing hit

#define R_HARVEST_HIT (0.008f)
#define R_HARVEST_BREAK (0.060f)
#define R_HARVEST_WASTE (-0.015f)      // harvest when nothing hit
#define R_HARVEST_NO_STAMINA (-0.006f) // tried to harvest but couldn't afford

#define R_FIRE_HIT (0.020f)
#define R_FIRE_WASTE (-0.030f) // fired and had no mob in ray
#define R_FIRE_NO_AMMO (-0.010f)

#define R_EAT_GOOD (0.120f)
#define R_EAT_WASTE (-0.040f) // ate when already basically full
#define R_EAT_NO_FOOD (-0.010f)

#define R_WANDER_PENALTY (-0.0008f); // punish random wandering

#define R_DEATH (-1.000f)

#define OBS_DIM 128
#define TRAIN_INTERVAL 1

#define MAX_WORLDS 4096
#define WORLD_NAME_MAX 4096

/* =======================
   GLOBAL STATE
======================= */
#define WORKER_COUNT 8

/* =======================
   ENUMS
======================= */

typedef enum {
  STATE_TITLE = 0,
  STATE_WORLD_SELECT,
  STATE_WORLD_CREATE,
  STATE_PLAYING,
  STATE_PAUSED,
  STATE_COUNT
} GameStateType;

typedef enum {
  TOOL_HAND = 0,
  TOOL_AXE = 1,
  TOOL_PICKAXE = 2,
  TOOL_SWORD = 3,
  TOOL_ARMOR = 4,
  TOOL_BOW = 5,
  TOOL_COUNT = 6,
  TOOL_NONE = -1
} ToolType;

typedef enum {
  RES_TREE = 0,
  RES_ROCK,
  RES_GOLD,
  RES_FOOD,
  RES_NONE
} ResourceType;

typedef enum {
  MOB_PIG = 0,
  MOB_SHEEP,
  MOB_SKELETON,
  MOB_ZOMBIE,
  MOB_COUNT
} MobType;

typedef enum {
  INTENT_NONE = 0,
  INTENT_ATTACK,
  INTENT_HARVEST,
  INTENT_COUNT
} IntentType;

typedef enum {
  HIT_NONE = 0,
  HIT_RESOURCE = 1,
  HIT_MOB = 2,
  HIT_BASE = 3
} HitKind;

typedef struct {
  HitKind kind;
  float t;         // distance along ray (world units)
  int cx, cy;      // chunk coords of hit
  int index;       // resource index OR mob index OR tribe index
  Vector2 hit_pos; // world-space hit point
} RayHit;

typedef enum {
  ACTION_UP = 0,
  ACTION_DOWN,
  ACTION_LEFT,
  ACTION_RIGHT,
  ACTION_ATTACK,
  ACTION_FIRE,
  ACTION_HARVEST,
  ACTION_EAT,
  // ACTION_CRAFT,
  ACTION_CRAFT_AXE,
  ACTION_CRAFT_PICKAXE,
  ACTION_CRAFT_SWORD,
  ACTION_CRAFT_ARMOR,
  ACTION_CRAFT_BOW,
  ACTION_CRAFT_ARROWS,
  ACTION_NONE,
  ACTION_COUNT
} ActionType;

typedef enum {
  PROJ_OWNER_PLAYER = 0,
  PROJ_OWNER_AGENT = 1,
  PROJ_OWNER_MOB = 2
} ProjOwner;

typedef enum { PICK_FOOD = 0, PICK_SHARD, PICK_ARROW } PickupType;

// ----- forward prototypes -----
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
  int owner_agent_id;
  ProjOwner owner;
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
  float integrity; // 0..100

  int wood, stone, gold, food;
  int shards, arrows;
} Tribe;

typedef struct Agent {
  Vector2 position;
  Vector2 facing;
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

  float attack_cd;
  float harvest_cd;
  float fire_cd;

  int inv_food;
  int inv_arrows;
  int inv_shards;

  bool has_axe;
  bool has_pickaxe;
  bool has_sword;
  bool has_armor;
  bool has_bow;

  int tool_selected;
  int last_craft_selected;

  // --- continuous actions (latches) ---
  int fire_latched;       // 0/1
  float fire_latch_timer; // seconds remaining to keep trying

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

typedef struct {
  char names[MAX_WORLDS][WORLD_NAME_MAX];
  int count;
  int selected; // -1 if none
  int scroll;   // index offset for drawing
} WorldList;

#pragma pack(push, 1)

typedef struct {
  // identity
  int32_t agent_id;
  int32_t alive;

  // pose
  float x, y;
  float fx, fy;

  // vitals
  float health, stamina;

  // bookkeeping
  int32_t age;
  int32_t last_action;
  float reward_accumulator;

  // cooldowns + latches
  float attack_cd, harvest_cd, fire_cd;
  int32_t fire_latched;
  float fire_latch_timer;

  // inventory
  int32_t inv_food, inv_arrows, inv_shards;

  // tools
  int32_t has_axe, has_pickaxe, has_sword, has_armor, has_bow;
  int32_t tool_selected;
  int32_t last_craft_selected;
} SaveAgent;

typedef struct {
  int32_t alive;
  float x, y;
  int32_t type;
  int32_t amount;
  float ttl;
  float bob_t;
} SavePickup;

typedef struct {
  int32_t alive;
  float x, y;
  float vx, vy;
  float ttl;
  int32_t damage;
  int32_t owner;
} SaveProjectile;

typedef struct {
  char magic[4];    // "SAMW"
  uint32_t version; // 1
  uint32_t seed;
  uint32_t world_size;
  uint32_t chunk_size;
  float time_of_day;

  // player
  float player_x, player_y;
  float player_health, player_stamina;

  // player inv/tools
  int32_t inv_wood, inv_stone, inv_gold, inv_food;
  int32_t inv_shards, inv_arrows;
  int32_t has_axe, has_pickaxe, has_sword, has_armor, has_bow;

  // tribes
  float tribe_integrity[TRIBE_COUNT];
  int32_t tribe_wood[TRIBE_COUNT], tribe_stone[TRIBE_COUNT],
      tribe_gold[TRIBE_COUNT], tribe_food[TRIBE_COUNT];
  int32_t tribe_shards[TRIBE_COUNT], tribe_arrows[TRIBE_COUNT];

  // chunk count follows
  uint32_t chunk_count;
  // ---- extra dynamic state ----
  uint32_t agent_count;
  uint32_t pickup_count;     // number of alive pickups written
  uint32_t projectile_count; // number of alive projectiles written

} SaveHeader;

typedef struct {
  int32_t cx, cy;
  int32_t biome_type;
  float mob_spawn_timer;

  uint32_t resource_count;
  uint32_t mob_count; // alive mobs written
} SaveChunkHeader;

typedef struct {
  float lx, ly; // local pos
  int32_t type;
  int32_t health;
  float hit_timer;
  float break_flash;
} SaveResource;

typedef struct {
  float lx, ly; // local pos
  int32_t type;
  int32_t health;
  float velx, vely;
  float ai_timer, aggro_timer, attack_cd, hurt_timer, lunge_timer;
} SaveMob;
#pragma pack(pop)

static uint32_t g_world_seed = 0;

// Forward declarations for functions used before their definitions (C99+)
struct Chunk; // forward decl
static inline float randf(float a, float b);
static inline int is_night(void);
static void init_tribes(void);
static void init_agents(void);
static void ensure_save_root(void);
static inline Vector2 clamp_local_to_chunk(Vector2 p);
static int find_free_mob_slot(struct Chunk *c);
static void spawn_projectile(Vector2 pos, Vector2 dir, float speed, float ttl,
                             int dmg, ProjOwner owner);
Chunk *get_chunk(int cx, int cy);
static inline float mob_radius_world(MobType t);
static void on_mob_killed(MobType t, Vector2 mob_world_pos);
static int world_pos_blocked_nearby(int cx, int cy, Vector2 worldPos,
                                    float radius, int self_cx, int self_cy);
static inline float res_radius_world(ResourceType t);

// ----------------------------------------------------------------------

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

static WorldList g_world_list = {0};

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
static float player_fire_cd = 0.0f;

static int bow_charging = 0;
static float bow_charge01 = 0.0f; // 0..1

bool crafting_open = false;

bool has_axe = false;
bool has_pickaxe = false;
bool has_sword = false;
bool has_armor = false;
bool has_bow = false;

static Pickup pickups[MAX_PICKUPS];

static int inv_shards = 0;
static int inv_arrows = 0;

// smooth zoom
static float target_world_scale = 50.0f;

// camera shake
static float cam_shake = 0.0f;

// store last hand positions (screen space) for cooldown rings
static Vector2 g_handL = {0}, g_handR = {0};
static float g_dt = 1.0f / 60.0f;

// raid
static float raid_timer = 0.0f;
static float raid_interval = 4.5f; // seconds between mini-waves at night
static int was_night = 0;

static Projectile projectiles[MAX_PROJECTILES];

static Recipe recipes[] = {
    {"Axe (Wood+Stone)", 3, 2, 0, 0, &has_axe},
    {"Pickaxe (Wood+Stone)", 3, 3, 0, 0, &has_pickaxe},
    {"Sword (Stone+Gold)", 0, 4, 2, 0, &has_sword},
    {"Armor (Stone+Gold)", 0, 5, 2, 0, &has_armor},
    {"Bow (Wood+Gold)", 4, 0, 1, 0, &has_bow},
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

static void make_world_path(char *out, size_t cap, const char *world_name) {
  snprintf(out, cap, "%s/%s", SAVE_ROOT, world_name);
}

static void make_save_file_path(char *out, size_t cap, const char *world_name) {
  snprintf(out, cap, "%s/%s/world.sav", SAVE_ROOT, world_name);
}

static void make_models_dir(char *out, size_t cap, const char *world_name) {
  snprintf(out, cap, "%s/%s/models", SAVE_ROOT, world_name);
}

static void make_agent_model_path(char *out, size_t cap, const char *world_name,
                                  int agent_id) {
  snprintf(out, cap, "%s/%s/models/agent_%03d.bin", SAVE_ROOT, world_name,
           agent_id);
}

static void world_reset(uint32_t seed) {
  srand(seed);

  // reset chunks but KEEP locks valid
  for (int x = 0; x < WORLD_SIZE; x++) {
    for (int y = 0; y < WORLD_SIZE; y++) {
      Chunk *c = &world[x][y];
      // NOTE: lock should already be init'd once at program start
      pthread_rwlock_wrlock(&c->lock);
      c->generated = false;
      c->resource_count = 0;
      c->biome_type = 0;
      c->mob_spawn_timer = randf(1.0f, 3.0f);
      for (int i = 0; i < MAX_MOBS; i++)
        c->mobs[i].health = 0;
      pthread_rwlock_unlock(&c->lock);
    }
  }

  // reset player (spawn near center)
  player.position = (Vector2){(WORLD_SIZE * CHUNK_SIZE) * 0.5f,
                              (WORLD_SIZE * CHUNK_SIZE) * 0.5f};
  player.health = 100.0f;
  player.stamina = 100.0f;

  // reset globals
  time_of_day = 0.25f;
  is_night_cached = is_night();

  inv_wood = inv_stone = inv_gold = inv_food = 0;
  inv_shards = inv_arrows = 0;

  has_axe = has_pickaxe = has_sword = has_armor = has_bow = false;

  init_tribes();
  init_agents();
}

static int save_world_to_disk(const char *world_name) {
  ensure_save_root();

  char world_dir[256];
  make_world_path(world_dir, sizeof(world_dir), world_name);
  if (mkdir(world_dir, 0755) != 0 && errno != EEXIST) {
    fprintf(stderr, "mkdir(%s) failed: %s\n", world_dir, strerror(errno));
    return 0;
  }

  char final_path[256];
  make_save_file_path(final_path, sizeof(final_path), world_name);

  char tmp_path[256];
  snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", final_path);

  FILE *f = fopen(tmp_path, "wb");
  if (!f) {
    fprintf(stderr, "fopen(%s) failed: %s\n", tmp_path, strerror(errno));
    return 0;
  }

  SaveHeader h = {0};
  h.magic[0] = 'S';
  h.magic[1] = 'A';
  h.magic[2] = 'M';
  h.magic[3] = 'W';
  h.version = 3;
  h.seed = g_world_seed;
  h.world_size = WORLD_SIZE;
  h.chunk_size = CHUNK_SIZE;
  h.time_of_day = time_of_day;

  // player
  h.player_x = player.position.x;
  h.player_y = player.position.y;
  h.player_health = player.health;
  h.player_stamina = player.stamina;

  // player inv/tools
  h.inv_wood = inv_wood;
  h.inv_stone = inv_stone;
  h.inv_gold = inv_gold;
  h.inv_food = inv_food;
  h.inv_shards = inv_shards;
  h.inv_arrows = inv_arrows;
  h.has_axe = has_axe;
  h.has_pickaxe = has_pickaxe;
  h.has_sword = has_sword;
  h.has_armor = has_armor;
  h.has_bow = has_bow;

  // tribes
  for (int t = 0; t < TRIBE_COUNT; t++) {
    h.tribe_integrity[t] = tribes[t].integrity;
    h.tribe_wood[t] = tribes[t].wood;
    h.tribe_stone[t] = tribes[t].stone;
    h.tribe_gold[t] = tribes[t].gold;
    h.tribe_food[t] = tribes[t].food;
    h.tribe_shards[t] = tribes[t].shards;
    h.tribe_arrows[t] = tribes[t].arrows;
  }

  // chunk_count
  uint32_t chunk_count = 0;
  for (int cx = 0; cx < WORLD_SIZE; cx++)
    for (int cy = 0; cy < WORLD_SIZE; cy++)
      if (world[cx][cy].generated)
        chunk_count++;
  h.chunk_count = chunk_count;

  // agents (always MAX_AGENTS)
  h.agent_count = MAX_AGENTS;

  // pickups (alive only)
  uint32_t pcount = 0;
  for (int i = 0; i < MAX_PICKUPS; i++)
    if (pickups[i].alive)
      pcount++;
  h.pickup_count = pcount;

  // projectiles (alive only)
  uint32_t prcount = 0;
  for (int i = 0; i < MAX_PROJECTILES; i++)
    if (projectiles[i].alive)
      prcount++;
  h.projectile_count = prcount;

  // ---- write header ONCE ----
  if (fwrite(&h, sizeof(h), 1, f) != 1) {
    fclose(f);
    return 0;
  }

  // ---- write agents ONCE ----
  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &agents[i];
    SaveAgent sa = {0};

    sa.agent_id = a->agent_id;
    sa.alive = a->alive ? 1 : 0;

    sa.x = a->position.x;
    sa.y = a->position.y;
    sa.fx = a->facing.x;
    sa.fy = a->facing.y;

    sa.health = a->health;
    sa.stamina = a->stamina;

    sa.age = a->age;
    sa.last_action = a->last_action;
    sa.reward_accumulator = a->reward_accumulator;

    sa.attack_cd = a->attack_cd;
    sa.harvest_cd = a->harvest_cd;
    sa.fire_cd = a->fire_cd;

    sa.fire_latched = a->fire_latched;
    sa.fire_latch_timer = a->fire_latch_timer;

    sa.inv_food = a->inv_food;
    sa.inv_arrows = a->inv_arrows;
    sa.inv_shards = a->inv_shards;

    sa.has_axe = a->has_axe;
    sa.has_pickaxe = a->has_pickaxe;
    sa.has_sword = a->has_sword;
    sa.has_armor = a->has_armor;
    sa.has_bow = a->has_bow;

    sa.tool_selected = a->tool_selected;
    sa.last_craft_selected = a->last_craft_selected;

    if (fwrite(&sa, sizeof(sa), 1, f) != 1) {
      fclose(f);
      return 0;
    }
  }

  // ---- write pickups ONCE ----
  for (int i = 0; i < MAX_PICKUPS; i++) {
    if (!pickups[i].alive)
      continue;
    SavePickup sp = {0};
    sp.alive = 1;
    sp.x = pickups[i].pos.x;
    sp.y = pickups[i].pos.y;
    sp.type = (int32_t)pickups[i].type;
    sp.amount = pickups[i].amount;
    sp.ttl = pickups[i].ttl;
    sp.bob_t = pickups[i].bob_t;
    if (fwrite(&sp, sizeof(sp), 1, f) != 1) {
      fclose(f);
      return 0;
    }
  }

  // ---- write projectiles ONCE ----
  for (int i = 0; i < MAX_PROJECTILES; i++) {
    if (!projectiles[i].alive)
      continue;
    SaveProjectile sp = {0};
    sp.alive = 1;
    sp.x = projectiles[i].pos.x;
    sp.y = projectiles[i].pos.y;
    sp.vx = projectiles[i].vel.x;
    sp.vy = projectiles[i].vel.y;
    sp.ttl = projectiles[i].ttl;
    sp.damage = projectiles[i].damage;
    sp.owner = (int32_t)projectiles[i].owner;
    if (fwrite(&sp, sizeof(sp), 1, f) != 1) {
      fclose(f);
      return 0;
    }
  }

  // ---- write chunks ----
  for (int cx = 0; cx < WORLD_SIZE; cx++) {
    for (int cy = 0; cy < WORLD_SIZE; cy++) {
      Chunk *c = &world[cx][cy];
      if (!c->generated)
        continue;

      pthread_rwlock_rdlock(&c->lock);

      SaveChunkHeader ch = {0};
      ch.cx = cx;
      ch.cy = cy;
      ch.biome_type = c->biome_type;
      ch.mob_spawn_timer = c->mob_spawn_timer;
      ch.resource_count = (uint32_t)c->resource_count;

      uint32_t alive_mobs = 0;
      for (int i = 0; i < MAX_MOBS; i++)
        if (c->mobs[i].health > 0)
          alive_mobs++;
      ch.mob_count = alive_mobs;

      if (fwrite(&ch, sizeof(ch), 1, f) != 1) {
        pthread_rwlock_unlock(&c->lock);
        fclose(f);
        return 0;
      }

      // resources
      for (int i = 0; i < c->resource_count; i++) {
        const Resource *r = &c->resources[i];
        SaveResource sr = {0};
        sr.lx = r->position.x;
        sr.ly = r->position.y;
        sr.type = (int32_t)r->type;
        sr.health = (int32_t)r->health;
        sr.hit_timer = r->hit_timer;
        sr.break_flash = r->break_flash;
        if (fwrite(&sr, sizeof(sr), 1, f) != 1) {
          pthread_rwlock_unlock(&c->lock);
          fclose(f);
          return 0;
        }
      }

      // mobs (alive only)
      for (int i = 0; i < MAX_MOBS; i++) {
        const Mob *m = &c->mobs[i];
        if (m->health <= 0)
          continue;

        SaveMob sm = {0};
        sm.lx = m->position.x;
        sm.ly = m->position.y;
        sm.type = (int32_t)m->type;
        sm.health = (int32_t)m->health;
        sm.velx = m->vel.x;
        sm.vely = m->vel.y;
        sm.ai_timer = m->ai_timer;
        sm.aggro_timer = m->aggro_timer;
        sm.attack_cd = m->attack_cd;
        sm.hurt_timer = m->hurt_timer;
        sm.lunge_timer = m->lunge_timer;

        if (fwrite(&sm, sizeof(sm), 1, f) != 1) {
          pthread_rwlock_unlock(&c->lock);
          fclose(f);
          return 0;
        }
      }

      pthread_rwlock_unlock(&c->lock);
    }
  }

  fflush(f);
  fclose(f);

  // atomic replace
  if (rename(tmp_path, final_path) != 0) {
    fprintf(stderr, "rename(%s -> %s) failed: %s\n", tmp_path, final_path,
            strerror(errno));
    return 0;
  }

  return 1;
}

static int load_world_from_disk(const char *world_name) {
  char path[256];
  make_save_file_path(path, sizeof(path), world_name);

  FILE *f = fopen(path, "rb");
  if (!f)
    return 0;

  SaveHeader h = {0};
  if (fread(&h, sizeof(h), 1, f) != 1) {
    fclose(f);
    return 0;
  }

  if (h.magic[0] != 'S' || h.magic[1] != 'A' || h.magic[2] != 'M' ||
      h.magic[3] != 'W') {
    fclose(f);
    return 0;
  }
  if (h.version != 3) {
    // you *can* keep your old v1/v2 loader if you want backwards compat,
    // but DO NOT mix layouts. For now: reject.
    fclose(f);
    fprintf(stderr, "Unsupported save version: %u\n", h.version);
    return 0;
  }

  g_world_seed = h.seed;
  world_reset(g_world_seed);

  time_of_day = h.time_of_day;
  is_night_cached = is_night();

  // player
  player.position = (Vector2){h.player_x, h.player_y};
  player.health = h.player_health;
  player.stamina = h.player_stamina;

  // inv/tools
  inv_wood = h.inv_wood;
  inv_stone = h.inv_stone;
  inv_gold = h.inv_gold;
  inv_food = h.inv_food;
  inv_shards = h.inv_shards;
  inv_arrows = h.inv_arrows;

  has_axe = h.has_axe;
  has_pickaxe = h.has_pickaxe;
  has_sword = h.has_sword;
  has_armor = h.has_armor;
  has_bow = h.has_bow;

  // tribes
  for (int t = 0; t < TRIBE_COUNT; t++) {
    tribes[t].integrity = h.tribe_integrity[t];
    tribes[t].wood = h.tribe_wood[t];
    tribes[t].stone = h.tribe_stone[t];
    tribes[t].gold = h.tribe_gold[t];
    tribes[t].food = h.tribe_food[t];
    tribes[t].shards = h.tribe_shards[t];
    tribes[t].arrows = h.tribe_arrows[t];
  }

  // ---- agents ONCE ----
  uint32_t acount = h.agent_count;
  if (acount != MAX_AGENTS) {
    // if you later allow variable agent_count, handle it; for now keep
    // fixed.
    fprintf(stderr, "Save has agent_count=%u but build expects %d\n", acount,
            MAX_AGENTS);
    fclose(f);
    return 0;
  }

  for (int i = 0; i < MAX_AGENTS; i++) {
    SaveAgent sa = {0};
    if (fread(&sa, sizeof(sa), 1, f) != 1) {
      fclose(f);
      return 0;
    }

    Agent *a = &agents[i];
    a->agent_id = sa.agent_id;
    a->alive = sa.alive ? true : false;

    a->position = (Vector2){sa.x, sa.y};
    a->facing = (Vector2){sa.fx, sa.fy};

    a->health = sa.health;
    a->stamina = sa.stamina;

    a->age = sa.age;
    a->last_action = sa.last_action;
    a->reward_accumulator = sa.reward_accumulator;

    a->attack_cd = sa.attack_cd;
    a->harvest_cd = sa.harvest_cd;
    a->fire_cd = sa.fire_cd;

    a->fire_latched = sa.fire_latched;
    a->fire_latch_timer = sa.fire_latch_timer;

    a->inv_food = sa.inv_food;
    a->inv_arrows = sa.inv_arrows;
    a->inv_shards = sa.inv_shards;

    a->has_axe = sa.has_axe;
    a->has_pickaxe = sa.has_pickaxe;
    a->has_sword = sa.has_sword;
    a->has_armor = sa.has_armor;
    a->has_bow = sa.has_bow;

    a->tool_selected = sa.tool_selected;
    a->last_craft_selected = sa.last_craft_selected;
  }

  // ---- pickups ONCE ----
  for (int i = 0; i < MAX_PICKUPS; i++)
    pickups[i].alive = false;

  uint32_t pcount = h.pickup_count;
  for (uint32_t k = 0; k < pcount; k++) {
    SavePickup sp = {0};
    if (fread(&sp, sizeof(sp), 1, f) != 1) {
      fclose(f);
      return 0;
    }

    for (int i = 0; i < MAX_PICKUPS; i++) {
      if (!pickups[i].alive) {
        pickups[i].alive = true;
        pickups[i].pos = (Vector2){sp.x, sp.y};
        pickups[i].type = (PickupType)sp.type;
        pickups[i].amount = sp.amount;
        pickups[i].ttl = sp.ttl;
        pickups[i].bob_t = sp.bob_t;
        break;
      }
    }
  }

  // ---- projectiles ONCE ----
  for (int i = 0; i < MAX_PROJECTILES; i++)
    projectiles[i].alive = false;

  uint32_t prcount = h.projectile_count;
  for (uint32_t k = 0; k < prcount; k++) {
    SaveProjectile sp = {0};
    if (fread(&sp, sizeof(sp), 1, f) != 1) {
      fclose(f);
      return 0;
    }

    for (int i = 0; i < MAX_PROJECTILES; i++) {
      if (!projectiles[i].alive) {
        projectiles[i].alive = true;
        projectiles[i].pos = (Vector2){sp.x, sp.y};
        projectiles[i].vel = (Vector2){sp.vx, sp.vy};
        projectiles[i].ttl = sp.ttl;
        projectiles[i].damage = sp.damage;
        projectiles[i].owner = (ProjOwner)sp.owner;
        break;
      }
    }
  }

  // ---- chunks ----
  for (uint32_t k = 0; k < h.chunk_count; k++) {
    SaveChunkHeader ch = {0};
    if (fread(&ch, sizeof(ch), 1, f) != 1) {
      fclose(f);
      return 0;
    }

    int cx = (int)ch.cx;
    int cy = (int)ch.cy;
    if (cx < 0 || cx >= WORLD_SIZE || cy < 0 || cy >= WORLD_SIZE) {
      fclose(f);
      return 0;
    }

    Chunk *c = &world[cx][cy];
    pthread_rwlock_wrlock(&c->lock);

    c->generated = true;
    c->biome_type = ch.biome_type;
    c->mob_spawn_timer = ch.mob_spawn_timer;

    for (int i = 0; i < CHUNK_SIZE; i++)
      for (int j = 0; j < CHUNK_SIZE; j++)
        c->terrain[i][j] = c->biome_type;

    c->resource_count = (int)ch.resource_count;
    if (c->resource_count > MAX_RESOURCES)
      c->resource_count = MAX_RESOURCES;

    // resources
    for (int i = 0; i < (int)ch.resource_count; i++) {
      SaveResource sr = {0};
      if (fread(&sr, sizeof(sr), 1, f) != 1) {
        pthread_rwlock_unlock(&c->lock);
        fclose(f);
        return 0;
      }

      if (i < MAX_RESOURCES) {
        Resource *r = &c->resources[i];
        r->position = clamp_local_to_chunk((Vector2){sr.lx, sr.ly});
        r->type = (ResourceType)sr.type;
        r->health = (int)sr.health;
        r->visited = false;
        r->hit_timer = sr.hit_timer;
        r->break_flash = sr.break_flash;
      }
    }

    // clear mobs then load alive mobs
    for (int i = 0; i < MAX_MOBS; i++)
      c->mobs[i].health = 0;

    for (uint32_t i = 0; i < ch.mob_count; i++) {
      SaveMob sm = {0};
      if (fread(&sm, sizeof(sm), 1, f) != 1) {
        pthread_rwlock_unlock(&c->lock);
        fclose(f);
        return 0;
      }

      int slot = find_free_mob_slot(c);
      if (slot >= 0) {
        Mob *m = &c->mobs[slot];
        m->position = clamp_local_to_chunk((Vector2){sm.lx, sm.ly});
        m->type = (MobType)sm.type;
        m->health = (int)sm.health;
        m->visited = false;
        m->vel = (Vector2){sm.velx, sm.vely};
        m->ai_timer = sm.ai_timer;
        m->aggro_timer = sm.aggro_timer;
        m->attack_cd = sm.attack_cd;
        m->hurt_timer = sm.hurt_timer;
        m->lunge_timer = sm.lunge_timer;
      }
    }

    pthread_rwlock_unlock(&c->lock);
  }

  fclose(f);
  return 1;
}

static int save_models_to_disk(const char *world_name) {
  char dir[256];
  make_models_dir(dir, sizeof(dir), world_name);
  mkdir(dir, 0755);

  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &agents[i];
    if (!a->sam)
      continue;

    char path[256];
    make_agent_model_path(path, sizeof(path), world_name, a->agent_id);

    FILE *f = fopen(path, "wb");
    if (!f)
      continue;

    // DO THIS (real API):
    if (!SAM_save(f, a->sam)) {
      fprintf(stderr, "Failed to save model for agent %d\n", a->agent_id);

      fclose(f);
      return 1;
    }

    fclose(f);
  }
  return 1;
}

static int load_models_from_disk(const char *world_name) {
  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &agents[i];
    if (!a->sam)
      continue;

    char path[256];
    make_agent_model_path(path, sizeof(path), world_name, a->agent_id);

    FILE *f = fopen(path, "rb");
    if (!f)
      continue;

    // DO THIS (real API):
    a->sam = SAM_load(f);

    fclose(f);
  }
  return 1;
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
static inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

static inline float wrap_angle(float a) {
  while (a > PI)
    a -= 2.0f * PI;
  while (a < -PI)
    a += 2.0f * PI;
  return a;
}

static inline float lerp_angle(float a, float b, float t) {
  float d = wrap_angle(b - a);
  return a + d * t;
}

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

static void player_fire_bow_charged(Vector2 dir, float charge01) {
  charge01 = clamp01(charge01);

  float spd = lerp(BOW_SPEED_MIN, BOW_SPEED_MAX, charge01);
  float ttl = lerp(BOW_TTL_MIN, BOW_TTL_MAX, charge01);

  int dmg = (int)roundf(lerp((float)BOW_DMG_MIN, (float)BOW_DMG_MAX, charge01));
  if (has_sword)
    dmg += 4; // keep your “upgrade” synergy if you want

  spawn_projectile(player.position, dir, spd, ttl, dmg, PROJ_OWNER_PLAYER);

  // tiny feedback
  cam_shake = fmaxf(cam_shake, 0.05f + 0.10f * charge01);
}

static void draw_bow_charge_fx(void) {
  if (!bow_charging || !has_bow)
    return;

  // bowstring between hands
  DrawLineEx(g_handL, g_handR, 3.0f, (Color){20, 20, 20, 180});

  // charge ring on right hand
  float r = 26.0f;
  DrawCircleLines((int)g_handR.x, (int)g_handR.y, r,
                  (Color){255, 255, 255, 140});

  // progress arc (simple: draw a few small segments)
  int segs = 22;
  float a0 = -PI / 2.0f;
  float a1 = a0 + (2.0f * PI * bow_charge01);
  for (int i = 0; i < segs; i++) {
    float t0 = (float)i / (float)segs;
    float t1 = (float)(i + 1) / (float)segs;
    float aa0 = lerp(a0, a1, t0);
    float aa1 = lerp(a0, a1, t1);

    Vector2 p0 =
        (Vector2){g_handR.x + cosf(aa0) * r, g_handR.y + sinf(aa0) * r};
    Vector2 p1 =
        (Vector2){g_handR.x + cosf(aa1) * r, g_handR.y + sinf(aa1) * r};
    DrawLineEx(p0, p1, 3.0f, (Color){255, 220, 120, 220});
  }

  // tiny text
  DrawText(TextFormat("Charge %d%%", (int)(bow_charge01 * 100.0f)),
           (int)g_handR.x + 18, (int)g_handR.y - 42, 14, RAYWHITE);
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

static inline int agent_in_base(const Agent *a, const Tribe *tr) {
  float d = Vector2Distance(a->position, tr->base.position);
  return (d < tr->base.radius + 0.35f);
}

static int agent_try_craft_action(Agent *a, Tribe *tr, ActionType craft_action,
                                  float *reward) {
  if (!agent_in_base(a, tr)) {
    if (reward)
      *reward += -0.006f;
    return 0;
  }

  // Helper macro: spend if possible
#define TRY_CRAFT(cond_have, w, s, g, f, on_success)                           \
  do {                                                                         \
    if (cond_have) {                                                           \
      if (reward)                                                              \
        *reward += -0.003f;                                                    \
      return 0;                                                                \
    }                                                                          \
    if (tr->wood < (w) || tr->stone < (s) || tr->gold < (g) ||                 \
        tr->food < (f)) {                                                      \
      if (reward)                                                              \
        *reward += -0.003f;                                                    \
      return 0;                                                                \
    }                                                                          \
    tr->wood -= (w);                                                           \
    tr->stone -= (s);                                                          \
    tr->gold -= (g);                                                           \
    tr->food -= (f);                                                           \
    on_success;                                                                \
    if (reward)                                                                \
      *reward += 0.10f;                                                        \
    return 1;                                                                  \
  } while (0)

  switch (craft_action) {
  case ACTION_CRAFT_AXE:
    TRY_CRAFT(a->has_axe, 3, 2, 0, 0, {
      a->has_axe = true;
      a->last_craft_selected = TOOL_AXE;
    });
    break;

  case ACTION_CRAFT_PICKAXE:
    TRY_CRAFT(a->has_pickaxe, 3, 3, 0, 0, {
      a->has_pickaxe = true;
      a->last_craft_selected = TOOL_PICKAXE;
    });
    break;

  case ACTION_CRAFT_SWORD:
    TRY_CRAFT(a->has_sword, 0, 4, 2, 0, {
      a->has_sword = true;
      a->last_craft_selected = TOOL_SWORD;
    });
    break;

  case ACTION_CRAFT_ARMOR:
    TRY_CRAFT(a->has_armor, 0, 5, 2, 0, {
      a->has_armor = true;
      a->last_craft_selected = TOOL_ARMOR;
    });
    break;

  case ACTION_CRAFT_BOW:
    TRY_CRAFT(a->has_bow, 4, 0, 1, 0, {
      a->has_bow = true;
      a->last_craft_selected = TOOL_BOW;
    });
    break;

  case ACTION_CRAFT_ARROWS:
    // explicit ammo craft (no “oh you probably need ammo”)
    if (tr->shards < 1 || tr->wood < 1) {
      if (reward)
        *reward += -0.003f;
      return 0;
    }
    tr->shards -= 1;
    tr->wood -= 1;
    {
      int made = 6;
      tr->arrows += made;
      a->inv_arrows += made;
    }
    if (reward)
      *reward += 0.06f;
    return 1;

  default:
    if (reward)
      *reward += -0.003f;
    return 0;
  }

  if (reward)
    *reward += -0.003f;
  return 0;

#undef TRY_CRAFT
}

static void spawn_projectile(Vector2 pos, Vector2 dir, float speed, float ttl,
                             int dmg, ProjOwner owner) {
  for (int i = 0; i < MAX_PROJECTILES; i++) {
    if (!projectiles[i].alive) {
      projectiles[i].alive = true;
      projectiles[i].pos = pos;
      projectiles[i].vel = Vector2Scale(Vector2Normalize(dir), speed);
      projectiles[i].ttl = ttl;
      projectiles[i].damage = dmg;
      projectiles[i].owner = owner;
      projectiles[i].owner_agent_id = -1;
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

    // ---------------------------
    // MOB projectiles: hit player
    // ---------------------------
    if (p->owner == PROJ_OWNER_MOB) {
      float d = Vector2Distance(p->pos, player.position);
      if (d < 0.55f) {
        player.health -= (float)p->damage;
        player_hurt_timer = 0.18f;
        p->alive = false;
        cam_shake = fmaxf(cam_shake, 0.10f);
        continue;
      }

      // hit agents too (simple circle)
      for (int aidx = 0; aidx < MAX_AGENTS; aidx++) {
        Agent *a = &agents[aidx];
        if (!a->alive)
          continue;
        if (Vector2Distance(p->pos, a->position) < 0.55f) {
          a->health -= (float)p->damage;
          a->flash_timer = 0.18f;
          p->alive = false;
          cam_shake = fmaxf(cam_shake, 0.08f);
          break;
        }
      }
      if (!p->alive)
        continue;
    }

    // -----------------------------------
    // PLAYER/AGENT projectiles: hit mobs
    // -----------------------------------
    if (p->owner == PROJ_OWNER_PLAYER || p->owner == PROJ_OWNER_AGENT) {
      int cx = (int)(p->pos.x / CHUNK_SIZE);
      int cy = (int)(p->pos.y / CHUNK_SIZE);

      // check a small neighborhood so fast arrows don't miss edge cases
      for (int dx = -1; dx <= 1 && p->alive; dx++) {
        for (int dy = -1; dy <= 1 && p->alive; dy++) {
          Chunk *c = get_chunk(cx + dx, cy + dy);

          pthread_rwlock_wrlock(&c->lock);

          Vector2 origin = (Vector2){(float)((cx + dx) * CHUNK_SIZE),
                                     (float)((cy + dy) * CHUNK_SIZE)};

          for (int mi = 0; mi < MAX_MOBS; mi++) {
            Mob *m = &c->mobs[mi];
            if (m->health <= 0)
              continue;

            Vector2 mw = Vector2Add(origin, m->position);
            float mr = mob_radius_world(m->type);

            // projectile collision radius ~ small
            if (Vector2Distance(p->pos, mw) < (mr * 0.85f)) {
              m->health -= p->damage;
              m->hurt_timer = 0.18f;
              m->aggro_timer = 3.0f;

              p->alive = false;
              cam_shake = fmaxf(cam_shake, 0.08f);

              if (m->health <= 0) {
                Vector2 mob_world_pos =
                    (Vector2){(float)((cx + dx) * CHUNK_SIZE) + m->position.x,
                              (float)((cy + dy) * CHUNK_SIZE) + m->position.y};
                on_mob_killed(m->type, mob_world_pos);
                m->health = 0;
              }
              break;
            }
          }

          pthread_rwlock_unlock(&c->lock);
        }
      }
    }
  }
}

static inline Vector2 world_to_screen(Vector2 wp) {
  Vector2 sp = Vector2Subtract(wp, camera_pos);
  sp = Vector2Scale(sp, WORLD_SCALE);
  sp.x += SCREEN_WIDTH / 2;
  sp.y += SCREEN_HEIGHT / 2;
  return sp;
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

static inline float agent_radius_world(void) { return 0.45f; }

static inline float agent_move_speed(void) {
  // tuned for your world scale / feel
  return 2.8f; // world units per second
}

static inline float agent_attack_range(void) { return 2.6f; }
static inline int agent_attack_damage(void) { return 8; }

static inline float agent_attack_cooldown(void) { return 0.35f; }
static inline float agent_harvest_cooldown(void) { return 0.28f; }
static inline float agent_fire_cooldown(void) { return 0.85f; }

static inline float agent_harvest_cost(ResourceType t) {
  if (t == RES_ROCK || t == RES_GOLD)
    return 3.0f;
  if (t == RES_TREE)
    return 2.0f;
  if (t == RES_FOOD)
    return 1.2f;
  return 2.0f;
}

static inline int agent_harvest_damage(ResourceType t) {
  if (t == RES_ROCK || t == RES_GOLD)
    return 14;
  if (t == RES_TREE)
    return 12;
  if (t == RES_FOOD)
    return 10;
  return 10;
}

static inline int mob_is_hostile(MobType t) {
  return (t == MOB_ZOMBIE || t == MOB_SKELETON);
}

static void agent_try_move(Agent *a, Vector2 dir) {
  // dir is expected to be -1/0/1 style
  float dt = g_dt;
  float spd = agent_move_speed();

  float len = Vector2Length(dir);
  if (len < 1e-6f)
    return;
  dir = Vector2Scale(dir, 1.0f / len);

  Vector2 old = a->position;
  Vector2 next = Vector2Add(a->position, Vector2Scale(dir, spd * dt));

  // simple blocked test (uses generated chunks + locks safely)
  int cx = (int)(next.x / CHUNK_SIZE);
  int cy = (int)(next.y / CHUNK_SIZE);

  if (world_pos_blocked_nearby(cx, cy, next, agent_radius_world(), -999999,
                               -999999)) {
    // try a tiny slide in x or y
    Vector2 tryX = (Vector2){next.x, old.y};
    Vector2 tryY = (Vector2){old.x, next.y};

    int cxX = (int)(tryX.x / CHUNK_SIZE);
    int cyX = (int)(tryX.y / CHUNK_SIZE);
    int cxY = (int)(tryY.x / CHUNK_SIZE);
    int cyY = (int)(tryY.y / CHUNK_SIZE);

    if (!world_pos_blocked_nearby(cxX, cyX, tryX, agent_radius_world(), -999999,
                                  -999999))
      a->position = tryX;
    else if (!world_pos_blocked_nearby(cxY, cyY, tryY, agent_radius_world(),
                                       -999999, -999999))
      a->position = tryY;
    else
      a->position = old;
  } else {
    a->position = next;
  }
}

static void agent_gain_loot_for_mob_kill(Agent *a, Tribe *tr, MobType t) {
  if (t == MOB_PIG || t == MOB_SHEEP) {
    int f = 1 + (rand() % 2);
    a->inv_food += f;
    tr->food += f;
  } else if (t == MOB_ZOMBIE) {
    int s = 1 + (rand() % 3);
    a->inv_shards += s;
    tr->shards += s;
  } else if (t == MOB_SKELETON) {
    int arr = 2 + (rand() % 3);
    a->inv_arrows += arr;
    tr->arrows += arr;
    a->inv_shards += 1;
    tr->shards += 1;
  }
}

static inline int ray_circle_intersect(Vector2 ro, Vector2 rd, Vector2 c,
                                       float r, float *out_t) {
  // rd must be normalized
  Vector2 oc = Vector2Subtract(ro, c);
  float b = Vector2DotProduct(oc, rd);
  float cc = Vector2DotProduct(oc, oc) - r * r;
  float disc = b * b - cc;
  if (disc < 0.0f)
    return 0;
  float s = sqrtf(disc);
  float t0 = -b - s;
  float t1 = -b + s;

  float t = (t0 > 1e-4f) ? t0 : ((t1 > 1e-4f) ? t1 : -1.0f);
  if (t < 0.0f)
    return 0;

  *out_t = t;
  return 1;
}

static RayHit raycast_world_objects(Vector2 ro, Vector2 rd, float maxT) {
  RayHit best = {0};
  best.kind = HIT_NONE;
  best.t = maxT;

  // check bases (global, cheap)
  for (int t = 0; t < TRIBE_COUNT; t++) {
    float thit = 0.0f;
    if (ray_circle_intersect(ro, rd, tribes[t].base.position,
                             tribes[t].base.radius, &thit)) {
      if (thit < best.t && thit <= maxT) {
        best.kind = HIT_BASE;
        best.t = thit;
        best.index = t;
        best.hit_pos = Vector2Add(ro, Vector2Scale(rd, thit));
      }
    }
  }

  // check nearby chunks (3x3 around ray origin chunk)
  int ocx = (int)(ro.x / CHUNK_SIZE);
  int ocy = (int)(ro.y / CHUNK_SIZE);

  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      int cx = ocx + dx;
      int cy = ocy + dy;

      Chunk *c = get_chunk(cx, cy);
      pthread_rwlock_rdlock(&c->lock);

      Vector2 origin =
          (Vector2){(float)(cx * CHUNK_SIZE), (float)(cy * CHUNK_SIZE)};

      // resources (circles)
      for (int i = 0; i < c->resource_count; i++) {
        Resource *r = &c->resources[i];
        if (r->health <= 0)
          continue;

        Vector2 cw = Vector2Add(origin, r->position);
        float rr = res_radius_world(r->type);

        float thit = 0.0f;
        if (ray_circle_intersect(ro, rd, cw, rr, &thit)) {
          if (thit < best.t && thit <= maxT) {
            best.kind = HIT_RESOURCE;
            best.t = thit;
            best.cx = cx;
            best.cy = cy;
            best.index = i;
            best.hit_pos = Vector2Add(ro, Vector2Scale(rd, thit));
          }
        }
      }

      // mobs (circles)
      for (int i = 0; i < MAX_MOBS; i++) {
        Mob *m = &c->mobs[i];
        if (m->health <= 0)
          continue;

        Vector2 mw = Vector2Add(origin, m->position);
        float mr = mob_radius_world(m->type);

        float thit = 0.0f;
        if (ray_circle_intersect(ro, rd, mw, mr, &thit)) {
          if (thit < best.t && thit <= maxT) {
            best.kind = HIT_MOB;
            best.t = thit;
            best.cx = cx;
            best.cy = cy;
            best.index = i;
            best.hit_pos = Vector2Add(ro, Vector2Scale(rd, thit));
          }
        }
      }

      pthread_rwlock_unlock(&c->lock);
    }
  }

  return best;
}

static int agent_try_attack_forward(Agent *a, Tribe *tr, float *reward) {
  (void)tr;

  if (a->attack_cd > 0.0f) {
    *reward += R_ATTACK_WASTE * 0.25f; // tiny slap for spamming on cooldown
    return 0;
  }

  Vector2 ro = a->position;
  Vector2 rd = Vector2Normalize(a->facing);
  if (Vector2Length(rd) < 1e-3f)
    rd = (Vector2){1, 0};

  bool using_sword = (a->tool_selected == TOOL_SWORD) && a->has_sword;
  float range = agent_attack_range() + (using_sword ? 0.45f : 0.0f);
  int dmg = agent_attack_damage() + (using_sword ? 10 : 0);

  RayHit hit = raycast_world_objects(ro, rd, range);

  if (hit.kind != HIT_MOB) {
    // wasted swing into air
    a->attack_cd = agent_attack_cooldown() * 0.35f; // small “whiff” cooldown
    *reward += R_ATTACK_WASTE;
    return 0;
  }

  Chunk *c = get_chunk(hit.cx, hit.cy);
  pthread_rwlock_wrlock(&c->lock);

  Mob *m = &c->mobs[hit.index];
  if (m->health > 0) {
    a->attack_cd = agent_attack_cooldown();

    m->health -= dmg;
    m->hurt_timer = 0.18f;
    m->aggro_timer = 3.0f;
    m->lunge_timer = 0.10f;

    *reward += R_ATTACK_HIT;

    if (m->health <= 0) {
      Vector2 mw = (Vector2){hit.cx * CHUNK_SIZE + m->position.x,
                             hit.cy * CHUNK_SIZE + m->position.y};
      on_mob_killed(m->type, mw);
      m->health = 0;

      agent_gain_loot_for_mob_kill(a, tr, m->type);
      *reward += R_ATTACK_KILL;
    }

    pthread_rwlock_unlock(&c->lock);
    return 1;
  }

  pthread_rwlock_unlock(&c->lock);

  // If we somehow hit a dead slot, still count as waste
  *reward += R_ATTACK_WASTE * 0.5f;
  return 0;
}

static int agent_try_harvest_forward(Agent *a, Tribe *tr, float *reward) {
  if (a->harvest_cd > 0.0f) {
    *reward += R_HARVEST_WASTE * 0.25f;
    return 0;
  }

  Vector2 ro = a->position;
  Vector2 rd = Vector2Normalize(a->facing);
  if (Vector2Length(rd) < 1e-3f)
    rd = (Vector2){1, 0};

  float range = HARVEST_DISTANCE;
  bool using_axe = (a->tool_selected == TOOL_AXE) && a->has_axe;
  bool using_pick = (a->tool_selected == TOOL_PICKAXE) && a->has_pickaxe;

  RayHit hit = raycast_world_objects(ro, rd, range);
  if (hit.kind != HIT_RESOURCE) {
    // wasted harvest swing
    a->harvest_cd = agent_harvest_cooldown() * 0.35f; // small whiff cooldown
    *reward += R_HARVEST_WASTE;
    return 0;
  }

  Chunk *c = get_chunk(hit.cx, hit.cy);
  pthread_rwlock_wrlock(&c->lock);

  Resource *r = &c->resources[hit.index];
  if (r->health <= 0) {
    pthread_rwlock_unlock(&c->lock);
    *reward += R_HARVEST_WASTE * 0.5f;
    return 0;
  }

  float cost = agent_harvest_cost(r->type);
  if (r->type == RES_TREE && using_axe)
    cost *= 0.78f;
  if ((r->type == RES_ROCK || r->type == RES_GOLD) && using_pick)
    cost *= 0.78f;

  if (a->stamina < cost) {
    pthread_rwlock_unlock(&c->lock);
    *reward += R_HARVEST_NO_STAMINA;
    return 0;
  }

  a->harvest_cd = agent_harvest_cooldown();
  a->stamina -= cost;

  int dmg = agent_harvest_damage(r->type);
  if (r->type == RES_TREE && using_axe)
    dmg += 10;
  if ((r->type == RES_ROCK || r->type == RES_GOLD) && using_pick)
    dmg += 14;

  r->health -= dmg;
  r->hit_timer = 0.14f;
  r->break_flash = 0.06f;

  *reward += R_HARVEST_HIT;

  if (r->health <= 0) {
    r->health = 0;

    switch (r->type) {
    case RES_TREE:
      tr->wood += 1;
      break;
    case RES_ROCK:
      tr->stone += 1;
      break;
    case RES_GOLD:
      tr->gold += 1;
      break;
    case RES_FOOD:
      a->inv_food += 1;
      tr->food += 1;
      break;
    default:
      break;
    }

    *reward += R_HARVEST_BREAK;
  }

  pthread_rwlock_unlock(&c->lock);
  return 1;
}

static void agent_try_fire_forward(Agent *a, float *reward,
                                   bool is_continuous) {
  if (!a->has_bow) {
    if (reward)
      *reward += -0.006f;
    return;
  }

  if (a->fire_cd > 0.0f) {
    if (!is_continuous && reward)
      *reward += R_FIRE_WASTE * 0.10f;
    return;
  }

  if (a->inv_arrows <= 0) {
    if (reward)
      *reward += R_FIRE_NO_AMMO;
    return;
  }

  Vector2 rd = Vector2Normalize(a->facing);
  if (Vector2Length(rd) < 1e-3f)
    rd = (Vector2){1, 0};

  // consume + spawn; outcome handled by projectile collision only
  a->inv_arrows--;
  a->fire_cd = agent_fire_cooldown();

  spawn_projectile(a->position, rd, 13.0f, 1.65f, 10, PROJ_OWNER_AGENT);

  // optional: slight penalty to discourage blind spam (NOT lookahead-based)
  if (reward && !is_continuous)
    *reward += -0.002f;
}
static void agent_try_eat(Agent *a, float *reward) {
  if (a->inv_food <= 0) {
    *reward += R_EAT_NO_FOOD;
    return;
  }

  // if basically full, it's waste
  if (a->health > 92.0f && a->stamina > 75.0f) {
    a->inv_food--; // still consumes -> true waste punishment
    *reward += R_EAT_WASTE;
    return;
  }

  a->inv_food--;

  a->health = fminf(100.0f, a->health + 18.0f);
  a->stamina = fminf(100.0f, a->stamina + 32.0f);

  *reward += R_EAT_GOOD;
}

static void tribe_try_repair_base(Tribe *tr, float *reward) {
  // simple repair rule: if damaged, spend mats to repair slowly
  if (tr->integrity >= 99.9f)
    return;

  // spend 1 wood + 1 stone for +6 integrity
  if (tr->wood >= 1 && tr->stone >= 1) {
    tr->wood -= 1;
    tr->stone -= 1;
    tr->integrity = fminf(100.0f, tr->integrity + 6.0f);
    *reward += 0.08f;
  }
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

static Vector2 ui_nearest_base_pos(Vector2 wp) {
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

static void player_try_attack_forward(Vector2 facing_dir) {
  if (player_attack_cd > 0.0f)
    return;

  Vector2 rd = Vector2Normalize(facing_dir);
  if (Vector2Length(rd) < 1e-3f)
    rd = (Vector2){1, 0};

  float range = player_attack_range();
  int dmg = player_attack_damage();

  RayHit hit = raycast_world_objects(player.position, rd, range);
  if (hit.kind != HIT_MOB) {
    player_attack_cd = player_attack_cooldown() * 0.35f; // whiff cooldown
    return;
  }

  Chunk *c = get_chunk(hit.cx, hit.cy);
  pthread_rwlock_wrlock(&c->lock);

  Mob *m = &c->mobs[hit.index];
  if (m->health > 0) {
    player_attack_cd = player_attack_cooldown();

    m->health -= dmg;
    m->hurt_timer = 0.18f;
    m->aggro_timer = 3.0f;
    m->lunge_timer = 0.10f;

    cam_shake = fmaxf(cam_shake, 0.10f);

    if (m->health <= 0) {
      Vector2 mob_world_pos = (Vector2){hit.cx * CHUNK_SIZE + m->position.x,
                                        hit.cy * CHUNK_SIZE + m->position.y};
      on_mob_killed(m->type, mob_world_pos);
      m->health = 0;
    }
  }

  pthread_rwlock_unlock(&c->lock);
}

static void player_try_harvest_forward(Vector2 facing_dir) {
  if (player_harvest_cd > 0.0f)
    return;

  Vector2 rd = Vector2Normalize(facing_dir);
  if (Vector2Length(rd) < 1e-3f)
    rd = (Vector2){1, 0};

  float range = HARVEST_DISTANCE;

  RayHit hit = raycast_world_objects(player.position, rd, range);
  if (hit.kind != HIT_RESOURCE) {
    player_harvest_cd = 0.10f; // small whiff cooldown
    return;
  }

  Chunk *c = get_chunk(hit.cx, hit.cy);
  pthread_rwlock_wrlock(&c->lock);

  Resource *r = &c->resources[hit.index];
  if (r->health <= 0) {
    pthread_rwlock_unlock(&c->lock);
    return;
  }

  float cd = player_resource_cooldown(r->type);
  float cost = player_resource_stamina_cost(r->type);
  int dmg = player_resource_damage(r->type);

  if (player.stamina < cost) {
    pthread_rwlock_unlock(&c->lock);
    return;
  }

  player_harvest_cd = cd;
  player.stamina -= cost;

  r->health -= dmg;
  r->hit_timer = 0.14f;
  r->break_flash = 0.06f;

  if (r->type == RES_ROCK || r->type == RES_GOLD) {
    cam_shake = fmaxf(cam_shake, 0.10f);
  }

  if (r->health <= 0) {
    give_drop(r->type);
    r->health = 0;
  }

  pthread_rwlock_unlock(&c->lock);
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

// ---- Mob radius in WORLD units (used for spacing / collisions during
// spawn)
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

// ---- Spacing test: is "worldPos" too close to ANY resource/mob nearby?
// ---- IMPORTANT: we do NOT call get_chunk() here (avoids recursive
// generation). We only check chunks that are already generated.
// ---- Spacing test: is "worldPos" too close to ANY resource/mob nearby?
// ---- IMPORTANT:
// - Does NOT call get_chunk() (no recursive generation)
// - Safely reads generated chunks by taking a read lock
// - If the caller already holds the lock for (self_cx,self_cy), pass those
// so
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

      // Avoid deadlock: if caller already holds this chunk's lock, don't
      // lock it again.
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

static inline void agent_set_facing_from(Vector2 v, Agent *a) {
  float len = Vector2Length(v);
  if (len > 1e-3f)
    a->facing = Vector2Scale(v, 1.0f / len);
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

static void draw_agent(const Agent *a, Vector2 sp, Color tribeColor) {
  float r = WORLD_SCALE * 0.60 * scale_size; // base agent size
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
  float ang = 0.0f;
  switch (a->last_action) {
  case ACTION_UP:
    ang = -PI / 2;
    break;
  case ACTION_DOWN:
    ang = PI / 2;
    break;
  case ACTION_LEFT:
    ang = PI;
    break;
  case ACTION_RIGHT:
    ang = 0.0f;
    break;
  case ACTION_ATTACK:
  case ACTION_HARVEST:
  case ACTION_FIRE:
    // small idle wobble for “doing stuff”
    ang = sinf(t * 6.0f + (float)a->agent_id) * 0.35f;
    break;
  default:
    ang = sinf(t * 1.5f + (float)a->agent_id) * 0.25f;
    break;
  }
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
    tr->wood = tr->stone = tr->gold = tr->food = 0;
    tr->shards = tr->arrows = 0;

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
  MuConfig cfg = {.obs_dim = OBS_DIM, // expandable, not fixed memory
                  .latent_dim = 64,
                  .action_count = ACTION_COUNT};

  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &agents[i];
    a->agent_id = i;
    a->facing = (Vector2){1, 0};
    a->alive = true;
    a->health = a->stamina = 100;
    a->flash_timer = 0;
    a->age = 0;
    a->agent_start = i;
    a->reward_accumulator = 0.0f;

    a->attack_cd = 0.0f;
    a->harvest_cd = 0.0f;
    a->fire_cd = 0.0f;

    a->fire_latched = 0;
    a->fire_latch_timer = 0.0f;

    a->inv_food = 0;
    a->inv_arrows = 0;
    a->inv_shards = 0;

    a->has_axe = a->has_pickaxe = a->has_sword = a->has_armor = a->has_bow =
        false;

    a->tool_selected = TOOL_HAND;
    a->last_craft_selected = -1;

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

  // --- nearest HOSTILE mob in this chunk ---
  float best_h_d = 1e9f;
  Vector2 best_h_dir = (Vector2){0, 0};
  int hostile_found = 0;

  // --- nearest PASSIVE mob in this chunk ---
  float best_p_d = 1e9f;
  Vector2 best_p_dir = (Vector2){0, 0};
  int passive_found = 0;

  int hostile_count_near = 0; // within 6 units
  int passive_count_near = 0;

  for (int i = 0; i < MAX_MOBS; i++) {
    Mob *m = &c->mobs[i];
    if (m->health <= 0)
      continue;

    Vector2 m_world = Vector2Add(chunk_origin, m->position);
    Vector2 dvec = Vector2Subtract(m_world, a->position);
    float d = Vector2Length(dvec);

    int hostile = mob_is_hostile(m->type);

    if (d < 6.0f) {
      if (hostile)
        hostile_count_near++;
      else
        passive_count_near++;
    }

    if (hostile) {
      if (d < best_h_d) {
        best_h_d = d;
        best_h_dir = dvec;
        hostile_found = 1;
      }
    } else {
      if (d < best_p_d) {
        best_p_d = d;
        best_p_dir = dvec;
        passive_found = 1;
      }
    }
  }

  // hostile features
  if (!hostile_found) {
    obs_push(obs, 1.0f);
    obs_push(obs, 0.0f);
    obs_push(obs, 0.0f);
    obs_push(obs, 0.0f);
  } else {
    obs_push(obs, clamp01(best_h_d / 32.0f));
    obs_push(obs, safe_norm(best_h_dir.x, best_h_d));
    obs_push(obs, safe_norm(best_h_dir.y, best_h_d));
    obs_push(obs, 1.0f);
  }

  // passive features
  if (!passive_found) {
    obs_push(obs, 1.0f);
    obs_push(obs, 0.0f);
    obs_push(obs, 0.0f);
    obs_push(obs, 0.0f);
  } else {
    obs_push(obs, clamp01(best_p_d / 32.0f));
    obs_push(obs, safe_norm(best_p_dir.x, best_p_d));
    obs_push(obs, safe_norm(best_p_dir.y, best_p_d));
    obs_push(obs, 1.0f);
  }

  // counts (soft normalized)
  obs_push(obs, clamp01((float)hostile_count_near / 6.0f));
  obs_push(obs, clamp01((float)passive_count_near / 6.0f));

  // --- player relative (only meaningful if player is in same chunk) ---
  int pcx = (int)(player.position.x / CHUNK_SIZE);
  int pcy = (int)(player.position.y / CHUNK_SIZE);
  int same_chunk = (pcx == cx && pcy == cy);

  Vector2 toP = Vector2Subtract(player.position, a->position);
  float dP = Vector2Length(toP);

  obs_push(obs, same_chunk ? 1.0f : 0.0f);
  if (!same_chunk) {
    obs_push(obs, 1.0f); // far
    obs_push(obs, 0.0f);
    obs_push(obs, 0.0f);
  } else {
    obs_push(obs, clamp01(dP / 32.0f));
    obs_push(obs, safe_norm(toP.x, dP));
    obs_push(obs, safe_norm(toP.y, dP));
  }

  // facing (so forward-actions make sense)
  obs_push(obs, a->facing.x);
  obs_push(obs, a->facing.y);

  // ----------------------------
  // INVENTORY / TOOLS (compact)
  // ----------------------------
  float in_base01 = (dbase < tr->base.radius) ? 1.0f : 0.0f;

  // inv open: for agents, "inventory/crafting context is open" == in base
  obs_push(obs, in_base01);

  // available tools for use (owned)
  obs_push(obs, a->has_axe ? 1.0f : 0.0f);
  obs_push(obs, a->has_pickaxe ? 1.0f : 0.0f);
  obs_push(obs, a->has_sword ? 1.0f : 0.0f);
  obs_push(obs, a->has_armor ? 1.0f : 0.0f);
  obs_push(obs, a->has_bow ? 1.0f : 0.0f);

  // selected tool one-hot across TOOL_COUNT
  for (int i = 0; i < TOOL_COUNT; i++) {
    obs_push(obs, (a->tool_selected == i) ? 1.0f : 0.0f);
  }

  // provide selected tool as a single continuous scalar (helps learning)
  obs_push(obs, (float)a->tool_selected / (float)(TOOL_COUNT - 1));

  // ----------------------------
  // EXTRA CONTEXT (small, helps policy)
  // ----------------------------

  // day/night (helps survival & raid behavior)
  obs_push(obs, is_night_cached ? 1.0f : 0.0f);

  // time of day as sin/cos (smooth cyclical signal)
  float ang = time_of_day * 2.0f * PI;
  obs_push(obs, sinf(ang));
  obs_push(obs, cosf(ang));

  // base integrity (per tribe)
  obs_push(obs, clamp01(tr->integrity / 100.0f));

  // tribe resources snapshot (normalize softly to avoid huge values
  // dominating)
  obs_push(obs, clamp01((float)tr->wood / 30.0f));
  obs_push(obs, clamp01((float)tr->stone / 30.0f));
  obs_push(obs, clamp01((float)tr->gold / 30.0f));
  obs_push(obs, clamp01((float)tr->food / 30.0f));
  obs_push(obs, clamp01((float)tr->shards / 30.0f));
  obs_push(obs, clamp01((float)tr->arrows / 60.0f));

  // agent personal inventory (important for eat/fire decisions)
  obs_push(obs, clamp01((float)a->inv_food / 10.0f));
  obs_push(obs, clamp01((float)a->inv_shards / 10.0f));
  obs_push(obs, clamp01((float)a->inv_arrows / 20.0f));

  // cooldown fractions (0=ready, 1=on cooldown)
  obs_push(obs, clamp01(a->attack_cd / agent_attack_cooldown()));
  obs_push(obs, clamp01(a->harvest_cd / agent_harvest_cooldown()));
  obs_push(obs, clamp01(a->fire_cd / agent_fire_cooldown()));

  // bias
  obs_push(obs, 1.0f);

  // final: enforce fixed-size vector
  obs_finalize_fixed(obs, OBS_DIM);
}

static int agent_face_nearest_mob_in_chunk(Agent *a, Chunk *c, int cx, int cy,
                                           float maxRange) {
  Vector2 origin =
      (Vector2){(float)(cx * CHUNK_SIZE), (float)(cy * CHUNK_SIZE)};
  float bestD = 1e9f;
  Vector2 bestDir = {0};

  for (int i = 0; i < MAX_MOBS; i++) {
    Mob *m = &c->mobs[i];
    if (m->health <= 0)
      continue;

    Vector2 mw = Vector2Add(origin, m->position);
    Vector2 dv = Vector2Subtract(mw, a->position);
    float d = Vector2Length(dv);
    if (d < bestD && d <= maxRange) {
      bestD = d;
      bestDir = dv;
    }
  }

  if (bestD >= 1e8f)
    return 0;
  agent_set_facing_from(bestDir, a);
  return 1;
}

/* =======================
   AGENT UPDATE
======================= */
void update_agent(Agent *a) {
  if (!a || !a->alive)
    return;

  float dt = g_dt;
  if (dt <= 0.0f)
    dt = 1.0f / 60.0f;

  Tribe *tr = &tribes[a->agent_id / AGENT_PER_TRIBE];

  // --- timers ---
  if (a->attack_cd > 0.0f)
    a->attack_cd -= dt;
  if (a->harvest_cd > 0.0f)
    a->harvest_cd -= dt;
  if (a->fire_cd > 0.0f)
    a->fire_cd -= dt;
  if (a->flash_timer > 0.0f)
    a->flash_timer -= dt;
  if (a->fire_latch_timer > 0.0f) {
    a->fire_latch_timer -= dt;
    if (a->fire_latch_timer <= 0.0f) {
      a->fire_latch_timer = 0.0f;
      a->fire_latched = 0;
    }
  }

  // --- passive environment dynamics (NOT policy overrides) ---
  float dBase = Vector2Distance(a->position, tr->base.position);
  bool in_base = (dBase < tr->base.radius + 0.35f);

  if (in_base) {
    a->health = fminf(100.0f, a->health + 10.0f * dt);
    a->stamina = fminf(100.0f, a->stamina + 24.0f * dt);
  } else {
    a->stamina = fminf(100.0f, a->stamina + (STAMINA_REGEN_RATE * 0.55f) * dt);
  }

  // --- build observation ---
  int cx = (int)(a->position.x / CHUNK_SIZE);
  int cy = (int)(a->position.y / CHUNK_SIZE);
  Chunk *c = get_chunk(cx, cy);

  ObsBuffer obs;
  obs_init(&obs);

  pthread_rwlock_rdlock(&c->lock);
  encode_observation(a, c, &obs);
  pthread_rwlock_unlock(&c->lock);

  // --- MuZero chooses action ---
  int action =
      muze_plan(a->cortex, obs.data, (size_t)obs.size, (size_t)ACTION_COUNT);

  obs_free(&obs);

  a->last_action = action;

  // tool selection is "sticky" but reacts to actions
  if (action == ACTION_ATTACK) {
    a->tool_selected = a->has_sword ? TOOL_SWORD : TOOL_HAND;
  } else if (action == ACTION_HARVEST) {
    if (a->has_pickaxe)
      a->tool_selected = TOOL_PICKAXE;
    else if (a->has_axe)
      a->tool_selected = TOOL_AXE;
    else
      a->tool_selected = TOOL_HAND;
  } else if (action == ACTION_FIRE) {
    a->tool_selected = (a->has_bow ? TOOL_BOW : TOOL_HAND);
  }
  if (action == ACTION_FIRE) {
    a->fire_latched = 1;
    a->fire_latch_timer = AGENT_FIRE_LATCH_TIME;
  }

  // --- execute ---
  float reward = 0.0f;

  Vector2 moveDir = {0, 0};
  switch (action) {
  case ACTION_UP:
    moveDir = (Vector2){0, -1};
    break;
  case ACTION_DOWN:
    moveDir = (Vector2){0, 1};
    break;
  case ACTION_LEFT:
    moveDir = (Vector2){-1, 0};
    break;
  case ACTION_RIGHT:
    moveDir = (Vector2){1, 0};
    break;

  case ACTION_ATTACK:
    // No targeting assist. Attack in facing direction.
    agent_try_attack_forward(a, tr, &reward);
    break;

  case ACTION_HARVEST:
    agent_try_harvest_forward(a, tr, &reward);
    break;

  /*
case ACTION_CRAFT:
  // agent_try_craft(a, tr, &reward);

  break;
  */
  case ACTION_CRAFT_AXE:
    if (!agent_in_base(a, tr)) {
      reward += -0.006f;
      return;
    }

    if (!a->has_axe && tr->wood >= 3 && tr->stone >= 2) {
      tr->wood -= 3;
      tr->stone -= 2;
      a->has_axe = true;
      a->last_craft_selected = TOOL_AXE;
      reward += 0.09f;
      return;
    }
    break;

  case ACTION_CRAFT_PICKAXE:
    break;

  case ACTION_CRAFT_SWORD:
    break;

  case ACTION_CRAFT_ARMOR:
    break;

  case ACTION_CRAFT_BOW:
    break;

  case ACTION_CRAFT_ARROWS:
    break;

  case ACTION_FIRE: {
    pthread_rwlock_rdlock(&c->lock);
    // agent_face_nearest_mob_in_chunk(a, c, cx, cy, 14.0f);
    pthread_rwlock_unlock(&c->lock);

    agent_try_fire_forward(a, &reward, false);
  } break;

  case ACTION_EAT:
    agent_try_eat(a, &reward);
    break;

  default:
    // If they do nothing, you can optionally punish "doing nothing"
    // reward += -0.002f;
    break;
  }

  // --- continuous fire: if latched, keep attempting to fire as soon as CD
  if (a->fire_latched) {
    if (action != ACTION_FIRE) {
      // auto-face while latched too (so it doesn't keep shooting the old
      // direction)
      pthread_rwlock_rdlock(&c->lock);
      // agent_face_nearest_mob_in_chunk(a, c, cx, cy, 14.0f);
      pthread_rwlock_unlock(&c->lock);

      agent_try_fire_forward(a, &reward, true);
    }
  }

  // Movement
  agent_try_move(a, moveDir);

  // Facing updates only when moved
  if (moveDir.x != 0.0f || moveDir.y != 0.0f) {
    agent_set_facing_from(moveDir, a);
    if (moveDir.x != 0.0f || moveDir.y != 0.0f)
      reward += R_WANDER_PENALTY;
  }

#if AGENT_FIRE_CANCEL_ON_MOVE
  if (moveDir.x != 0.0f || moveDir.y != 0.0f) {
    a->fire_latched = 0;
    a->fire_latch_timer = 0.0f;
  }
#endif

  // --- survival shaping ---
  if (a->health <= 0.0f) {
    a->health = 0.0f;
    a->alive = false;
    reward += R_DEATH;
  } else {
    reward += R_SURVIVE_PER_TICK;
  }

  a->reward_accumulator += reward;
  a->age++;
}

// ------------------------------------------------------------
// Nearest agent lookup (WORLD space) in a given chunk
// ------------------------------------------------------------
// ------------------------------------------------------------
// Nearest agent finder (same chunk only)
// ------------------------------------------------------------
static Agent *nearest_agent_in_chunk(int cx, int cy, Vector2 mob_world_pos,
                                     float *outDist) {
  Agent *best = NULL;
  float bestD = 1e9f;

  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &agents[i];
    if (!a->alive)
      continue;

    int acx = (int)(a->position.x / CHUNK_SIZE);
    int acy = (int)(a->position.y / CHUNK_SIZE);
    if (acx != cx || acy != cy)
      continue;

    float d = Vector2Distance(mob_world_pos, a->position);
    if (d < bestD) {
      bestD = d;
      best = a;
    }
  }

  if (outDist)
    *outDist = bestD;
  return best;
}

// Small helper: nearest base (pos + index)
static int nearest_base_idx(Vector2 wp, Vector2 *outPos, float *outDist) {
  int bestI = 0;
  float bestD = 1e9f;
  Vector2 bestP = tribes[0].base.position;

  for (int t = 0; t < TRIBE_COUNT; t++) {
    float d = Vector2Distance(wp, tribes[t].base.position);
    if (d < bestD) {
      bestD = d;
      bestI = t;
      bestP = tribes[t].base.position;
    }
  }

  if (outPos)
    *outPos = bestP;
  if (outDist)
    *outDist = bestD;
  return bestI;
}

// ------------------------------------------------------------
// Mob AI (rewrite)
//  - passive: wander, flee player/agents if close or recently hit
//  (aggro_timer)
//  - hostile: chase nearest player/agent in SAME chunk; otherwise drift to
//  nearest base
//  - hostile attacks: player/agent if in range, else base if close
//  - keeps mobs chunk-local (no cross-chunk migration)
// ------------------------------------------------------------
static void update_mob_ai(Mob *m, Vector2 chunk_origin, float dt) {
  if (!m || m->health <= 0)
    return;

  // ---- timers ----
  if (m->ai_timer > 0.0f)
    m->ai_timer -= dt;
  if (m->aggro_timer > 0.0f)
    m->aggro_timer -= dt;
  if (m->attack_cd > 0.0f)
    m->attack_cd -= dt;
  if (m->hurt_timer > 0.0f)
    m->hurt_timer -= dt;
  if (m->lunge_timer > 0.0f)
    m->lunge_timer -= dt;

  // ---- world pos ----
  Vector2 mw = Vector2Add(chunk_origin, m->position);

  int mcx = (int)(mw.x / CHUNK_SIZE);
  int mcy = (int)(mw.y / CHUNK_SIZE);

  // player chunk check (cheap)
  int pcx = (int)(player.position.x / CHUNK_SIZE);
  int pcy = (int)(player.position.y / CHUNK_SIZE);
  bool player_same_chunk = (pcx == mcx && pcy == mcy);

  // hostile?
  bool hostile = (m->type == MOB_ZOMBIE || m->type == MOB_SKELETON);

  // ---- wander direction refresh ----
  if (m->ai_timer <= 0.0f) {
    m->ai_timer = randf(0.35f, 1.25f);
    float ang = randf(0.0f, 2.0f * PI);
    m->vel = (Vector2){cosf(ang), sinf(ang)};
  }

  // ---- find nearest base ----
  Vector2 basePos = {0};
  float dBase = 0.0f;
  int baseIdx = nearest_base_idx(mw, &basePos, &dBase);

  Vector2 toBase = Vector2Subtract(basePos, mw);
  float baseLen = Vector2Length(toBase);
  Vector2 dirBase = (baseLen > 1e-3f) ? Vector2Scale(toBase, 1.0f / baseLen)
                                      : (Vector2){0, 0};

  // ---- find nearest agent in this chunk (hostiles care) ----
  float dA = 1e9f;
  Agent *targetA = nearest_agent_in_chunk(mcx, mcy, mw, &dA);

  // ---- compute player vector ----
  Vector2 toP = Vector2Subtract(player.position, mw);
  float dP = Vector2Length(toP);
  Vector2 dirP = (dP > 1e-3f) ? Vector2Scale(toP, 1.0f / dP) : (Vector2){0, 0};

  // ---- decide target (hostiles) ----
  bool player_targetable = player_same_chunk && (dP <= MOB_AGGRO_RANGE);
  bool agent_targetable = (targetA != NULL) && (dA <= MOB_AGGRO_RANGE);

  // choose nearer of (player, agent) if both exist
  int target_kind = 0; // 0 none, 1 player, 2 agent, 3 base
  Vector2 targetPos = mw;
  float targetDist = 1e9f;
  Vector2 targetDir = (Vector2){0, 0};

  if (hostile) {
    if (player_targetable && (!agent_targetable || dP <= dA)) {
      target_kind = 1;
      targetPos = player.position;
      targetDist = dP;
      targetDir = dirP;
    } else if (agent_targetable) {
      target_kind = 2;
      targetPos = targetA->position;
      targetDist = dA;

      Vector2 toA = Vector2Subtract(targetA->position, mw);
      float al = Vector2Length(toA);
      targetDir = (al > 1e-3f) ? Vector2Scale(toA, 1.0f / al) : (Vector2){0, 0};
    } else {
      // nobody in chunk -> pressure base (especially at night)
      target_kind = 3;
      targetPos = basePos;
      targetDist = dBase;
      targetDir = dirBase;
    }
  }

  // ---- speed + steering ----
  float speed = MOB_SPEED_PASSIVE;

  if (!hostile) {
    // Passive: flee if player/agent close OR recently hit (aggro_timer)
    bool scared = (m->aggro_timer > 0.0f);

    // if player close in same chunk, scared
    if (player_same_chunk && dP < 4.0f)
      scared = true;

    // also flee nearest agent if close (helps agents feel “real”)
    if (targetA && dA < 4.0f)
      scared = true;

    if (scared) {
      speed = MOB_SPEED_SCARED;

      // flee from the closer threat (player vs agent)
      Vector2 fleeDir = {0};
      if (player_same_chunk && (!targetA || dP <= dA)) {
        fleeDir = Vector2Scale(dirP, -1.0f);
      } else if (targetA) {
        Vector2 toA = Vector2Subtract(targetA->position, mw);
        float al = Vector2Length(toA);
        Vector2 dirA =
            (al > 1e-3f) ? Vector2Scale(toA, 1.0f / al) : (Vector2){0, 0};
        fleeDir = Vector2Scale(dirA, -1.0f);
      } else {
        // no known threat direction, just keep wander vel
        fleeDir = m->vel;
      }

      m->vel = fleeDir;
    } else {
      speed = MOB_SPEED_PASSIVE;
      // keep wander vel
    }
  } else {
    // Hostile: chase/pressure
    speed = MOB_SPEED_HOSTILE;

    // if targeting base but it's daytime, chill a bit
    if (!is_night_cached && target_kind == 3)
      speed *= 0.65f;

    // strong steering toward target
    if (target_kind != 0) {
      m->vel = targetDir;
    }
  }

  // normalize vel if needed
  float vlen = Vector2Length(m->vel);
  if (vlen > 1e-3f)
    m->vel = Vector2Scale(m->vel, 1.0f / vlen);

  // ---- attacks (hostiles only) ----
  if (hostile) {
    // attack player/agent if in range
    if (target_kind == 1 && player_same_chunk &&
        targetDist <= MOB_ATTACK_RANGE) {
      if (m->attack_cd <= 0.0f) {
        m->attack_cd = 0.80f;
        m->lunge_timer = 0.12f;

        player.health -= (float)PLAYER_TAKEN_DAMAGE;
        player_hurt_timer = 0.16f;
        cam_shake = fmaxf(cam_shake, 0.10f);
      }
    } else if (target_kind == 2 && targetA && targetDist <= MOB_ATTACK_RANGE) {
      if (m->attack_cd <= 0.0f) {
        m->attack_cd = 0.85f;
        m->lunge_timer = 0.12f;

        targetA->health -= (float)AGENT_TAKEN_DAMAGE;
        targetA->flash_timer = 0.16f;
        cam_shake = fmaxf(cam_shake, 0.08f);

        // keep them “aggro” after hitting
        m->aggro_timer = fmaxf(m->aggro_timer, 2.0f);
      }
    } else if (target_kind == 3) {
      // base pressure if close
      Tribe *tr = &tribes[baseIdx];
      float reach = tr->base.radius + 1.05f;
      if (dBase <= reach && m->attack_cd <= 0.0f) {
        m->attack_cd = 1.10f;
        m->lunge_timer = 0.10f;

        tr->integrity = fmaxf(0.0f, tr->integrity - 2.25f);
        cam_shake = fmaxf(cam_shake, 0.06f);
      }
    }
  }

  // ---- move (chunk-local; avoid crossing chunks) ----
  Vector2 step = Vector2Scale(m->vel, speed * dt);
  Vector2 nextW = Vector2Add(mw, step);

  // collision test against nearby generated things (simple)
  if (world_pos_blocked_nearby(mcx, mcy, nextW, mob_radius_world(m->type), mcx,
                               mcy)) {
    // bounce: flip direction and try a smaller step
    m->vel = Vector2Scale(m->vel, -1.0f);
    step = Vector2Scale(m->vel, speed * dt * 0.35f);
    nextW = Vector2Add(mw, step);
  }

  // convert to local position and clamp in chunk
  Vector2 nextLocal = Vector2Subtract(nextW, chunk_origin);

  // if trying to leave the chunk, bounce back inward
  if (nextLocal.x < 0.25f || nextLocal.x > (float)CHUNK_SIZE - 0.25f) {
    m->vel.x = -m->vel.x;
    nextLocal.x = clampf(nextLocal.x, 0.25f, (float)CHUNK_SIZE - 0.25f);
  }
  if (nextLocal.y < 0.25f || nextLocal.y > (float)CHUNK_SIZE - 0.25f) {
    m->vel.y = -m->vel.y;
    nextLocal.y = clampf(nextLocal.y, 0.25f, (float)CHUNK_SIZE - 0.25f);
  }

  m->position = nextLocal;
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
  if (dt <= 0.0f)
    dt = 1.0f / 60.0f;

  // --- cooldown timers ---
  if (player_harvest_cd > 0.0f)
    player_harvest_cd -= dt;
  if (player_attack_cd > 0.0f)
    player_attack_cd -= dt;
  if (player_fire_cd > 0.0f)
    player_fire_cd -= dt;
  if (player_hurt_timer > 0.0f)
    player_hurt_timer -= dt;

  // --- stamina regen (player) ---
  // tweak as you like:
  player.stamina = fminf(100.0f, player.stamina + STAMINA_REGEN_RATE * dt);

  // --- movement ---
  float speed = 0.6f;
  float move = speed; // (or speed * dt * 60.0f for true time-based)

  bool moving = false;
  if (IsKeyDown(KEY_W)) {
    player.position.y -= move;
    moving = true;
  }
  if (IsKeyDown(KEY_S)) {
    player.position.y += move;
    moving = true;
  }
  if (IsKeyDown(KEY_A)) {
    player.position.x -= move;
    moving = true;
  }
  if (IsKeyDown(KEY_D)) {
    player.position.x += move;
    moving = true;
  }
  // --- zoom controls ---
  if (IsKeyDown(KEY_EQUAL))
    target_world_scale += 60.0f * dt;
  if (IsKeyDown(KEY_MINUS))
    target_world_scale -= 60.0f * dt;
  target_world_scale = clampf(target_world_scale, 0.0f, 100.0f);

  // optional: tiny stamina drain while moving
  if (moving) {
    player.stamina = fmaxf(0.0f, player.stamina - STAMINA_DRAIN_RATE * dt);
  }

  // --- crafting toggle ---
  if (IsKeyPressed(KEY_TAB)) {
    crafting_open = !crafting_open;
  }

  // --- crafting input (1..9) only when crafting menu is open ---
  if (crafting_open) {
    for (int i = 0; i < recipe_count && i < 9; i++) {
      bool pressed = IsKeyPressed((KeyboardKey)(KEY_ONE + i));
      pressed = pressed || IsKeyPressed((KeyboardKey)(KEY_KP_1 + i));
      if (pressed)
        craft(&recipes[i]);
    }
  }

  // =========================
  // CONTINUOUS HOLD ACTIONS
  // =========================

  // (A) Continuous melee attack (hold LEFT mouse)
  if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
    int cx = (int)(player.position.x / CHUNK_SIZE);
    int cy = (int)(player.position.y / CHUNK_SIZE);
    Chunk *c = get_chunk(cx, cy);

    // player_try_attack_mob_in_chunk modifies mobs -> take write lock
    pthread_rwlock_wrlock(&c->lock);
    player_try_attack_(c, wrap(cx), wrap(cy));
    pthread_rwlock_unlock(&c->lock);
  }

  // (B) Continuous harvest/mine (hold RIGHT mouse)
  if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
    int cx = (int)(player.position.x / CHUNK_SIZE);
    int cy = (int)(player.position.y / CHUNK_SIZE);
    Chunk *c = get_chunk(cx, cy);

    // player_try_harvest_resource_in_chunk modifies resources -> write lock
    pthread_rwlock_wrlock(&c->lock);
    player_try_harvest_resource_in_chunk(c, wrap(cx), wrap(cy));
    pthread_rwlock_unlock(&c->lock);
  }

  // =========================
  // BOW CHARGE + RELEASE (F)
  // =========================
  // Hold F to charge, release F to fire (if enough charge).
  if (has_bow) {
    if (IsKeyDown(KEY_F)) {
      bow_charging = 1;
      bow_charge01 += dt / BOW_CHARGE_TIME;
      bow_charge01 = clamp01(bow_charge01);
    }

    // Release to fire (continuous-ready through player_fire_cd)
    if (bow_charging && IsKeyReleased(KEY_F)) {
      bow_charging = 0;

      if (player_fire_cd <= 0.0f && inv_arrows > 0 &&
          bow_charge01 >= BOW_CHARGE_MIN01) {
        // aim from player -> mouse in WORLD space
        Vector2 mouse = GetMousePosition();
        Vector2 mouse_world = {
            (mouse.x - SCREEN_WIDTH * 0.5f) / WORLD_SCALE + camera_pos.x,
            (mouse.y - SCREEN_HEIGHT * 0.5f) / WORLD_SCALE + camera_pos.y};

        Vector2 dir = Vector2Subtract(mouse_world, player.position);
        if (Vector2Length(dir) < 1e-3f)
          dir = (Vector2){1, 0};
        dir = Vector2Normalize(dir);

        // consume ammo + set cooldown
        inv_arrows--;
        player_fire_cd = PLAYER_FIRE_COOLDOWN;

        // uses your existing charged fire helper
        player_fire_bow_charged(dir, bow_charge01);
      }

      // reset charge after release regardless
      bow_charge01 = 0.0f;
    }
  } else {
    // if you don't have bow, ensure charge is off
    bow_charging = 0;
    bow_charge01 = 0.0f;
  }
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

static int is_dir_path(const char *path) {
  struct stat st;
  if (stat(path, &st) != 0)
    return 0;
  return S_ISDIR(st.st_mode);
}

static int is_file_path(const char *path) {
  struct stat st;
  if (stat(path, &st) != 0)
    return 0;
  return S_ISREG(st.st_mode);
}

// optional: only list folders that contain world.sav
static int world_dir_has_save(const char *world_name) {
  char p[256];
  make_save_file_path(p, sizeof(p), world_name);
  return is_file_path(p);
}

static void ensure_save_root(void) {
  struct stat st = {0};
  if (stat(SAVE_ROOT, &st) == -1) {
    if (mkdir(SAVE_ROOT, 0755) != 0 && errno != EEXIST) {
      fprintf(stderr, "mkdir(%s) failed: %s\n", SAVE_ROOT, strerror(errno));
    }
  }
}

static void world_list_refresh(WorldList *wl) {
  ensure_save_root();

  wl->count = 0;
  wl->selected = -1;
  wl->scroll = 0;

  DIR *d = opendir(SAVE_ROOT);
  if (!d)
    return;

  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    if (e->d_name[0] == '.')
      continue;

    char full[256];
    snprintf(full, sizeof(full), "%s/%s", SAVE_ROOT, e->d_name);

    if (!is_dir_path(full))
      continue;

    // If you want *every* folder listed, comment this out.
    if (!world_dir_has_save(e->d_name))
      continue;

    if (wl->count < MAX_WORLDS) {
      snprintf(wl->names[wl->count], WORLD_NAME_MAX, "%s", e->d_name);
      wl->count++;
    }
  }

  closedir(d);

  // simple sort (lexicographic)
  for (int i = 0; i < wl->count; i++) {
    for (int j = i + 1; j < wl->count; j++) {
      if (strcmp(wl->names[j], wl->names[i]) < 0) {
        char tmp[WORLD_NAME_MAX];
        strcpy(tmp, wl->names[i]);
        strcpy(wl->names[i], wl->names[j]);
        strcpy(wl->names[j], tmp);
      }
    }
  }

  if (wl->count > 0)
    wl->selected = 0;
}

static void world_list_ensure_valid(WorldList *wl) {
  if (wl->count <= 0) {
    wl->selected = -1;
    wl->scroll = 0;
    return;
  }
  if (wl->selected < 0)
    wl->selected = 0;
  if (wl->selected >= wl->count)
    wl->selected = wl->count - 1;
  if (wl->scroll < 0)
    wl->scroll = 0;
  if (wl->scroll > wl->count - 1)
    wl->scroll = wl->count - 1;
}

// =======================
// DELETE WORLD (rm -r saves/<world>)
// =======================
static int delete_dir_recursive(const char *path) {
  DIR *d = opendir(path);
  if (!d) {
    // If it doesn't exist, treat as success-ish
    return (errno == ENOENT) ? 1 : 0;
  }

  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, ".."))
      continue;

    char child[512];
    snprintf(child, sizeof(child), "%s/%s", path, e->d_name);

    struct stat st;
    if (stat(child, &st) != 0)
      continue;

    if (S_ISDIR(st.st_mode)) {
      if (!delete_dir_recursive(child)) {
        closedir(d);
        return 0;
      }
      if (rmdir(child) != 0) {
        closedir(d);
        return 0;
      }
    } else {
      if (remove(child) != 0) {
        closedir(d);
        return 0;
      }
    }
  }

  closedir(d);
  // remove the top directory
  if (rmdir(path) != 0)
    return 0;
  return 1;
}

static int delete_world_by_name(const char *world_name) {
  char world_dir[256];
  make_world_path(world_dir, sizeof(world_dir), world_name);
  return delete_dir_recursive(world_dir);
}

static void draw_pause_overlay(void) {
  DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, (Color){0, 0, 0, 160});
  const char *title = "PAUSED";
  int fs = 52;
  int tw = MeasureText(title, fs);
  DrawText(title, (SCREEN_WIDTH - tw) / 2, (int)(SCREEN_HEIGHT * 0.18f), fs,
           RAYWHITE);
}

// tiny button helper
static int ui_button(Rectangle r, const char *text) {
  Vector2 m = GetMousePosition();
  int hot = CheckCollisionPointRec(m, r);
  Color bg = hot ? (Color){70, 70, 90, 255} : (Color){50, 50, 70, 255};
  DrawRectangleRounded(r, 0.25f, 8, bg);
  DrawRectangleRoundedLines(r, 0.25f, 8, (Color){0, 0, 0, 160});
  int fs = 20;
  int tw = MeasureText(text, fs);
  DrawText(text, (int)(r.x + (r.width - tw) / 2),
           (int)(r.y + (r.height - fs) / 2), fs, RAYWHITE);
  return hot && IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
}

static void ui_textbox(Rectangle r, char *buf, int cap, int *active,
                       int digits_only) {
  Vector2 m = GetMousePosition();
  int hot = CheckCollisionPointRec(m, r);
  if (hot && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
    *active = 1;
  if (!hot && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
    *active = 0;

  Color bg = *active ? (Color){35, 35, 45, 255} : (Color){25, 25, 35, 255};
  DrawRectangleRounded(r, 0.2f, 8, bg);
  DrawRectangleRoundedLines(r, 0.2f, 8, (Color){0, 0, 0, 170});

  // input
  if (*active) {
    int key = GetCharPressed();
    while (key > 0) {
      int len = (int)strlen(buf);
      if (key == 32 || (key >= 33 && key <= 126)) {
        if (digits_only && !(key >= '0' && key <= '9')) {
          key = GetCharPressed();
          continue;
        }
        if (len < cap - 1) {
          buf[len] = (char)key;
          buf[len + 1] = 0;
        }
      }
      key = GetCharPressed();
    }
    if (IsKeyPressed(KEY_BACKSPACE)) {
      int len = (int)strlen(buf);
      if (len > 0)
        buf[len - 1] = 0;
    }
  }

  DrawText(buf, (int)r.x + 10, (int)r.y + 10, 20, RAYWHITE);
}

static void do_pause_menu(void) {
  draw_pause_overlay();

  float cx = SCREEN_WIDTH * 0.5f;
  float y = SCREEN_HEIGHT * 0.35f;

  Rectangle rResume = {cx - 140, y + 0, 280, 54};
  Rectangle rSave = {cx - 140, y + 70, 280, 54};
  Rectangle rExit = {cx - 140, y + 140, 280, 54};

  if (ui_button(rResume, "Resume (ESC)")) {
    g_state = STATE_PLAYING;
  }

  if (ui_button(rSave, "Save World")) {
    save_world_to_disk(g_world_name);
    save_models_to_disk(g_world_name);
  }

  if (ui_button(rExit, "Exit to World Select")) {
    // Optional: save before leaving
    save_world_to_disk(g_world_name);
    save_models_to_disk(g_world_name);

    world_list_refresh(&g_world_list);
    g_state = STATE_WORLD_SELECT;
  }

  DrawText("Tip: ESC toggles pause", (int)(cx - 160), (int)(y + 220), 18,
           RAYWHITE);
}

static void save_current_world_session(void) {
  if (g_world_name[0] == '\0')
    return;
  save_world_to_disk(g_world_name);
  save_models_to_disk(g_world_name);
}

static void do_world_select_screen(void) {
  // Refresh list on first entry or when empty
  static int initialized = 0;
  if (!initialized) {
    world_list_refresh(&g_world_list);
    initialized = 1;
  }

  world_list_ensure_valid(&g_world_list);

  DrawRectangle(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, (Color){18, 18, 28, 255});
  DrawText("Select World", 40, 30, 44, RAYWHITE);

  Rectangle listBox = {40, 100, 520, (float)SCREEN_HEIGHT - 180};
  DrawRectangleRounded(listBox, 0.12f, 8, (Color){25, 25, 40, 255});
  DrawRectangleRoundedLines(listBox, 0.12f, 8, (Color){0, 0, 0, 160});

  int itemH = 44;
  int visible = (int)(listBox.height / itemH);
  if (visible < 1)
    visible = 1;

  // mouse wheel scroll
  float wheel = GetMouseWheelMove();
  if (wheel != 0.0f && g_world_list.count > 0) {
    g_world_list.scroll -= (int)wheel;
    if (g_world_list.scroll < 0)
      g_world_list.scroll = 0;
    int maxScroll =
        (g_world_list.count > visible) ? (g_world_list.count - visible) : 0;
    if (g_world_list.scroll > maxScroll)
      g_world_list.scroll = maxScroll;
  }

  // keyboard nav
  if (IsKeyPressed(KEY_UP) && g_world_list.selected > 0)
    g_world_list.selected--;
  if (IsKeyPressed(KEY_DOWN) && g_world_list.selected < g_world_list.count - 1)
    g_world_list.selected++;

  // keep selected in view
  if (g_world_list.selected >= 0) {
    if (g_world_list.selected < g_world_list.scroll)
      g_world_list.scroll = g_world_list.selected;
    if (g_world_list.selected >= g_world_list.scroll + visible)
      g_world_list.scroll = g_world_list.selected - visible + 1;
  }

  // draw items
  for (int i = 0; i < visible; i++) {
    int idx = g_world_list.scroll + i;
    if (idx >= g_world_list.count)
      break;

    Rectangle row = {listBox.x + 10, listBox.y + 10 + i * itemH,
                     listBox.width - 20, (float)itemH - 6};
    int hot = CheckCollisionPointRec(GetMousePosition(), row);

    Color bg = (Color){35, 35, 55, 255};
    if (idx == g_world_list.selected)
      bg = (Color){60, 60, 95, 255};
    else if (hot)
      bg = (Color){45, 45, 70, 255};

    DrawRectangleRounded(row, 0.18f, 8, bg);

    DrawText(g_world_list.names[idx], (int)(row.x + 12), (int)(row.y + 10), 22,
             RAYWHITE);

    if (hot && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
      g_world_list.selected = idx;
    }
  }

  Rectangle rPlay = {600, 140, 260, 54};
  Rectangle rDelete = {600, 210, 260, 54};
  Rectangle rCreate = {600, 280, 260, 54};
  Rectangle rBack = {600, 350, 260, 54};

  int hasSelection = (g_world_list.selected >= 0 &&
                      g_world_list.selected < g_world_list.count);

  if (ui_button(rPlay, hasSelection ? "Play Selected" : "Play (no world)")) {
    if (hasSelection) {
      snprintf(g_world_name, sizeof(g_world_name), "%s",
               g_world_list.names[g_world_list.selected]);

      // load
      if (!load_world_from_disk(g_world_name)) {
        // if load fails, create fresh using current seed
        world_reset(g_world_seed);
        save_world_to_disk(g_world_name);
        save_models_to_disk(g_world_name);
      } else {
        load_models_from_disk(g_world_name);
      }
      g_state = STATE_PLAYING;
    }
  }

  if (ui_button(rDelete, hasSelection ? "Delete World" : "Delete (no world)")) {
    if (hasSelection) {
      const char *wname = g_world_list.names[g_world_list.selected];
      delete_world_by_name(wname);

      world_list_refresh(&g_world_list);
      world_list_ensure_valid(&g_world_list);
    }
  }

  if (ui_button(rCreate, "Create New World")) {
    // go to your create UI
    g_state = STATE_WORLD_CREATE;
  }

  if (ui_button(rBack, "Back")) {
    g_state = STATE_TITLE;
  }

  // Enter to play
  if (hasSelection && IsKeyPressed(KEY_ENTER)) {
    snprintf(g_world_name, sizeof(g_world_name), "%s",
             g_world_list.names[g_world_list.selected]);
    if (!load_world_from_disk(g_world_name)) {
      world_reset(g_world_seed);
      save_world_to_disk(g_world_name);
    }
    g_state = STATE_PLAYING;
  }
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
    // Wait for a job batch to become active (or quit)
    pthread_mutex_lock(&job_mtx);
    while (!job_active && !job_quit) {
      pthread_cond_wait(&job_cv, &job_mtx);
    }
    if (job_quit) {
      pthread_mutex_unlock(&job_mtx);
      break;
    }
    pthread_mutex_unlock(&job_mtx);

    // Work loop: grab next agent index atomically under mutex
    for (;;) {
      int idx;

      pthread_mutex_lock(&job_mtx);
      idx = job_next_agent++;
      pthread_mutex_unlock(&job_mtx);

      if (idx >= MAX_AGENTS)
        break;

      update_agent(&agents[idx]);
    }

    // Signal completion for this worker
    pthread_mutex_lock(&job_mtx);
    job_done_workers++;

    if (job_done_workers >= WORKER_COUNT) {
      job_active = 0;                // batch finished
      pthread_cond_signal(&done_cv); // wake main thread
    }
    pthread_mutex_unlock(&job_mtx);
  }

  return NULL;
}

static void start_workers(void) {
  pthread_mutex_lock(&job_mtx);
  job_quit = 0;
  job_active = 0;
  job_next_agent = 0;
  job_done_workers = 0;
  pthread_mutex_unlock(&job_mtx);

  for (int i = 0; i < WORKER_COUNT; i++) {
    pthread_create(&workers[i], NULL, agent_worker, NULL);
  }
}

static void stop_workers(void) {
  pthread_mutex_lock(&job_mtx);
  job_quit = 1;
  pthread_cond_broadcast(&job_cv);
  pthread_mutex_unlock(&job_mtx);

  for (int i = 0; i < WORKER_COUNT; i++) {
    pthread_join(workers[i], NULL);
  }
}

/* =======================
   MAIN
======================= */
int main(void) {
  srand(time(NULL));

  InitWindow(1280, 800, "MUZE Tribal Simulation");
  SetExitKey(KEY_NULL); // <- ESC will NOT close the window anymore
  SCREEN_WIDTH = GetScreenWidth();
  SCREEN_HEIGHT = GetScreenHeight();
  TILE_SIZE = SCREEN_HEIGHT / 18.0f;
  SetTargetFPS(60);

  init_tribes();
  init_agents();
  init_player();

  start_workers();

  for (int x = 0; x < WORLD_SIZE; x++) {
    for (int y = 0; y < WORLD_SIZE; y++) {
      pthread_rwlock_init(&world[x][y].lock, NULL);
      world[x][y].generated = false;
      world[x][y].resource_count = 0;
      world[x][y].mob_spawn_timer = 0.0f;
    }
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
    collect_nearby_pickups();

    update_visible_world(dt);

    g_dt = dt;
    run_agent_jobs();

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

    BeginDrawing();
    ClearBackground(BLACK);

    if (g_state == STATE_PLAYING) {
      // ---- your current game draw/update ----
      // update_daynight(dt);
      // update agents/mobs, draw_chunks/resources/mobs/player etc

      // quick save hotkey
      if (IsKeyPressed(KEY_F5))
        save_world_to_disk(g_world_name);
      if (IsKeyPressed(KEY_P))
        g_state = STATE_PAUSED;
    } else if (g_state == STATE_TITLE) {

      DrawText("SAMCRAFT", 40, 40, 52, RAYWHITE);
      DrawText("F5 = Save while playing", 44, 100, 18,
               (Color){200, 200, 200, 180});

      Rectangle b1 = (Rectangle){60, 160, 260, 50};
      Rectangle b2 = (Rectangle){60, 220, 260, 50};
      Rectangle b3 = (Rectangle){60, 280, 260, 50};

      if (ui_button(b1, "Play (Load/Select)"))
        g_state = STATE_WORLD_SELECT;
      if (ui_button(b2, "Create World"))
        g_state = STATE_WORLD_CREATE;
      if (ui_button(b3, "Quit"))
        CloseWindow();
    } else if (g_state == STATE_WORLD_CREATE) {

      DrawText("Create World", 60, 50, 34, RAYWHITE);

      DrawText("World Name", 60, 120, 18, RAYWHITE);
      ui_textbox((Rectangle){60, 145, 360, 45}, g_world_name,
                 sizeof(g_world_name), &g_typing_name, 0);

      DrawText("Seed", 60, 205, 18, RAYWHITE);
      ui_textbox((Rectangle){60, 230, 200, 45}, g_seed_text,
                 sizeof(g_seed_text), &g_typing_seed, 1);

      if (ui_button((Rectangle){60, 300, 200, 50}, "Create & Play")) {
        g_world_seed = (uint32_t)strtoul(g_seed_text, NULL, 10);
        world_reset(g_world_seed);
        save_world_to_disk(g_world_name); // create initial save
        g_state = STATE_PLAYING;
      }

      if (ui_button((Rectangle){280, 300, 140, 50}, "Back")) {
        g_state = STATE_TITLE;
      }
    } else if (g_state == STATE_WORLD_SELECT) {
      DrawText("Select World", 60, 50, 34, RAYWHITE);
      DrawText("(This screen next: list saves/ folders)", 60, 95, 18,
               (Color){200, 200, 200, 180});

      // For now: quick load the current name
      if (ui_button((Rectangle){60, 140, 260, 50}, "Load World Name")) {
        if (load_world_from_disk(g_world_name))
          load_models_from_disk(g_world_name);
        g_state = STATE_PLAYING;
      }

      if (ui_button((Rectangle){60, 200, 260, 50}, "Back"))
        g_state = STATE_TITLE;
    }

    draw_chunks();
    draw_resources();
    draw_mobs();
    draw_projectiles();
    draw_pickups();

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
      draw_agent(&agents[i], ap, tc);
    }

    // player
    Vector2 pp = world_to_screen(player.position);
    draw_player(pp);
    draw_bow_charge_fx();

    // UI + debug
    draw_ui();
    draw_hover_label();
    draw_minimap();
    draw_daynight_overlay(); // AFTER world draw, before EndDrawing
    draw_hurt_vignette();
    draw_crafting_ui();

    DrawText("MUZE Tribal Simulation", 20, 160, 20, RAYWHITE);
    DrawText(TextFormat("FPS: %d", GetFPS()), 20, 185, 20, RAYWHITE);

    EndDrawing();
  }

  stop_workers();

  CloseWindow();
  return 0;
}
