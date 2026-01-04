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
#define MAX_BASE_PARTICLES 32

#define HARVEST_DISTANCE 1.0f
#define HARVEST_AMOUNT 1
#define ATTACK_DISTANCE 1.0f
#define ATTACK_DAMAGE 1

#define TRAIN_INTERVAL 1

int SCREEN_WIDTH;
int SCREEN_HEIGHT;
float TILE_SIZE;

Color pink = (Color){255, 192, 203, 255};
Color yellow = (Color){255, 255, 0, 255};
Color gray = (Color){100, 100, 100, 255};

/* =======================
   ENUMS
======================= */

typedef enum {
  TOOL_HAND = 0,
  TOOL_AXE,
  TOOL_PICKAXE,
  TOOL_SHOVEL,
  TOOL_NONE,
  TOOL_COUNT
} ToolType;

typedef enum {
  RES_TREE = 0,
  RES_ROCK,
  RES_GOLD,
  RES_FOOD,
  RES_NONE,
  RES_COUNT
} ResourceType;

typedef enum {
  MOB_PIG = 0,
  MOB_SHEEP,
  MOB_SKELETON,
  MOB_ZOMBIE,
  MOB_NONE,
  MOB_COUNT
} MobType;

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
  int health;
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
  float reward_accumulator;
  int age;
  int steps_alive;
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
  float hand_angle;
  float hand_swing;
} Player;

typedef struct {
  Vector2 position;
  float radius;
} Base;

typedef struct {
  Vector2 pos;
  float lifetime;
} BaseParticle;

typedef struct {
  int tribe_id;
  Color color;

  Base base;

  int agent_start;
  int agent_count;

  // future
  // int wood, stone, food;
  // MuSharedMemory *collective_memory;
} Tribe;

/* =======================
   GLOBAL STATE
======================= */

Chunk world[WORLD_SIZE][WORLD_SIZE];
Player player;
Base agent_base;
BaseParticle base_particles[MAX_BASE_PARTICLES];
Tribe tribes[TRIBE_COUNT];

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

  for (int i = 0; i < MAX_MOBS; i++) {
    MobType type = rand() % MOB_COUNT;
    printf("%d\n", type);
    c->mobs[i].type = type;
    c->mobs[i].position = (Vector2){rand() % CHUNK_SIZE, rand() % CHUNK_SIZE};
    c->mobs[i].health = 100;
  }

  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &c->agents[i];
    a->alive = true;
    a->health = a->stamina = 100;
    a->flash_timer = 0;
    a->agent_id = i;
    a->reward_accumulator = 0;
    a->age = 0;
    a->steps_alive = 0;

    int tribe_id = i / AGENT_PER_TRIBE;
    Tribe *tr = &tribes[tribe_id];

    a->tribe_color = tr->color;
    a->agent_id = i;

    int tribe_id = i / AGENT_PER_TRIBE;
    Tribe *tr = &tribes[tribe_id];

    if (cx == (int)tr->base.position.x && cy == (int)tr->base.position.y) {

      float ang = randf(0, 2 * PI);
      float d = randf(2, tr->base.radius - 1);

      a->position = (Vector2){tr->base.position.x + cosf(ang) * d,
                              tr->base.position.y + sinf(ang) * d};
    }
    float ang = (float)i / MAX_AGENTS * 2 * PI;
    float d = randf(2, BASE_RADIUS - 1);
    a->position = (Vector2){agent_base.position.x + cosf(ang) * d,
                            agent_base.position.y + sinf(ang) * d};
  }
  else {
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
   HARVESTING
======================= */
void agent_harvest(Agent *a, Chunk *c) {
  for (int i = 0; i < c->resource_count; i++) {
    Resource *r = &c->resources[i];
    if (r->type == RES_NONE)
      continue;

    float dist = Vector2Distance(a->position, r->position);
    if (dist < HARVEST_DISTANCE) {
      // Harvest amount can depend on tool
      int amount = HARVEST_AMOUNT;
      if ((a->tribe_color.r & a->tribe_color.g & a->tribe_color.b) ==
          (RED.r & RED.g & RED.b))
        amount *= 1; // example for tribe bonuses

      r->health -= amount;
      r->visited = true; // flash effect

      // Reward agent
      a->reward_accumulator += 0.05f;

      // If resource depleted
      if (r->health <= 0) {
        r->type = RES_NONE;
        a->reward_accumulator += 0.1f; // extra reward
      }

      break; // harvest one resource per step
    }
  }
}

/* =======================
   ATTACK
======================= */
void agent_attack(Agent *a, Chunk *c) {
  for (int i = 0; i < MAX_MOBS; i++) {
    Mob *m = &c->mobs[i];
    if (m->health <= 0)
      continue;

    float dist = Vector2Distance(a->position, m->position);
    if (dist < ATTACK_DISTANCE) {
      m->health -= ATTACK_DAMAGE;
      a->reward_accumulator += 0.05f;

      if (m->health <= 0) {
        m->health = 0;
        a->reward_accumulator += 0.1f; // bonus for killing
      }

      break; // attack one mob per step
    }
  }
}

void respawn_agent(Agent *a) {
  a->alive = true;
  a->health = a->stamina = 100;
  a->position = agent_base.position;
  a->reward_accumulator = 0;
  a->age = 0;
  a->steps_alive = 0;

  mu_model_reset_episode(a->brain);
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
  // --- Movement ---
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

  // --- Face mouse ---
  Vector2 mouse_screen = GetMousePosition();
  Camera2D cam = {0};
  cam.zoom = 1.0f;
  cam.offset = (Vector2){SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2};
  cam.target = player.position;
  Vector2 mouse_world = GetScreenToWorld2D(mouse_screen, cam);

  Vector2 dir = Vector2Subtract(mouse_world, player.position);
  player.hand_angle = atan2f(dir.y, dir.x);

  // --- Mouse actions ---
  int cx = (int)(player.position.x / (CHUNK_SIZE * TILE_SIZE));
  int cy = (int)(player.position.y / (CHUNK_SIZE * TILE_SIZE));
  Chunk *c = get_chunk(cx, cy);

  bool action_performed = false;

  // Left click = attack
  if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
    for (int i = 0; i < MAX_MOBS; i++) {
      Mob *m = &c->mobs[i];
      if (m->health <= 0)
        continue;

      float dist = Vector2Distance(player.position, m->position) * TILE_SIZE;
      if (dist < ATTACK_DISTANCE * TILE_SIZE) {
        m->health -= ATTACK_DAMAGE;
        m->visited = true; // shake
        action_performed = true;
        break;
      }
    }
  }

  // Right click = harvest
  if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
    for (int i = 0; i < c->resource_count; i++) {
      Resource *r = &c->resources[i];
      if (r->type == RES_NONE)
        continue;

      float dist = Vector2Distance(player.position, r->position) * TILE_SIZE;
      if (dist < HARVEST_DISTANCE * TILE_SIZE) {
        r->health -= HARVEST_AMOUNT;
        r->visited = true; // shake
        if (r->health <= 0)
          r->type = RES_NONE;
        action_performed = true;
        break;
      }
    }
  }

  // --- Swing animation ---
  if (action_performed)
    player.hand_swing = 0.2f; // duration of swing
  if (player.hand_swing > 0)
    player.hand_swing -= GetFrameTime();
  else
    player.hand_swing = 0;
}

void draw_player(Player *p) {
  // --- Body ---
  DrawCircleV(p->position, TILE_SIZE * 0.35f, (Color){245, 222, 179, 255});

  // --- Swing amplitude ---
  float swing_offset = sinf((0.2f - p->hand_swing) * 10.0f) * 0.3f;

  // --- Left hand points exactly to mouse ---
  Vector2 left_hand = {
      p->position.x +
          cosf(p->hand_angle) * (TILE_SIZE * 0.45f + swing_offset * TILE_SIZE),
      p->position.y +
          sinf(p->hand_angle) * (TILE_SIZE * 0.45f + swing_offset * TILE_SIZE)};

  // --- Right hand swings slightly opposite ---
  Vector2 right_hand = {
      p->position.x + cosf(p->hand_angle + PI) *
                          (TILE_SIZE * 0.45f - swing_offset * TILE_SIZE),
      p->position.y + sinf(p->hand_angle + PI) *
                          (TILE_SIZE * 0.45f - swing_offset * TILE_SIZE)};

  DrawCircleV(left_hand, TILE_SIZE * 0.15f, RED);
  DrawCircleV(right_hand, TILE_SIZE * 0.15f, RED);
}

/* =======================
   TRIBES
======================= */
void init_tribes(void) {
  float spacing = 24.0f;

  Color tribe_colors[] = {RED, BLUE, GREEN, ORANGE};

  for (int t = 0; t < TRIBE_COUNT; t++) {
    Tribe *tr = &tribes[t];

    tr->tribe_id = t;
    tr->color = tribe_colors[t % 4];
    tr->agent_start = t * AGENT_PER_TRIBE;
    tr->agent_count = AGENT_PER_TRIBE;

    tr->base.position =
        (Vector2){WORLD_SIZE / 2 + cosf(t * 2 * PI / TRIBE_COUNT) * spacing,
                  WORLD_SIZE / 2 + sinf(t * 2 * PI / TRIBE_COUNT) * spacing};

    tr->base.radius = BASE_RADIUS;
  }
}

/* =======================
   AGENTS
======================= */

MCTSParams mcts_params = {.num_simulations = 40,
                          .c_puct = 1.2f,
                          .discount = 0.95f,
                          .temperature = 1.0f};

void encode_observation(Agent *a, Chunk *c, float *obs) {
  // --- Health & stamina ---
  obs[0] = a->health / 100.0f;
  obs[1] = a->stamina / 100.0f;

  // --- Base relation ---
  Vector2 to_base = Vector2Subtract(agent_base.position, a->position);
  float base_dist = Vector2Length(to_base);

  obs[2] = fminf(base_dist / (BASE_RADIUS * 4.0f), 1.0f);
  obs[3] = (base_dist > 0) ? to_base.x / base_dist : 0;
  obs[4] = (base_dist > 0) ? to_base.y / base_dist : 0;

  // --- Nearest resource ---
  float nearest = 9999.0f;
  Vector2 nearest_dir = {0, 0};

  for (int i = 0; i < c->resource_count; i++) {
    Resource *r = &c->resources[i];
    Vector2 delta = Vector2Subtract(r->position, a->position);
    float d = Vector2Length(delta);

    if (d < nearest) {
      nearest = d;
      nearest_dir = delta;
    }
  }

  if (nearest < 9999.0f) {
    obs[5] = fminf(nearest / (float)CHUNK_SIZE, 1.0f);
    obs[6] = nearest_dir.x / (nearest + 0.0001f);
    obs[7] = nearest_dir.y / (nearest + 0.0001f);
  } else {
    obs[5] = 1.0f;
    obs[6] = 0.0f;
    obs[7] = 0.0f;
  }

  // --- Inside base ---
  obs[8] = (base_dist < BASE_RADIUS) ? 1.0f : 0.0f;

  // --- Bias / time ---
  obs[9] = 1.0f;
}

float compute_reward(Agent *a, Chunk *c, float *obs) {
  float r = 0.0f;

  // survival incentive
  r += 0.001f;

  // low health penalty
  if (a->health < 30)
    r -= 0.02f;

  // inside base = healing
  if (obs[8] > 0.5f)
    r += 0.05f;

  // encourage resource seeking
  r += (1.0f - obs[5]) * 0.01f;

  return r;
}

int decide_action(Agent *a, float *obs) {
  if (!a || !a->brain)
    return rand() % ACTION_COUNT;

  MCTSResult r = mcts_run(a->brain, obs, &mcts_params);
  int act = r.chosen_action;
  mcts_result_free(&r);

  return (act >= 0 && act < ACTION_COUNT) ? act : rand() % ACTION_COUNT;
}

void update_agent(Agent *a, Chunk *c) {
  if (!a->alive)
    return;

  float obs[a->input_size];
  encode_observation(a, c, obs);

  int action = decide_action(a, obs);

  // --- apply action ---
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
  case ACTION_HARVEST:
    agent_harvest(a, c);
    break;
  case ACTION_ATTACK:
    agent_attack(a, c);
    break;
  default:
    break;
  }

  // --- Base healing ---
  float dist = Vector2Distance(a->position, agent_base.position);
  if (dist < BASE_RADIUS) {
    a->health = fminf(a->health + 0.5f, 100);
    a->stamina = fminf(a->stamina + 0.5f, 100);
    a->flash_timer += 0.1f;
  } else {
    a->flash_timer = 0;
    a->stamina -= 0.05f;
  }

  // --- Death ---
  if (a->stamina <= 0 || a->health <= 0) {
    a->alive = false;
    mu_model_end_episode(a->brain, -1.0f);
    return;
  }

  // --- Reward ---
  float reward = compute_reward(a, c, obs);
  a->reward_accumulator += reward;

  mu_model_step(a->brain, obs, action, reward);

  a->age++;
  a->steps_alive++;
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
   RESOURCES
======================= */
void draw_resource(Resource *r) {
  Vector2 pos = Vector2Scale(r->position, TILE_SIZE);

  // shake effect
  if (r->visited) {
    pos.x += rand() % 3 - 1; // small shake
    pos.y += rand() % 3 - 1;
  }

  float size_multiplier = 1.0f;

  switch (r->type) {
  case RES_TREE:
    size_multiplier = 3.0f;
    DrawRectangleV((Vector2){pos.x - TILE_SIZE * 0.15f, pos.y},
                   (Vector2){TILE_SIZE * 0.3f, TILE_SIZE * 0.7f},
                   (Color){101, 67, 33, 255});
    DrawCircleV((Vector2){pos.x, pos.y - TILE_SIZE * 0.35f}, TILE_SIZE * 0.5f,
                GREEN);
    break;
  case RES_ROCK:
    size_multiplier = 1.5f;
    DrawCircleV(pos, TILE_SIZE * 0.35f, GRAY);
    break;
  case RES_GOLD:
    DrawCircleV(pos, TILE_SIZE * 0.25f, YELLOW);
    break;
  case RES_FOOD:
    DrawCircleV(pos, TILE_SIZE * 0.2f, RED);
    break;
  default:
    break;
  }

  if (r->visited)
    r->visited = false; // reset after flash
}

/* =======================
  MOBS
======================= */
/* =======================
  MOBS
======================= */
void draw_pig(Vector2 pos, float size) {
  // Body
  DrawCircleV(pos, size * 0.5f, pink);

  // Head
  DrawCircleV((Vector2){pos.x, pos.y - size * 0.4f}, size * 0.3f, pink);

  // Snout
  DrawCircleV((Vector2){pos.x, pos.y - size * 0.4f}, size * 0.15f,
              (Color){255, 160, 160, 255});

  // Legs
  DrawRectangleV((Vector2){pos.x - size * 0.35f, pos.y + size * 0.3f},
                 (Vector2){size * 0.2f, size * 0.2f}, pink);
  DrawRectangleV((Vector2){pos.x + size * 0.15f, pos.y + size * 0.3f},
                 (Vector2){size * 0.2f, size * 0.2f}, pink);
}

void draw_sheep(Vector2 pos, float size) {
  DrawCircleV(pos, size * 0.5f, Fade(WHITE, 0.5f));
  DrawCircleV((Vector2){pos.x, pos.y - size * 0.4f}, size * 0.3f,
              Fade(WHITE, 0.5f));
  DrawRectangleV((Vector2){pos.x - size * 0.35f, pos.y + size * 0.3f},
                 (Vector2){size * 0.2f, size * 0.2f}, Fade(WHITE, 0.5f));
  DrawRectangleV((Vector2){pos.x + size * 0.15f, pos.y + size * 0.3f},
                 (Vector2){size * 0.2f, size * 0.2f}, Fade(WHITE, 0.5f));
}

void draw_skeleton(Vector2 pos, float size) {
  DrawCircleV(pos, size * 0.5f, WHITE); // Body
  DrawCircleV((Vector2){pos.x, pos.y - size * 0.4f}, size * 0.3f,
              WHITE); // Head
  // Bones (lines)
  DrawLine(pos.x - size * 0.25f, pos.y, pos.x + size * 0.25f, pos.y, GRAY);
  DrawLine(pos.x, pos.y - size * 0.25f, pos.x, pos.y + size * 0.25f, GRAY);
}

void draw_zombie(Vector2 pos, float size) {
  DrawRectangleV((Vector2){pos.x - size * 0.4f, pos.y - size * 0.4f},
                 (Vector2){size * 0.8f, size * 0.8f}, GREEN); // Body
  DrawRectangleV((Vector2){pos.x - size * 0.2f, pos.y - size * 0.6f},
                 (Vector2){size * 0.4f, size * 0.2f}, GREEN); // Head
  DrawRectangleV((Vector2){pos.x - size * 0.35f, pos.y + size * 0.3f},
                 (Vector2){size * 0.15f, size * 0.3f}, GREEN); // Left leg
  DrawRectangleV((Vector2){pos.x + size * 0.2f, pos.y + size * 0.3f},
                 (Vector2){size * 0.15f, size * 0.3f}, GREEN); // Right leg
}

void draw_mob(Mob *m, Vector2 chunk_offset) {
  Vector2 p = Vector2Add(Vector2Scale(m->position, TILE_SIZE), chunk_offset);

  // shake effect
  if (m->visited) {
    p.x += rand() % 3 - 1;
    p.y += rand() % 3 - 1;
    m->visited = false;
  }

  switch (m->type) {
  case MOB_PIG:
    draw_pig(p, TILE_SIZE * 0.8f);
    break;
  case MOB_SHEEP:
    draw_sheep(p, TILE_SIZE * 0.8f);
    break;
  case MOB_SKELETON:
    draw_skeleton(p, TILE_SIZE * 0.8f);
    break;
  case MOB_ZOMBIE:
    draw_zombie(p, TILE_SIZE * 0.8f);
    break;
  default:
    break;
  }
}

void draw_agent(Agent *a, Vector2 chunk_offset) {
  if (!a->alive)
    return;

  Vector2 p = Vector2Add(Vector2Scale(a->position, TILE_SIZE), chunk_offset);

  DrawCircleV(p, TILE_SIZE * 0.35f, a->tribe_color);
  if (a->flash_timer > 0)
    DrawCircleV(p, TILE_SIZE * 0.25f, Fade(WHITE, 0.6f));
}

void draw_chunk(Chunk *c, int cx, int cy) {
  Vector2 offset = {cx * CHUNK_SIZE * TILE_SIZE, cy * CHUNK_SIZE * TILE_SIZE};

  // Terrain tiles
  for (int i = 0; i < CHUNK_SIZE; i++)
    for (int j = 0; j < CHUNK_SIZE; j++)
      DrawRectangle(offset.x + i * TILE_SIZE, offset.y + j * TILE_SIZE,
                    TILE_SIZE, TILE_SIZE, biome_color(c->terrain[i][j]));

  // Resources
  for (int i = 0; i < c->resource_count; i++)
    draw_resource(&c->resources[i]);

  // Mobs
  for (int i = 0; i < MAX_MOBS; i++)
    draw_mob(&c->mobs[i], offset);

  // Agents
  for (int i = 0; i < MAX_AGENTS; i++)
    draw_agent(&c->agents[i], offset);
}

/* =======================
   UI
======================= */
/* =======================
   UI
======================= */
void draw_ui(Player player) {
  const int margin = 10;     // distance from screen edges
  const int row_height = 20; // spacing between UI rows
  const int bar_width = 120; // width of health/stamina bars
  const int bar_height = 16; // height of health/stamina bars
  int x = margin;
  int y = margin;

  // --- Health ---
  DrawText("Health:", x, y, 16, RED);
  DrawRectangle(x + 70, y, bar_width, bar_height, DARKGRAY);
  DrawRectangle(x + 70, y,
                (int)(bar_width * (player.health / player.max_health)),
                bar_height, RED);
  y += row_height;

  // --- Stamina ---
  DrawText("Stamina:", x, y, 16, YELLOW);
  DrawRectangle(x + 70, y, bar_width, bar_height, DARKGRAY);

  static float displayed_stamina = 100;
  if (displayed_stamina < player.stamina)
    displayed_stamina += 0.5f;
  else if (displayed_stamina > player.stamina)
    displayed_stamina -= 0.5f;

  DrawRectangle(x + 70, y,
                (int)(bar_width * (displayed_stamina / player.max_stamina)),
                bar_height, YELLOW);
  y += row_height;

  // --- Current Tool ---
  const char *tool_names[] = {"Hand", "Axe", "Pickaxe", "Shovel", "None"};
  DrawText(TextFormat("Tool: %s", tool_names[player.tool]), x, y, 16, GREEN);
  y += row_height;

  // --- FPS ---
  DrawText(TextFormat("FPS: %d", (int)GetFPS()), x, y, 16, BLUE);
}

/* =======================
   MAIN
======================= */

int main(void) {
  srand(time(NULL));

  // -----------------------
  // Window & camera setup
  // -----------------------
  InitWindow(1280, 800, "MUZE Game");

  SCREEN_WIDTH = GetScreenWidth();
  SCREEN_HEIGHT = GetScreenHeight();
  TILE_SIZE = SCREEN_HEIGHT / 18.0f;

  SetTargetFPS(60);

  Camera2D cam = {0};
  cam.zoom = 1.0f;
  cam.offset = (Vector2){SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2};

  // -----------------------
  // Game initialization
  // -----------------------
  init_base();
  init_player();

  int train_timer = 0;

  // -----------------------
  // Main loop
  // -----------------------
  while (!WindowShouldClose()) {
    // --- Update ---
    update_player();

    int cx = (int)(player.position.x / (CHUNK_SIZE * TILE_SIZE));
    int cy = (int)(player.position.y / (CHUNK_SIZE * TILE_SIZE));
    Chunk *c = get_chunk(cx, cy);

    // Train agents periodically
    train_timer++;
    if (train_timer > TRAIN_INTERVAL) {
      for (int i = 0; i < MAX_AGENTS; i++)
        mu_model_train(c->agents[i].brain);
      train_timer = 0;
    }

    // Update agents
    for (int i = 0; i < MAX_AGENTS; i++) {
      Agent *a = &c->agents[i];
      if (!a->alive)
        respawn_agent(a);
      else
        update_agent(a, c);
    }

    cam.target = player.position;

    // --- Draw ---
    BeginDrawing();
    ClearBackground(SKYBLUE);
    BeginMode2D(cam);

    // Draw surrounding chunks
    for (int dx = -1; dx <= 1; dx++) {
      for (int dy = -1; dy <= 1; dy++) {
        draw_chunk(get_chunk(cx + dx, cy + dy), cx + dx, cy + dy);
      }
    }

    // Draw base
    DrawCircle(agent_base.position.x * TILE_SIZE,
               agent_base.position.y * TILE_SIZE, agent_base.radius * TILE_SIZE,
               DARKGRAY);

    // Draw player
    draw_player(&player);

    EndMode2D();

    // Draw UI
    draw_ui(player);

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
