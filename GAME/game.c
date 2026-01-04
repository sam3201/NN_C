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

int SCREEN_WIDTH;
int SCREEN_HEIGHT;
float TILE_SIZE;

Color pink = (Color){255, 192, 203, 255};
Color yellow = (Color){255, 255, 0, 255};
Color gray = (Color){100, 100, 100, 255};

#define WORLD_SIZE 128
#define CHUNK_SIZE 32

#define MAX_RESOURCES 512
#define MAX_MOBS 10
#define MAX_AGENTS 8
#define BASE_RADIUS 8
#define MAX_BASE_PARTICLES 32

#define TRAIN_INTERVAL 1

/* =======================
   ENUMS
======================= */

typedef enum {
  TOOL_HAND = 0,
  TOOL_AXE,
  TOOL_PICKAXE,
  TOOL_SHOVEL,
  TOOL_NONE
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
  MOB_NONE
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

  for (int i = 0; i < MAX_MOBS; i++) {
    Mob *m = &c->mobs[i];
    m->type = MOB_PIG;
    m->position = (Vector2){rand() % CHUNK_SIZE, rand() % CHUNK_SIZE};
    m->health = 10;
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

  // --- Apply action ---
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

  float size_multiplier = 1.0f;

  switch (r->type) {
  case RES_TREE:
    size_multiplier = 3.0f; // Trees big
    // Trunk
    DrawRectangleV((Vector2){pos.x - TILE_SIZE * 0.15f, pos.y},
                   (Vector2){TILE_SIZE * 0.3f, TILE_SIZE * 0.7f},
                   (Color){101, 67, 33, 255});
    // Leaves
    DrawCircleV((Vector2){pos.x, pos.y - TILE_SIZE * 0.35f}, TILE_SIZE * 0.5f,
                GREEN);
    break;
  case RES_ROCK:
    size_multiplier = 1.5f; // Rocks smaller than trees but bigger than player
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

  if (r->visited) {
    DrawCircleV(pos, TILE_SIZE * 0.35f * size_multiplier, Fade(WHITE, 0.5f));
    r->visited = false; // reset after flash
  }
}

/* =======================
  MOBS
======================= */
void draw_pig(Vector2 pos, float size) {
  // Body
  DrawCircleV(pos, size * 0.5f, pink);

  // Head (small circle on top)
  DrawCircleV((Vector2){pos.x, pos.y - size * 0.4f}, size * 0.3f, pink);

  // Snout
  DrawCircleV((Vector2){pos.x, pos.y - size * 0.4f}, size * 0.15f,
              (Color){255, 160, 160, 255});

  // Legs (simple rectangles)
  DrawRectangleV((Vector2){pos.x - size * 0.35f, pos.y + size * 0.3f},
                 (Vector2){size * 0.2f, size * 0.2f}, pink);
  DrawRectangleV((Vector2){pos.x + size * 0.15f, pos.y + size * 0.3f},
                 (Vector2){size * 0.2f, size * 0.2f}, pink);
}

void draw_mob(Mob *m, Vector2 chunk_offset) {
  Vector2 p = Vector2Add(Vector2Scale(m->position, TILE_SIZE), chunk_offset);

  if (m->type == MOB_PIG) {
    draw_pig(p, TILE_SIZE * 0.8f);
  }
  // future mobs like sheep, zombies can be added here
}

void draw_agent(Agent *a, Vector2 chunk_offset) {
  if (!a->alive)
    return;

  Vector2 p = Vector2Add(Vector2Scale(a->position, TILE_SIZE), chunk_offset);

  DrawCircleV(p, TILE_SIZE * 0.35f, a->tribe_color);
  if (a->flash_timer > 0)
    DrawCircleV(p, TILE_SIZE * 0.25f, Fade(WHITE, 0.6f));
}

void draw_player(Player *p) {
  // Body (tan)
  DrawCircleV(p->position, TILE_SIZE * 0.35f, (Color){245, 222, 179, 255});

  float hand_offset = TILE_SIZE * 0.4; // moved slightly outside body
  DrawCircleV((Vector2){p->position.x - hand_offset, p->position.y},
              TILE_SIZE * 0.15f, RED);
  DrawCircleV((Vector2){p->position.x + hand_offset, p->position.y},
              TILE_SIZE * 0.15f, RED);
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
void draw_ui(Player player) {
  // --- Health ---
  DrawText("Health:", 10, 10, 16, RED);
  DrawRectangle(80, 10, 100, 16, DARKGRAY);
  DrawRectangle(80, 10, (int)(100 * (player.health / player.max_health)), 16,
                RED);

  // --- Stamina ---
  DrawText("Stamina:", 10, 30, 16, YELLOW);
  DrawRectangle(80, 30, 100, 16, DARKGRAY);

  // Smooth stamina visualization
  static float displayed_stamina = 100;
  if (displayed_stamina < player.stamina)
    displayed_stamina += 0.5f; // regen smoothing
  else if (displayed_stamina > player.stamina)
    displayed_stamina -= 0.5f; // decrease smoothing

  DrawRectangle(80, 30, (int)(100 * (displayed_stamina / player.max_stamina)),
                16, YELLOW);

  // --- Current Tool ---
  const char *tool_names[] = {"Hand", "Axe", "Pickaxe", "Shovel", "None"};
  DrawText(TextFormat("Tool: %s", tool_names[player.tool]), 10, 50, 16, GREEN);

  // --- FPS ---
  DrawText(TextFormat("FPS: %d", (int)GetFPS()), 10, 70, 16, BLUE);
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

    // Draw UI
    draw_ui(player);

    EndMode2D();
    EndDrawing();
  }

  CloseWindow();
  return 0;
}
