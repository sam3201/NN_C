#include "../utils/NN/MUZE/all.h"
#include "../utils/Raylib/src/raylib.h"
#include "../utils/Raylib/src/raymath.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

int SCREEN_WIDTH;
int SCREEN_HEIGHT;

float TILE_SIZE;

#define BASE_TILE_SIZE 32

#define WORLD_SIZE 128
#define CHUNK_SIZE 32

#define MAX_RESOURCES 512
#define MAX_MOBS 10
#define MAX_AGENTS 8
#define BASE_RADIUS 8
#define MAX_BASE_PARTICLES 32

// ---------- ENUMS ----------
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

// ---------- STRUCTS ----------
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

typedef struct {
  Vector2 pos;
  float lifetime;
  bool flash_white;
} BaseParticle;

// ---------- GLOBALS ----------
Chunk world[WORLD_SIZE][WORLD_SIZE];
Player player;
Base agent_base;
BaseParticle base_particles[MAX_BASE_PARTICLES];

// ---------- HELPERS ----------
static inline int wrap(int v) { return (v + WORLD_SIZE) % WORLD_SIZE; }
static inline float randf(float min, float max) {
  return min + (float)rand() / RAND_MAX * (max - min);
}

Color biome_color(int type) {
  switch (type) {
  case 0:
    return (Color){120 + rand() % 20, 200, 120 + rand() % 20, 255}; // grass
  case 1:
    return (Color){34, 139 + rand() % 30, 34, 255}; // forest
  case 2:
    return (Color){139, 137, 137, 255}; // rock
  default:
    return RAYWHITE;
  }
}

Color resource_color(ResourceType type) {
  switch (type) {
  case RES_TREE:
    return DARKGREEN;
  case RES_ROCK:
    return GRAY;
  case RES_FOOD:
    return ORANGE;
  default:
    return WHITE;
  }
}

// ---------- WORLD ----------
Chunk *get_chunk(int cx, int cy) {
  cx = wrap(cx);
  cy = wrap(cy);
  Chunk *c = &world[cx][cy];
  if (!c->generated) {
    c->generated = true;
    c->biome_type = (abs(cx) + abs(cy)) % 3;

    for (int i = 0; i < CHUNK_SIZE; i++)
      for (int j = 0; j < CHUNK_SIZE; j++)
        c->terrain[i][j] = c->biome_type;

    int target = (c->biome_type == 0) ? 6 : (c->biome_type == 1) ? 12 : 3;
    c->resource_count = target;

    for (int i = 0; i < target; i++) {
      int roll = rand() % 100;
      if (c->biome_type == 1 && roll < 70)
        c->resources[i].type = RES_TREE;
      else if (roll < 40)
        c->resources[i].type = RES_ROCK;
      else
        c->resources[i].type = RES_FOOD;
      c->resources[i].position =
          (Vector2){rand() % CHUNK_SIZE, rand() % CHUNK_SIZE};
      c->resources[i].health = 100;
      c->resources[i].visited = false;
    }

    for (int i = 0; i < MAX_MOBS; i++) {
      c->mobs[i].position = (Vector2){rand() % CHUNK_SIZE, rand() % CHUNK_SIZE};
      c->mobs[i].value = 10;
      c->mobs[i].type = rand() % 2;
      c->mobs[i].visited = false;
    }

    for (int i = 0; i < MAX_AGENTS; i++) {
      Agent *a = &c->agents[i];
      a->health = a->stamina = 100;
      a->agent_id = i;
      a->alive = true;
      a->flash_timer = 0;
      a->tribe_color = (rand() % 4 == 0)   ? RED
                       : (rand() % 4 == 1) ? BLUE
                       : (rand() % 4 == 2) ? GREEN
                                           : YELLOW;

      if (cx == WORLD_SIZE / 2 && cy == WORLD_SIZE / 2) {
        float angle = ((float)i / MAX_AGENTS) * 2 * PI;
        float dist = rand() % (BASE_RADIUS - 2) + 2;
        a->position.x = agent_base.position.x + cosf(angle) * dist;
        a->position.y = agent_base.position.y + sinf(angle) * dist;
      } else {
        a->position.x = rand() % CHUNK_SIZE;
        a->position.y = rand() % CHUNK_SIZE;
      }

      MuConfig cfg = {
          .obs_dim = 10, .latent_dim = 32, .action_count = ACTION_COUNT};
      a->brain = mu_model_create(&cfg);
      a->input_size = cfg.obs_dim;
    }
  }
  return c;
}

// ---------- PLAYER ----------
void init_player() {
  player.position = (Vector2){0, 0};
  player.max_health = player.health = 100;
  player.max_stamina = player.stamina = 100;
  player.move_speed = 2.0f;
  player.attack_damage = 10;
  player.attack_range = 10;
  player.wood = player.stone = player.food = 0;
  player.alive = true;
  player.tool = TOOL_HAND;
}

void update_player() {
  if (!player.alive)
    return;
  Vector2 move = {0, 0};
  if (IsKeyDown(KEY_W))
    move.y -= 1;
  if (IsKeyDown(KEY_S))
    move.y += 1;
  if (IsKeyDown(KEY_A))
    move.x -= 1;
  if (IsKeyDown(KEY_D))
    move.x += 1;

  if (move.x != 0 && move.y != 0) {
    move.x *= 0.7071f;
    move.y *= 0.7071f;
  }

  float speed = player.move_speed * (player.stamina / player.max_stamina);
  player.position.x += move.x * speed;
  player.position.y += move.y * speed;
}
// Configure MCTS for this decision
MCTSParams mcts_params = {
    .num_simulations = 40, // can tune later
    .c_puct = 1.2f,
    .discount = 0.95f,
    .temperature = 1.0f,
    .max_depth = 20,         // optional
    .dirichlet_alpha = 0.3f, // optional
    .dirichlet_eps = 0.25f   // optional
};

// ---------- AGENT ----------
int decide_action(Agent *agent, long double *inputs) {
  if (!agent || !agent->brain)
    return rand() % ACTION_COUNT;

  // Convert long double inputs to float for MUZE
  int obs_dim = agent->brain->cfg.obs_dim;
  float obs[obs_dim];
  for (int i = 0; i < obs_dim; i++) {
    obs[i] = (float)inputs[i];
  }

  // Run MUZE + MCTS
  MCTSResult res = mcts_run(agent->brain, obs, &mcts_params);
  int action = res.chosen_action;

  // Free resources
  mcts_result_free(&res);

  // Safety fallback
  if (action < 0 || action >= ACTION_COUNT)
    action = rand() % ACTION_COUNT;

  return action;
}

void update_agent(Agent *a) {
  if (!a->alive)
    return;
  float obs[a->input_size];
  for (size_t i = 0; i < a->input_size; i++)
    obs[i] = randf(0, 1);
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

  float dist = Vector2Distance(a->position, agent_base.position);
  if (dist < BASE_RADIUS) {
    a->health = fminf(a->health + 0.5f, 100);
    a->stamina = fminf(a->stamina + 0.5f, 100);
    a->flash_timer += 0.1f;
    if (a->flash_timer > 1.0f)
      a->flash_timer = 0;
  } else
    a->flash_timer = 0;
}

// ---------- BASE ----------
void init_base() {
  agent_base.position = (Vector2){WORLD_SIZE / 2, WORLD_SIZE / 2};
  agent_base.radius = BASE_RADIUS;

  for (int i = 0; i < MAX_BASE_PARTICLES; i++) {
    base_particles[i].pos = agent_base.position;
    base_particles[i].lifetime = randf(0, 1);
    base_particles[i].flash_white = false;
  }
}

// ---------- MAIN ----------
int main() {
  SCREEN_WIDTH = GetScreenWidth();
  SCREEN_HEIGHT = GetScreenHeight();

  TILE_SIZE =
      ((float)SCREEN_WIDTH + (float)SCREEN_HEIGHT) / (float)WORLD_SIZE * 2;

  InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "MUZE Game");
  SetTargetFPS(60);

  init_base();
  init_player();

  Camera2D camera = {0};
  camera.target = player.position;
  camera.zoom = 1.0f;
  camera.offset = (Vector2){SCREEN_WIDTH / 2.0f, SCREEN_HEIGHT / 2.0f};

  srand(time(NULL));

  while (!WindowShouldClose()) {
    // --- Update ---
    update_player();

    int cx = player.position.x / (CHUNK_SIZE * TILE_SIZE);
    int cy = player.position.y / (CHUNK_SIZE * TILE_SIZE);
    Chunk *c = get_chunk(cx, cy);

    for (int i = 0; i < MAX_AGENTS; i++)
      update_agent(&c->agents[i]);

    // Update camera
    camera.target = player.position;

    // Update base particles
    for (int i = 0; i < MAX_BASE_PARTICLES; i++) {
      BaseParticle *p = &base_particles[i];
      p->pos.x += randf(-0.2f, 0.2f);
      p->pos.y += randf(-0.2f, 0.2f);
      p->lifetime -= 0.01f;
      if (p->lifetime <= 0) {
        p->pos = agent_base.position;
        p->lifetime = randf(0, 1);
      }
    }

    // --- Draw ---
    BeginDrawing();
    ClearBackground(SKYBLUE);
    BeginMode2D(camera);

    // Draw chunks
    for (int dx = -1; dx <= 1; dx++)
      for (int dy = -1; dy <= 1; dy++) {
        Chunk *ch = get_chunk(cx + dx, cy + dy);
        // Draw terrain
        for (int i = 0; i < CHUNK_SIZE; i++)
          for (int j = 0; j < CHUNK_SIZE; j++) {
            int sx = (cx + dx) * CHUNK_SIZE * TILE_SIZE + i * TILE_SIZE;
            int sy = (cy + dy) * CHUNK_SIZE * TILE_SIZE + j * TILE_SIZE;
            DrawRectangle(sx, sy, TILE_SIZE, TILE_SIZE,
                          biome_color(ch->terrain[i][j]));
          }

        // Draw resources (trees, rocks, food) with more style
        for (int i = 0; i < ch->resource_count; i++) {
          Resource *r = &ch->resources[i];
          Vector2 s = {(cx + dx) * CHUNK_SIZE * TILE_SIZE +
                           r->position.x * TILE_SIZE + TILE_SIZE / 2,
                       (cy + dy) * CHUNK_SIZE * TILE_SIZE +
                           r->position.y * TILE_SIZE + TILE_SIZE / 2};
          float tree_radius = TILE_SIZE * 0.45f;
          float rock_radius = TILE_SIZE * 0.35f;
          float food_radius = TILE_SIZE * 0.25f;

          switch (r->type) {

          case RES_TREE: {
            // trunk
            DrawRectangle(s.x - TILE_SIZE * 0.08f, s.y + TILE_SIZE * 0.1f,
                          TILE_SIZE * 0.16f, TILE_SIZE * 0.45f, BROWN);

            // canopy
            DrawCircleV((Vector2){s.x, s.y - TILE_SIZE * 0.1f}, tree_radius,
                        DARKGREEN);
            break;
          }

          case RES_ROCK: {
            DrawCircleV(s, rock_radius, DARKGRAY);
            DrawCircleV((Vector2){s.x + 3, s.y - 3}, rock_radius * 0.5f, GRAY);
            break;
          }

          case RES_FOOD:
            DrawCircleV(s, food_radius, ORANGE);
            break;
          }
        }
        // Draw agents
        float body_radius = TILE_SIZE * 0.35f;
        float band_radius = TILE_SIZE * 0.38f;

        Vector2 body_pos = s;

        // body
        DrawCircleV(body_pos, body_radius, LIGHTGRAY);

        // headband (tribe color)
        DrawRing(body_pos, band_radius * 0.85f, band_radius, 200, 340, 16,
                 a->tribe_color);

        // flash when healing
        if (a->flash_timer > 0.0f) {
          DrawCircleV(body_pos, body_radius * 0.9f, Fade(WHITE, 0.6f));
        }

        // Draw base with particles
        DrawCircle(agent_base.position.x * TILE_SIZE,
                   agent_base.position.y * TILE_SIZE,
                   agent_base.radius * TILE_SIZE, DARKGRAY);

        for (int i = 0; i < MAX_BASE_PARTICLES; i++) {
          BaseParticle *p = &base_particles[i];
          DrawCircleV(p->pos, 1 + rand() % 2,
                      p->flash_white ? WHITE : LIGHTGRAY);
        }

        // Draw player
        DrawCircle(player.position.x, player.position.y, 6, RED);

        EndMode2D();
        EndDrawing();
      }

    CloseWindow();
    return 0;
  }
