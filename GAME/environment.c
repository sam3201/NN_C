#include "environment.h"
#include "../utils/Raylib/src/raylib.h"
#include "../utils/Raylib/src/raymath.h"
#include <stdio.h>
#include <stdlib.h>

#define WORLD_SIZE 128 // 128×128 chunks = enough for “infinite feel”

static Chunk world[WORLD_SIZE][WORLD_SIZE];
static bool world_initialized = false;

Resource resources[MAX_RESOURCES];
int resource_count = 0;

Base agent_base;

// ------------------------ helpers --------------------------
static inline int wrap(int v) { return (v + WORLD_SIZE) % WORLD_SIZE; }

// ------------------------ init base --------------------------
void init_base(void) {
  agent_base.position = (Vector2){WORLD_SIZE / 2, WORLD_SIZE / 2};
  agent_base.radius = BASE_RADIUS;

  Chunk *c = get_chunk(agent_base.position.x, agent_base.position.y);
  for (int i = 0; i < MAX_AGENTS; i++) {
    float angle = ((float)i / MAX_AGENTS) * 6.28319f; // circle around base
    float dist =
        (float)(rand() % (BASE_RADIUS - 2) + 2); // avoid overlapping center
    c->agents[i].position.x = agent_base.position.x + cosf(angle) * dist;
    c->agents[i].position.y = agent_base.position.y + sinf(angle) * dist;
    c->agents[i].health = 100;
    c->agents[i].stamina = 100;
    c->agents[i].agent_id = i;
    c->agents[i].alive = true;

    // random tribe color
    switch (rand() % 4) {
    case 0:
      c->agents[i].tribe_color = RED;
      break;
    case 1:
      c->agents[i].tribe_color = BLUE;
      break;
    case 2:
      c->agents[i].tribe_color = GREEN;
      break;
    case 3:
      c->agents[i].tribe_color = YELLOW;
      break;
    }
  }
}

// ------------------------ draw chunk resources --------------------------
void draw_chunk_resources(Chunk *c, int cx, int cy, Vector2 camera) {
  for (int i = 0; i < c->resource_count; i++) {
    Resource *r = &c->resources[i];
    if (r->visited || r->health <= 0)
      continue;

    Vector2 world_pos = {(cx * CHUNK_SIZE + r->position.x) * TILE_SIZE,
                         (cy * CHUNK_SIZE + r->position.y) * TILE_SIZE};

    Vector2 s = {world_pos.x - camera.x, world_pos.y - camera.y};

    float hp_ratio = r->health / 100.0f;

    switch (r->type) {
    case RES_TREE:
      if (hp_ratio < 0)
        hp_ratio = 0;

      Color trunk = ColorLerp(DARKBROWN, GRAY, 1.0f - hp_ratio);
      Color leaves = ColorLerp(GREEN, BROWN, 1.0f - hp_ratio);

      // outline
      DrawRectangle(s.x - 6, s.y - 16, 12, 20, BLACK);
      // trunk
      DrawRectangle(s.x - 4, s.y - 14, 8, 14, trunk);
      // canopy
      DrawCircle(s.x, s.y - 18, 9, BLACK);
      DrawCircle(s.x, s.y - 18, 7, leaves);

      break;

    case RES_ROCK:
      if (hp_ratio < 0)
        hp_ratio = 0;

      Color rock = ColorLerp(GRAY, LIGHTGRAY, 1.0f - hp_ratio);

      DrawCircle(s.x, s.y, 7, BLACK);
      DrawCircle(s.x, s.y, 5, rock);

      break;

    case RES_FOOD:
      DrawCircle(s.x, s.y, 4, BLACK);
      DrawCircle(s.x, s.y, 3, RED);
      break;

    default:
      break;
    }
  }
}

// ------------------------ initialize entire world --------------------------
void init_world() {
  if (world_initialized)
    return;

  for (int x = 0; x < WORLD_SIZE; x++) {
    for (int y = 0; y < WORLD_SIZE; y++) {
      world[x][y].generated = false;
    }
  }

  world_initialized = true;
}

// ------------------------ chunk access --------------------------
Chunk *get_chunk(int cx, int cy) {
  cx = wrap(cx);
  cy = wrap(cy);

  Chunk *chunk = &world[cx][cy];

  if (!chunk->generated)
    generate_chunk(chunk, cx, cy);

  return chunk;
}

// ------------------------ chunk generation --------------------------
void generate_chunk(Chunk *c, int cx, int cy) {
  c->generated = true;

  // biome type from world coordinates
  // very simple — will expand later
  c->biome_type = (abs(cx) + abs(cy)) % 3;

  // fill terrain
  for (int i = 0; i < CHUNK_SIZE; i++) {
    for (int j = 0; j < CHUNK_SIZE; j++) {
      switch (c->biome_type) {
      case 0:
        c->terrain[i][j] = 1;
        break; // grass
      case 1:
        c->terrain[i][j] = 2;
        break; // forest
      case 2:
        c->terrain[i][j] = 3;
        break; // desert
      }
    }
  }

  // generate some resources
  int target = 0;

  switch (c->biome_type) {
  case 0:
    target = 6;
    break; // grassland
  case 1:
    target = 12;
    break; // forest
  case 2:
    target = 3;
    break; // desert
  }

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

  // mark remaining as inactive
  for (int i = target; i < MAX_RESOURCES; i++) {
    c->resources[i].type = RES_NONE;
  }

  // generate mobs
  for (int i = 0; i < MAX_MOBS; i++) {
    c->mobs[i].position =
        (Vector2){(float)(rand() % CHUNK_SIZE), (float)(rand() % CHUNK_SIZE)};
    c->mobs[i].value = 10;
    c->mobs[i].type = rand() % 2;
    c->mobs[i].visited = false;
  }

  // generate agents
  for (int i = 0; i < MAX_AGENTS; i++) {
    c->agents[i].health = 100;
    c->agents[i].stamina = 100;
    c->agents[i].agent_id = i;
    c->agents[i].alive = true;
    c->agents[i].flash_timer = 0;

    // assign a random tribe color
    switch (rand() % 4) {
    case 0:
      c->agents[i].tribe_color = RED;
      break;
    case 1:
      c->agents[i].tribe_color = BLUE;
      break;
    case 2:
      c->agents[i].tribe_color = GREEN;
      break;
    case 3:
      c->agents[i].tribe_color = YELLOW;
      break;
    }

    // Initial chunk: spawn in circle around base
    if (cx == WORLD_SIZE / 2 && cy == WORLD_SIZE / 2) {
      float angle = ((float)i / MAX_AGENTS) * 6.28319f; // full circle
      float dist = (float)(rand() % (BASE_RADIUS - 2) + 2);
      c->agents[i].position.x = agent_base.position.x + cosf(angle) * dist;
      c->agents[i].position.y = agent_base.position.y + sinf(angle) * dist;
    } else {
      // other chunks: random positions
      c->agents[i].position.x = (float)(rand() % CHUNK_SIZE);
      c->agents[i].position.y = (float)(rand() % CHUNK_SIZE);
    }
  }
}

// --- PLAYER ---
void init_player(void);
void update_player(void);
void draw_player(Vector2 camera);
void draw_ui(void);

// --- INTERACTION ---
void harvest_resources(void);
void attack_mobs(void);

Player player;

void init_player() {
  player.position = (Vector2){0, 0};
  player.max_health = 100;
  player.health = 100;
  player.max_stamina = 100;
  player.stamina = 100;
  player.move_speed = 2.0f;
  player.attack_damage = 10.0f;
  player.attack_range = 10.0f;

  player.wood = 0;
  player.stone = 0;
  player.food = 0;

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
  if (IsKeyPressed(KEY_E)) {
    harvest_resources();
  }
  if (IsKeyPressed(KEY_SPACE)) {
    attack_mobs();
  }
  if (IsKeyPressed(KEY_ONE))
    player.tool = TOOL_HAND;
  if (IsKeyPressed(KEY_TWO))
    player.tool = TOOL_AXE;
  if (IsKeyPressed(KEY_THREE))
    player.tool = TOOL_PICKAXE;

  bool moving = (move.x != 0 || move.y != 0);

  // normalize diagonal movement
  if (move.x != 0 && move.y != 0) {
    move.x *= 0.7071f;
    move.y *= 0.7071f;
  }

  float stamina_ratio = player.stamina / player.max_stamina;
  if (stamina_ratio < 0.1f)
    stamina_ratio = 0.1f;

  float speed = player.move_speed * stamina_ratio;

  if (moving) {
    player.position.x += move.x * speed;
    player.position.y += move.y * speed;

    player.stamina -= 0.4f;
    if (player.stamina < 0)
      player.stamina = 0;
  } else {
    player.stamina += 0.6f;
  }

  if (player.stamina > player.max_stamina)
    player.stamina = player.max_stamina;
}

void update_agent(Agent *a) {
  if (!a->alive)
    return;

  float dist = Vector2Distance(a->position, agent_base.position);

  if (dist < BASE_RADIUS) {
    // Heal agent
    a->health += 0.5f;
    a->stamina += 0.5f;
    if (a->health > 100)
      a->health = 100;
    if (a->stamina > 100)
      a->stamina = 100;

    // flash timer
    a->flash_timer += 0.1f;
    if (a->flash_timer > 1.0f)
      a->flash_timer = 0;
  } else {
    // reset flash when outside
    a->flash_timer = 0;
  }
}

void harvest_resources() {
  int tool_power = 1;

  int cx = (int)(player.position.x / (CHUNK_SIZE * TILE_SIZE));
  int cy = (int)(player.position.y / (CHUNK_SIZE * TILE_SIZE));

  Chunk *chunk = get_chunk(cx, cy);

  for (int i = 0; i < chunk->resource_count; i++) {
    Resource *r = &chunk->resources[i];
    if (r->visited || r->health <= 0)
      continue;

    Vector2 world_pos = {(cx * CHUNK_SIZE + r->position.x) * TILE_SIZE,
                         (cy * CHUNK_SIZE + r->position.y) * TILE_SIZE};

    float hit_radius = (r->type == RES_TREE)   ? 18.0f
                       : (r->type == RES_ROCK) ? 14.0f
                                               : 10.0f;

    if (r->type == RES_TREE) {
      if (player.tool == TOOL_AXE)
        tool_power = 4;
      else
        tool_power = 1;
    }

    if (r->type == RES_ROCK) {
      if (player.tool == TOOL_PICKAXE)
        tool_power = 5;
      else
        tool_power = 0;
    }

    if (r->type == RES_FOOD) {
      tool_power = 2;
    }

    if (Vector2Distance(player.position, world_pos) < hit_radius) {

      // tool restrictions
      if ((r->type == RES_TREE && player.tool != TOOL_AXE) ||
          (r->type == RES_ROCK && player.tool != TOOL_PICKAXE))
        continue;

      r->health -= tool_power;

      r->position.x += (rand() % 3 - 1) * 0.1f;
      r->position.y += (rand() % 3 - 1) * 0.1f;

      player.stamina -= 2;

      if (r->health <= 0) {
        for (int k = 0; k < 6; k++) {
          DrawCircle(world_pos.x + rand() % 6 - 3, world_pos.y + rand() % 6 - 3,
                     1, (r->type == RES_TREE) ? BROWN : GRAY);
        }

        r->visited = true;

        if (r->type == RES_TREE)
          player.wood += 3;
        if (r->type == RES_ROCK)
          player.stone += 2;
        if (r->type == RES_FOOD)
          player.food += 1;
      }
    }
  }
}

void attack_mobs() {
  int cx = (int)(player.position.x / (CHUNK_SIZE * TILE_SIZE));
  int cy = (int)(player.position.y / (CHUNK_SIZE * TILE_SIZE));

  Chunk *chunk = get_chunk(cx, cy);

  for (int i = 0; i < MAX_MOBS; i++) {
    Mob *m = &chunk->mobs[i];
    if (m->visited)
      continue;

    Vector2 world_pos = {(cx * CHUNK_SIZE + m->position.x) * TILE_SIZE,
                         (cy * CHUNK_SIZE + m->position.y) * TILE_SIZE};

    if (Vector2Distance(player.position, world_pos) < player.attack_range) {
      m->value -= player.attack_damage;
      player.stamina -= 10;

      if (m->value <= 0) {
        m->visited = true;
        player.food += 2;
      }
    }
  }
}

void draw_player(Vector2 camera) {
  Vector2 screen = {player.position.x - camera.x, player.position.y - camera.y};

  DrawCircle(screen.x, screen.y, 6, BLACK); // outline
  DrawCircle(screen.x, screen.y, 4, RED);   // body
}

void draw_ui() {
  DrawRectangleRounded((Rectangle){10, 10, 220, 80}, 0.2f, 8,
                       Fade((Color){20, 20, 20, 255}, 0.8f));

  DrawText(TextFormat("HP: %.0f", player.health), 20, 18, 16, RED);
  DrawText(TextFormat("STA: %.0f", player.stamina), 20, 36, 16, SKYBLUE);
  DrawText(TextFormat("Wood:%d Stone:%d Food:%d", player.wood, player.stone,
                      player.food),
           20, 56, 14, GREEN);

  const char *tool_name = (player.tool == TOOL_HAND)  ? "HAND"
                          : (player.tool == TOOL_AXE) ? "AXE"
                                                      : "PICK";

  DrawText(TextFormat("Tool: %s", tool_name), 20, 74, 14, YELLOW);
}

void spawn_resource(Vector2 pos, ResourceType type) {
  if (resource_count >= MAX_RESOURCES)
    return;

  Resource *r = &resources[resource_count++];
  r->position = pos;
  r->type = type;
  r->health = (type == RES_TREE) ? 5 : 8;
  r->visited = false;
}

void draw_resources(Vector2 camera) {
  for (int i = 0; i < resource_count; i++) {
    Resource *r = &resources[i];
    if (r->health <= 0)
      continue;

    Vector2 s = {r->position.x - camera.x, r->position.y - camera.y};

    if (r->type == RES_TREE) {
      DrawRectangle(s.x - 4, s.y - 12, 8, 16, DARKGREEN);
      DrawCircle(s.x, s.y - 16, 10, GREEN);
    } else {
      DrawCircle(s.x, s.y, 8, GRAY);
    }
  }
}

Resource *find_nearest_resource(Vector2 pos, float max_dist) {
  Resource *best = NULL;
  float best_dist = max_dist;

  for (int i = 0; i < resource_count; i++) {
    Resource *r = &resources[i];
    if (r->health <= 0)
      continue;

    float d = Vector2Distance(pos, r->position);
    if (d < best_dist) {
      best_dist = d;
      best = r;
    }
  }
  return best;
}

void draw_chunk_agents(Chunk *c, int cx, int cy, Vector2 camera) {
  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &c->agents[i];
    if (!a->alive)
      continue;

    Vector2 world_pos = {(cx * CHUNK_SIZE + a->position.x) * TILE_SIZE,
                         (cy * CHUNK_SIZE + a->position.y) * TILE_SIZE};
    Vector2 s = {world_pos.x - camera.x, world_pos.y - camera.y};

    // body
    DrawCircle(s.x, s.y, 6, BLACK);                       // outline
    DrawCircle(s.x, s.y, 5, (Color){245, 222, 179, 255}); // tan body

    // hands
    DrawCircle(s.x - 6, s.y, 2, BLACK);
    DrawCircle(s.x + 6, s.y, 2, BLACK);

    // bandana / headband
    DrawRectangle(s.x - 4, s.y - 6, 8, 2, a->tribe_color);

    // healing + flash
    if (a->flash_timer > 0) {
      // oscillate color between green and white
      float t = a->flash_timer;
      Color col = ColorLerp(GREEN, WHITE, (sinf(t * 6.28319f) + 1) / 2);
      DrawText("+", s.x - 4, s.y - 14, 12, col);
    }
  }
}

// ------------------------ draw world --------------------------
void draw_world(Vector2 camera) {
  int cx = (int)(camera.x / (CHUNK_SIZE * TILE_SIZE));
  int cy = (int)(camera.y / (CHUNK_SIZE * TILE_SIZE));

  Color grass = (Color){60, 160, 80, 255};
  Color forest = (Color){30, 110, 60, 255};
  Color desert = (Color){210, 185, 140, 255};

  // draw surrounding 3×3 chunks
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {

      Chunk *c = get_chunk(cx + dx, cy + dy);
      int world_x = (cx + dx) * CHUNK_SIZE;
      int world_y = (cy + dy) * CHUNK_SIZE;

      // ---------- draw terrain tiles ----------
      for (int i = 0; i < CHUNK_SIZE; i++) {
        for (int j = 0; j < CHUNK_SIZE; j++) {

          Color col;
          switch (c->terrain[i][j]) {
          case 1:
            col = grass;
            break;
          case 2:
            col = forest;
            break;
          case 3:
            col = desert;
            break;
          default:
            col = GRAY;
            break;
          }

          int sx = (world_x + i) * TILE_SIZE - camera.x;
          int sy = (world_y + j) * TILE_SIZE - camera.y;
          {
            DrawRectangle(sx, sy, TILE_SIZE, TILE_SIZE, col);

            // subtle grid lines
            DrawRectangleLines(sx, sy, TILE_SIZE, TILE_SIZE,
                               Fade(BLACK, 0.05f));
          }
        }

        // in draw_world, only for initial chunk
        if (cx == 0 && cy == 0) {
          Vector2 s = {agent_base.position.x * TILE_SIZE - camera.x,
                       agent_base.position.y * TILE_SIZE - camera.y};
          DrawCircle(s.x, s.y, agent_base.radius * TILE_SIZE,
                     Fade(SKYBLUE, 0.2f)); // translucent base
        }

        // ---------- draw resources ONCE per chunk ----------
        draw_chunk_resources(c, cx + dx, cy + dy, camera);

        // ---------- draw agents ONCE per chunk ----------
        draw_chunk_agents(c, cx + dx, cy + dy, camera);
      }
    }
  }
}
