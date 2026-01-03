#include "environment.h"
#include "../utils/Raylib/src/raylib.h"
#include "../utils/Raylib/src/raymath.h"
#include <stdio.h>
#include <stdlib.h>

#define WORLD_SIZE 128 // 128×128 chunks = enough for “infinite feel”

static Chunk world[WORLD_SIZE][WORLD_SIZE];
static bool world_initialized = false;

// ------------------------ helpers --------------------------
static inline int wrap(int v) { return (v + WORLD_SIZE) % WORLD_SIZE; }

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
  for (int i = 0; i < MAX_RESOURCES; i++) {
    c->resources[i].position =
        (Vector2){(float)(rand() % CHUNK_SIZE), (float)(rand() % CHUNK_SIZE)};
    c->resources[i].value = rand() % 10;
    c->resources[i].type = rand() % 3;
    c->resources[i].visited = false;
  }

  // generate mobs
  for (int i = 0; i < MAX_MOBS; i++) {
    c->mobs[i].position =
        (Vector2){(float)(rand() % CHUNK_SIZE), (float)(rand() % CHUNK_SIZE)};
    c->mobs[i].value = 10;
    c->mobs[i].type = rand() % 2;
    c->mobs[i].visited = false;
  }
}

// ------------------------ draw world --------------------------
void draw_world(Vector2 camera) {
  int cx = (int)(camera.x / (CHUNK_SIZE * TILE_SIZE));
  int cy = (int)(camera.y / (CHUNK_SIZE * TILE_SIZE));

  // draw surrounding 3×3 chunks
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {

      Chunk *c = get_chunk(cx + dx, cy + dy);
      int world_x = (cx + dx) * CHUNK_SIZE;
      int world_y = (cy + dy) * CHUNK_SIZE;

      // draw terrain tiles
      for (int i = 0; i < CHUNK_SIZE; i++) {
        for (int j = 0; j < CHUNK_SIZE; j++) {
          Color col;

          switch (c->terrain[i][j]) {
          case 1:
            col = GREEN;
            break;
          case 2:
            col = DARKGREEN;
            break;
          case 3:
            col = BEIGE;
            break;
          default:
            col = GRAY;
            break;
          }

          DrawRectangle((world_x + i) * TILE_SIZE - camera.x,
                        (world_y + j) * TILE_SIZE - camera.y, TILE_SIZE,
                        TILE_SIZE, col);
        }
      }
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

  if ((move.x != 0 || move.y != 0) && player.stamina > 0) {
    player.position.x += move.x * player.move_speed;
    player.position.y += move.y * player.move_speed;
    player.stamina -= 0.5f;
  } else {
    // stamina regen
    player.stamina += 0.25f;
  }

  if (player.stamina > player.max_stamina)
    player.stamina = player.max_stamina;
}

void harvest_resources() {
  int cx = (int)(player.position.x / (CHUNK_SIZE * TILE_SIZE));
  int cy = (int)(player.position.y / (CHUNK_SIZE * TILE_SIZE));

  Chunk *chunk = get_chunk(cx, cy);

  for (int i = 0; i < MAX_RESOURCES; i++) {
    Resource *r = &chunk->resources[i];
    if (r->visited)
      continue;

    Vector2 world_pos = {(cx * CHUNK_SIZE + r->position.x) * TILE_SIZE,
                         (cy * CHUNK_SIZE + r->position.y) * TILE_SIZE};

    float dist = Vector2Distance(player.position, world_pos);
    if (dist < 12) {
      r->visited = true;
      player.stamina -= 5;

      if (r->type == 0)
        player.wood++;
      if (r->type == 1)
        player.stone++;
      if (r->type == 2)
        player.food++;
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
  DrawCircle(player.position.x - camera.x, player.position.y - camera.y, 5,
             RED);
}

void draw_ui() {
  DrawRectangle(10, 10, 200, 70, Fade(BLACK, 0.6f));

  DrawText(TextFormat("HP: %.0f", player.health), 20, 20, 16, RED);
  DrawText(TextFormat("STA: %.0f", player.stamina), 20, 40, 16, BLUE);

  DrawText(TextFormat("Wood:%d Stone:%d Food:%d", player.wood, player.stone,
                      player.food),
           20, 60, 16, GREEN);
}
