#include "environment.h"
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
