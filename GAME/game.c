#include "../utils/NN/MUZE/all.h"
#include "../utils/Raylib/src/raylib.h"
#include "environment.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define MAX_BASE_PARTICLES 32

typedef struct {
  Vector2 pos;
  float lifetime;
  bool flash_white;
} BaseParticle;

BaseParticle base_particles[MAX_BASE_PARTICLES];

void init_base_particles(void) {
  for (int i = 0; i < MAX_BASE_PARTICLES; i++) {
    float angle = ((float)i / MAX_BASE_PARTICLES) * 6.28319f;
    float dist = (float)(rand() % (BASE_RADIUS * TILE_SIZE));
    base_particles[i].pos.x =
        agent_base.position.x * TILE_SIZE + cosf(angle) * dist;
    base_particles[i].pos.y =
        agent_base.position.y * TILE_SIZE + sinf(angle) * dist;
    base_particles[i].lifetime = (float)(rand() % 60);
    base_particles[i].flash_white = false;
  }
}

void update_base_particles(void) {
  for (int i = 0; i < MAX_BASE_PARTICLES; i++) {
    base_particles[i].lifetime += 1.0f;
    // flash white every 30 frames
    if (((int)base_particles[i].lifetime) % 30 < 15)
      base_particles[i].flash_white = true;
    else
      base_particles[i].flash_white = false;
  }
}

void draw_base_particles(Vector2 camera) {
  for (int i = 0; i < MAX_BASE_PARTICLES; i++) {
    Vector2 s = {base_particles[i].pos.x - camera.x,
                 base_particles[i].pos.y - camera.y};
    DrawCircle(s.x, s.y, 2, base_particles[i].flash_white ? WHITE : GREEN);
  }
}

int main() {
  srand((unsigned int)time(NULL));

  SetConfigFlags(FLAG_WINDOW_TOPMOST | FLAG_WINDOW_UNDECORATED);
  InitWindow(GetScreenWidth(), GetScreenHeight(), "MUZE — Infinite 2D World")

      SetTargetFPS(60);

  init_world();
  init_base();
  init_base_particles(); // initialize particles
  init_player();

  Vector2 camera = {0, 0};

  while (!WindowShouldClose()) {
    update_player();

    // Update agents: heal if in base
    for (int dx = -1; dx <= 1; dx++) {
      for (int dy = -1; dy <= 1; dy++) {
        Chunk *c = get_chunk(WORLD_SIZE / 2 + dx, WORLD_SIZE / 2 + dy);
        for (int i = 0; i < MAX_AGENTS; i++) {
          if (c->agents[i].alive)
            update_agent(&c->agents[i]);
        }
      }
    }

    update_base_particles(); // update particle flashing

    camera.x = player.position.x - SCREEN_WIDTH / 2;
    camera.y = player.position.y - SCREEN_HEIGHT / 2;

    BeginDrawing();
    ClearBackground(BLACK);

    draw_world(camera);
    draw_base_particles(camera); // draw green/white healing particles
    draw_player(camera);
    draw_ui();

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
