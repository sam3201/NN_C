#include "../utils/NN/MEMORY/memory.h"
#include "../utils/NN/MUZE/all.h"
#include "../utils/NN/NEAT.h"
#include "../utils/NN/NN.h"
#include "../utils/Raylib/src/raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define FRAME_RATE 60

#define MEMORY_CAPACITY infinity

typedef enum {
  ACTION_NONE = 0,
  ACTION_MOVE_LEFT,
  ACTION_MOVE_RIGHT,
  ACTION_MOVE_UP,
  ACTION_MOVE_DOWN,
  ACTION_COUNT
} Action;

typedef struct {
  Vector2 position;
  unsigned int radius;
  unsigned int size;
  float time_alive;
  int agent_id;
  float breeding_timer;
  Color color;
  NEAT_t *brain;
  Memory memory;
  size_t input_size;
} Agent;

// ---------------- MuZero AI -----------------
MuModel *ai_model = NULL;

void init_ai() {
  MuConfig cfg = {.obs_dim = 4, .latent_dim = 16, .action_count = 5};
  ai_model = mu_model_create(&cfg);
}

void free_ai() {
  if (ai_model) {
    mu_model_free(ai_model);
    ai_model = NULL;
  }
}

int ai_choose_action(Agent *ai, Tank *player) {
  if (!ai_model)
    return 0; // fallback safety

  float obs[4] = {ai->position.x, ai->position.y, player->position.x,
                  player->position.y};

  MCTSParams params = {.num_simulations = 25,
                       .c_puct = 1.0f,
                       .max_depth = 10,
                       .dirichlet_alpha = 0.3f,
                       .dirichlet_eps = 0.25f,
                       .temperature = 1.0f,
                       .discount = 0.99f};

  MCTSResult res = mcts_run(ai_model, obs, &params);
  return res.chosen_action;
}

// ---------------- Tank Update -----------------
void tank_update(Tank *t, Tank *target, int screen_width, int screen_height) {
  if (t->isAI) {
    int action = ai_choose_action(t, target);

    switch (action) {
    case 0:
      move_up(t);
      break;
    case 1:
      move_down(t, screen_height);
      break;
    case 2:
      move_left(t);
      break;
    case 3:
      move_right(t, screen_width);
      break;
    case 4:
      tank_shoot(t);
      break;
    }
    aim_turret(t, target);
  } else {
    Vector2 center = {t->position.x + TANK_SIZE / 2,
                      t->position.y + TANK_SIZE / 2};
    Vector2 mouse = GetMousePosition();
    t->turretAngle = atan2f(mouse.y - center.y, mouse.x - center.x);
  }
}

// ---------------- Title / Menu -----------------
void title_screen(int *topAI, int *bottomAI) {
  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);
    DrawText("Tank NN_C Testbed", 220, 50, 40, LIGHTGRAY);
    DrawText("Top Tank:", 100, 200, 20, LIGHTGRAY);
    DrawText(*topAI ? "AI" : "PLAYER", 250, 200, 20, *topAI ? RED : GREEN);
    DrawText("Press 1 to toggle", 400, 200, 20, GRAY);
    DrawText("Bottom Tank:", 100, 300, 20, LIGHTGRAY);
    DrawText(*bottomAI ? "AI" : "PLAYER", 250, 300, 20,
             *bottomAI ? RED : GREEN);
    DrawText("Press 2 to toggle", 400, 300, 20, GRAY);
    DrawText("Press ENTER to start", 200, 450, 30, YELLOW);
    EndDrawing();

    if (IsKeyPressed(KEY_ONE))
      *topAI = !(*topAI);
    if (IsKeyPressed(KEY_TWO))
      *bottomAI = !(*bottomAI);
    if (IsKeyPressed(KEY_ENTER))
      break;
  }
}

// ---------------- Main -----------------
int main() {
  InitWindow(800, 600, "Tank Game");
  SetTargetFPS(60);

  bottom = (Tank){{100, 500}, 5, 3, 0, 0};
  top = (Tank){{700, 100}, 5, 3, 0, 0};

  title_screen(&top.isAI, &bottom.isAI);

  init_ai(); // MuZero init

  while (!WindowShouldClose()) {
    // Player Input
    if (!bottom.isAI) {
      if (IsKeyDown(KEY_A))
        move_left(&bottom);
      if (IsKeyDown(KEY_D))
        move_right(&bottom, 800);
      if (IsKeyDown(KEY_W))
        move_up(&bottom);
      if (IsKeyDown(KEY_S))
        move_down(&bottom, 600);
      if (IsKeyPressed(KEY_SPACE))
        tank_shoot(&bottom);
    }
    if (!top.isAI) {
      if (IsKeyDown(KEY_LEFT))
        move_left(&top);
      if (IsKeyDown(KEY_RIGHT))
        move_right(&top, 800);
      if (IsKeyDown(KEY_UP))
        move_up(&top);
      if (IsKeyDown(KEY_DOWN))
        move_down(&top, 600);
      if (IsKeyPressed(KEY_ENTER))
        tank_shoot(&top);
    }

    // AI Update
    tank_update(&bottom, &top, 800, 600);
    tank_update(&top, &bottom, 800, 600);

    // Update bullets
    update_bullets();

    // ---------------- Rendering ----------------
    BeginDrawing();
    ClearBackground(BLACK);

    DrawRectangle(bottom.position.x, bottom.position.y, TANK_SIZE, TANK_SIZE,
                  BLUE);
    DrawRectangle(top.position.x, top.position.y, TANK_SIZE, TANK_SIZE, RED);

    Vector2 b_center = {bottom.position.x + TANK_SIZE / 2,
                        bottom.position.y + TANK_SIZE / 2};
    DrawLineV(b_center,
              (Vector2){b_center.x + cosf(bottom.turretAngle) * TANK_SIZE,
                        b_center.y + sinf(bottom.turretAngle) * TANK_SIZE},
              DARKBLUE);

    Vector2 t_center = {top.position.x + TANK_SIZE / 2,
                        top.position.y + TANK_SIZE / 2};
    DrawLineV(t_center,
              (Vector2){t_center.x + cosf(top.turretAngle) * TANK_SIZE,
                        t_center.y + sinf(top.turretAngle) * TANK_SIZE},
              RED);

    for (int i = 0; i < bullet_count; i++)
      if (bullets[i].active)
        DrawCircleV(bullets[i].pos, BULLET_SIZE / 2, YELLOW);

    DrawText(TextFormat("Bottom HP: %d", bottom.health), 10, 10, 20, LIGHTGRAY);
    DrawText(TextFormat("Top HP: %d", top.health), 650, 10, 20, LIGHTGRAY);

    EndDrawing();
  }

  free_ai();
  CloseWindow();
  return 0;
}
