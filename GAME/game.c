#include "../utils/NN/MUZE/all.h"
#include "../utils/Raylib/src/raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------- Constants ----------------
#define ROWS 75
#define COLS 75
#define NUM_CHANNELS 3
#define CELL_LIFETIME 120
#define TANK_SIZE 40
#define MAX_STEPS 200

// ---------------- Structs -----------------
typedef struct {
  Vector2 position;
  float speed;
  int health;
  int isAI; // 0 = player, 1 = AI
} Tank;

typedef struct {
  int dummy;
} Policy;

// ---------------- Global Variables --------
Tank bottom;
Tank top;
Policy tank_policy;

// Screen representation for RL
long double screen[ROWS * COLS * NUM_CHANNELS];

// ---------------- Movement -----------------
void move_left(Tank *t) {
  t->position.x -= t->speed;
  if (t->position.x < 0)
    t->position.x = 0;
}
void move_right(Tank *t, int screen_width) {
  t->position.x += t->speed;
  if (t->position.x + TANK_SIZE > screen_width)
    t->position.x = screen_width - TANK_SIZE;
}
void move_up(Tank *t) {
  t->position.y -= t->speed;
  if (t->position.y < 0)
    t->position.y = 0;
}
void move_down(Tank *t, int screen_height) {
  t->position.y += t->speed;
  if (t->position.y + TANK_SIZE > screen_height)
    t->position.y = screen_height - TANK_SIZE;
}

void tank_update(Tank *t, Vector2 target) {
  if (t->isAI) {
    if (t->position.x < target.x)
      move_right(t, 800);
    if (t->position.x > target.x)
      move_left(t);
    if (t->position.y < target.y)
      move_down(t, 600);
    if (t->position.y > target.y)
      move_up(t);
  }
}

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

int main() {
  InitWindow(800, 600, "Tank Game");
  bottom = (Tank){{100, 500}, 5, 3, 0};
  top = (Tank){{700, 100}, 5, 3, 0};
  SetTargetFPS(60);

  title_screen(&top.isAI, &bottom.isAI);

  ToyEnvState env;
  env.size = ROWS;
  float obs[ROWS];
  toy_env_reset(&env, obs);
  env_step_fn step_fn = toy_env_step;
  env_reset_fn reset_fn = toy_env_reset;

  while (!WindowShouldClose()) {
    if (!bottom.isAI) {
      if (IsKeyDown(KEY_A))
        move_left(&bottom);
      if (IsKeyDown(KEY_D))
        move_right(&bottom, 800);
      if (IsKeyDown(KEY_W))
        move_up(&bottom);
      if (IsKeyDown(KEY_S))
        move_down(&bottom, 600);
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
    }

    tank_update(&bottom, top.position);
    tank_update(&top, bottom.position);

    int action = 0;
    float reward = 0.0f;
    int done = 0;
    step_fn(&env, action, obs, &reward, &done);

    BeginDrawing();
    ClearBackground(BLACK);
    DrawRectangle(bottom.position.x, bottom.position.y, TANK_SIZE, TANK_SIZE,
                  BLUE);
    DrawRectangle(top.position.x, top.position.y, TANK_SIZE, TANK_SIZE, RED);
    EndDrawing();
  }

  CloseWindow();
  return 0;
}
