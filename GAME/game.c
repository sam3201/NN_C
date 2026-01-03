#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ---------------- Constants ----------------
#define ROWS 75
#define COLS 75
#define NUM_CHANNELS 3
#define CELL_LIFETIME 120

// ---------------- Structs -----------------
typedef struct {
  Vector2 position;
  float speed;
  int health;
} Tank;

// ---------------- Global Variables --------
Tank bottom;
Tank top;

// Placeholder policy struct
typedef struct {
  // RL policy parameters go here later
  int dummy;
} Policy;

Policy tank_policy;

// Screen representation for RL
long double screen[ROWS * COLS * NUM_CHANNELS];

// ---------------- Function Stubs ----------
void move_left(Tank *t) { t->position.x -= t->speed; }
void move_right(Tank *t) { t->position.x += t->speed; }
void move_up(Tank *t) { t->position.y -= t->speed; }
void move_down(Tank *t) { t->position.y += t->speed; }
void shoot(Tank *t) { /* implement shooting */ }

long double compute_reward(Tank *player, Tank *enemy) {
  // placeholder reward
  return 0;
}

// ---------------- Tank Update -------------
void tank_update(Tank *t, int isTop, Vector2 target) {
  // implement tank movement or RL logic here
  // currently, just a stub
  (void)isTop; // suppress unused warnings
  (void)target;
}

// ---------------- Main Game Loop -----------
int main() {
  InitWindow(800, 600, "Tank Game");

  // initialize tanks
  bottom.position = (Vector2){100, 500};
  bottom.speed = 5;
  bottom.health = 3;

  top.position = (Vector2){700, 100};
  top.speed = 5;
  top.health = 3;

  SetTargetFPS(60);

  while (!WindowShouldClose()) {

    // ---- Tank updates ----
    tank_update(&bottom, 0, bottom.position); // player control
    tank_update(&top, 1, top.position);       // player control

    // ---- RL placeholders ----
    long double *state = screen; // encode_screen placeholder
    int action = 0;              // policy_select_action placeholder

    // ---- Move based on action (placeholder) ----
    // Example: just move bottom tank randomly
    move_left(&bottom);
    move_right(&top);

    // Compute reward placeholder
    long double reward = compute_reward(&bottom, &top);

    // ---- Rendering ----
    BeginDrawing();
    ClearBackground(BLACK);

    DrawRectangle(bottom.position.x, bottom.position.y, 40, 40, BLUE);
    DrawRectangle(top.position.x, top.position.y, 40, 40, RED);

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
