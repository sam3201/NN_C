#include "../utils/NN/NN.h"
#include "../utils/Raylib/src/raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 75
#define COLS 75
#define NUM_CHANNELS 3

long double screen[ROWS][COLS][NUM_CHANNELS];

typedef struct {
  NN_t *nn;
  size_t action_size;
} Policy;

typedef struct {
  Vector2 position;
  float radius;
  float rotation;
  float armLength;
  float armWidth;
  int health;
  int maxHealth;
  Color baseColor;
  Color armColor;
} Tank;

Tank tank_new(Vector2 pos, Color base, Color arm) {
  Tank t = {0};
  t.position = pos;
  t.radius = 25;
  t.rotation = 0;
  t.armLength = 35;
  t.armWidth = 8;
  t.maxHealth = 100;
  t.health = 100;
  t.baseColor = base;
  t.armColor = arm;
  return t;
}

void tank_draw(Tank *t, Vector2 aim) {
  // Draw the base
  DrawCircleV(t->position, t->radius, t->baseColor);

  // Calculate the angle to aim at (mouse or AI)
  float angle = atan2f(aim.y - t->position.y, aim.x - t->position.x);

  t->rotation = angle; // store rotation for logic

  // Turret dimensions
  float armLength = t->radius * 1.5f;
  float armWidth = t->radius / 2;

  // Compute offsets for two arms (left/right of center)
  Vector2 offset = {0, armWidth / 2 + 2};

  // Draw left arm
  Rectangle arm1 = {t->position.x - offset.x, t->position.y - offset.y,
                    armLength, armWidth};
  DrawRectanglePro(arm1, (Vector2){0, armWidth / 2}, angle * 180 / PI,
                   t->armColor);

  // Draw right arm
  Rectangle arm2 = {t->position.x + offset.x, t->position.y + offset.y,
                    armLength, armWidth};
  DrawRectanglePro(arm2, (Vector2){0, armWidth / 2}, angle * 180 / PI,
                   t->armColor);
}

void tank_update(Tank *t, int isTop, Vector2 target) {
  float move = 2.5f;

  // Movement (keyboard)
  if (!isTop) {
    if (IsKeyDown(KEY_A))
      t->position.x -= move;
    if (IsKeyDown(KEY_D))
      t->position.x += move;
    if (IsKeyDown(KEY_W))
      t->position.y -= move;
    if (IsKeyDown(KEY_S))
      t->position.y += move;
  } else {
    if (IsKeyDown(KEY_LEFT))
      t->position.x -= move;
    if (IsKeyDown(KEY_RIGHT))
      t->position.x += move;
    if (IsKeyDown(KEY_UP))
      t->position.y -= move;
    if (IsKeyDown(KEY_DOWN))
      t->position.y += move;
  }

  // Rotation automatically toward target (mouse for player, enemy for AI)
  t->rotation = atan2f(target.y - t->position.y, target.x - t->position.x);
}

long double *encode_screen(int **screen, size_t rows, size_t cols,
                           size_t num_channels) {
  size_t input_size = rows * cols * num_channels;
  long double *input_vector =
      (long double *)calloc(input_size, sizeof(long double));
  if (!input_vector)
    return NULL;

  for (size_t r = 0; r < rows; r++) {
    for (size_t c = 0; c < cols; c++) {
      int val = screen[r][c];
      if (val >= 0 && val < num_channels) {
        size_t idx = r * cols * num_channels + c * num_channels + val;
        input_vector[idx] = 1.0L; // one-hot
      }
    }
  }
  return input_vector;
}

// Sample action from softmax output
int policy_select_action(Policy *policy, long double *state) {
  long double *probs = NN_forward_softmax(policy->nn, state);

  // Sample action based on probability distribution
  long double r = (long double)rand() / RAND_MAX;
  long double cumulative = 0.0L;
  int action = 0;
  for (size_t i = 0; i < policy->action_size; i++) {
    cumulative += probs[i];
    if (r <= cumulative) {
      action = i;
      break;
    }
  }

  free(probs);
  return action;
}

void policy_update(Policy *policy, long double *state, int action,
                   long double reward) {
  size_t out_size = policy->action_size;

  // Forward pass
  long double *probs = NN_forward_softmax(policy->nn, state);

  // Create one-hot vector for chosen action
  long double *action_one_hot = create_one_hot(action, out_size);

  // Gradient: dL/dz = (p - 1) for chosen action scaled by reward
  for (size_t i = 0; i < out_size; i++) {
    action_one_hot[i] = reward * (probs[i] - action_one_hot[i]);
  }

  // Backprop through last layer
  NN_backprop_argmax(policy->nn, state, action_one_hot, probs, ce_derivative);

  free(probs);
  free(action_one_hot);
}

// Draws the title screen and allows selection of AI/Player tanks
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

int main(void) {
  InitWindow(800, 600, "Tank NN_C Testbed");
  SetTargetFPS(60);

  int topAI = 0;    // 0 = Player, 1 = AI
  int bottomAI = 0; // 0 = Player, 1 = AI

  title_screen(&topAI, &bottomAI); // select AI/Player

  Tank bottom = tank_new((Vector2){400, 500}, DARKGREEN, GREEN);
  Tank top = tank_new((Vector2){400, 100}, MAROON, RED);

  while (!WindowShouldClose()) {
    if (!bottomAI)
      tank_update(&bottom, 0, bottom.position); // player control
    if (!topAI)
      tank_update(&top, 1, top.position); // player control

    // TODO: replace AI logic here for tanks with AI

    BeginDrawing();
    ClearBackground(BLACK);

    DrawLine(0, 300, 800, 300, GRAY); // battlefield divider

    long double *state = encode_screen(screen, ROWS, COLS, NUM_CHANNELS);
    int action = policy_select_action(&tank_policy, state);

    // Map action index to tank movement/shooting
    switch (action) {
    case 0:
      move_left(&tank);
      break;
    case 1:
      move_right(&tank);
      break;
    case 2:
      move_up(&tank);
      break;
    case 3:
      move_down(&tank);
      break;
    case 4:
      shoot(&tank);
      break;
    }

    long double reward = compute_reward(&tank, &enemy); // +1/-1 etc
    policy_update(&tank_policy, state, action, reward);

    free(state);

    Vector2 mouse = GetMousePosition();

    // Bottom tank aims at mouse
    tank_draw(&bottom, mouse);

    // Top tank (player or AI)
    Vector2 topAim;
    if (!topAI) {
      topAim = GetMousePosition();
    } else {
      // For now, aim at bottom tank
      topAim = bottom.position;
    }
    tank_draw(&top, topAim);

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
