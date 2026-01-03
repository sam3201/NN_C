#include "../utils/Raylib/src/raylib.h"
#include "../utils/Raylib/src/raymath.h"
#include <stdio.h>
#include <stdlib.h>

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

void tank_update(Tank *t, int isTop) {
  float move = 2.5f;
  float rot = 2.0f;

  if (!isTop) {
    if (IsKeyDown(KEY_A))
      t->position.x -= move;
    if (IsKeyDown(KEY_D))
      t->position.x += move;
    if (IsKeyDown(KEY_Q))
      t->rotation -= rot;
    if (IsKeyDown(KEY_E))
      t->rotation += rot;
  } else {
    if (IsKeyDown(KEY_LEFT))
      t->position.x -= move;
    if (IsKeyDown(KEY_RIGHT))
      t->position.x += move;
    if (IsKeyDown(KEY_KP_1))
      t->rotation -= rot;
    if (IsKeyDown(KEY_KP_2))
      t->rotation += rot;
  }
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
      tank_update(&bottom, 0); // player control
    if (!topAI)
      tank_update(&top, 1); // player control

    // TODO: replace AI logic here for tanks with AI

    BeginDrawing();
    ClearBackground(BLACK);

    DrawLine(0, 300, 800, 300, GRAY); // battlefield divider

    tank_draw(&bottom);
    tank_draw(&top);

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
