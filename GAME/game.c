#include "../utils/Raylib/src/raylib.h"
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

void tank_draw(Tank *t) {
  DrawCircleV(t->position, t->radius, t->baseColor);

  Vector2 origin = {t->armLength, t->armWidth / 2};

  Rectangle arm = {t->position.x, t->position.y, t->armLength, t->armWidth};

  DrawRectanglePro(arm, origin, t->rotation, t->armColor);

  Rectangle arm2 = arm;
  arm2.y += t->armWidth + 4;
  DrawRectanglePro(arm2, origin, t->rotation, t->armColor);
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

int main(void) {
  InitWindow(800, 600, "Tank NN_C Testbed");
  SetTargetFPS(60);

  Tank bottom = tank_new((Vector2){400, 500}, DARKGREEN, GREEN);
  Tank top = tank_new((Vector2){400, 100}, MAROON, RED);

  while (!WindowShouldClose()) {
    tank_update(&bottom, 0);
    tank_update(&top, 1);

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
