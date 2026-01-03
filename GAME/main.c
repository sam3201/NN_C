#include <stdio.h>

#include "../utils/Raylib/src/raylib.h"
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
  t.radius = 25.0f;

  t.rotation = 0.0f;
  t.armLength = 35.0f;
  t.armWidth = 8.0f;

  t.maxHealth = 100;
  t.health = 100;

  t.baseColor = base;
  t.armColor = arm;
  return t;
}

void tank_free(Tank *tank) { free(tank); }

void tank_draw(Tank *t) {
  DrawCircleV(t->position, t->radius, t->baseColor);

  Vector2 origin = {0, t->armWidth / 2};

  Rectangle arm = {t->position.x, t->position.y, t->armLength, t->armWidth};

  DrawRectanglePro(arm, origin, t->rotation, t->armColor);

  Rectangle arm2 = arm;
  arm2.y += t->armWidth + 4;
  DrawRectanglePro(arm2, origin, t->rotation, t->armColor);
}

int main(void) {
  printf("Hello, World!");

  return 0;
}
