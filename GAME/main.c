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

int main(void) {
  printf("Hello, World!");

  return 0;
}
