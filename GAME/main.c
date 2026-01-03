#include <stdio.h>

#include "../utils/Raylib/src/raylib.h"
#include <stdlib.h>

typedef struct {
  Vector2 position;
  float radius;

  float rotation; // turret rotation
  float armLength;
  float armWidth;

  int health;
  int maxHealth;

  Color baseColor;
  Color armColor;
} Tank;

Tank *tank_new(int x, int y) {
  Tank *tank = malloc(sizeof(Tank));
  tank->x = x;
  tank->y = y;
  return tank;
}

void tank_free(Tank *tank) { free(tank); }

int main(void) {
  printf("Hello, World!");

  return 0;
}
