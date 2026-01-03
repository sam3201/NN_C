#include <stdio.h>

#include "../utils/Raylib/src/raylib.h"
#include <stdlib.h>

typedef struct {
  int x;
  int y;
  int width;
  int height;
  int health;
  int maxHealth;
  int speed;
  int damage;
  int direction;
  int xVelocity;
  int yVelocity;

} Tank;

Tank *tank_new(int x, int y) {
  Tank *tank = malloc(sizeof(Tank));
  tank->x = x;
  tank->y = y;
  return tank;
}

void tank_free(Tank *tank) { free(tank); }

void tank_move(Tank *tank, int x, int y) {
  tank->x = x;
  tank->y = y;
}

int main(void) {
  printf("Hello, World!");

  return 0;
}
