#include <stdio.h>

#include "../utils/Raylib/src/raylib.h"

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

int main(void) {
  printf("Hello, World!");

  return 0;
}
