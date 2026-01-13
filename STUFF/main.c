#include "../utils/Raylib/src/raylib.h"
#include <stdio.h>

int main(void) {
  InitWinow(800, 450, "Neural Network");

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(RAYWHITE);
    EndDrawing();
  }

  CloseWindow();

  return 0;
}
