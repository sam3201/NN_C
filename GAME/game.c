#include "../utils/Raylib/src/raylib.h"
#include "environment.h"

int main() {
  InitWindow(800, 600, "MUZE — Infinite 2D World");
  SetTargetFPS(60);

  init_world();

  Vector2 camera = {0, 0};

  while (!WindowShouldClose()) {

    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
      camera.x += 2;
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
      camera.x -= 2;
    if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W))
      camera.y -= 2;
    if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
      camera.y += 2;

    BeginDrawing();
    ClearBackground(BLACK);

    draw_world(camera);

    DrawText("WASD/ARROWS to move", 10, 10, 20, WHITE);

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
