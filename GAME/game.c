#include "../utils/Raylib/src/raylib.h"
#include "environment.h"

int main() {
  InitWindow(800, 600, "MUZE — Infinite 2D World");
  SetTargetFPS(60);

  init_world();
  init_player();

  while (!WindowShouldClose()) {
    update_player();

    camera.x = player.position.x - SCREEN_WIDTH / 2;
    camera.y = player.position.y - SCREEN_HEIGHT / 2;

    BeginDrawing();
    ClearBackground(BLACK);

    draw_world(camera);
    draw_player(camera);
    draw_ui();

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
