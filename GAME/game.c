#include "../utils/NN/MUZE/all.h"
#include "../utils/Raylib/src/raylib.h"
#include "environment.h"

/* MuConfig config =
    {
        .obs_dim = CHUNK_SIZE * CHUNK_SIZE * 3,
        .latent_dim = 64,
        .action_count = 4,

}

muze_t *muze = NULL;
muze = mu_model_create(config);

*/

int main() {
  InitWindow(800, 600, "MUZE — Infinite 2D World");
  SetTargetFPS(60);

  init_world();
  init_player();

  Vector2 camera = {0, 0};

  Vector2 target = {player.position.x - SCREEN_WIDTH / 2,
                    player.position.y - SCREEN_HEIGHT / 2};

  camera.x += (target.x - camera.x) * 0.1f;
  camera.y += (target.y - camera.y) * 0.1f;

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
