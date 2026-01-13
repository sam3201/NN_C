#include "../utils/NN/CONVOLUTION.h"
#include "../utils/NN/MUZE/all.h"
#include "../utils/NN/TRANSFORMER.h"
#include "../utils/Raylib/src/raylib.h"
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define FPS 60

#define MAX_PATH_LENGTH 1024

#define BG_COLOR BLACK

#define BRUSH_SIZE 5
#define BRUSH_COLOR RED

int main(void) {

  InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Paint");
  SetTargetFPS(FPS);

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_ESCAPE))
      break;

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
      int x = GetMouseX();
      int y = GetMouseY();
      DrawCircle(x, y, BRUSH_COLOR, RED);
    }

    if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
      int x = GetMouseX();
      int y = GetMouseY();
      DrawCircle(x, y, BRUSH_COLOR, BLUE);
    }

    BeginDrawing();
    ClearBackground(BG_COLOR);
    EndDrawing();
  }

  CloseWindow();

  return 0;
}
