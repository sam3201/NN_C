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

void DrawColorWheel() {
  DrawCircleLines(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, 1, BG_COLOR);
  DrawCircle(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, 1, BRUSH_COLOR);
}

void DrawBrush(int x, int y, Color color) {
  DrawCircleLines(x, y, 1, BG_COLOR);
  DrawCircle(x, y, BRUSH_SIZE, BRUSH_COLOR);
}

Color invert(Color color) {
  return (Color){255 - color.r, 255 - color.g, 255 - color.b, color.a};
}

int main(void) {

  InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Paint");
  SetTargetFPS(FPS);

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_ESCAPE))
      break;

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
      int x = GetMouseX();
      int y = GetMouseY();
      DrawBrush(x, y, BRUSH_COLOR);
    }

    if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
      int x = GetMouseX();
      int y = GetMouseY();
      DrawBrush(x, y, invert(BRUSH_COLOR));
    }

    BeginDrawing();
    ClearBackground(BG_COLOR);
    EndDrawing();
  }

  CloseWindow();

  return 0;
}
