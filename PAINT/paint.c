#include "../utils/NN/CONVOLUTION.h"
#include "../utils/NN/MUZE/all.h"
#include "../utils/NN/TRANSFORMER.h"
#include "../utils/SDL3/SDL3_compat.h"
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

#define SCREEN_WIDTH 1200
#define SCREEN_HEIGHT 800
#define FPS 60

#define MAX_PATH_LENGTH 1024

#define BG_COLOR BLACK
#define CANVAS_SIZE 64
#define PIXEL_SIZE 8

#define BRUSH_SIZE 3
#define BRUSH_COLOR WHITE

// 3D paint canvas
static unsigned char g_canvas[CANVAS_SIZE][CANVAS_SIZE][3]; // RGB
static float g_depth_map[CANVAS_SIZE][CANVAS_SIZE];
static bool g_3d_mode = true;

// Neural network components
static CONVNet *g_conv = NULL;
static Transformer_t *g_transformer = NULL;
static float *g_nn_output = NULL;

void init_canvas() {
  memset(g_canvas, 0, sizeof(g_canvas));
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      g_depth_map[y][x] = 0.5f; // Default depth
    }
  }
}

void init_neural_networks() {
  // Initialize convolution network
  g_conv = CONV_create(CANVAS_SIZE, CANVAS_SIZE, 3, 0.01f);
  CONV_add_conv2d(g_conv, 16, 3, 3, 1, 1);
  CONV_add_flatten(g_conv);
  CONV_add_dense(g_conv, 128);

  // Initialize transformer
  g_transformer = TRANSFORMER_init(128, 4, 2);

  // Allocate output buffer
  g_nn_output = malloc(CANVAS_SIZE * CANVAS_SIZE * 128 * sizeof(float));

  printf("Neural networks initialized\n");
}

void process_with_neural_networks() {
  if (!g_conv || !g_transformer)
    return;

  // Flatten canvas for processing
  float *input = malloc(CANVAS_SIZE * CANVAS_SIZE * 3 * sizeof(float));
  int idx = 0;
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      input[idx++] = g_canvas[y][x][0] / 255.0f;
      input[idx++] = g_canvas[y][x][1] / 255.0f;
      input[idx++] = g_canvas[y][x][2] / 255.0f;
    }
  }

  // Process through convolution
  const float *conv_output = CONV_forward(g_conv, input);

  // Convert to double for transformer
  long double **transformer_input =
      malloc(CANVAS_SIZE * CANVAS_SIZE * sizeof(long double *));
  for (int i = 0; i < CANVAS_SIZE * CANVAS_SIZE; i++) {
    transformer_input[i] = malloc(128 * sizeof(long double));
    for (int j = 0; j < 128; j++) {
      transformer_input[i][j] = (long double)conv_output[i * 128 + j];
    }
  }

  // Process through transformer
  long double **transformer_output = TRANSFORMER_forward(
      g_transformer, transformer_input, CANVAS_SIZE * CANVAS_SIZE);

  // Update depth map from neural network output
  idx = 0;
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      // Use first dimension of output as depth
      g_depth_map[y][x] = (float)(transformer_output[idx][0] + 1.0L) *
                          0.5f; // Normalize to [0,1]
      idx++;
    }
  }

  // Cleanup
  free(input);
  for (int i = 0; i < CANVAS_SIZE * CANVAS_SIZE; i++) {
    free(transformer_input[i]);
    free(transformer_output[i]);
  }
  free(transformer_input);
  free(transformer_output);

  printf("Neural network processing complete\n");
}

void draw_3d_canvas() {
  int canvas_x = (SCREEN_WIDTH - CANVAS_SIZE * PIXEL_SIZE) / 2;
  int canvas_y = (SCREEN_HEIGHT - CANVAS_SIZE * PIXEL_SIZE) / 2;

  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      Color color = {g_canvas[y][x][0], g_canvas[y][x][1], g_canvas[y][x][2],
                     255};

      if (g_3d_mode) {
        // Apply depth shading
        float depth = g_depth_map[y][x];
        color.r = (unsigned char)(color.r * (0.3f + depth * 0.7f));
        color.g = (unsigned char)(color.g * (0.3f + depth * 0.7f));
        color.b = (unsigned char)(color.b * (0.3f + depth * 0.7f));
      }

      DrawRectangle(canvas_x + x * PIXEL_SIZE, canvas_y + y * PIXEL_SIZE,
                    PIXEL_SIZE, PIXEL_SIZE, color);
    }
  }

  // Draw grid
  for (int i = 0; i <= CANVAS_SIZE; i++) {
    DrawLine(canvas_x, canvas_y + i * PIXEL_SIZE,
             canvas_x + CANVAS_SIZE * PIXEL_SIZE, canvas_y + i * PIXEL_SIZE,
             GRAY);
    DrawLine(canvas_x + i * PIXEL_SIZE, canvas_y, canvas_x + i * PIXEL_SIZE,
             canvas_y + CANVAS_SIZE * PIXEL_SIZE, GRAY);
  }
}

void paint_at(int screen_x, int screen_y, Color color) {
  int canvas_x =
      (screen_x - (SCREEN_WIDTH - CANVAS_SIZE * PIXEL_SIZE) / 2) / PIXEL_SIZE;
  int canvas_y =
      (screen_y - (SCREEN_HEIGHT - CANVAS_SIZE * PIXEL_SIZE) / 2) / PIXEL_SIZE;

  for (int dy = -BRUSH_SIZE; dy <= BRUSH_SIZE; dy++) {
    for (int dx = -BRUSH_SIZE; dx <= BRUSH_SIZE; dx++) {
      int px = canvas_x + dx;
      int py = canvas_y + dy;

      if (px >= 0 && px < CANVAS_SIZE && py >= 0 && py < CANVAS_SIZE) {
        float dist = sqrt(dx * dx + dy * dy);
        if (dist <= BRUSH_SIZE) {
          g_canvas[py][px][0] = color.r;
          g_canvas[py][px][1] = color.g;
          g_canvas[py][px][2] = color.b;
        }
      }
    }
  }
}

void erase_at(int screen_x, int screen_y) {
  paint_at(screen_x, screen_y, BG_COLOR);
}

int main(void) {
  InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "3D Paint with Neural Networks");
  SetTargetFPS(FPS);

  init_canvas();
  init_neural_networks();

  bool nn_processed = false;

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_ESCAPE))
      break;

    if (IsKeyPressed(KEY_SPACE)) {
      g_3d_mode = !g_3d_mode;
      printf("3D mode: %s\n", g_3d_mode ? "ON" : "OFF");
    }

    if (IsKeyPressed(KEY_ENTER)) {
      process_with_neural_networks();
      nn_processed = true;
    }

    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
      Vector2 mouse = GetMousePosition();
      paint_at((int)mouse.x, (int)mouse.y, BRUSH_COLOR);
    }

    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
      Vector2 mouse = GetMousePosition();
      erase_at((int)mouse.x, (int)mouse.y);
    }

    BeginDrawing();
    ClearBackground(BG_COLOR);

    draw_3d_canvas();

    // Draw UI
    DrawText("3D Paint with Neural Networks", 10, 10, 20, WHITE);
    DrawText("Left Click: Paint | Right Click: Erase", 10, 40, 16, WHITE);
    DrawText("SPACE: Toggle 3D Mode | ENTER: Process with NN", 10, 60, 16,
             WHITE);
    DrawText(TextFormat("3D Mode: %s", g_3d_mode ? "ON" : "OFF"), 10, 80, 16,
             g_3d_mode ? GREEN : RED);

    if (nn_processed) {
      DrawText("Neural Network Processing Complete!", 10, 100, 16, GREEN);
    }

    EndDrawing();
  }

  // Cleanup
  if (g_conv)
    CONV_free(g_conv);
  if (g_transformer)
    free(g_transformer);
  if (g_nn_output)
    free(g_nn_output);

  CloseWindow();
  return 0;
}
