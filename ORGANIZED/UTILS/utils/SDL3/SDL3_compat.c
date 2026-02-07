#define SDL_MAIN_HANDLED

#include "SDL3_compat.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_opengl.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
  int size;
  TTF_Font *font;
} FontEntry;

static SDL_Window *g_window = NULL;
static SDL_Renderer *g_renderer = NULL;
static SDL_GLContext g_gl_ctx = NULL;
static bool g_use_gl = false;
static bool g_should_close = false;
static KeyboardKey g_exit_key = KEY_ESCAPE;
static int g_target_fps = 60;
static float g_last_dt = 1.0f / 60.0f;
static uint64_t g_last_tick = 0;
static uint64_t g_frame_start_tick = 0;
static bool g_frame_active = false;
static uint64_t g_start_tick = 0;

static bool g_key_down[SDL_SCANCODE_COUNT];
static bool g_key_pressed[SDL_SCANCODE_COUNT];
static bool g_key_released[SDL_SCANCODE_COUNT];

static bool g_mouse_down[8];
static bool g_mouse_pressed[8];
static bool g_mouse_released[8];
static float g_mouse_wheel = 0.0f;
static float g_mouse_dx = 0.0f;
static float g_mouse_dy = 0.0f;
static unsigned char g_text_queue[128];
static int g_text_head = 0;
static int g_text_tail = 0;

static FontEntry g_fonts[32];
static int g_font_count = 0;
static char g_text_buffers[8][2048];
static int g_text_buffer_index = 0;

static SDL_Scancode sdl_scancode_from_key(KeyboardKey key) {
  if (key >= KEY_ONE && key <= KEY_ZERO) {
    static const SDL_Scancode map[] = {SDL_SCANCODE_1, SDL_SCANCODE_2,
                                       SDL_SCANCODE_3, SDL_SCANCODE_4,
                                       SDL_SCANCODE_5, SDL_SCANCODE_6,
                                       SDL_SCANCODE_7, SDL_SCANCODE_8,
                                       SDL_SCANCODE_9, SDL_SCANCODE_0};
    return map[key - KEY_ONE];
  }
  if (key >= KEY_KP_1 && key <= KEY_KP_0) {
    static const SDL_Scancode map[] = {SDL_SCANCODE_KP_1, SDL_SCANCODE_KP_2,
                                       SDL_SCANCODE_KP_3, SDL_SCANCODE_KP_4,
                                       SDL_SCANCODE_KP_5, SDL_SCANCODE_KP_6,
                                       SDL_SCANCODE_KP_7, SDL_SCANCODE_KP_8,
                                       SDL_SCANCODE_KP_9, SDL_SCANCODE_KP_0};
    return map[key - KEY_KP_1];
  }
  if (key >= KEY_A && key <= KEY_Z) {
    return (SDL_Scancode)(SDL_SCANCODE_A + (key - KEY_A));
  }
  switch (key) {
  case KEY_SPACE:
    return SDL_SCANCODE_SPACE;
  case KEY_TAB:
    return SDL_SCANCODE_TAB;
  case KEY_ENTER:
    return SDL_SCANCODE_RETURN;
  case KEY_BACKSPACE:
    return SDL_SCANCODE_BACKSPACE;
  case KEY_ESCAPE:
    return SDL_SCANCODE_ESCAPE;
  case KEY_P:
    return SDL_SCANCODE_P;
  case KEY_UP:
    return SDL_SCANCODE_UP;
  case KEY_DOWN:
    return SDL_SCANCODE_DOWN;
  case KEY_LEFT:
    return SDL_SCANCODE_LEFT;
  case KEY_RIGHT:
    return SDL_SCANCODE_RIGHT;
  case KEY_MINUS:
    return SDL_SCANCODE_MINUS;
  case KEY_EQUAL:
    return SDL_SCANCODE_EQUALS;
  case KEY_F5:
    return SDL_SCANCODE_F5;
  case KEY_F6:
    return SDL_SCANCODE_F6;
  case KEY_F7:
    return SDL_SCANCODE_F7;
  case KEY_LEFT_SHIFT:
    return SDL_SCANCODE_LSHIFT;
  default:
    return SDL_SCANCODE_UNKNOWN;
  }
}

static void sdl3_poll_events(void) {
  memset(g_key_pressed, 0, sizeof(g_key_pressed));
  memset(g_key_released, 0, sizeof(g_key_released));
  memset(g_mouse_pressed, 0, sizeof(g_mouse_pressed));
  memset(g_mouse_released, 0, sizeof(g_mouse_released));
  g_mouse_wheel = 0.0f;
  g_mouse_dx = 0.0f;
  g_mouse_dy = 0.0f;

  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    switch (event.type) {
    case SDL_EVENT_QUIT:
      g_should_close = true;
      break;
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
      g_should_close = true;
      break;
    case SDL_EVENT_KEY_DOWN:
      if (!event.key.repeat && event.key.scancode < SDL_SCANCODE_COUNT) {
        g_key_down[event.key.scancode] = true;
        g_key_pressed[event.key.scancode] = true;
      }
      break;
    case SDL_EVENT_KEY_UP:
      if (event.key.scancode < SDL_SCANCODE_COUNT) {
        g_key_down[event.key.scancode] = false;
        g_key_released[event.key.scancode] = true;
      }
      break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
      if (event.button.button < (int)(sizeof(g_mouse_down) /
                                      sizeof(g_mouse_down[0]))) {
        g_mouse_down[event.button.button] = true;
        g_mouse_pressed[event.button.button] = true;
      }
      break;
    case SDL_EVENT_MOUSE_BUTTON_UP:
      if (event.button.button < (int)(sizeof(g_mouse_down) /
                                      sizeof(g_mouse_down[0]))) {
        g_mouse_down[event.button.button] = false;
        g_mouse_released[event.button.button] = true;
      }
      break;
    case SDL_EVENT_MOUSE_WHEEL:
      g_mouse_wheel += (float)event.wheel.y;
      break;
    case SDL_EVENT_MOUSE_MOTION:
      g_mouse_dx += event.motion.xrel;
      g_mouse_dy += event.motion.yrel;
      break;
    case SDL_EVENT_TEXT_INPUT: {
      if (event.text.text) {
        const unsigned char *p = (const unsigned char *)event.text.text;
        while (*p) {
          int next_tail = (g_text_tail + 1) %
                          (int)(sizeof(g_text_queue) /
                                sizeof(g_text_queue[0]));
          if (next_tail == g_text_head) {
            break;
          }
          g_text_queue[g_text_tail] = *p;
          g_text_tail = next_tail;
          p++;
        }
      }
    } break;
    default:
      break;
    }
  }

  if (g_exit_key != KEY_NULL && IsKeyPressed(g_exit_key)) {
    g_should_close = true;
  }
}

static void sdl3_frame_start(void) {
  if (g_frame_active) {
    return;
  }
  uint64_t now = SDL_GetTicks();
  if (g_last_tick == 0) {
    g_last_tick = now;
  }
  g_last_dt = (float)(now - g_last_tick) / 1000.0f;
  g_last_tick = now;
  g_frame_start_tick = now;
  g_frame_active = true;
}

static TTF_Font *sdl3_get_font(int size) {
  for (int i = 0; i < g_font_count; i++) {
    if (g_fonts[i].size == size) {
      return g_fonts[i].font;
    }
  }

  const char *path = getenv("SDL_FONT_PATH");
  const char *fallbacks[] = {
      path,
      "/System/Library/Fonts/Supplemental/Arial.ttf",
      "/System/Library/Fonts/Supplemental/Helvetica.ttf",
      "/System/Library/Fonts/SFNS.ttf",
      "/Library/Fonts/Arial.ttf",
      "/Library/Fonts/Helvetica.ttf"};

  TTF_Font *font = NULL;
  for (size_t i = 0; i < sizeof(fallbacks) / sizeof(fallbacks[0]); i++) {
    if (!fallbacks[i] || fallbacks[i][0] == '\0') {
      continue;
    }
    font = TTF_OpenFont(fallbacks[i], size);
    if (font) {
      break;
    }
  }

  if (!font) {
    fprintf(stderr, "SDL_ttf: failed to load a font at size %d\n", size);
    return NULL;
  }

  if (g_font_count < (int)(sizeof(g_fonts) / sizeof(g_fonts[0]))) {
    g_fonts[g_font_count++] = (FontEntry){size, font};
  }
  return font;
}

void SDL3_SetUseGL(int enable) { g_use_gl = (enable != 0); }

void InitWindow(int width, int height, const char *title) {
  SDL_SetMainReady();
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    const char *err = SDL_GetError();
    fprintf(stderr, "SDL_Init failed: %s\n", err && err[0] ? err : "(no error)");
    exit(1);
  }
  if (!TTF_Init()) {
    fprintf(stderr, "TTF_Init failed: %s\n", SDL_GetError());
    exit(1);
  }

  Uint64 window_flags = g_use_gl ? SDL_WINDOW_OPENGL : 0;
  g_window = SDL_CreateWindow(title, width, height, window_flags);
  if (!g_window) {
    fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
    exit(1);
  }

  if (g_use_gl) {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    g_gl_ctx = SDL_GL_CreateContext(g_window);
    if (!g_gl_ctx) {
      fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError());
      exit(1);
    }
    SDL_GL_MakeCurrent(g_window, g_gl_ctx);
    SDL_GL_SetSwapInterval(1);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
  } else {
    g_renderer = SDL_CreateRenderer(g_window, NULL);
    if (!g_renderer) {
      fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
      exit(1);
    }
    SDL_SetRenderVSync(g_renderer, 1);
    SDL_SetRenderDrawBlendMode(g_renderer, SDL_BLENDMODE_BLEND);
  }
  SDL_StartTextInput(g_window);

  g_start_tick = SDL_GetTicks();
}

void CloseWindow(void) {
  for (int i = 0; i < g_font_count; i++) {
    if (g_fonts[i].font) {
      TTF_CloseFont(g_fonts[i].font);
    }
  }
  g_font_count = 0;

  if (g_renderer) {
    SDL_DestroyRenderer(g_renderer);
    g_renderer = NULL;
  }
  if (g_gl_ctx) {
    SDL_GL_DestroyContext(g_gl_ctx);
    g_gl_ctx = NULL;
  }
  if (g_window) {
    SDL_StopTextInput(g_window);
    SDL_DestroyWindow(g_window);
    g_window = NULL;
  }
  TTF_Quit();
  SDL_Quit();
}

bool WindowShouldClose(void) {
  sdl3_poll_events();
  return g_should_close;
}

void BeginDrawing(void) { sdl3_frame_start(); }

void EndDrawing(void) {
  if (g_use_gl) {
    SDL_GL_SwapWindow(g_window);
  } else if (g_renderer) {
    SDL_RenderPresent(g_renderer);
  }

  if (g_target_fps > 0) {
    float target_ms = 1000.0f / (float)g_target_fps;
    uint64_t now = SDL_GetTicks();
    float elapsed = (float)(now - g_frame_start_tick);
    if (elapsed < target_ms) {
      SDL_Delay((uint32_t)(target_ms - elapsed));
    }
  }
  g_frame_active = false;
}

void ClearBackground(Color color) {
  if (g_use_gl) {
    glClearColor(color.r / 255.0f, color.g / 255.0f, color.b / 255.0f,
                 color.a / 255.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  } else if (g_renderer) {
    SDL_SetRenderDrawColor(g_renderer, color.r, color.g, color.b, color.a);
    SDL_RenderClear(g_renderer);
  }
}

void SetTargetFPS(int fps) { g_target_fps = fps; }

float GetFrameTime(void) {
  sdl3_frame_start();
  return g_last_dt;
}

int GetFPS(void) {
  if (g_last_dt <= 0.000001f) {
    return 0;
  }
  return (int)lroundf(1.0f / g_last_dt);
}

double GetTime(void) {
  uint64_t now = SDL_GetTicks();
  return (double)(now - g_start_tick) / 1000.0;
}

void SetExitKey(KeyboardKey key) { g_exit_key = key; }

int GetScreenWidth(void) {
  int w = 0;
  int h = 0;
  if (g_window) {
    SDL_GetWindowSize(g_window, &w, &h);
  }
  return w;
}

int GetScreenHeight(void) {
  int w = 0;
  int h = 0;
  if (g_window) {
    SDL_GetWindowSize(g_window, &w, &h);
  }
  return h;
}

bool IsKeyDown(KeyboardKey key) {
  SDL_Scancode sc = sdl_scancode_from_key(key);
  if (sc == SDL_SCANCODE_UNKNOWN) {
    return false;
  }
  return g_key_down[sc];
}

bool IsKeyPressed(KeyboardKey key) {
  SDL_Scancode sc = sdl_scancode_from_key(key);
  if (sc == SDL_SCANCODE_UNKNOWN) {
    return false;
  }
  return g_key_pressed[sc];
}

bool IsKeyReleased(KeyboardKey key) {
  SDL_Scancode sc = sdl_scancode_from_key(key);
  if (sc == SDL_SCANCODE_UNKNOWN) {
    return false;
  }
  return g_key_released[sc];
}

bool IsMouseButtonDown(int button) {
  if (button < 0 ||
      button >= (int)(sizeof(g_mouse_down) / sizeof(g_mouse_down[0]))) {
    return false;
  }
  return g_mouse_down[button];
}

bool IsMouseButtonPressed(int button) {
  if (button < 0 ||
      button >= (int)(sizeof(g_mouse_pressed) / sizeof(g_mouse_pressed[0]))) {
    return false;
  }
  return g_mouse_pressed[button];
}

Vector2 GetMousePosition(void) {
  float x = 0.0f;
  float y = 0.0f;
  SDL_GetMouseState(&x, &y);
  return (Vector2){x, y};
}

float GetMouseWheelMove(void) { return g_mouse_wheel; }

Vector2 GetMouseDelta(void) { return (Vector2){g_mouse_dx, g_mouse_dy}; }

void SetRelativeMouseMode(int enabled) {
  if (g_window) {
    SDL_SetWindowRelativeMouseMode(g_window, enabled ? true : false);
  }
}

void SetMousePosition(int x, int y) {
  if (g_window) {
    SDL_WarpMouseInWindow(g_window, (float)x, (float)y);
  }
}

void SetMouseVisible(int visible) {
  if (visible) {
    SDL_ShowCursor();
  } else {
    SDL_HideCursor();
  }
}

int GetCharPressed(void) {
  if (g_text_head == g_text_tail) {
    return 0;
  }
  unsigned char c = g_text_queue[g_text_head];
  g_text_head =
      (g_text_head + 1) %
      (int)(sizeof(g_text_queue) / sizeof(g_text_queue[0]));
  return (int)c;
}

static void sdl3_set_color(Color color) {
  SDL_SetRenderDrawColor(g_renderer, color.r, color.g, color.b, color.a);
}

static SDL_FColor sdl3_color_f(Color color) {
  return (SDL_FColor){color.r / 255.0f, color.g / 255.0f, color.b / 255.0f,
                      color.a / 255.0f};
}

void DrawLine(int startPosX, int startPosY, int endPosX, int endPosY,
              Color color) {
  sdl3_set_color(color);
  SDL_RenderLine(g_renderer, (float)startPosX, (float)startPosY,
                 (float)endPosX, (float)endPosY);
}

void DrawLineEx(Vector2 startPos, Vector2 endPos, float thick, Color color) {
  if (thick <= 1.0f) {
    DrawLine((int)startPos.x, (int)startPos.y, (int)endPos.x, (int)endPos.y,
             color);
    return;
  }

  Vector2 dir = Vector2Subtract(endPos, startPos);
  float len = Vector2Length(dir);
  if (len <= 0.00001f) {
    return;
  }
  dir = Vector2Scale(dir, 1.0f / len);
  Vector2 normal = (Vector2){-dir.y, dir.x};
  Vector2 offset = Vector2Scale(normal, thick * 0.5f);

  SDL_Vertex verts[4];
  SDL_FColor sdlc = sdl3_color_f(color);

  verts[0].position.x = startPos.x + offset.x;
  verts[0].position.y = startPos.y + offset.y;
  verts[1].position.x = startPos.x - offset.x;
  verts[1].position.y = startPos.y - offset.y;
  verts[2].position.x = endPos.x - offset.x;
  verts[2].position.y = endPos.y - offset.y;
  verts[3].position.x = endPos.x + offset.x;
  verts[3].position.y = endPos.y + offset.y;

  for (int i = 0; i < 4; i++) {
    verts[i].color = sdlc;
    verts[i].tex_coord.x = 0.0f;
    verts[i].tex_coord.y = 0.0f;
  }

  int indices[6] = {0, 1, 2, 0, 2, 3};
  SDL_RenderGeometry(g_renderer, NULL, verts, 4, indices, 6);
}

void DrawPixel(int x, int y, Color color) {
  sdl3_set_color(color);
  SDL_RenderPoint(g_renderer, (float)x, (float)y);
}

static void sdl3_draw_circle_filled(Vector2 center, float radius, Color color) {
  if (radius <= 0.0f) {
    return;
  }
  int segments = (int)fmaxf(12.0f, radius * 0.6f);
  float step = (float)(2.0 * M_PI / segments);

  SDL_Vertex *verts = malloc(sizeof(SDL_Vertex) * (segments + 2));
  int *indices = malloc(sizeof(int) * segments * 3);
  if (!verts || !indices) {
    free(verts);
    free(indices);
    return;
  }

  SDL_FColor sdlc = sdl3_color_f(color);
  verts[0].position.x = center.x;
  verts[0].position.y = center.y;
  verts[0].color = sdlc;
  verts[0].tex_coord.x = 0.0f;
  verts[0].tex_coord.y = 0.0f;

  for (int i = 0; i <= segments; i++) {
    float a = step * i;
    verts[i + 1].position.x = center.x + cosf(a) * radius;
    verts[i + 1].position.y = center.y + sinf(a) * radius;
    verts[i + 1].color = sdlc;
    verts[i + 1].tex_coord.x = 0.0f;
    verts[i + 1].tex_coord.y = 0.0f;
  }

  for (int i = 0; i < segments; i++) {
    indices[i * 3 + 0] = 0;
    indices[i * 3 + 1] = i + 1;
    indices[i * 3 + 2] = i + 2;
  }

  SDL_RenderGeometry(g_renderer, NULL, verts, segments + 2, indices,
                     segments * 3);
  free(verts);
  free(indices);
}

static void sdl3_draw_circle_lines(Vector2 center, float radius, Color color) {
  if (radius <= 0.0f) {
    return;
  }
  int segments = (int)fmaxf(16.0f, radius * 0.6f);
  float step = (float)(2.0 * M_PI / segments);
  sdl3_set_color(color);

  float prev_x = center.x + cosf(0.0f) * radius;
  float prev_y = center.y + sinf(0.0f) * radius;
  for (int i = 1; i <= segments; i++) {
    float a = step * i;
    float x = center.x + cosf(a) * radius;
    float y = center.y + sinf(a) * radius;
    SDL_RenderLine(g_renderer, prev_x, prev_y, x, y);
    prev_x = x;
    prev_y = y;
  }
}

void DrawCircle(int centerX, int centerY, float radius, Color color) {
  DrawCircleV((Vector2){(float)centerX, (float)centerY}, radius, color);
}

void DrawCircleV(Vector2 center, float radius, Color color) {
  sdl3_draw_circle_filled(center, radius, color);
}

void DrawCircleLines(int centerX, int centerY, float radius, Color color) {
  DrawCircleLinesV((Vector2){(float)centerX, (float)centerY}, radius, color);
}

void DrawCircleLinesV(Vector2 center, float radius, Color color) {
  sdl3_draw_circle_lines(center, radius, color);
}

void DrawEllipse(int centerX, int centerY, int radiusH, int radiusV,
                 Color color) {
  if (radiusH <= 0 || radiusV <= 0) {
    return;
  }
  int segments = (int)fmaxf(16.0f, (float)(radiusH + radiusV) * 0.3f);
  float step = (float)(2.0 * M_PI / segments);

  SDL_Vertex *verts = malloc(sizeof(SDL_Vertex) * (segments + 2));
  int *indices = malloc(sizeof(int) * segments * 3);
  if (!verts || !indices) {
    free(verts);
    free(indices);
    return;
  }

  SDL_FColor sdlc = sdl3_color_f(color);
  verts[0].position.x = (float)centerX;
  verts[0].position.y = (float)centerY;
  verts[0].color = sdlc;
  verts[0].tex_coord.x = 0.0f;
  verts[0].tex_coord.y = 0.0f;

  for (int i = 0; i <= segments; i++) {
    float a = step * i;
    verts[i + 1].position.x = centerX + cosf(a) * radiusH;
    verts[i + 1].position.y = centerY + sinf(a) * radiusV;
    verts[i + 1].color = sdlc;
    verts[i + 1].tex_coord.x = 0.0f;
    verts[i + 1].tex_coord.y = 0.0f;
  }

  for (int i = 0; i < segments; i++) {
    indices[i * 3 + 0] = 0;
    indices[i * 3 + 1] = i + 1;
    indices[i * 3 + 2] = i + 2;
  }

  SDL_RenderGeometry(g_renderer, NULL, verts, segments + 2, indices,
                     segments * 3);
  free(verts);
  free(indices);
}

void DrawTriangle(Vector2 v1, Vector2 v2, Vector2 v3, Color color) {
  SDL_Vertex verts[3];
  SDL_FColor sdlc = sdl3_color_f(color);

  verts[0].position.x = v1.x;
  verts[0].position.y = v1.y;
  verts[1].position.x = v2.x;
  verts[1].position.y = v2.y;
  verts[2].position.x = v3.x;
  verts[2].position.y = v3.y;

  for (int i = 0; i < 3; i++) {
    verts[i].color = sdlc;
    verts[i].tex_coord.x = 0.0f;
    verts[i].tex_coord.y = 0.0f;
  }

  int indices[3] = {0, 1, 2};
  SDL_RenderGeometry(g_renderer, NULL, verts, 3, indices, 3);
}

void DrawPoly(Vector2 center, int sides, float radius, float rotation,
              Color color) {
  if (sides < 3 || radius <= 0.0f) {
    return;
  }

  float rot = rotation * (float)M_PI / 180.0f;
  float step = (float)(2.0 * M_PI / sides);
  int vertex_count = sides + 2;

  SDL_Vertex *verts = malloc(sizeof(SDL_Vertex) * vertex_count);
  int *indices = malloc(sizeof(int) * sides * 3);
  if (!verts || !indices) {
    free(verts);
    free(indices);
    return;
  }

  SDL_FColor sdlc = sdl3_color_f(color);
  verts[0].position.x = center.x;
  verts[0].position.y = center.y;
  verts[0].color = sdlc;
  verts[0].tex_coord.x = 0.0f;
  verts[0].tex_coord.y = 0.0f;

  for (int i = 0; i <= sides; i++) {
    float a = rot + step * i;
    verts[i + 1].position.x = center.x + cosf(a) * radius;
    verts[i + 1].position.y = center.y + sinf(a) * radius;
    verts[i + 1].color = sdlc;
    verts[i + 1].tex_coord.x = 0.0f;
    verts[i + 1].tex_coord.y = 0.0f;
  }

  for (int i = 0; i < sides; i++) {
    indices[i * 3 + 0] = 0;
    indices[i * 3 + 1] = i + 1;
    indices[i * 3 + 2] = i + 2;
  }

  SDL_RenderGeometry(g_renderer, NULL, verts, vertex_count, indices,
                     sides * 3);
  free(verts);
  free(indices);
}

void DrawPolyLines(Vector2 center, int sides, float radius, float rotation,
                   Color color) {
  if (sides < 3 || radius <= 0.0f) {
    return;
  }
  float rot = rotation * (float)M_PI / 180.0f;
  float step = (float)(2.0 * M_PI / sides);
  sdl3_set_color(color);

  float prev_x = center.x + cosf(rot) * radius;
  float prev_y = center.y + sinf(rot) * radius;
  for (int i = 1; i <= sides; i++) {
    float a = rot + step * i;
    float x = center.x + cosf(a) * radius;
    float y = center.y + sinf(a) * radius;
    SDL_RenderLine(g_renderer, prev_x, prev_y, x, y);
    prev_x = x;
    prev_y = y;
  }
}

void DrawRectangle(int posX, int posY, int width, int height, Color color) {
  SDL_FRect rect = {(float)posX, (float)posY, (float)width, (float)height};
  sdl3_set_color(color);
  SDL_RenderFillRect(g_renderer, &rect);
}

void DrawRectangleLines(int posX, int posY, int width, int height,
                        Color color) {
  SDL_FRect rect = {(float)posX, (float)posY, (float)width, (float)height};
  sdl3_set_color(color);
  SDL_RenderRect(g_renderer, &rect);
}

static void sdl3_draw_rounded_rect(Rectangle rec, float roundness, Color color,
                                   bool outline) {
  float radius = fminf(rec.width, rec.height) * 0.5f * roundness;
  if (radius <= 0.5f) {
    if (outline) {
      DrawRectangleLines((int)rec.x, (int)rec.y, (int)rec.width,
                         (int)rec.height, color);
    } else {
      DrawRectangle((int)rec.x, (int)rec.y, (int)rec.width, (int)rec.height,
                    color);
    }
    return;
  }

  float left = rec.x;
  float right = rec.x + rec.width;
  float top = rec.y;
  float bottom = rec.y + rec.height;

  if (!outline) {
    DrawRectangle((int)(left + radius), (int)top,
                  (int)(rec.width - 2.0f * radius), (int)rec.height, color);
    DrawRectangle((int)left, (int)(top + radius), (int)radius,
                  (int)(rec.height - 2.0f * radius), color);
    DrawRectangle((int)(right - radius), (int)(top + radius), (int)radius,
                  (int)(rec.height - 2.0f * radius), color);

    sdl3_draw_circle_filled((Vector2){left + radius, top + radius}, radius,
                            color);
    sdl3_draw_circle_filled((Vector2){right - radius, top + radius}, radius,
                            color);
    sdl3_draw_circle_filled((Vector2){left + radius, bottom - radius}, radius,
                            color);
    sdl3_draw_circle_filled((Vector2){right - radius, bottom - radius}, radius,
                            color);
  } else {
    DrawLine((int)(left + radius), (int)top, (int)(right - radius), (int)top,
             color);
    DrawLine((int)(left + radius), (int)bottom, (int)(right - radius),
             (int)bottom, color);
    DrawLine((int)left, (int)(top + radius), (int)left, (int)(bottom - radius),
             color);
    DrawLine((int)right, (int)(top + radius), (int)right,
             (int)(bottom - radius), color);

    sdl3_draw_circle_lines((Vector2){left + radius, top + radius}, radius,
                           color);
    sdl3_draw_circle_lines((Vector2){right - radius, top + radius}, radius,
                           color);
    sdl3_draw_circle_lines((Vector2){left + radius, bottom - radius}, radius,
                           color);
    sdl3_draw_circle_lines((Vector2){right - radius, bottom - radius}, radius,
                           color);
  }
}

void DrawRectangleRounded(Rectangle rec, float roundness, int segments,
                          Color color) {
  (void)segments;
  sdl3_draw_rounded_rect(rec, roundness, color, false);
}

void DrawRectangleRoundedLines(Rectangle rec, float roundness, int segments,
                               Color color) {
  (void)segments;
  sdl3_draw_rounded_rect(rec, roundness, color, true);
}

void DrawRing(Vector2 center, float innerRadius, float outerRadius,
              float startAngle, float endAngle, int segments, Color color) {
  if (outerRadius <= 0.0f || innerRadius < 0.0f ||
      outerRadius <= innerRadius) {
    return;
  }
  if (segments < 3) {
    segments = 3;
  }

  float start = startAngle * (float)M_PI / 180.0f;
  float end = endAngle * (float)M_PI / 180.0f;
  float step = (end - start) / segments;

  int vert_count = (segments + 1) * 2;
  SDL_Vertex *verts = malloc(sizeof(SDL_Vertex) * vert_count);
  int *indices = malloc(sizeof(int) * segments * 6);
  if (!verts || !indices) {
    free(verts);
    free(indices);
    return;
  }

  SDL_FColor sdlc = sdl3_color_f(color);
  for (int i = 0; i <= segments; i++) {
    float a = start + step * i;
    float cos_a = cosf(a);
    float sin_a = sinf(a);
    float x_outer = center.x + cos_a * outerRadius;
    float y_outer = center.y + sin_a * outerRadius;
    float x_inner = center.x + cos_a * innerRadius;
    float y_inner = center.y + sin_a * innerRadius;

    verts[i * 2].position.x = x_outer;
    verts[i * 2].position.y = y_outer;
    verts[i * 2].color = sdlc;
    verts[i * 2].tex_coord.x = 0.0f;
    verts[i * 2].tex_coord.y = 0.0f;

    verts[i * 2 + 1].position.x = x_inner;
    verts[i * 2 + 1].position.y = y_inner;
    verts[i * 2 + 1].color = sdlc;
    verts[i * 2 + 1].tex_coord.x = 0.0f;
    verts[i * 2 + 1].tex_coord.y = 0.0f;
  }

  for (int i = 0; i < segments; i++) {
    int idx = i * 2;
    indices[i * 6 + 0] = idx;
    indices[i * 6 + 1] = idx + 1;
    indices[i * 6 + 2] = idx + 2;
    indices[i * 6 + 3] = idx + 1;
    indices[i * 6 + 4] = idx + 3;
    indices[i * 6 + 5] = idx + 2;
  }

  SDL_RenderGeometry(g_renderer, NULL, verts, vert_count, indices,
                     segments * 6);
  free(verts);
  free(indices);
}

const char *TextFormat(const char *text, ...) {
  va_list args;
  va_start(args, text);
  g_text_buffer_index = (g_text_buffer_index + 1) % 8;
  vsnprintf(g_text_buffers[g_text_buffer_index],
            sizeof(g_text_buffers[g_text_buffer_index]), text, args);
  va_end(args);
  return g_text_buffers[g_text_buffer_index];
}

void DrawText(const char *text, int posX, int posY, int fontSize,
              Color color) {
  if (!text || !text[0]) {
    return;
  }

  TTF_Font *font = sdl3_get_font(fontSize);
  if (!font) {
    return;
  }

  SDL_Color sdlc = {color.r, color.g, color.b, color.a};
  size_t len = strlen(text);
  SDL_Surface *surface = TTF_RenderText_Blended(font, text, len, sdlc);
  if (!surface) {
    return;
  }

  SDL_Texture *texture = SDL_CreateTextureFromSurface(g_renderer, surface);
  if (texture) {
    SDL_FRect dst = {(float)posX, (float)posY, (float)surface->w,
                     (float)surface->h};
    SDL_RenderTexture(g_renderer, texture, NULL, &dst);
    SDL_DestroyTexture(texture);
  }

  SDL_DestroySurface(surface);
}

int MeasureText(const char *text, int fontSize) {
  if (!text || !text[0]) {
    return 0;
  }
  TTF_Font *font = sdl3_get_font(fontSize);
  if (!font) {
    return 0;
  }
  int w = 0;
  int h = 0;
  size_t len = strlen(text);
  if (!TTF_GetStringSize(font, text, len, &w, &h)) {
    return 0;
  }
  return w;
}

Color Fade(Color color, float alpha) {
  if (alpha < 0.0f) {
    alpha = 0.0f;
  }
  if (alpha > 1.0f) {
    alpha = 1.0f;
  }
  color.a = (unsigned char)lroundf((float)color.a * alpha);
  return color;
}

bool CheckCollisionPointRec(Vector2 point, Rectangle rec) {
  return point.x >= rec.x && point.x <= rec.x + rec.width && point.y >= rec.y &&
         point.y <= rec.y + rec.height;
}
