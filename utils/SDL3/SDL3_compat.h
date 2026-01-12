#ifndef SDL3_COMPAT_H
#define SDL3_COMPAT_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#ifndef PI
#define PI 3.14159265358979323846f
#endif

typedef struct {
  float x;
  float y;
} Vector2;

typedef struct {
  float x;
  float y;
  float width;
  float height;
} Rectangle;

typedef struct {
  unsigned char r;
  unsigned char g;
  unsigned char b;
  unsigned char a;
} Color;

typedef enum {
  KEY_NULL = 0,
  KEY_ONE = 1,
  KEY_TWO,
  KEY_THREE,
  KEY_FOUR,
  KEY_FIVE,
  KEY_SIX,
  KEY_SEVEN,
  KEY_EIGHT,
  KEY_NINE,
  KEY_ZERO,
  KEY_KP_1 = 100,
  KEY_KP_2,
  KEY_KP_3,
  KEY_KP_4,
  KEY_KP_5,
  KEY_KP_6,
  KEY_KP_7,
  KEY_KP_8,
  KEY_KP_9,
  KEY_KP_0,
  KEY_A = 200,
  KEY_B,
  KEY_C,
  KEY_D,
  KEY_E,
  KEY_F,
  KEY_G,
  KEY_H,
  KEY_I,
  KEY_J,
  KEY_K,
  KEY_L,
  KEY_M,
  KEY_N,
  KEY_O,
  KEY_P,
  KEY_Q,
  KEY_R,
  KEY_S,
  KEY_T,
  KEY_U,
  KEY_V,
  KEY_W,
  KEY_X,
  KEY_Y,
  KEY_Z,
  KEY_SPACE = 260,
  KEY_TAB,
  KEY_ENTER,
  KEY_BACKSPACE,
  KEY_ESCAPE,
  KEY_UP,
  KEY_DOWN,
  KEY_LEFT,
  KEY_RIGHT,
  KEY_MINUS,
  KEY_EQUAL,
  KEY_F5
} KeyboardKey;

typedef enum {
  MOUSE_BUTTON_LEFT = 1,
  MOUSE_BUTTON_MIDDLE = 2,
  MOUSE_BUTTON_RIGHT = 3
} MouseButton;

#define MOUSE_LEFT_BUTTON MOUSE_BUTTON_LEFT
#define MOUSE_RIGHT_BUTTON MOUSE_BUTTON_RIGHT

static const Color BLACK = {0, 0, 0, 255};
static const Color WHITE = {255, 255, 255, 255};
static const Color RAYWHITE = {245, 245, 245, 255};
static const Color RED = {230, 41, 55, 255};
static const Color GREEN = {0, 228, 48, 255};
static const Color BLUE = {0, 121, 241, 255};
static const Color DARKGREEN = {0, 117, 44, 255};
static const Color GRAY = {130, 130, 130, 255};
static const Color DARKGRAY = {80, 80, 80, 255};
static const Color GOLD = {255, 203, 0, 255};
static const Color ORANGE = {255, 161, 0, 255};
static const Color PINK = {255, 109, 194, 255};

void InitWindow(int width, int height, const char *title);
void SDL3_SetUseGL(int enable);
void CloseWindow(void);
bool WindowShouldClose(void);
void BeginDrawing(void);
void EndDrawing(void);
void ClearBackground(Color color);
void SetTargetFPS(int fps);
float GetFrameTime(void);
int GetFPS(void);
double GetTime(void);
void SetExitKey(KeyboardKey key);
int GetScreenWidth(void);
int GetScreenHeight(void);

bool IsKeyDown(KeyboardKey key);
bool IsKeyPressed(KeyboardKey key);
bool IsKeyReleased(KeyboardKey key);

bool IsMouseButtonDown(int button);
bool IsMouseButtonPressed(int button);
Vector2 GetMousePosition(void);
float GetMouseWheelMove(void);
int GetCharPressed(void);
Vector2 GetMouseDelta(void);
void SetRelativeMouseMode(int enabled);
void SetMousePosition(int x, int y);
void SetMouseVisible(int visible);

void DrawLine(int startPosX, int startPosY, int endPosX, int endPosY, Color color);
void DrawLineEx(Vector2 startPos, Vector2 endPos, float thick, Color color);
void DrawPixel(int x, int y, Color color);
void DrawCircle(int centerX, int centerY, float radius, Color color);
void DrawCircleV(Vector2 center, float radius, Color color);
void DrawCircleLines(int centerX, int centerY, float radius, Color color);
void DrawCircleLinesV(Vector2 center, float radius, Color color);
void DrawEllipse(int centerX, int centerY, int radiusH, int radiusV, Color color);
void DrawTriangle(Vector2 v1, Vector2 v2, Vector2 v3, Color color);
void DrawPoly(Vector2 center, int sides, float radius, float rotation, Color color);
void DrawPolyLines(Vector2 center, int sides, float radius, float rotation,
                   Color color);
void DrawRectangle(int posX, int posY, int width, int height, Color color);
void DrawRectangleLines(int posX, int posY, int width, int height, Color color);
void DrawRectangleRounded(Rectangle rec, float roundness, int segments,
                          Color color);
void DrawRectangleRoundedLines(Rectangle rec, float roundness, int segments,
                               Color color);
void DrawRing(Vector2 center, float innerRadius, float outerRadius,
              float startAngle, float endAngle, int segments, Color color);

const char *TextFormat(const char *text, ...);
void DrawText(const char *text, int posX, int posY, int fontSize, Color color);
int MeasureText(const char *text, int fontSize);

Color Fade(Color color, float alpha);

bool CheckCollisionPointRec(Vector2 point, Rectangle rec);

static inline Vector2 Vector2Add(Vector2 v1, Vector2 v2) {
  return (Vector2){v1.x + v2.x, v1.y + v2.y};
}

static inline Vector2 Vector2Subtract(Vector2 v1, Vector2 v2) {
  return (Vector2){v1.x - v2.x, v1.y - v2.y};
}

static inline Vector2 Vector2Scale(Vector2 v, float scale) {
  return (Vector2){v.x * scale, v.y * scale};
}

static inline float Vector2Length(Vector2 v) {
  return sqrtf(v.x * v.x + v.y * v.y);
}

static inline float Vector2Distance(Vector2 v1, Vector2 v2) {
  return Vector2Length(Vector2Subtract(v1, v2));
}

static inline float Vector2DotProduct(Vector2 v1, Vector2 v2) {
  return v1.x * v2.x + v1.y * v2.y;
}

static inline Vector2 Vector2Normalize(Vector2 v) {
  float len = Vector2Length(v);
  if (len > 0.000001f) {
    return Vector2Scale(v, 1.0f / len);
  }
  return (Vector2){0.0f, 0.0f};
}

#endif
