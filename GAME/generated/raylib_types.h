#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// --- Raylib Type declarations ---
#define API_TYPE(vis, name, body) typedef struct name { body } name;
#include "raylib_types.def"
#undef API_TYPE

// --- Raylib Constants ---
#define BLACK     (Color){  0,   0,   0,   255 }
#define WHITE     (Color){ 255, 255, 255, 255 }
#define RAYWHITE  (Color){ 245, 245, 245, 255 }
#define RED       (Color){ 255,   0,   0,   255 }
#define GREEN     (Color){   0, 255,   0,   255 }
#define DARKGREEN (Color){   0, 128,   0,   255 }
#define GOLD      (Color){ 255, 203,   0,   255 }
#define GRAY      (Color){ 128, 128, 128, 255 }
#define DARKGRAY  (Color){  80,  80,  80, 255 }
#define PINK      (Color){ 255, 192, 203, 255 }
#define YELLOW    (Color){ 253, 249,   0,   255 }
#define ORANGE    (Color){ 255, 161,   0,   255 }
#define PURPLE    (Color){ 161,   0, 255,   255 }
#define LIME      (Color){   0, 255,   0,   255 }
#define SKYBLUE   (Color){ 102, 191, 255,   255 }
#define BLUE      (Color){   0, 121, 241,   255 }
#define MAROON    (Color){ 190,  33,  55,   255 }
#define BROWN     (Color){ 165,  42,  42,   255 }

#ifdef __cplusplus
}
#endif
