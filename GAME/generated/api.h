#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// --- Function prototypes ---
#define API_FN(ret, name, sig) ret name sig;
#include "api.def"
#undef API_FN

// --- Constants ---
#ifndef PI
#define PI 3.14159265358979323846f
#endif

// --- Color Constants ---
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

#ifdef __cplusplus
}
#endif
