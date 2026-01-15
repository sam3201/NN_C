#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// --- Type declarations ---
#define API_TYPE(vis, name, body) typedef struct name { body } name;

// --- Function prototypes ---
#define API_FN(ret, name, sig) ret name sig;

// Include the complete API definition
#include "complete_api.def"

#undef API_TYPE
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
#define YELLOW    (Color){ 253, 249,   0,   255 }
#define ORANGE    (Color){ 255, 161,   0,   255 }
#define PURPLE    (Color){ 161,   0, 255,   255 }
#define LIME      (Color){   0, 255,   0,   255 }
#define SKYBLUE   (Color){ 102, 191, 255,   255 }
#define BLUE      (Color){   0, 121, 241,   255 }
#define MAROON    (Color){ 190,  33,  55,   255 }
#define BROWN     (Color){ 165,  42,  42,   255 }

// --- Keyboard Key Constants ---
#define KEY_NULL          (0)
#define KEY_SPACE         (SDL_SCANCODE_SPACE)
#define KEY_APOSTROPHE    (SDL_SCANCODE_APOSTROPHE)
#define KEY_COMMA         (SDL_SCANCODE_COMMA)
#define KEY_MINUS         (SDL_SCANCODE_MINUS)
#define KEY_PERIOD        (SDL_SCANCODE_PERIOD)
#define KEY_SLASH         (SDL_SCANCODE_SLASH)
#define KEY_ZERO          (SDL_SCANCODE_0)
#define KEY_ONE           (SDL_SCANCODE_1)
#define KEY_TWO           (SDL_SCANCODE_2)
#define KEY_THREE         (SDL_SCANCODE_3)
#define KEY_FOUR          (SDL_SCANCODE_4)
#define KEY_FIVE          (SDL_SCANCODE_5)
#define KEY_SIX           (SDL_SCANCODE_6)
#define KEY_SEVEN         (SDL_SCANCODE_7)
#define KEY_EIGHT         (SDL_SCANCODE_8)
#define KEY_NINE          (SDL_SCANCODE_9)
#define KEY_SEMICOLON     (SDL_SCANCODE_SEMICOLON)
#define KEY_EQUAL         (SDL_SCANCODE_EQUALS)
#define KEY_A             (SDL_SCANCODE_A)
#define KEY_B             (SDL_SCANCODE_B)
#define KEY_C             (SDL_SCANCODE_C)
#define KEY_D             (SDL_SCANCODE_D)
#define KEY_E             (SDL_SCANCODE_E)
#define KEY_F             (SDL_SCANCODE_F)
#define KEY_G             (SDL_SCANCODE_G)
#define KEY_H             (SDL_SCANCODE_H)
#define KEY_I             (SDL_SCANCODE_I)
#define KEY_J             (SDL_SCANCODE_J)
#define KEY_K             (SDL_SCANCODE_K)
#define KEY_L             (SDL_SCANCODE_L)
#define KEY_M             (SDL_SCANCODE_M)
#define KEY_N             (SDL_SCANCODE_N)
#define KEY_O             (SDL_SCANCODE_O)
#define KEY_P             (SDL_SCANCODE_P)
#define KEY_Q             (SDL_SCANCODE_Q)
#define KEY_R             (SDL_SCANCODE_R)
#define KEY_S             (SDL_SCANCODE_S)
#define KEY_T             (SDL_SCANCODE_T)
#define KEY_U             (SDL_SCANCODE_U)
#define KEY_V             (SDL_SCANCODE_V)
#define KEY_W             (SDL_SCANCODE_W)
#define KEY_X             (SDL_SCANCODE_X)
#define KEY_Y             (SDL_SCANCODE_Y)
#define KEY_Z             (SDL_SCANCODE_Z)
#define KEY_LEFT_SHIFT     (SDL_SCANCODE_LSHIFT)
#define KEY_RIGHT_SHIFT    (SDL_SCANCODE_RSHIFT)
#define KEY_LEFT_CONTROL   (SDL_SCANCODE_LCTRL)
#define KEY_RIGHT_CONTROL  (SDL_SCANCODE_RCTRL)
#define KEY_LEFT_ALT       (SDL_SCANCODE_LALT)
#define KEY_RIGHT_ALT      (SDL_SCANCODE_RALT)
#define KEY_ENTER          (SDL_SCANCODE_RETURN)
#define KEY_BACKSPACE      (SDL_SCANCODE_BACKSPACE)
#define KEY_TAB            (SDL_SCANCODE_TAB)

// --- Mouse Button Constants ---
#define MOUSE_LEFT_BUTTON     (1)  // SDL_BUTTON_LEFT
#define MOUSE_RIGHT_BUTTON    (3)  // SDL_BUTTON_RIGHT
#define MOUSE_MIDDLE_BUTTON   (2)  // SDL_BUTTON_MIDDLE

#ifdef __cplusplus
}
#endif
