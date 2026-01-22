// Diep.io Tank Game with SDL graphics and one-hot vector AI inputs
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// SDL includes
#include <SDL3/SDL.h>
#include <SDL3/SDL_ttf.h>

// Game constants
#define SCREEN_WIDTH 1200
#define SCREEN_HEIGHT 800
#define TANK_SIZE 30
#define BULLET_SIZE 8
#define BULLET_SPEED 8.0f
#define TANK_SPEED 3.0f
#define TANK_ROTATION_SPEED 0.05f
#define MAX_BULLETS 50
#define FIRE_RATE 300 // milliseconds
#define ARENA_MARGIN 100

// Entity types for one-hot encoding
#define ENTITY_EMPTY 0.0f
#define ENTITY_SELF 1.0f
#define ENTITY_ENEMY 2.0f
#define ENTITY_BULLET_SELF 3.0f
#define ENTITY_BULLET_ENEMY 4.0f

// Colors
#define COLOR_TANK_PLAYER 0x00FF00FF
#define COLOR_TANK_AI 0xFF0000FF
#define COLOR_BULLET 0xFFFF00FF
#define COLOR_ARENA 0x808080FF

// Grid dimensions for AI observation
#define GRID_WIDTH 40
#define GRID_HEIGHT 30
#define AI_INPUT_SIZE (GRID_WIDTH * GRID_HEIGHT * 5) // 5 channels per cell

typedef struct {
  float x, y;
  float vx, vy;
  float angle;
  float radius;
  uint32_t color;
  int health;
  bool alive;
  int entity_type; // For one-hot encoding
} Tank;

typedef struct {
  float x, y;
  float vx, vy;
  float radius;
  uint32_t color;
  bool active;
  int owner;       // 0 = player, 1 = AI
  int entity_type; // For one-hot encoding
} Bullet;

typedef struct {
  Tank player;
  Tank ai_tank;
  uint32_t last_fire_time[2]; // 0 = player, 1 = AI
  int score[2];
  bool game_running;
  bool paused;

  // SDL components
  SDL_Window *window;
  SDL_Renderer *renderer;
  TTF_Font *font;

  // AI observation grid
  Bullet bullets[MAX_BULLETS];
  float ai_observation[AI_INPUT_SIZE];
} GameState;

// Function prototypes
void init_game(GameState *game);
void handle_input(GameState *game);
void update_game(GameState *game);
void render_game(GameState *game);
void cleanup_game(GameState *game);
void fire_bullet(GameState *game, int owner);
void update_bullets(GameState *game);
void check_collisions(GameState *game);
void respawn_tank(Tank *tank);
float get_distance(float x1, float y1, float x2, float y2);
uint32_t get_current_time_ms();

// One-hot encoding functions
void clear_observation_grid(GameState *game);
void update_observation_grid(GameState *game);
void add_entity_to_grid(GameState *game, float x, float y, float entity_type);
void get_ai_input_vector(GameState *game, float *input_vector);

// AI functions
void init_ai();
void update_ai_decision(GameState *game);
int get_ai_action_from_observation(GameState *game);

// Utility functions
void draw_circle(SDL_Renderer *renderer, int x, int y, int radius,
                 uint32_t color);
void draw_tank(SDL_Renderer *renderer, Tank *tank);
void draw_bullet(SDL_Renderer *renderer, Bullet *bullet);
void draw_ui(SDL_Renderer *renderer, TTF_Font *font, GameState *game);

int main(int argc, char *argv[]) {
  GameState game;

  // Initialize SDL
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    printf("SDL initialization failed: %s\n", SDL_GetError());
    return 1;
  }

  // Initialize TTF
  if (TTF_Init() < 0) {
    printf("TTF initialization failed: %s\n", TTF_GetError());
    SDL_Quit();
    return 1;
  }

  // Create window
  game.window = SDL_CreateWindow("Diep.io Tank Game - SDL Version",
                                 SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                 SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
  if (!game.window) {
    printf("Window creation failed: %s\n", SDL_GetError());
    TTF_Quit();
    SDL_Quit();
    return 1;
  }

  // Create renderer
  game.renderer = SDL_CreateRenderer(
      game.window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
  if (!game.renderer) {
    printf("Renderer creation failed: %s\n", SDL_GetError());
    SDL_DestroyWindow(game.window);
    TTF_Quit();
    SDL_Quit();
    return 1;
  }

  // Load font - try multiple font paths
  const char *font_paths[] = {"/System/Library/Fonts/Arial.ttf",
                              "/System/Library/Fonts/Helvetica.ttc",
                              "/System/Library/Fonts/Geneva.ttf",
                              "/System/Library/Fonts/SFNS.ttf", NULL};

  game.font = NULL;
  for (int i = 0; font_paths[i] != NULL; i++) {
    game.font = TTF_OpenFont(font_paths[i], 24);
    if (game.font) {
      break;
    }
  }

  if (!game.font) {
    printf(
        "Warning: Could not load any system font, continuing without text\n");
    // Continue without font - game will still work
  }

  // Initialize game
  init_game(&game);

  // Initialize AI
  init_ai();

  // Main game loop
  while (game.game_running) {
    handle_input(&game);
    if (!game.paused) {
      update_game(&game);
    }
    render_game(&game);
    SDL_Delay(16); // ~60 FPS
  }

  // Cleanup
  cleanup_game(&game);

  return 0;
}

void init_game(GameState *game) {
  // Initialize player tank
  game->player.x = SCREEN_WIDTH / 4;
  game->player.y = SCREEN_HEIGHT / 2;
  game->player.vx = 0;
  game->player.vy = 0;
  game->player.angle = 0;
  game->player.radius = TANK_SIZE;
  game->player.color = COLOR_TANK_PLAYER;
  game->player.health = 100;
  game->player.alive = true;
  game->player.entity_type = ENTITY_SELF;

  // Initialize AI tank
  game->ai_tank.x = 3 * SCREEN_WIDTH / 4;
  game->ai_tank.y = SCREEN_HEIGHT / 2;
  game->ai_tank.vx = 0;
  game->ai_tank.vy = 0;
  game->ai_tank.angle = M_PI;
  game->ai_tank.radius = TANK_SIZE;
  game->ai_tank.color = COLOR_TANK_AI;
  game->ai_tank.health = 100;
  game->ai_tank.alive = true;
  game->ai_tank.entity_type = ENTITY_ENEMY;

  // Initialize bullets
  for (int i = 0; i < MAX_BULLETS; i++) {
    game->bullets[i].active = false;
    game->bullets[i].entity_type =
        ENTITY_BULLET_ENEMY; // Default, will be updated
  }

  // Initialize game state
  game->last_fire_time[0] = 0;
  game->last_fire_time[1] = 0;
  game->score[0] = 0;
  game->score[1] = 0;
  game->game_running = true;
  game->paused = false;

  // Initialize AI observation
  memset(game->ai_observation, 0, sizeof(game->ai_observation));
}

void handle_input(GameState *game) {
  SDL_Event event;

  while (SDL_PollEvent(&event)) {
    switch (event.type) {
    case SDL_QUIT:
      game->game_running = false;
      break;
    case SDL_KEYDOWN:
      switch (event.key.keysym.sym) {
      case SDLK_ESCAPE:
      case SDLK_q:
        game->game_running = false;
        break;
      case SDLK_p:
        game->paused = !game->paused;
        break;
      }
      break;
    }
  }

  // Get keyboard state for continuous input
  const Uint8 *keys = SDL_GetKeyboardState(NULL);

  if (game->player.alive && !game->paused) {
    // Movement
    if (keys[SDL_SCANCODE_W] || keys[SDL_SCANCODE_UP]) {
      game->player.vx = cosf(game->player.angle) * TANK_SPEED;
      game->player.vy = sinf(game->player.angle) * TANK_SPEED;
    } else if (keys[SDL_SCANCODE_S] || keys[SDL_SCANCODE_DOWN]) {
      game->player.vx = -cosf(game->player.angle) * TANK_SPEED;
      game->player.vy = -sinf(game->player.angle) * TANK_SPEED;
    } else {
      game->player.vx *= 0.9f; // Friction
      game->player.vy *= 0.9f;
    }

    // Rotation
    if (keys[SDL_SCANCODE_A] || keys[SDL_SCANCODE_LEFT]) {
      game->player.angle -= TANK_ROTATION_SPEED;
    }
    if (keys[SDL_SCANCODE_D] || keys[SDL_SCANCODE_RIGHT]) {
      game->player.angle += TANK_ROTATION_SPEED;
    }

    // Shooting
    if (keys[SDL_SCANCODE_SPACE]) {
      fire_bullet(game, 0);
    }
  }
}

void update_game(GameState *game) {
  // Update player position
  if (game->player.alive) {
    game->player.x += game->player.vx;
    game->player.y += game->player.vy;

    // Keep player in bounds
    if (game->player.x < ARENA_MARGIN + game->player.radius) {
      game->player.x = ARENA_MARGIN + game->player.radius;
      game->player.vx = 0;
    }
    if (game->player.x > SCREEN_WIDTH - ARENA_MARGIN - game->player.radius) {
      game->player.x = SCREEN_WIDTH - ARENA_MARGIN - game->player.radius;
      game->player.vx = 0;
    }
    if (game->player.y < ARENA_MARGIN + game->player.radius) {
      game->player.y = ARENA_MARGIN + game->player.radius;
      game->player.vy = 0;
    }
    if (game->player.y > SCREEN_HEIGHT - ARENA_MARGIN - game->player.radius) {
      game->player.y = SCREEN_HEIGHT - ARENA_MARGIN - game->player.radius;
      game->player.vy = 0;
    }
  }

  // Update AI
  if (game->ai_tank.alive) {
    update_ai_decision(game);

    game->ai_tank.x += game->ai_tank.vx;
    game->ai_tank.y += game->ai_tank.vy;

    // Keep AI in bounds
    if (game->ai_tank.x < ARENA_MARGIN + game->ai_tank.radius) {
      game->ai_tank.x = ARENA_MARGIN + game->ai_tank.radius;
      game->ai_tank.vx = 0;
    }
    if (game->ai_tank.x > SCREEN_WIDTH - ARENA_MARGIN - game->ai_tank.radius) {
      game->ai_tank.x = SCREEN_WIDTH - ARENA_MARGIN - game->ai_tank.radius;
      game->ai_tank.vx = 0;
    }
    if (game->ai_tank.y < ARENA_MARGIN + game->ai_tank.radius) {
      game->ai_tank.y = ARENA_MARGIN + game->ai_tank.radius;
      game->ai_tank.vy = 0;
    }
    if (game->ai_tank.y > SCREEN_HEIGHT - ARENA_MARGIN - game->ai_tank.radius) {
      game->ai_tank.y = SCREEN_HEIGHT - ARENA_MARGIN - game->ai_tank.radius;
      game->ai_tank.vy = 0;
    }
  }

  // Update bullets
  update_bullets(game);

  // Check collisions
  check_collisions(game);

  // Update AI observation grid
  update_observation_grid(game);

  // Respawn tanks if needed
  if (!game->player.alive) {
    respawn_tank(&game->player);
  }
  if (!game->ai_tank.alive) {
    respawn_tank(&game->ai_tank);
  }
}

void render_game(GameState *game) {
  // Clear screen
  SDL_SetRenderDrawColor(game->renderer, 20, 20, 30, 255);
  SDL_RenderClear(game->renderer);

  // Draw arena boundaries
  SDL_SetRenderDrawColor(game->renderer, 128, 128, 128, 255);
  SDL_Rect arena = {ARENA_MARGIN, ARENA_MARGIN, SCREEN_WIDTH - 2 * ARENA_MARGIN,
                    SCREEN_HEIGHT - 2 * ARENA_MARGIN};
  SDL_RenderDrawRect(game->renderer, &arena);

  // Draw grid lines (for debugging AI observation)
  if (game->paused) {
    SDL_SetRenderDrawColor(game->renderer, 40, 40, 50, 255);
    int cell_width = (SCREEN_WIDTH - 2 * ARENA_MARGIN) / GRID_WIDTH;
    int cell_height = (SCREEN_HEIGHT - 2 * ARENA_MARGIN) / GRID_HEIGHT;

    for (int x = 0; x <= GRID_WIDTH; x++) {
      int line_x = ARENA_MARGIN + x * cell_width;
      SDL_RenderDrawLine(game->renderer, line_x, ARENA_MARGIN, line_x,
                         SCREEN_HEIGHT - ARENA_MARGIN);
    }
    for (int y = 0; y <= GRID_HEIGHT; y++) {
      int line_y = ARENA_MARGIN + y * cell_height;
      SDL_RenderDrawLine(game->renderer, ARENA_MARGIN, line_y,
                         SCREEN_WIDTH - ARENA_MARGIN, line_y);
    }
  }

  // Draw player tank
  if (game->player.alive) {
    draw_tank(game->renderer, &game->player);
  }

  // Draw AI tank
  if (game->ai_tank.alive) {
    draw_tank(game->renderer, &game->ai_tank);
  }

  // Draw bullets
  for (int i = 0; i < MAX_BULLETS; i++) {
    if (game->bullets[i].active) {
      draw_bullet(game->renderer, &game->bullets[i]);
    }
  }

  // Draw UI
  draw_ui(game->renderer, game->font, game);

  // Present
  SDL_RenderPresent(game->renderer);
}

void cleanup_game(GameState *game) {
  if (game->font) {
    TTF_CloseFont(game->font);
  }
  if (game->renderer) {
    SDL_DestroyRenderer(game->renderer);
  }
  if (game->window) {
    SDL_DestroyWindow(game->window);
  }

  TTF_Quit();
  SDL_Quit();
}

void fire_bullet(GameState *game, int owner) {
  uint32_t current_time = get_current_time_ms();

  if (current_time - game->last_fire_time[owner] < FIRE_RATE) {
    return; // Can't fire yet
  }

  Tank *tank = (owner == 0) ? &game->player : &game->ai_tank;

  if (!tank->alive) {
    return;
  }

  // Find inactive bullet
  for (int i = 0; i < MAX_BULLETS; i++) {
    if (!game->bullets[i].active) {
      game->bullets[i].x = tank->x + cosf(tank->angle) * (tank->radius + 15);
      game->bullets[i].y = tank->y + sinf(tank->angle) * (tank->radius + 15);
      game->bullets[i].vx = cosf(tank->angle) * BULLET_SPEED;
      game->bullets[i].vy = sinf(tank->angle) * BULLET_SPEED;
      game->bullets[i].radius = BULLET_SIZE;
      game->bullets[i].color = COLOR_BULLET;
      game->bullets[i].active = true;
      game->bullets[i].owner = owner;
      game->bullets[i].entity_type =
          (owner == 0) ? ENTITY_BULLET_SELF : ENTITY_BULLET_ENEMY;
      game->last_fire_time[owner] = current_time;
      break;
    }
  }
}

void update_bullets(GameState *game) {
  for (int i = 0; i < MAX_BULLETS; i++) {
    if (game->bullets[i].active) {
      game->bullets[i].x += game->bullets[i].vx;
      game->bullets[i].y += game->bullets[i].vy;

      // Check if bullet is out of bounds
      if (game->bullets[i].x < 0 || game->bullets[i].x > SCREEN_WIDTH ||
          game->bullets[i].y < 0 || game->bullets[i].y > SCREEN_HEIGHT) {
        game->bullets[i].active = false;
      }
    }
  }
}

void check_collisions(GameState *game) {
  // Check bullet-tank collisions
  for (int i = 0; i < MAX_BULLETS; i++) {
    if (!game->bullets[i].active)
      continue;

    // Check collision with player tank
    if (game->player.alive && game->bullets[i].owner == 1) {
      float dist = get_distance(game->bullets[i].x, game->bullets[i].y,
                                game->player.x, game->player.y);
      if (dist < game->bullets[i].radius + game->player.radius) {
        game->player.health -= 20;
        game->bullets[i].active = false;

        if (game->player.health <= 0) {
          game->player.alive = false;
          game->score[1]++;
        }
      }
    }

    // Check collision with AI tank
    if (game->ai_tank.alive && game->bullets[i].owner == 0) {
      float dist = get_distance(game->bullets[i].x, game->bullets[i].y,
                                game->ai_tank.x, game->ai_tank.y);
      if (dist < game->bullets[i].radius + game->ai_tank.radius) {
        game->ai_tank.health -= 20;
        game->bullets[i].active = false;

        if (game->ai_tank.health <= 0) {
          game->ai_tank.alive = false;
          game->score[0]++;
        }
      }
    }
  }
}

void respawn_tank(Tank *tank) {
  static uint32_t respawn_time = 0;
  uint32_t current_time = get_current_time_ms();

  if (respawn_time == 0) {
    respawn_time = current_time;
    return;
  }

  if (current_time - respawn_time > 3000) { // 3 second respawn
    if (tank->entity_type == ENTITY_SELF) {
      tank->x = SCREEN_WIDTH / 4;
      tank->y = SCREEN_HEIGHT / 2;
    } else {
      tank->x = 3 * SCREEN_WIDTH / 4;
      tank->y = SCREEN_HEIGHT / 2;
    }
    tank->vx = 0;
    tank->vy = 0;
    tank->health = 100;
    tank->alive = true;
    respawn_time = 0;
  }
}

float get_distance(float x1, float y1, float x2, float y2) {
  return sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

uint32_t get_current_time_ms() { return (uint32_t)SDL_GetTicks(); }

// One-hot encoding functions
void clear_observation_grid(GameState *game) {
  memset(game->ai_observation, 0, sizeof(game->ai_observation));
}

void add_entity_to_grid(GameState *game, float x, float y, float entity_type) {
  // Convert world coordinates to grid coordinates
  int grid_x = (int)((x - ARENA_MARGIN) / (SCREEN_WIDTH - 2 * ARENA_MARGIN) *
                     GRID_WIDTH);
  int grid_y = (int)((y - ARENA_MARGIN) / (SCREEN_HEIGHT - 2 * ARENA_MARGIN) *
                     GRID_HEIGHT);

  // Clamp to grid bounds
  if (grid_x < 0 || grid_x >= GRID_WIDTH || grid_y < 0 ||
      grid_y >= GRID_HEIGHT) {
    return;
  }

  // Calculate index in the one-hot encoded array
  int cell_index = grid_y * GRID_WIDTH + grid_x;
  int channel_index = (int)entity_type;

  if (channel_index >= 0 && channel_index < 5) {
    int array_index = cell_index * 5 + channel_index;
    if (array_index < AI_INPUT_SIZE) {
      game->ai_observation[array_index] = 1.0f;
    }
  }
}

void update_observation_grid(GameState *game) {
  clear_observation_grid(game);

  // Add AI tank (self)
  if (game->ai_tank.alive) {
    add_entity_to_grid(game, game->ai_tank.x, game->ai_tank.y, ENTITY_SELF);
  }

  // Add player tank (enemy)
  if (game->player.alive) {
    add_entity_to_grid(game, game->player.x, game->player.y, ENTITY_ENEMY);
  }

  // Add bullets
  for (int i = 0; i < MAX_BULLETS; i++) {
    if (game->bullets[i].active) {
      add_entity_to_grid(game, game->bullets[i].x, game->bullets[i].y,
                         game->bullets[i].entity_type);
    }
  }
}

void get_ai_input_vector(GameState *game, float *input_vector) {
  memcpy(input_vector, game->ai_observation, sizeof(game->ai_observation));
}

// AI functions
void init_ai() {
  // Initialize AI - placeholder for future MUZE integration
  printf("AI System Initialized\n");
}

void update_ai_decision(GameState *game) {
  if (!game->ai_tank.alive)
    return;

  // Get current observation
  float input_vector[AI_INPUT_SIZE];
  get_ai_input_vector(game, input_vector);

  // Simple AI behavior for now - will be replaced with MUZE
  int action = get_ai_action_from_observation(game);

  // Execute action
  switch (action) {
  case 0: // Move forward
    game->ai_tank.vx = cosf(game->ai_tank.angle) * TANK_SPEED;
    game->ai_tank.vy = sinf(game->ai_tank.angle) * TANK_SPEED;
    break;
  case 1: // Move backward
    game->ai_tank.vx = -cosf(game->ai_tank.angle) * TANK_SPEED;
    game->ai_tank.vy = -sinf(game->ai_tank.angle) * TANK_SPEED;
    break;
  case 2: // Rotate left
    game->ai_tank.angle -= TANK_ROTATION_SPEED * 2;
    game->ai_tank.vx *= 0.9f;
    game->ai_tank.vy *= 0.9f;
    break;
  case 3: // Rotate right
    game->ai_tank.angle += TANK_ROTATION_SPEED * 2;
    game->ai_tank.vx *= 0.9f;
    game->ai_tank.vy *= 0.9f;
    break;
  case 4: // Shoot
    fire_bullet(game, 1);
    break;
  }

  // Aim towards player
  if (game->player.alive) {
    float dx = game->player.x - game->ai_tank.x;
    float dy = game->player.y - game->ai_tank.y;
    float target_angle = atan2f(dy, dx);

    // Smooth rotation towards target
    float angle_diff = target_angle - game->ai_tank.angle;
    while (angle_diff > M_PI)
      angle_diff -= 2 * M_PI;
    while (angle_diff < -M_PI)
      angle_diff += 2 * M_PI;

    game->ai_tank.angle += angle_diff * 0.1f;
  }
}

int get_ai_action_from_observation(GameState *game) {
  // Simple rule-based AI for now
  // Will be replaced with neural network decision

  if (!game->player.alive)
    return 0; // Move forward if player dead

  float dx = game->player.x - game->ai_tank.x;
  float dy = game->player.y - game->ai_tank.y;
  float distance = sqrtf(dx * dx + dy * dy);

  // Aim towards player
  float target_angle = atan2f(dy, dx);
  float angle_diff = target_angle - game->ai_tank.angle;
  while (angle_diff > M_PI)
    angle_diff -= 2 * M_PI;
  while (angle_diff < -M_PI)
    angle_diff += 2 * M_PI;

  // Decision logic
  if (fabs(angle_diff) > 0.3f) {
    return (angle_diff > 0) ? 3 : 2; // Rotate towards player
  } else if (distance > 200) {
    return 0; // Move forward if far
  } else if (distance < 150) {
    return 4; // Shoot if close and aimed
  } else {
    return 0; // Move forward
  }
}

// Utility functions
void draw_circle(SDL_Renderer *renderer, int x, int y, int radius,
                 uint32_t color) {
  Uint8 r = (color >> 24) & 0xFF;
  Uint8 g = (color >> 16) & 0xFF;
  Uint8 b = (color >> 8) & 0xFF;
  Uint8 a = color & 0xFF;

  SDL_SetRenderDrawColor(renderer, r, g, b, a);

  for (int w = 0; w < radius * 2; w++) {
    for (int h = 0; h < radius * 2; h++) {
      int dx = radius - w;
      int dy = radius - h;
      if ((dx * dx + dy * dy) <= (radius * radius)) {
        SDL_RenderDrawPoint(renderer, x + dx, y + dy);
      }
    }
  }
}

void draw_tank(SDL_Renderer *renderer, Tank *tank) {
  // Draw tank body
  draw_circle(renderer, (int)tank->x, (int)tank->y, (int)tank->radius,
              tank->color);

  // Draw cannon
  Uint8 r = (tank->color >> 24) & 0xFF;
  Uint8 g = (tank->color >> 16) & 0xFF;
  Uint8 b = (tank->color >> 8) & 0xFF;
  Uint8 a = tank->color & 0xFF;

  SDL_SetRenderDrawColor(renderer, r, g, b, a);

  float cannon_end_x = tank->x + cosf(tank->angle) * (tank->radius + 15);
  float cannon_end_y = tank->y + sinf(tank->angle) * (tank->radius + 15);

  SDL_RenderDrawLine(renderer, (int)tank->x, (int)tank->y, (int)cannon_end_x,
                     (int)cannon_end_y);
}

void draw_bullet(SDL_Renderer *renderer, Bullet *bullet) {
  draw_circle(renderer, (int)bullet->x, (int)bullet->y, (int)bullet->radius,
              bullet->color);
}

void draw_ui(SDL_Renderer *renderer, TTF_Font *font, GameState *game) {
  if (!font) {
    // Draw simple text using rectangles when no font is available
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

    // Draw health bars
    SDL_Rect player_health = {10, 10, (int)(game->player.health * 2), 20};
    SDL_Rect ai_health = {10, 40, (int)(game->ai_tank.health * 2), 20};

    SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
    SDL_RenderFillRect(renderer, &player_health);
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    SDL_RenderFillRect(renderer, &ai_health);

    // Draw health bar borders
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_Rect player_border = {10, 10, 200, 20};
    SDL_Rect ai_border = {10, 40, 200, 20};
    SDL_RenderDrawRect(renderer, &player_border);
    SDL_RenderDrawRect(renderer, &ai_border);

    // Draw pause indicator
    if (game->paused) {
      SDL_Rect pause_box = {SCREEN_WIDTH / 2 - 100, SCREEN_HEIGHT / 2 - 15, 200,
                            30};
      SDL_RenderDrawRect(renderer, &pause_box);
    }
    return;
  }

  // Create score text
  char score_text[100];
  snprintf(score_text, sizeof(score_text),
           "Player: %d HP  |  AI: %d HP  |  Score - P: %d, AI: %d",
           game->player.health, game->ai_tank.health, game->score[0],
           game->score[1]);

  SDL_Color textColor = {255, 255, 255, 255};
  SDL_Surface *textSurface = TTF_RenderText_Solid(font, score_text, textColor);

  if (textSurface) {
    SDL_Texture *textTexture =
        SDL_CreateTextureFromSurface(renderer, textSurface);
    if (textTexture) {
      SDL_Rect textRect = {10, 10, textSurface->w, textSurface->h};
      SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
      SDL_DestroyTexture(textTexture);
    }
    SDL_FreeSurface(textSurface);
  }

  // Draw controls info
  char controls_text[] =
      "W/S: Move  A/D: Rotate  Space: Shoot  P: Pause  Q: Quit";
  textSurface = TTF_RenderText_Solid(font, controls_text, textColor);

  if (textSurface) {
    SDL_Texture *textTexture =
        SDL_CreateTextureFromSurface(renderer, textSurface);
    if (textTexture) {
      SDL_Rect textRect = {10, SCREEN_HEIGHT - 30, textSurface->w,
                           textSurface->h};
      SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
      SDL_DestroyTexture(textTexture);
    }
    SDL_FreeSurface(textSurface);
  }

  // Draw pause indicator
  if (game->paused) {
    char pause_text[] = "PAUSED - Grid visualization enabled";
    textSurface = TTF_RenderText_Solid(font, pause_text, textColor);

    if (textSurface) {
      SDL_Texture *textTexture =
          SDL_CreateTextureFromSurface(renderer, textSurface);
      if (textTexture) {
        SDL_Rect textRect = {SCREEN_WIDTH / 2 - 150, SCREEN_HEIGHT / 2 - 15,
                             textSurface->w, textSurface->h};
        SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
        SDL_DestroyTexture(textTexture);
      }
      SDL_FreeSurface(textSurface);
    }
  }
}
