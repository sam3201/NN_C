// Diep.io-style tank game with AI opponent
#include "../RL_AGENT/rl_agent.h"
#include "../utils/SDL3/SDL3_compat.h"
#include <SDL3/SDL.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

// Colors
#define COLOR_TANK_PLAYER 0x00FF00FF
#define COLOR_TANK_AI 0xFF0000FF
#define COLOR_BULLET 0xFFFF00FF
#define COLOR_ARENA 0x808080FF

typedef struct {
    float x, y;
    float vx, vy;
    float angle;
    float radius;
    uint32_t color;
    int health;
    bool alive;
} Tank;

typedef struct {
    float x, y;
    float vx, vy;
    float radius;
    uint32_t color;
    bool active;
    int owner; // 0 = player, 1 = AI
} Bullet;

typedef struct {
    Tank player;
    Tank ai_tank;
    Bullet bullets[MAX_BULLETS];
    uint32_t last_fire_time[2]; // 0 = player, 1 = AI
    int score[2];
    bool game_running;
    SDL_Window* window;
    SDL_Renderer* renderer;
    TTF_Font* font;
} GameState;

// Function prototypes
void init_game(GameState* game);
void handle_input(GameState* game);
void update_game(GameState* game);
void render_game(GameState* game);
void cleanup_game(GameState* game);
void fire_bullet(GameState* game, int owner);
void update_bullets(GameState* game);
void check_collisions(GameState* game);
void respawn_tank(Tank* tank);
float get_distance(float x1, float y1, float x2, float y2);
uint32_t get_current_time_ms();

// AI functions
void init_ai_agent();
void update_ai_decision(GameState* game);
void get_game_state_for_ai(GameState* game, float* state);
int get_ai_action(float* state);

// Global AI agent
RLAgent* ai_agent = NULL;

int main(int argc, char* argv[]) {
    GameState game;
    
    // Initialize SDL
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        printf("SDL initialization failed: %s\n", SDL_GetError());
        return 1;
    }
    
    // Initialize TTF
    if (!TTF_Init()) {
        printf("TTF initialization failed: %s\n", TTF_GetError());
        SDL_Quit();
        return 1;
    }
    
    // Create window
    game.window = SDL_CreateWindow("Diep.io Tank Game", SCREEN_WIDTH, SCREEN_HEIGHT, 0);
    if (!game.window) {
        printf("Window creation failed: %s\n", SDL_GetError());
        TTF_Quit();
        SDL_Quit();
        return 1;
    }
    
    // Create renderer
    game.renderer = SDL_CreateRenderer(game.window, NULL);
    if (!game.renderer) {
        printf("Renderer creation failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(game.window);
        TTF_Quit();
        SDL_Quit();
        return 1;
    }
    
    // Load font
    game.font = TTF_OpenFont("/System/Library/Fonts/Arial.ttf", 24);
    if (!game.font) {
        printf("Font loading failed: %s\n", TTF_GetError());
        SDL_DestroyRenderer(game.renderer);
        SDL_DestroyWindow(game.window);
        TTF_Quit();
        SDL_Quit();
        return 1;
    }
    
    // Initialize game
    init_game(&game);
    
    // Initialize AI agent
    init_ai_agent();
    
    // Main game loop
    while (game.game_running) {
        handle_input(&game);
        update_game(&game);
        render_game(&game);
        SDL_Delay(16); // ~60 FPS
    }
    
    // Cleanup
    cleanup_game(&game);
    
    return 0;
}

void init_game(GameState* game) {
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
    
    // Initialize bullets
    for (int i = 0; i < MAX_BULLETS; i++) {
        game->bullets[i].active = false;
    }
    
    // Initialize game state
    game->last_fire_time[0] = 0;
    game->last_fire_time[1] = 0;
    game->score[0] = 0;
    game->score[1] = 0;
    game->game_running = true;
}

void handle_input(GameState* game) {
    SDL_Event event;
    
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_EVENT_QUIT) {
            game->game_running = false;
        }
    }
    
    // Get keyboard state
    const bool* keys = SDL_GetKeyboardState(NULL);
    
    if (game->player.alive) {
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

void update_game(GameState* game) {
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
    
    // Respawn tanks if needed
    if (!game->player.alive) {
        respawn_tank(&game->player);
    }
    if (!game->ai_tank.alive) {
        respawn_tank(&game->ai_tank);
    }
}

void render_game(GameState* game) {
    // Clear screen
    SDL_SetRenderDrawColor(game->renderer, 20, 20, 30, 255);
    SDL_RenderClear(game->renderer);
    
    // Draw arena boundaries
    SDL_SetRenderDrawColor(game->renderer, 128, 128, 128, 255);
    SDL_FRect arena = {ARENA_MARGIN, ARENA_MARGIN, 
                      SCREEN_WIDTH - 2 * ARENA_MARGIN, 
                      SCREEN_HEIGHT - 2 * ARENA_MARGIN};
    SDL_RenderRect(game->renderer, &arena);
    
    // Draw player tank
    if (game->player.alive) {
        SDL_SetRenderDrawColor(game->renderer, 
                              (game->player.color >> 24) & 0xFF,
                              (game->player.color >> 16) & 0xFF,
                              (game->player.color >> 8) & 0xFF,
                              game->player.color & 0xFF);
        SDL_FCircle player_circle = {game->player.x, game->player.y, game->player.radius};
        SDL_RenderFillCircle(game->renderer, &player_circle);
        
        // Draw tank cannon
        float cannon_end_x = game->player.x + cosf(game->player.angle) * (game->player.radius + 15);
        float cannon_end_y = game->player.y + sinf(game->player.angle) * (game->player.radius + 15);
        SDL_RenderLine(game->renderer, game->player.x, game->player.y, cannon_end_x, cannon_end_y);
    }
    
    // Draw AI tank
    if (game->ai_tank.alive) {
        SDL_SetRenderDrawColor(game->renderer, 
                              (game->ai_tank.color >> 24) & 0xFF,
                              (game->ai_tank.color >> 16) & 0xFF,
                              (game->ai_tank.color >> 8) & 0xFF,
                              game->ai_tank.color & 0xFF);
        SDL_FCircle ai_circle = {game->ai_tank.x, game->ai_tank.y, game->ai_tank.radius};
        SDL_RenderFillCircle(game->renderer, &ai_circle);
        
        // Draw tank cannon
        float cannon_end_x = game->ai_tank.x + cosf(game->ai_tank.angle) * (game->ai_tank.radius + 15);
        float cannon_end_y = game->ai_tank.y + sinf(game->ai_tank.angle) * (game->ai_tank.radius + 15);
        SDL_RenderLine(game->renderer, game->ai_tank.x, game->ai_tank.y, cannon_end_x, cannon_end_y);
    }
    
    // Draw bullets
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (game->bullets[i].active) {
            SDL_SetRenderDrawColor(game->renderer, 
                                  (game->bullets[i].color >> 24) & 0xFF,
                                  (game->bullets[i].color >> 16) & 0xFF,
                                  (game->bullets[i].color >> 8) & 0xFF,
                                  game->bullets[i].color & 0xFF);
            SDL_FCircle bullet_circle = {game->bullets[i].x, game->bullets[i].y, game->bullets[i].radius};
            SDL_RenderFillCircle(game->renderer, &bullet_circle);
        }
    }
    
    // Draw scores
    char score_text[100];
    snprintf(score_text, sizeof(score_text), "Player: %d  AI: %d", game->score[0], game->score[1]);
    SDL_Surface* text_surface = TTF_RenderText_Solid(game->font, score_text, (SDL_Color){255, 255, 255, 255});
    if (text_surface) {
        SDL_Texture* text_texture = SDL_CreateTextureFromSurface(game->renderer, text_surface);
        if (text_texture) {
            SDL_FRect text_rect = {10, 10, text_surface->w, text_surface->h};
            SDL_RenderTexture(game->renderer, text_texture, NULL, &text_rect);
            SDL_DestroyTexture(text_texture);
        }
        SDL_DestroySurface(text_surface);
    }
    
    // Present
    SDL_RenderPresent(game->renderer);
}

void cleanup_game(GameState* game) {
    if (ai_agent) {
        rl_agent_free(ai_agent);
    }
    
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

void fire_bullet(GameState* game, int owner) {
    uint32_t current_time = get_current_time_ms();
    
    if (current_time - game->last_fire_time[owner] < FIRE_RATE) {
        return; // Can't fire yet
    }
    
    Tank* tank = (owner == 0) ? &game->player : &game->ai_tank;
    
    if (!tank->alive) {
        return;
    }
    
    // Find inactive bullet
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (!game->bullets[i].active) {
            game->bullets[i].x = tank->x + cosf(tank->angle) * (tank->radius + 10);
            game->bullets[i].y = tank->y + sinf(tank->angle) * (tank->radius + 10);
            game->bullets[i].vx = cosf(tank->angle) * BULLET_SPEED;
            game->bullets[i].vy = sinf(tank->angle) * BULLET_SPEED;
            game->bullets[i].radius = BULLET_SIZE;
            game->bullets[i].color = COLOR_BULLET;
            game->bullets[i].active = true;
            game->bullets[i].owner = owner;
            game->last_fire_time[owner] = current_time;
            break;
        }
    }
}

void update_bullets(GameState* game) {
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

void check_collisions(GameState* game) {
    // Check bullet-tank collisions
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (!game->bullets[i].active) continue;
        
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

void respawn_tank(Tank* tank) {
    static uint32_t respawn_time = 0;
    uint32_t current_time = get_current_time_ms();
    
    if (respawn_time == 0) {
        respawn_time = current_time;
        return;
    }
    
    if (current_time - respawn_time > 3000) { // 3 second respawn
        tank->x = (tank == &game->player) ? SCREEN_WIDTH / 4 : 3 * SCREEN_WIDTH / 4;
        tank->y = SCREEN_HEIGHT / 2;
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

uint32_t get_current_time_ms() {
    return (uint32_t)(SDL_GetTicks());
}

// AI Functions
void init_ai_agent() {
    // Initialize the RL agent for the AI opponent
    ai_agent = rl_agent_create(10, 4); // 10 state dimensions, 4 actions
    
    if (!ai_agent) {
        printf("Failed to create AI agent\n");
        return;
    }
    
    // Load existing model if available
    rl_agent_load_model(ai_agent, "diep_ai_model.bin");
}

void update_ai_decision(GameState* game) {
    if (!ai_agent || !game->ai_tank.alive) return;
    
    // Get current game state
    float state[10];
    get_game_state_for_ai(game, state);
    
    // Get AI action
    int action = rl_agent_predict_action(ai_agent, state);
    
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
        case 3: // Rotate right and shoot
            game->ai_tank.angle += TANK_ROTATION_SPEED * 2;
            game->ai_tank.vx *= 0.9f;
            game->ai_tank.vy *= 0.9f;
            fire_bullet(game, 1);
            break;
    }
    
    // Aim towards player
    float dx = game->player.x - game->ai_tank.x;
    float dy = game->player.y - game->ai_tank.y;
    float target_angle = atan2f(dy, dx);
    
    // Smooth rotation towards target
    float angle_diff = target_angle - game->ai_tank.angle;
    while (angle_diff > M_PI) angle_diff -= 2 * M_PI;
    while (angle_diff < -M_PI) angle_diff += 2 * M_PI;
    
    game->ai_tank.angle += angle_diff * 0.1f;
}

void get_game_state_for_ai(GameState* game, float* state) {
    // Normalize positions (0-1)
    state[0] = game->ai_tank.x / SCREEN_WIDTH;
    state[1] = game->ai_tank.y / SCREEN_HEIGHT;
    state[2] = game->player.x / SCREEN_WIDTH;
    state[3] = game->player.y / SCREEN_HEIGHT;
    
    // Normalize velocities (-1 to 1)
    state[4] = game->ai_tank.vx / TANK_SPEED;
    state[5] = game->ai_tank.vy / TANK_SPEED;
    state[6] = game->player.vx / TANK_SPEED;
    state[7] = game->player.vy / TANK_SPEED;
    
    // Health (0-1)
    state[8] = game->ai_tank.health / 100.0f;
    state[9] = game->player.health / 100.0f;
}
