// Enhanced Diep.io Tank Game - Part 1: Core Systems and Data Structures
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// SDL includes
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

// Game constants
#define SCREEN_WIDTH 1200
#define SCREEN_HEIGHT 800
#define TANK_SIZE 30
#define BULLET_SIZE 8
#define BULLET_SPEED 8.0f
#define TANK_SPEED 3.0f
#define TANK_ROTATION_SPEED 0.05f
#define MAX_BULLETS 200
#define MAX_NPCS 20
#define FIRE_RATE 300 // milliseconds
#define ARENA_MARGIN 50

// World constants
#define WORLD_WIDTH 4000
#define WORLD_HEIGHT 3000

// Entity types for one-hot encoding
#define ENTITY_EMPTY 0.0f
#define ENTITY_SELF 1.0f
#define ENTITY_ENEMY 2.0f
#define ENTITY_NPC 3.0f
#define ENTITY_BULLET_SELF 4.0f
#define ENTITY_BULLET_ENEMY 5.0f
#define ENTITY_BULLET_NPC 6.0f
#define ENTITY_SHAPE 7.0f

// Colors
#define COLOR_TANK_PLAYER 0x00FF00FF
#define COLOR_TANK_AI 0xFF0000FF
#define COLOR_TANK_NPC 0x0000FFFF
#define COLOR_BULLET 0xFFFF00FF
#define COLOR_SHAPE 0xFF00FFFF

// Grid dimensions for AI observation
#define GRID_WIDTH 60
#define GRID_HEIGHT 45
#define AI_INPUT_SIZE (GRID_WIDTH * GRID_HEIGHT * 8) // 8 channels per cell

// Tank evolution types
typedef enum {
    TANK_BASIC = 0,
    TANK_TWIN,
    TANK_SNIPER,
    TANK_MACHINE,
    TANK_DESTROYER,
    TANK_MAX_TYPES
} TankType;

// Tank stats structure
typedef struct {
    float speed;
    float bullet_speed;
    float bullet_damage;
    float fire_rate;
    float max_health;
    int bullet_count;
    float spread_angle;
} TankStats;

// Camera structure
typedef struct {
    float x, y;
    float zoom;
} Camera;

typedef struct {
    float x, y;
    float vx, vy;
    float angle;
    float radius;
    uint32_t color;
    int health;
    float max_health;
    bool alive;
    int entity_type;
    TankType tank_type;
    TankStats stats;
    int level;
    int experience;
    int experience_to_next_level;
} Tank;

typedef struct {
    float x, y;
    float vx, vy;
    float radius;
    uint32_t color;
    bool active;
    int owner;
    int entity_type;
    float damage;
} Bullet;

typedef struct {
    float x, y;
    float radius;
    uint32_t color;
    bool active;
    int health;
    int experience_value;
} Shape;

typedef struct {
    Tank player;
    Tank ai_tank;
    Tank npcs[MAX_NPCS];
    uint32_t last_fire_time[2 + MAX_NPCS]; // player + ai + npcs
    int score[2];
    bool game_running;
    bool paused;
    bool training_mode;

    // Camera
    Camera camera;

    // SDL components
    SDL_Window *window;
    SDL_Renderer *renderer;
    TTF_Font *font;

    // Game entities
    Bullet bullets[MAX_BULLETS];
    Shape shapes[50]; // Static shapes to farm
    float ai_observation[AI_INPUT_SIZE];
} GameState;

// Tank evolution stats
static const TankStats TANK_STATS[TANK_MAX_TYPES] = {
    // TANK_BASIC
    {3.0f, 8.0f, 20.0f, 300.0f, 100.0f, 1, 0.0f},
    // TANK_TWIN
    {2.8f, 8.5f, 15.0f, 250.0f, 100.0f, 2, 0.3f},
    // TANK_SNIPER
    {3.2f, 12.0f, 40.0f, 500.0f, 100.0f, 1, 0.0f},
    // TANK_MACHINE
    {2.5f, 7.0f, 10.0f, 100.0f, 100.0f, 1, 0.1f},
    // TANK_DESTROYER
    {2.2f, 6.0f, 60.0f, 800.0f, 120.0f, 1, 0.0f}
};

// Function prototypes
void init_game(GameState *game);
void handle_input(GameState *game);
void update_game(GameState *game);
void render_game(GameState *game);
void cleanup_game(GameState *game);
void fire_bullet(GameState *game, int owner);
void update_bullets(GameState *game);
void update_shapes(GameState *game);
void check_collisions(GameState *game);
void respawn_tank(Tank *tank, GameState *game);
float get_distance(float x1, float y1, float x2, float y2);
uint32_t get_current_time_ms();

// Camera functions
void update_camera(GameState *game);
float world_to_screen_x(GameState *game, float world_x);
float world_to_screen_y(GameState *game, float world_y);
bool is_on_screen(GameState *game, float x, float y, float radius);

// One-hot encoding functions
void clear_observation_grid(GameState *game);
void update_observation_grid(GameState *game);
void add_entity_to_grid(GameState *game, float x, float y, float entity_type);

// Level and evolution functions
void add_experience(Tank *tank, int exp);
void level_up(Tank *tank);
void evolve_tank(Tank *tank, TankType new_type);
void update_tank_stats(Tank *tank);

// AI functions
void init_simple_ai(GameState *game);
void update_simple_ai(GameState *game);
void update_npc_ai(GameState *game, int npc_index);

// Utility functions
void draw_circle(SDL_Renderer *renderer, int x, int y, int radius, uint32_t color);
void draw_tank(GameState *game, Tank *tank);
void draw_bullet(GameState *game, Bullet *bullet);
void draw_shape(GameState *game, Shape *shape);
void draw_ui(SDL_Renderer *renderer, TTF_Font *font, GameState *game);
void draw_minimap(SDL_Renderer *renderer, GameState *game);

int main(int argc, char *argv[]) {
    (void)argc; // Unused
    (void)argv; // Unused
    
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
    game.window = SDL_CreateWindow("Enhanced Diep.io Tank Game", 
                                   SDL_WINDOWPOS_CENTERED, 
                                   SDL_WINDOWPOS_CENTERED,
                                   SCREEN_WIDTH, SCREEN_HEIGHT, 
                                   SDL_WINDOW_SHOWN);
    if (!game.window) {
        printf("Window creation failed: %s\n", SDL_GetError());
        TTF_Quit();
        SDL_Quit();
        return 1;
    }
    
    // Create renderer
    game.renderer = SDL_CreateRenderer(game.window, -1, 
                                       SDL_RENDERER_ACCELERATED | 
                                       SDL_RENDERER_PRESENTVSYNC);
    if (!game.renderer) {
        printf("Renderer creation failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(game.window);
        TTF_Quit();
        SDL_Quit();
        return 1;
    }
    
    // Load font - try multiple font paths
    const char* font_paths[] = {
        "/System/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Geneva.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        NULL
    };
    
    game.font = NULL;
    for (int i = 0; font_paths[i] != NULL; i++) {
        game.font = TTF_OpenFont(font_paths[i], 24);
        if (game.font) {
            break;
        }
    }
    
    if (!game.font) {
        printf("Warning: Could not load any system font, continuing without text\n");
    }
    
    // Initialize game
    init_game(&game);
    
    // Initialize simple AI
    init_simple_ai(&game);
    
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
    // Initialize camera
    game->camera.x = WORLD_WIDTH / 2;
    game->camera.y = WORLD_HEIGHT / 2;
    game->camera.zoom = 1.0f;
    
    // Initialize player tank
    game->player.x = WORLD_WIDTH / 4;
    game->player.y = WORLD_HEIGHT / 2;
    game->player.vx = 0;
    game->player.vy = 0;
    game->player.angle = 0;
    game->player.radius = TANK_SIZE;
    game->player.color = COLOR_TANK_PLAYER;
    game->player.health = 100;
    game->player.max_health = 100;
    game->player.alive = true;
    game->player.entity_type = ENTITY_SELF;
    game->player.tank_type = TANK_BASIC;
    game->player.level = 1;
    game->player.experience = 0;
    game->player.experience_to_next_level = 100;
    update_tank_stats(&game->player);
    
    // Initialize AI tank
    game->ai_tank.x = 3 * WORLD_WIDTH / 4;
    game->ai_tank.y = WORLD_HEIGHT / 2;
    game->ai_tank.vx = 0;
    game->ai_tank.vy = 0;
    game->ai_tank.angle = M_PI;
    game->ai_tank.radius = TANK_SIZE;
    game->ai_tank.color = COLOR_TANK_AI;
    game->ai_tank.health = 100;
    game->ai_tank.max_health = 100;
    game->ai_tank.alive = true;
    game->ai_tank.entity_type = ENTITY_ENEMY;
    game->ai_tank.tank_type = TANK_BASIC;
    game->ai_tank.level = 1;
    game->ai_tank.experience = 0;
    game->ai_tank.experience_to_next_level = 100;
    update_tank_stats(&game->ai_tank);
    
    // Initialize NPC tanks
    for (int i = 0; i < MAX_NPCS; i++) {
        game->npcs[i].x = (float)(rand() % (WORLD_WIDTH - 200)) + 100;
        game->npcs[i].y = (float)(rand() % (WORLD_HEIGHT - 200)) + 100;
        game->npcs[i].vx = 0;
        game->npcs[i].vy = 0;
        game->npcs[i].angle = (float)(rand() % 360) * M_PI / 180.0f;
        game->npcs[i].radius = TANK_SIZE;
        game->npcs[i].color = COLOR_TANK_NPC;
        game->npcs[i].health = 80;
        game->npcs[i].max_health = 80;
        game->npcs[i].alive = true;
        game->npcs[i].entity_type = ENTITY_NPC;
        game->npcs[i].tank_type = TANK_BASIC;
        game->npcs[i].level = 1;
        game->npcs[i].experience = 0;
        game->npcs[i].experience_to_next_level = 100;
        update_tank_stats(&game->npcs[i]);
    }
    
    // Initialize bullets
    for (int i = 0; i < MAX_BULLETS; i++) {
        game->bullets[i].active = false;
        game->bullets[i].entity_type = ENTITY_BULLET_ENEMY;
        game->bullets[i].damage = 20.0f;
    }
    
    // Initialize shapes
    for (int i = 0; i < 50; i++) {
        game->shapes[i].x = (float)(rand() % (WORLD_WIDTH - 100)) + 50;
        game->shapes[i].y = (float)(rand() % (WORLD_HEIGHT - 100)) + 50;
        game->shapes[i].radius = (float)(rand() % 20) + 10;
        game->shapes[i].color = COLOR_SHAPE;
        game->shapes[i].active = true;
        game->shapes[i].health = (int)game->shapes[i].radius;
        game->shapes[i].experience_value = (int)game->shapes[i].radius * 2;
    }
    
    // Initialize game state
    for (int i = 0; i < 2 + MAX_NPCS; i++) {
        game->last_fire_time[i] = 0;
    }
    game->score[0] = 0;
    game->score[1] = 0;
    game->game_running = true;
    game->paused = false;
    game->training_mode = false;
    
    // Initialize AI observation
    memset(game->ai_observation, 0, sizeof(game->ai_observation));
}

void init_simple_ai(GameState *game) {
    (void)game; // Suppress unused parameter warning
    printf("Enhanced AI System Initialized with %d NPCs\n", MAX_NPCS);
    printf("Features: Large world, Camera system, Leveling, Evolution, Training mode\n");
}

void update_tank_stats(Tank *tank) {
    TankStats base_stats = TANK_STATS[tank->tank_type];
    
    // Apply level bonuses
    float level_multiplier = 1.0f + (tank->level - 1) * 0.1f;
    
    tank->stats.speed = base_stats.speed * level_multiplier;
    tank->stats.bullet_speed = base_stats.bullet_speed;
    tank->stats.bullet_damage = base_stats.bullet_damage * level_multiplier;
    tank->stats.fire_rate = base_stats.fire_rate;
    tank->stats.max_health = base_stats.max_health * level_multiplier;
    tank->stats.bullet_count = base_stats.bullet_count;
    tank->stats.spread_angle = base_stats.spread_angle;
    
    tank->max_health = tank->stats.max_health;
}

void add_experience(Tank *tank, int exp) {
    tank->experience += exp;
    while (tank->experience >= tank->experience_to_next_level) {
        tank->experience -= tank->experience_to_next_level;
        level_up(tank);
    }
}

void level_up(Tank *tank) {
    tank->level++;
    tank->experience_to_next_level = tank->level * 100;
    update_tank_stats(tank);
    
    // Evolve at certain levels
    if (tank->level == 5 && tank->tank_type == TANK_BASIC) {
        evolve_tank(tank, TANK_TWIN);
    } else if (tank->level == 10 && tank->tank_type == TANK_TWIN) {
        evolve_tank(tank, TANK_SNIPER);
    } else if (tank->level == 15 && tank->tank_type == TANK_SNIPER) {
        evolve_tank(tank, TANK_MACHINE);
    } else if (tank->level == 20 && tank->tank_type == TANK_MACHINE) {
        evolve_tank(tank, TANK_DESTROYER);
    }
}

void evolve_tank(Tank *tank, TankType new_type) {
    tank->tank_type = new_type;
    update_tank_stats(tank);
    
    // Adjust appearance based on tank type
    switch (new_type) {
        case TANK_TWIN:
            tank->radius = TANK_SIZE * 1.1f;
            break;
        case TANK_SNIPER:
            tank->radius = TANK_SIZE * 0.9f;
            break;
        case TANK_MACHINE:
            tank->radius = TANK_SIZE * 1.2f;
            break;
        case TANK_DESTROYER:
            tank->radius = TANK_SIZE * 1.4f;
            break;
        default:
            tank->radius = TANK_SIZE;
            break;
    }
}

// Camera functions
void update_camera(GameState *game) {
    // Follow player
    float target_x = game->player.x;
    float target_y = game->player.y;
    
    // Smooth camera movement
    game->camera.x += (target_x - game->camera.x) * 0.1f;
    game->camera.y += (target_y - game->camera.y) * 0.1f;
}

float world_to_screen_x(GameState *game, float world_x) {
    return (world_x - game->camera.x) * game->camera.zoom + SCREEN_WIDTH / 2;
}

float world_to_screen_y(GameState *game, float world_y) {
    return (world_y - game->camera.y) * game->camera.zoom + SCREEN_HEIGHT / 2;
}

bool is_on_screen(GameState *game, float x, float y, float radius) {
    float screen_x = world_to_screen_x(game, x);
    float screen_y = world_to_screen_y(game, y);
    float screen_radius = radius * game->camera.zoom;
    
    return (screen_x + screen_radius >= 0 && screen_x - screen_radius <= SCREEN_WIDTH &&
            screen_y + screen_radius >= 0 && screen_y - screen_radius <= SCREEN_HEIGHT);
}

// Utility functions
float get_distance(float x1, float y1, float x2, float y2) {
    return sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

uint32_t get_current_time_ms() {
    return (uint32_t)SDL_GetTicks();
}
// Enhanced Diep.io Tank Game - Full Version
// All parts combined into single file

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
                    case SDLK_t:
                        game->training_mode = !game->training_mode;
                        printf("Training mode: %s\n", game->training_mode ? "ON" : "OFF");
                        break;
                    case SDLK_KP_PLUS:
                    case SDLK_EQUALS:
                        game->camera.zoom = fminf(2.0f, game->camera.zoom + 0.1f);
                        break;
                    case SDLK_KP_MINUS:
                    case SDLK_MINUS:
                        game->camera.zoom = fmaxf(0.5f, game->camera.zoom - 0.1f);
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
            game->player.vx = cosf(game->player.angle) * game->player.stats.speed;
            game->player.vy = sinf(game->player.angle) * game->player.stats.speed;
        } else if (keys[SDL_SCANCODE_S] || keys[SDL_SCANCODE_DOWN]) {
            game->player.vx = -cosf(game->player.angle) * game->player.stats.speed;
            game->player.vy = -sinf(game->player.angle) * game->player.stats.speed;
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
            fire_bullet(game, 0); // Player is index 0
        }
    }
}

void update_game(GameState *game) {
    // Update player
    if (game->player.alive) {
        game->player.x += game->player.vx;
        game->player.y += game->player.vy;
        
        // Keep player in world bounds
        game->player.x = fmaxf(ARENA_MARGIN + game->player.radius, 
                              fminf(WORLD_WIDTH - ARENA_MARGIN - game->player.radius, game->player.x));
        game->player.y = fmaxf(ARENA_MARGIN + game->player.radius, 
                              fminf(WORLD_HEIGHT - ARENA_MARGIN - game->player.radius, game->player.y));
    }
    
    // Update AI tank
    if (game->ai_tank.alive) {
        update_simple_ai(game);
        
        game->ai_tank.x += game->ai_tank.vx;
        game->ai_tank.y += game->ai_tank.vy;
        
        // Keep AI in world bounds
        game->ai_tank.x = fmaxf(ARENA_MARGIN + game->ai_tank.radius, 
                               fminf(WORLD_WIDTH - ARENA_MARGIN - game->ai_tank.radius, game->ai_tank.x));
        game->ai_tank.y = fmaxf(ARENA_MARGIN + game->ai_tank.radius, 
                               fminf(WORLD_HEIGHT - ARENA_MARGIN - game->ai_tank.radius, game->ai_tank.y));
    }
    
    // Update NPCs
    for (int i = 0; i < MAX_NPCS; i++) {
        if (game->npcs[i].alive) {
            update_npc_ai(game, i);
            
            game->npcs[i].x += game->npcs[i].vx;
            game->npcs[i].y += game->npcs[i].vy;
            
            // Keep NPC in world bounds
            game->npcs[i].x = fmaxf(ARENA_MARGIN + game->npcs[i].radius, 
                                   fminf(WORLD_WIDTH - ARENA_MARGIN - game->npcs[i].radius, game->npcs[i].x));
            game->npcs[i].y = fmaxf(ARENA_MARGIN + game->npcs[i].radius, 
                                   fminf(WORLD_HEIGHT - ARENA_MARGIN - game->npcs[i].radius, game->npcs[i].y));
        }
    }
    
    // Update bullets
    update_bullets(game);
    
    // Update shapes
    update_shapes(game);
    
    // Check collisions
    check_collisions(game);
    
    // Update camera
    update_camera(game);
    
    // Update AI observation grid
    update_observation_grid(game);
    
    // Respawn tanks if needed
    if (!game->player.alive) {
        respawn_tank(&game->player, game);
    }
    if (!game->ai_tank.alive) {
        respawn_tank(&game->ai_tank, game);
    }
    
    // Respawn NPCs
    for (int i = 0; i < MAX_NPCS; i++) {
        if (!game->npcs[i].alive) {
            respawn_tank(&game->npcs[i], game);
        }
    }
}

void update_simple_ai(GameState *game) {
    if (!game->ai_tank.alive || !game->player.alive) {
        game->ai_tank.vx *= 0.9f;
        game->ai_tank.vy *= 0.9f;
        return;
    }
    
    // Enhanced AI: aim and move towards player with continuous aiming
    float dx = game->player.x - game->ai_tank.x;
    float dy = game->player.y - game->ai_tank.y;
    float distance = sqrtf(dx * dx + dy * dy);
    
    // Continuous aiming angle
    float target_angle = atan2f(dy, dx);
    float angle_diff = target_angle - game->ai_tank.angle;
    
    // Normalize angle difference
    while (angle_diff > M_PI) angle_diff -= 2 * M_PI;
    while (angle_diff < -M_PI) angle_diff += 2 * M_PI;
    
    // Smooth rotation towards target
    game->ai_tank.angle += angle_diff * 0.1f;
    
    // Movement strategy based on tank type and distance
    if (game->ai_tank.tank_type == TANK_SNIPER) {
        // Sniper keeps distance
        if (distance < 300) {
            game->ai_tank.vx = -cosf(game->ai_tank.angle) * game->ai_tank.stats.speed * 0.8f;
            game->ai_tank.vy = -sinf(game->ai_tank.angle) * game->ai_tank.stats.speed * 0.8f;
        } else if (distance > 500) {
            game->ai_tank.vx = cosf(game->ai_tank.angle) * game->ai_tank.stats.speed * 0.6f;
            game->ai_tank.vy = sinf(game->ai_tank.angle) * game->ai_tank.stats.speed * 0.6f;
        } else {
            game->ai_tank.vx *= 0.9f;
            game->ai_tank.vy *= 0.9f;
        }
    } else {
        // Other tanks move closer
        if (distance > 200) {
            game->ai_tank.vx = cosf(game->ai_tank.angle) * game->ai_tank.stats.speed * 0.8f;
            game->ai_tank.vy = sinf(game->ai_tank.angle) * game->ai_tank.stats.speed * 0.8f;
        } else if (distance < 150) {
            game->ai_tank.vx = -cosf(game->ai_tank.angle) * game->ai_tank.stats.speed * 0.5f;
            game->ai_tank.vy = -sinf(game->ai_tank.angle) * game->ai_tank.stats.speed * 0.5f;
        } else {
            game->ai_tank.vx *= 0.9f;
            game->ai_tank.vy *= 0.9f;
        }
    }
    
    // Shoot if aimed correctly and in range
    float fire_threshold = (game->ai_tank.tank_type == TANK_SNIPER) ? 0.1f : 0.3f;
    float max_range = (game->ai_tank.tank_type == TANK_SNIPER) ? 600 : 400;
    
    if (fabs(angle_diff) < fire_threshold && distance < max_range) {
        fire_bullet(game, 1); // AI is index 1
    }
}

void update_npc_ai(GameState *game, int npc_index) {
    Tank *npc = &game->npcs[npc_index];
    
    // Simple wandering AI with occasional targeting
    if (rand() % 100 < 2) { // 2% chance to change direction
        npc->angle += (float)(rand() % 200 - 100) * M_PI / 180.0f;
    }
    
    // Move in current direction
    npc->vx = cosf(npc->angle) * npc->stats.speed * 0.5f;
    npc->vy = sinf(npc->angle) * npc->stats.speed * 0.5f;
    
    // Occasionally shoot at nearby targets
    if (rand() % 100 < 5) { // 5% chance to shoot
        // Find nearest target
        float min_dist = 300.0f;
        Tank *target = NULL;
        
        if (game->player.alive) {
            float dist = get_distance(npc->x, npc->y, game->player.x, game->player.y);
            if (dist < min_dist) {
                min_dist = dist;
                target = &game->player;
            }
        }
        
        if (game->ai_tank.alive) {
            float dist = get_distance(npc->x, npc->y, game->ai_tank.x, game->ai_tank.y);
            if (dist < min_dist) {
                target = &game->ai_tank;
            }
        }
        
        if (target) {
            // Aim at target
            float dx = target->x - npc->x;
            float dy = target->y - npc->y;
            npc->angle = atan2f(dy, dx);
            fire_bullet(game, 2 + npc_index); // NPCs start at index 2
        }
    }
}

void fire_bullet(GameState *game, int owner) {
    uint32_t current_time = get_current_time_ms();
    
    Tank *tank;
    if (owner == 0) {
        tank = &game->player;
    } else if (owner == 1) {
        tank = &game->ai_tank;
    } else {
        tank = &game->npcs[owner - 2];
    }
    
    if (!tank->alive || current_time - game->last_fire_time[owner] < tank->stats.fire_rate) {
        return;
    }
    
    // Fire bullets based on tank type
    for (int i = 0; i < tank->stats.bullet_count; i++) {
        // Find inactive bullet
        for (int j = 0; j < MAX_BULLETS; j++) {
            if (!game->bullets[j].active) {
                float angle_offset = 0;
                if (tank->stats.bullet_count > 1) {
                    angle_offset = (i - (tank->stats.bullet_count - 1) / 2.0f) * tank->stats.spread_angle;
                }
                
                float fire_angle = tank->angle + angle_offset;
                game->bullets[j].x = tank->x + cosf(fire_angle) * (tank->radius + 15);
                game->bullets[j].y = tank->y + sinf(fire_angle) * (tank->radius + 15);
                game->bullets[j].vx = cosf(fire_angle) * tank->stats.bullet_speed;
                game->bullets[j].vy = sinf(fire_angle) * tank->stats.bullet_speed;
                game->bullets[j].radius = BULLET_SIZE;
                game->bullets[j].color = tank->color;
                game->bullets[j].active = true;
                game->bullets[j].owner = owner;
                game->bullets[j].damage = tank->stats.bullet_damage;
                
                if (owner == 0) {
                    game->bullets[j].entity_type = ENTITY_BULLET_SELF;
                } else if (owner == 1) {
                    game->bullets[j].entity_type = ENTITY_BULLET_ENEMY;
                } else {
                    game->bullets[j].entity_type = ENTITY_BULLET_NPC;
                }
                
                break;
            }
        }
    }
    
    game->last_fire_time[owner] = current_time;
}

void update_bullets(GameState *game) {
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (game->bullets[i].active) {
            game->bullets[i].x += game->bullets[i].vx;
            game->bullets[i].y += game->bullets[i].vy;
            
            // Check if bullet is out of world bounds
            if (game->bullets[i].x < ARENA_MARGIN || 
                game->bullets[i].x > WORLD_WIDTH - ARENA_MARGIN ||
                game->bullets[i].y < ARENA_MARGIN || 
                game->bullets[i].y > WORLD_HEIGHT - ARENA_MARGIN) {
                game->bullets[i].active = false;
            }
        }
    }
}

void update_shapes(GameState *game) {
    for (int i = 0; i < 50; i++) {
        if (!game->shapes[i].active) {
            // Respawn shape
            game->shapes[i].x = (float)(rand() % (WORLD_WIDTH - 100)) + 50;
            game->shapes[i].y = (float)(rand() % (WORLD_HEIGHT - 100)) + 50;
            game->shapes[i].radius = (float)(rand() % 20) + 10;
            game->shapes[i].active = true;
            game->shapes[i].health = (int)game->shapes[i].radius;
            game->shapes[i].experience_value = (int)game->shapes[i].radius * 2;
        }
    }
}

void check_collisions(GameState *game) {
    // Check bullet-tank collisions
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (!game->bullets[i].active) continue;
        
        Bullet *bullet = &game->bullets[i];
        
        // Check collision with player
        if (bullet->owner != 0 && game->player.alive) {
            float dist = get_distance(bullet->x, bullet->y, 
                                   game->player.x, game->player.y);
            if (dist < bullet->radius + game->player.radius) {
                game->player.health -= (int)bullet->damage;
                bullet->active = false;
                
                if (game->player.health <= 0) {
                    game->player.alive = false;
                    game->score[bullet->owner]++;
                }
            }
        }
        
        // Check collision with AI
        if (bullet->owner != 1 && game->ai_tank.alive) {
            float dist = get_distance(bullet->x, bullet->y, 
                                   game->ai_tank.x, game->ai_tank.y);
            if (dist < bullet->radius + game->ai_tank.radius) {
                game->ai_tank.health -= (int)bullet->damage;
                bullet->active = false;
                
                if (game->ai_tank.health <= 0) {
                    game->ai_tank.alive = false;
                    game->score[bullet->owner]++;
                    add_experience(bullet->owner == 0 ? &game->player : &game->npcs[bullet->owner - 2], 50);
                }
            }
        }
        
        // Check collision with NPCs
        for (int j = 0; j < MAX_NPCS; j++) {
            if (bullet->owner != (2 + j) && game->npcs[j].alive) {
                float dist = get_distance(bullet->x, bullet->y, 
                                       game->npcs[j].x, game->npcs[j].y);
                if (dist < bullet->radius + game->npcs[j].radius) {
                    game->npcs[j].health -= (int)bullet->damage;
                    bullet->active = false;
                    
                    if (game->npcs[j].health <= 0) {
                        game->npcs[j].alive = false;
                        game->score[bullet->owner]++;
                        add_experience(bullet->owner == 0 ? &game->player : 
                                     (bullet->owner == 1 ? &game->ai_tank : &game->npcs[bullet->owner - 2]), 30);
                    }
                }
            }
        }
    }
    
    // Check bullet-shape collisions
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (!game->bullets[i].active) continue;
        
        Bullet *bullet = &game->bullets[i];
        
        for (int j = 0; j < 50; j++) {
            if (!game->shapes[j].active) continue;
            
            float dist = get_distance(bullet->x, bullet->y, 
                                   game->shapes[j].x, game->shapes[j].y);
            if (dist < bullet->radius + game->shapes[j].radius) {
                game->shapes[j].health -= (int)bullet->damage;
                bullet->active = false;
                
                if (game->shapes[j].health <= 0) {
                    game->shapes[j].active = false;
                    
                    // Give experience to the bullet owner
                    if (bullet->owner == 0) {
                        add_experience(&game->player, game->shapes[j].experience_value);
                    } else if (bullet->owner == 1) {
                        add_experience(&game->ai_tank, game->shapes[j].experience_value);
                    } else {
                        add_experience(&game->npcs[bullet->owner - 2], game->shapes[j].experience_value);
                    }
                }
            }
        }
    }
}

void respawn_tank(Tank *tank, GameState *game) {
    static uint32_t respawn_time[2 + MAX_NPCS] = {0};
    uint32_t current_time = get_current_time_ms();
    
    int tank_index;
    if (tank == &game->player) {
        tank_index = 0;
    } else if (tank == &game->ai_tank) {
        tank_index = 1;
    } else {
        tank_index = 2 + (tank - game->npcs);
    }
    
    if (respawn_time[tank_index] == 0) {
        respawn_time[tank_index] = current_time;
        return;
    }
    
    if (current_time - respawn_time[tank_index] > 3000) { // 3 second respawn
        if (tank == &game->player) {
            tank->x = WORLD_WIDTH / 4;
            tank->y = WORLD_HEIGHT / 2;
        } else if (tank == &game->ai_tank) {
            tank->x = 3 * WORLD_WIDTH / 4;
            tank->y = WORLD_HEIGHT / 2;
        } else {
            tank->x = (float)(rand() % (WORLD_WIDTH - 200)) + 100;
            tank->y = (float)(rand() % (WORLD_HEIGHT - 200)) + 100;
        }
        
        tank->vx = 0;
        tank->vy = 0;
        tank->health = (int)tank->max_health;
        tank->alive = true;
        respawn_time[tank_index] = 0;
    }
}

// One-hot encoding functions
void clear_observation_grid(GameState *game) {
    memset(game->ai_observation, 0, sizeof(game->ai_observation));
}

void add_entity_to_grid(GameState *game, float x, float y, float entity_type) {
    // Convert world coordinates to grid coordinates
    int grid_x = (int)((x - ARENA_MARGIN) / (WORLD_WIDTH - 2 * ARENA_MARGIN) * GRID_WIDTH);
    int grid_y = (int)((y - ARENA_MARGIN) / (WORLD_HEIGHT - 2 * ARENA_MARGIN) * GRID_HEIGHT);
    
    // Clamp to grid bounds
    if (grid_x < 0 || grid_x >= GRID_WIDTH || grid_y < 0 || grid_y >= GRID_HEIGHT) {
        return;
    }
    
    // Calculate index in one-hot encoded array
    int cell_index = grid_y * GRID_WIDTH + grid_x;
    int channel_index = (int)entity_type;
    
    if (channel_index >= 0 && channel_index < 8) {
        int array_index = cell_index * 8 + channel_index;
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
    
    // Add player (enemy)
    if (game->player.alive) {
        add_entity_to_grid(game, game->player.x, game->player.y, ENTITY_ENEMY);
    }
    
    // Add NPCs
    for (int i = 0; i < MAX_NPCS; i++) {
        if (game->npcs[i].alive) {
            add_entity_to_grid(game, game->npcs[i].x, game->npcs[i].y, ENTITY_NPC);
        }
    }
    
    // Add bullets
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (game->bullets[i].active) {
            add_entity_to_grid(game, game->bullets[i].x, game->bullets[i].y, 
                             game->bullets[i].entity_type);
        }
    }
    
    // Add shapes
    for (int i = 0; i < 50; i++) {
        if (game->shapes[i].active) {
            add_entity_to_grid(game, game->shapes[i].x, game->shapes[i].y, ENTITY_SHAPE);
        }
    }
}
// Enhanced Diep.io Tank Game - Full Version
// All parts combined into single file

void render_game(GameState *game) {
    // Clear screen
    SDL_SetRenderDrawColor(game->renderer, 20, 20, 30, 255);
    SDL_RenderClear(game->renderer);
    
    // Draw grid lines (for debugging AI observation)
    if (game->paused) {
        SDL_SetRenderDrawColor(game->renderer, 40, 40, 50, 255);
        
        for (int x = 0; x <= GRID_WIDTH; x++) {
            float world_x = ARENA_MARGIN + x * (WORLD_WIDTH - 2 * ARENA_MARGIN) / GRID_WIDTH;
            float screen_x1 = world_to_screen_x(game, world_x);
            float screen_y1 = world_to_screen_y(game, ARENA_MARGIN);
            float screen_y2 = world_to_screen_y(game, WORLD_HEIGHT - ARENA_MARGIN);
            SDL_RenderDrawLine(game->renderer, (int)screen_x1, (int)screen_y1, 
                             (int)screen_x1, (int)screen_y2);
        }
        for (int y = 0; y <= GRID_HEIGHT; y++) {
            float world_y = ARENA_MARGIN + y * (WORLD_HEIGHT - 2 * ARENA_MARGIN) / GRID_HEIGHT;
            float screen_x1 = world_to_screen_x(game, ARENA_MARGIN);
            float screen_x2 = world_to_screen_x(game, WORLD_WIDTH - ARENA_MARGIN);
            float screen_y1 = world_to_screen_y(game, world_y);
            SDL_RenderDrawLine(game->renderer, (int)screen_x1, (int)screen_y1, 
                             (int)screen_x2, (int)screen_y1);
        }
    }
    
    // Draw world boundaries
    SDL_SetRenderDrawColor(game->renderer, 128, 128, 128, 255);
    SDL_Rect world_bounds = {
        (int)world_to_screen_x(game, ARENA_MARGIN),
        (int)world_to_screen_y(game, ARENA_MARGIN),
        (int)((WORLD_WIDTH - 2 * ARENA_MARGIN) * game->camera.zoom),
        (int)((WORLD_HEIGHT - 2 * ARENA_MARGIN) * game->camera.zoom)
    };
    SDL_RenderDrawRect(game->renderer, &world_bounds);
    
    // Draw shapes
    for (int i = 0; i < 50; i++) {
        if (game->shapes[i].active && is_on_screen(game, game->shapes[i].x, game->shapes[i].y, game->shapes[i].radius)) {
            draw_shape(game, &game->shapes[i]);
        }
    }
    
    // Draw player tank
    if (game->player.alive && is_on_screen(game, game->player.x, game->player.y, game->player.radius)) {
        draw_tank(game, &game->player);
    }
    
    // Draw AI tank
    if (game->ai_tank.alive && is_on_screen(game, game->ai_tank.x, game->ai_tank.y, game->ai_tank.radius)) {
        draw_tank(game, &game->ai_tank);
    }
    
    // Draw NPCs
    for (int i = 0; i < MAX_NPCS; i++) {
        if (game->npcs[i].alive && is_on_screen(game, game->npcs[i].x, game->npcs[i].y, game->npcs[i].radius)) {
            draw_tank(game, &game->npcs[i]);
        }
    }
    
    // Draw bullets
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (game->bullets[i].active && is_on_screen(game, game->bullets[i].x, game->bullets[i].y, game->bullets[i].radius)) {
            draw_bullet(game, &game->bullets[i]);
        }
    }
    
    // Draw UI
    draw_ui(game->renderer, game->font, game);
    
    // Draw minimap
    draw_minimap(game->renderer, game);
    
    // Present
    SDL_RenderPresent(game->renderer);
}

void cleanup_game(GameState *game) {
    if (game->font) {
        TTF_CloseFont(game->font);
        game->font = NULL;
    }
    if (game->renderer) {
        SDL_DestroyRenderer(game->renderer);
        game->renderer = NULL;
    }
    if (game->window) {
        SDL_DestroyWindow(game->window);
        game->window = NULL;
    }
    
    TTF_Quit();
    SDL_Quit();
}

// Utility functions
void draw_circle(SDL_Renderer *renderer, int x, int y, int radius, uint32_t color) {
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

void draw_tank(GameState *game, Tank *tank) {
    float screen_x = world_to_screen_x(game, tank->x);
    float screen_y = world_to_screen_y(game, tank->y);
    float screen_radius = tank->radius * game->camera.zoom;
    
    // Draw tank body
    draw_circle(game->renderer, (int)screen_x, (int)screen_y, (int)screen_radius, tank->color);
    
    // Draw cannon(s) based on tank type
    Uint8 r = (tank->color >> 24) & 0xFF;
    Uint8 g = (tank->color >> 16) & 0xFF;
    Uint8 b = (tank->color >> 8) & 0xFF;
    Uint8 a = tank->color & 0xFF;
    
    SDL_SetRenderDrawColor(game->renderer, r, g, b, a);
    
    float cannon_length = (tank->radius + 15) * game->camera.zoom;
    
    switch (tank->tank_type) {
        case TANK_TWIN:
            // Draw two cannons
            for (int i = -1; i <= 1; i += 2) {
                float offset = i * screen_radius * 0.5f;
                float cannon_start_x = screen_x + cosf(tank->angle + M_PI/2) * offset;
                float cannon_start_y = screen_y + sinf(tank->angle + M_PI/2) * offset;
                float cannon_end_x = cannon_start_x + cosf(tank->angle) * cannon_length;
                float cannon_end_y = cannon_start_y + sinf(tank->angle) * cannon_length;
                SDL_RenderDrawLine(game->renderer, (int)cannon_start_x, (int)cannon_start_y, 
                                 (int)cannon_end_x, (int)cannon_end_y);
            }
            break;
            
        case TANK_MACHINE:
            // Draw machine gun barrel
            {
                float cannon_end_x = screen_x + cosf(tank->angle) * cannon_length * 1.2f;
                float cannon_end_y = screen_y + sinf(tank->angle) * cannon_length * 1.2f;
                SDL_RenderDrawLine(game->renderer, (int)screen_x, (int)screen_y, 
                                 (int)cannon_end_x, (int)cannon_end_y);
            }
            break;
            
        case TANK_DESTROYER:
            // Draw large cannon
            {
                // Draw thicker cannon by drawing multiple lines
                for (int i = -2; i <= 2; i++) {
                    float offset = i * 2;
                    float start_x = screen_x + cosf(tank->angle + M_PI/2) * offset;
                    float start_y = screen_y + sinf(tank->angle + M_PI/2) * offset;
                    float end_x = start_x + cosf(tank->angle) * cannon_length * 1.5f;
                    float end_y = start_y + sinf(tank->angle) * cannon_length * 1.5f;
                    SDL_RenderDrawLine(game->renderer, (int)start_x, (int)start_y, 
                                     (int)end_x, (int)end_y);
                }
            }
            break;
            
        default: // TANK_BASIC, TANK_SNIPER
            {
                float cannon_end_x = screen_x + cosf(tank->angle) * cannon_length;
                float cannon_end_y = screen_y + sinf(tank->angle) * cannon_length;
                SDL_RenderDrawLine(game->renderer, (int)screen_x, (int)screen_y, 
                                 (int)cannon_end_x, (int)cannon_end_y);
            }
            break;
    }
    
    // Draw level indicator
    if (tank->level > 1) {
        SDL_SetRenderDrawColor(game->renderer, 255, 255, 255, 255);
        char level_text[10];
        snprintf(level_text, sizeof(level_text), "%d", tank->level);
        
        if (game->font) {
            SDL_Color textColor = {255, 255, 255, 255};
            SDL_Surface *textSurface = TTF_RenderText_Solid(game->font, level_text, textColor);
            if (textSurface) {
                SDL_Texture *textTexture = SDL_CreateTextureFromSurface(game->renderer, textSurface);
                if (textTexture) {
                    SDL_Rect textRect = {(int)(screen_x - textSurface->w/2), 
                                       (int)(screen_y - screen_radius - 20), 
                                       textSurface->w, textSurface->h};
                    SDL_RenderCopy(game->renderer, textTexture, NULL, &textRect);
                    SDL_DestroyTexture(textTexture);
                }
                SDL_FreeSurface(textSurface);
            }
        }
    }
}

void draw_bullet(GameState *game, Bullet *bullet) {
    float screen_x = world_to_screen_x(game, bullet->x);
    float screen_y = world_to_screen_y(game, bullet->y);
    float screen_radius = bullet->radius * game->camera.zoom;
    
    draw_circle(game->renderer, (int)screen_x, (int)screen_y, 
              (int)screen_radius, bullet->color);
}

void draw_shape(GameState *game, Shape *shape) {
    float screen_x = world_to_screen_x(game, shape->x);
    float screen_y = world_to_screen_y(game, shape->y);
    float screen_radius = shape->radius * game->camera.zoom;
    
    draw_circle(game->renderer, (int)screen_x, (int)screen_y, 
              (int)screen_radius, shape->color);
}

void draw_ui(SDL_Renderer *renderer, TTF_Font *font, GameState *game) {
    if (!font) {
        // Draw simple UI using rectangles when no font is available
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
        
        // Draw experience bar for player
        float exp_ratio = (float)game->player.experience / game->player.experience_to_next_level;
        SDL_Rect exp_bar = {10, 70, (int)(exp_ratio * 200), 10};
        SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255);
        SDL_RenderFillRect(renderer, &exp_bar);
        SDL_Rect exp_border = {10, 70, 200, 10};
        SDL_RenderDrawRect(renderer, &exp_border);
        
        // Draw pause indicator
        if (game->paused) {
            SDL_Rect pause_box = {SCREEN_WIDTH/2 - 100, SCREEN_HEIGHT/2 - 15, 200, 30};
            SDL_RenderDrawRect(renderer, &pause_box);
        }
        
        // Draw training mode indicator
        if (game->training_mode) {
            SDL_Rect train_box = {SCREEN_WIDTH - 200, 10, 180, 30};
            SDL_SetRenderDrawColor(renderer, 0, 255, 255, 255);
            SDL_RenderDrawRect(renderer, &train_box);
        }
        
        return;
    }
    
    // Draw text with font
    char score_text[100];
    snprintf(score_text, sizeof(score_text), "Player: %d  AI: %d", 
             game->score[0], game->score[1]);
    
    SDL_Color textColor = {255, 255, 255, 255};
    SDL_Surface *textSurface = TTF_RenderText_Solid(font, score_text, textColor);
    if (textSurface) {
        SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
        if (textTexture) {
            SDL_Rect textRect = {10, 10, textSurface->w, textSurface->h};
            SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
            SDL_DestroyTexture(textTexture);
        }
        SDL_FreeSurface(textSurface);
    }
    
    // Draw player stats
    char stats_text[200];
    snprintf(stats_text, sizeof(stats_text), 
             "Level: %d  XP: %d/%d  Type: %s", 
             game->player.level, game->player.experience, game->player.experience_to_next_level,
             game->player.tank_type == TANK_BASIC ? "Basic" :
             game->player.tank_type == TANK_TWIN ? "Twin" :
             game->player.tank_type == TANK_SNIPER ? "Sniper" :
             game->player.tank_type == TANK_MACHINE ? "Machine" : "Destroyer");
    
    textSurface = TTF_RenderText_Solid(font, stats_text, textColor);
    if (textSurface) {
        SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
        if (textTexture) {
            SDL_Rect textRect = {10, 40, textSurface->w, textSurface->h};
            SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
            SDL_DestroyTexture(textTexture);
        }
        SDL_FreeSurface(textSurface);
    }
    
    // Draw controls
    char controls_text[] = "Controls: WASD=Move, Space=Shoot, P=Pause, T=Training, +/-=Zoom";
    textSurface = TTF_RenderText_Solid(font, controls_text, textColor);
    if (textSurface) {
        SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
        if (textTexture) {
            SDL_Rect textRect = {10, SCREEN_HEIGHT - 25, textSurface->w, textSurface->h};
            SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
            SDL_DestroyTexture(textTexture);
        }
        SDL_FreeSurface(textSurface);
    }
    
    // Draw pause indicator
    if (game->paused) {
        char pause_text[] = "PAUSED - Grid Visualization Enabled";
        SDL_Color pauseColor = {255, 255, 0, 255};
        SDL_Surface *pauseSurface = TTF_RenderText_Solid(font, pause_text, pauseColor);
        if (pauseSurface) {
            SDL_Texture *pauseTexture = SDL_CreateTextureFromSurface(renderer, pauseSurface);
            if (pauseTexture) {
                SDL_Rect pauseRect = {SCREEN_WIDTH / 2 - 150, SCREEN_HEIGHT / 2 - 15, 
                                   pauseSurface->w, pauseSurface->h};
                SDL_RenderCopy(renderer, pauseTexture, NULL, &pauseRect);
                SDL_DestroyTexture(pauseTexture);
            }
            SDL_FreeSurface(pauseSurface);
        }
    }
    
    // Draw training mode indicator
    if (game->training_mode) {
        char train_text[] = "TRAINING MODE";
        SDL_Color trainColor = {0, 255, 255, 255};
        SDL_Surface *trainSurface = TTF_RenderText_Solid(font, train_text, trainColor);
        if (trainSurface) {
            SDL_Texture *trainTexture = SDL_CreateTextureFromSurface(renderer, trainSurface);
            if (trainTexture) {
                SDL_Rect trainRect = {SCREEN_WIDTH - 180, 10, trainSurface->w, trainSurface->h};
                SDL_RenderCopy(renderer, trainTexture, NULL, &trainRect);
                SDL_DestroyTexture(trainTexture);
            }
            SDL_FreeSurface(trainSurface);
        }
    }
}

void draw_minimap(SDL_Renderer *renderer, GameState *game) {
    // Minimap settings
    int minimap_width = 200;
    int minimap_height = 150;
    int minimap_x = SCREEN_WIDTH - minimap_width - 10;
    int minimap_y = SCREEN_HEIGHT - minimap_height - 10;
    
    // Draw minimap background
    SDL_SetRenderDrawColor(renderer, 40, 40, 50, 200);
    SDL_Rect minimap_bg = {minimap_x, minimap_y, minimap_width, minimap_height};
    SDL_RenderFillRect(renderer, &minimap_bg);
    
    // Draw minimap border
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderDrawRect(renderer, &minimap_bg);
    
    // Calculate scale factors
    float scale_x = (float)minimap_width / WORLD_WIDTH;
    float scale_y = (float)minimap_height / WORLD_HEIGHT;
    
    // Draw player on minimap
    if (game->player.alive) {
        int player_x = minimap_x + (int)(game->player.x * scale_x);
        int player_y = minimap_y + (int)(game->player.y * scale_y);
        SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
        SDL_Rect player_dot = {player_x - 2, player_y - 2, 4, 4};
        SDL_RenderFillRect(renderer, &player_dot);
    }
    
    // Draw AI on minimap
    if (game->ai_tank.alive) {
        int ai_x = minimap_x + (int)(game->ai_tank.x * scale_x);
        int ai_y = minimap_y + (int)(game->ai_tank.y * scale_y);
        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
        SDL_Rect ai_dot = {ai_x - 2, ai_y - 2, 4, 4};
        SDL_RenderFillRect(renderer, &ai_dot);
    }
    
    // Draw NPCs on minimap
    for (int i = 0; i < MAX_NPCS; i++) {
        if (game->npcs[i].alive) {
            int npc_x = minimap_x + (int)(game->npcs[i].x * scale_x);
            int npc_y = minimap_y + (int)(game->npcs[i].y * scale_y);
            SDL_SetRenderDrawColor(renderer, 0, 255, 255, 255);
            SDL_Rect npc_dot = {npc_x - 1, npc_y - 1, 2, 2};
            SDL_RenderFillRect(renderer, &npc_dot);
        }
    }
    
    // Draw camera view rectangle
    float view_width = SCREEN_WIDTH / game->camera.zoom;
    float view_height = SCREEN_HEIGHT / game->camera.zoom;
    int cam_x = minimap_x + (int)((game->camera.x - view_width/2) * scale_x);
    int cam_y = minimap_y + (int)((game->camera.y - view_height/2) * scale_y);
    int cam_w = (int)(view_width * scale_x);
    int cam_h = (int)(view_height * scale_y);
    
    SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255);
    SDL_Rect cam_rect = {cam_x, cam_y, cam_w, cam_h};
    SDL_RenderDrawRect(renderer, &cam_rect);
}
