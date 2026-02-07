// Diep.io Tank Simulation - All NPCs are RL Agents
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#define SCREEN_WIDTH 1200
#define SCREEN_HEIGHT 800
#define WORLD_WIDTH 4000
#define WORLD_HEIGHT 3000
#define ARENA_MARGIN 100
#define TANK_SIZE 20
#define BULLET_SIZE 15
#define MAX_BULLETS 500
#define MAX_AGENTS 20
#define GRID_WIDTH 60
#define GRID_HEIGHT 45
#define AI_INPUT_SIZE (GRID_WIDTH * GRID_HEIGHT * 8)

// Colors
#define COLOR_PLAYER 0xFF00FF00
#define COLOR_AGENT1 0xFF0000FF
#define COLOR_AGENT2 0xFFFFFF00
#define COLOR_AGENT3 0xFFFF00FF
#define COLOR_AGENT4 0xFF00FFFF
#define COLOR_BULLET 0xFFFFFFFF
#define COLOR_SHAPE 0xFF808080

// Entity types for one-hot encoding
typedef enum {
    ENTITY_EMPTY = 0,
    ENTITY_SELF = 1,
    ENTITY_AGENT = 2,
    ENTITY_BULLET_SELF = 3,
    ENTITY_BULLET_AGENT = 4,
    ENTITY_SHAPE = 5
} EntityType;

// Tank types
typedef enum {
    TANK_BASIC = 0,
    TANK_TWIN = 1,
    TANK_SNIPER = 2,
    TANK_MACHINE = 3,
    TANK_DESTROYER = 4
} TankType;

typedef struct {
    float speed;
    float bullet_speed;
    float bullet_damage;
    uint32_t fire_rate;
    int bullet_count;
    float spread_angle;
} TankStats;

typedef struct {
    float x, y;
    float vx, vy;
    float angle;
    float radius;
    uint32_t color;
    bool alive;
    int health;
    int max_health;
    int level;
    int experience;
    int experience_to_next_level;
    TankType tank_type;
    TankStats stats;
    int score;
    int agent_id;
    float ai_observation[AI_INPUT_SIZE];
    float ai_action[4]; // [move_x, move_y, aim_angle, shoot]
    uint32_t last_fire_time;
    uint32_t last_decision_time;
} Agent;

typedef struct {
    float x, y;
    float radius;
    uint32_t color;
    bool active;
    int health;
    int experience_value;
} Shape;

typedef struct {
    float x, y;
    float vx, vy;
    float radius;
    uint32_t color;
    bool active;
    int damage;
    int owner_id;
    EntityType entity_type;
} Bullet;

typedef struct {
    float x, y;
    float zoom;
    float target_x, target_y;
} Camera;

typedef struct {
    SDL_Window *window;
    SDL_Renderer *renderer;
    TTF_Font *font;
    Agent agents[MAX_AGENTS];
    Bullet bullets[MAX_BULLETS];
    Shape shapes[50];
    Camera camera;
    bool game_running;
    bool paused;
    uint32_t frame_count;
    uint32_t total_time;
    int generation;
    float fitness_scores[MAX_AGENTS];
} GameState;

// Tank stats by type
static const TankStats TANK_STATS[] = {
    // speed, bullet_speed, damage, fire_rate, bullet_count, spread_angle
    {3.0f, 4.0f, 20.0f, 1500, 1, 0.0f},      // Basic
    {2.8f, 4.5f, 15.0f, 1200, 2, 0.3f},      // Twin
    {3.2f, 6.0f, 40.0f, 2000, 1, 0.0f},     // Sniper
    {2.5f, 3.5f, 10.0f, 400, 1, 0.0f},     // Machine
    {2.2f, 3.0f, 60.0f, 3000, 1, 0.0f}      // Destroyer
};

// Function prototypes
int init_game(GameState *game);
void cleanup_game(GameState *game);
void handle_input(GameState *game);
void update_game(GameState *game);
void render_game(GameState *game);
void update_camera(GameState *game);
float world_to_screen_x(GameState *game, float x);
float world_to_screen_y(GameState *game, float y);
bool is_on_screen(GameState *game, float x, float y, float radius);
float get_distance(float x1, float y1, float x2, float y2);
uint32_t get_current_time_ms();

// AI functions
void init_agents(GameState *game);
void update_agent_ai(GameState *game, int agent_index);
void update_observation_grid(GameState *game, int agent_index);
void clear_observation_grid(float observation[AI_INPUT_SIZE]);
void add_entity_to_grid(float observation[AI_INPUT_SIZE], float x, float y, float entity_type);
void calculate_ai_action(GameState *game, int agent_index);
void execute_ai_action(GameState *game, int agent_index);
void update_fitness_scores(GameState *game);

// Game mechanics
void update_agent_stats(Agent *agent);
void add_experience(Agent *agent, int amount);
void level_up(Agent *agent);
void evolve_tank(Agent *agent, TankType new_type);
void fire_bullet(GameState *game, int agent_index);
void update_bullets(GameState *game);
void update_shapes(GameState *game);
void check_collisions(GameState *game);
void respawn_agent(Agent *agent, GameState *game);

// Rendering
void draw_circle(SDL_Renderer *renderer, int x, int y, int radius, uint32_t color);
void draw_agent(GameState *game, Agent *agent);
void draw_bullet(GameState *game, Bullet *bullet);
void draw_shape(GameState *game, Shape *shape);
void draw_ui(SDL_Renderer *renderer, TTF_Font *font, GameState *game);
void draw_minimap(SDL_Renderer *renderer, GameState *game);

int main(int argc, char *argv[]) {
    (void)argc; // Suppress unused parameter warning
    (void)argv; // Suppress unused parameter warning
    GameState game = {0};
    
    if (!init_game(&game)) {
        return 1;
    }
    
    printf("Diep.io Tank Simulation Started\n");
    printf("Features: %d RL Agents, Evolution, Training Mode\n", MAX_AGENTS);
    
    while (game.game_running) {
        handle_input(&game);
        update_game(&game);
        render_game(&game);
        
        SDL_Delay(16); // ~60 FPS
        game.frame_count++;
        game.total_time += 16;
    }
    
    printf("Simulation ended. Final fitness scores:\n");
    for (int i = 0; i < MAX_AGENTS; i++) {
        printf("Agent %d: Score: %d, Fitness: %.2f\n", i, game.agents[i].score, game.fitness_scores[i]);
    }
    
    cleanup_game(&game);
    return 0;
}

int init_game(GameState *game) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL initialization failed: %s\n", SDL_GetError());
        return false;
    }
    
    // Initialize SDL_ttf
    if (TTF_Init() < 0) {
        printf("SDL_ttf initialization failed: %s\n", TTF_GetError());
        return false;
    }
    
    // Create window
    game->window = SDL_CreateWindow("Diep.io Tank Simulation - RL Agents", 
                                   SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                   SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (!game->window) {
        printf("Window creation failed: %s\n", SDL_GetError());
        return false;
    }
    
    // Create renderer
    game->renderer = SDL_CreateRenderer(game->window, -1, SDL_RENDERER_ACCELERATED);
    if (!game->renderer) {
        printf("Renderer creation failed: %s\n", SDL_GetError());
        return false;
    }
    
    // Load font
    game->font = TTF_OpenFont("/System/Library/Fonts/Arial.ttf", 16);
    if (!game->font) {
        printf("Font loading failed, using fallback UI\n");
    }
    
    // Initialize camera
    game->camera.x = WORLD_WIDTH / 2;
    game->camera.y = WORLD_HEIGHT / 2;
    game->camera.zoom = 0.5f;
    game->camera.target_x = game->camera.x;
    game->camera.target_y = game->camera.y;
    
    // Initialize agents
    init_agents(game);
    
    // Initialize shapes
    for (int i = 0; i < 50; i++) {
        game->shapes[i].x = (float)(rand() % (WORLD_WIDTH - 200)) + 100;
        game->shapes[i].y = (float)(rand() % (WORLD_HEIGHT - 200)) + 100;
        game->shapes[i].radius = (float)(rand() % 20) + 10;
        game->shapes[i].active = true;
        game->shapes[i].health = (int)game->shapes[i].radius;
        game->shapes[i].experience_value = (int)game->shapes[i].radius * 2;
        game->shapes[i].color = COLOR_SHAPE;
    }
    
    // Initialize bullets
    for (int i = 0; i < MAX_BULLETS; i++) {
        game->bullets[i].active = false;
    }
    
    game->game_running = true;
    game->paused = false;
    game->frame_count = 0;
    game->total_time = 0;
    game->generation = 1;
    
    srand((unsigned int)time(NULL));
    
    return true;
}

void init_agents(GameState *game) {
    uint32_t agent_colors[] = {
        COLOR_AGENT1, COLOR_AGENT2, COLOR_AGENT3, COLOR_AGENT4,
        0xFF800000, 0xFF008000, 0xFF000080, 0xFF808000,
        0xFF800080, 0xFF008080, 0xFF404040, 0xFF804040,
        0xFF408040, 0xFF404080, 0xFF808040, 0xFF804080,
        0xFF408080, 0xFF606060, 0xFFA0A0A0, 0xFFC0C0C0
    };
    
    for (int i = 0; i < MAX_AGENTS; i++) {
        Agent *agent = &game->agents[i];
        
        // Random spawn position
        agent->x = (float)(rand() % (WORLD_WIDTH - 400)) + 200;
        agent->y = (float)(rand() % (WORLD_HEIGHT - 400)) + 200;
        agent->vx = 0;
        agent->vy = 0;
        agent->angle = (float)(rand() % 360) * M_PI / 180.0f;
        
        // Agent properties
        agent->radius = TANK_SIZE;
        agent->color = agent_colors[i % 20];
        agent->alive = true;
        agent->health = 100;
        agent->max_health = 100;
        agent->level = 1;
        agent->experience = 0;
        agent->experience_to_next_level = 100;
        agent->tank_type = TANK_BASIC;
        agent->score = 0;
        agent->agent_id = i;
        agent->last_fire_time = 0;
        agent->last_decision_time = 0;
        
        // Initialize AI
        update_agent_stats(agent);
        memset(agent->ai_observation, 0, sizeof(agent->ai_observation));
        memset(agent->ai_action, 0, sizeof(agent->ai_action));
        
        game->fitness_scores[i] = 0.0f;
    }
    
    printf("Initialized %d RL Agents\n", MAX_AGENTS);
}

void update_agent_ai(GameState *game, int agent_index) {
    Agent *agent = &game->agents[agent_index];
    
    if (!agent->alive) {
        return;
    }
    
    uint32_t current_time = get_current_time_ms();
    
    // Update AI decision every 100ms
    if (current_time - agent->last_decision_time > 100) {
        update_observation_grid(game, agent_index);
        calculate_ai_action(game, agent_index);
        execute_ai_action(game, agent_index);
        agent->last_decision_time = current_time;
    }
}

void update_observation_grid(GameState *game, int agent_index) {
    Agent *agent = &game->agents[agent_index];
    
    clear_observation_grid(agent->ai_observation);
    
    // Add self
    add_entity_to_grid(agent->ai_observation, agent->x, agent->y, ENTITY_SELF);
    
    // Add other agents
    for (int i = 0; i < MAX_AGENTS; i++) {
        if (i != agent_index && game->agents[i].alive) {
            add_entity_to_grid(agent->ai_observation, game->agents[i].x, game->agents[i].y, ENTITY_AGENT);
        }
    }
    
    // Add bullets
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (game->bullets[i].active) {
            if (game->bullets[i].owner_id == agent_index) {
                add_entity_to_grid(agent->ai_observation, game->bullets[i].x, game->bullets[i].y, ENTITY_BULLET_SELF);
            } else {
                add_entity_to_grid(agent->ai_observation, game->bullets[i].x, game->bullets[i].y, ENTITY_BULLET_AGENT);
            }
        }
    }
    
    // Add shapes
    for (int i = 0; i < 50; i++) {
        if (game->shapes[i].active) {
            add_entity_to_grid(agent->ai_observation, game->shapes[i].x, game->shapes[i].y, ENTITY_SHAPE);
        }
    }
}

void clear_observation_grid(float observation[AI_INPUT_SIZE]) {
    memset(observation, 0, AI_INPUT_SIZE * sizeof(float));
}

void add_entity_to_grid(float observation[AI_INPUT_SIZE], float x, float y, float entity_type) {
    int grid_x = (int)((x - ARENA_MARGIN) / (WORLD_WIDTH - 2 * ARENA_MARGIN) * GRID_WIDTH);
    int grid_y = (int)((y - ARENA_MARGIN) / (WORLD_HEIGHT - 2 * ARENA_MARGIN) * GRID_HEIGHT);
    
    if (grid_x < 0 || grid_x >= GRID_WIDTH || grid_y < 0 || grid_y >= GRID_HEIGHT) {
        return;
    }
    
    int cell_index = grid_y * GRID_WIDTH + grid_x;
    int channel_index = (int)entity_type;
    
    if (channel_index >= 0 && channel_index < 8) {
        int array_index = cell_index * 8 + channel_index;
        if (array_index < AI_INPUT_SIZE) {
            observation[array_index] = 1.0f;
        }
    }
}

void calculate_ai_action(GameState *game, int agent_index) {
    Agent *agent = &game->agents[agent_index];
    
    // MUZE RL Agent - Simple neural network simulation
    // In a real implementation, this would call the actual MUZE neural network
    
    // Process observation through simulated neural network
    float input_sum = 0.0f;
    for (int i = 0; i < AI_INPUT_SIZE; i++) {
        input_sum += agent->ai_observation[i];
    }
    
    // Simple heuristic based on observation density
    float threat_level = input_sum / (AI_INPUT_SIZE * 0.1f); // Normalize
    
    // Find nearest enemy and shape for targeting
    float nearest_enemy_dist = 999999.0f;
    float nearest_enemy_x = 0, nearest_enemy_y = 0;
    float nearest_shape_dist = 999999.0f;
    float nearest_shape_x = 0, nearest_shape_y = 0;
    
    for (int i = 0; i < MAX_AGENTS; i++) {
        if (i != agent_index && game->agents[i].alive) {
            float dist = get_distance(agent->x, agent->y, game->agents[i].x, game->agents[i].y);
            if (dist < nearest_enemy_dist) {
                nearest_enemy_dist = dist;
                nearest_enemy_x = game->agents[i].x;
                nearest_enemy_y = game->agents[i].y;
            }
        }
    }
    
    for (int i = 0; i < 50; i++) {
        if (game->shapes[i].active) {
            float dist = get_distance(agent->x, agent->y, game->shapes[i].x, game->shapes[i].y);
            if (dist < nearest_shape_dist) {
                nearest_shape_dist = dist;
                nearest_shape_x = game->shapes[i].x;
                nearest_shape_y = game->shapes[i].y;
            }
        }
    }
    
    // MUZE-inspired decision making with exploration vs exploitation
    float exploration_factor = 0.1f + (rand() % 100) / 1000.0f; // Random exploration
    float target_x, target_y;
    bool should_shoot = false;
    
    // Neural network-like decision based on tank type and situation
    float aggression_score = 1.0f - (threat_level * 0.5f); // Less aggressive when threatened
    
    if (agent->tank_type == TANK_SNIPER) {
        // Snipers prefer long range, calculated shots
        if (nearest_enemy_dist < 600 && nearest_enemy_dist > 300 && aggression_score > 0.3f) {
            target_x = nearest_enemy_x;
            target_y = nearest_enemy_y;
            should_shoot = true;
        } else if (nearest_shape_dist < 500 && aggression_score > 0.2f) {
            target_x = nearest_shape_x;
            target_y = nearest_shape_y;
            should_shoot = true;
        } else {
            // Strategic positioning
            float wander_angle = agent->angle + exploration_factor * M_PI;
            target_x = agent->x + cosf(wander_angle) * 200;
            target_y = agent->y + sinf(wander_angle) * 200;
        }
    } else if (agent->tank_type == TANK_DESTROYER) {
        // Destroyers are aggressive but strategic
        if (nearest_enemy_dist < 400 && aggression_score > 0.4f) {
            target_x = nearest_enemy_x;
            target_y = nearest_enemy_y;
            should_shoot = true;
        } else if (nearest_shape_dist < 250 && aggression_score > 0.3f) {
            target_x = nearest_shape_x;
            target_y = nearest_shape_y;
            should_shoot = true;
        } else {
            // Hunt mode
            if (nearest_enemy_dist < 800) {
                target_x = nearest_enemy_x;
                target_y = nearest_enemy_y;
            } else {
                target_x = agent->x + cosf(agent->angle) * 150;
                target_y = agent->y + sinf(agent->angle) * 150;
            }
        }
    } else if (agent->tank_type == TANK_MACHINE) {
        // Machine guns focus on sustained fire
        if (nearest_enemy_dist < 350 && aggression_score > 0.2f) {
            target_x = nearest_enemy_x;
            target_y = nearest_enemy_y;
            should_shoot = true;
        } else if (nearest_shape_dist < 200) {
            target_x = nearest_shape_x;
            target_y = nearest_shape_y;
            should_shoot = true;
        } else {
            // Area control
            target_x = nearest_shape_x;
            target_y = nearest_shape_y;
        }
    } else if (agent->tank_type == TANK_TWIN) {
        // Twin tanks balance offense and mobility
        if (nearest_enemy_dist < 300 && aggression_score > 0.35f) {
            target_x = nearest_enemy_x;
            target_y = nearest_enemy_y;
            should_shoot = true;
        } else if (nearest_shape_dist < 180) {
            target_x = nearest_shape_x;
            target_y = nearest_shape_y;
            should_shoot = true;
        } else {
            // Flanking behavior
            float flank_angle = atan2f(nearest_enemy_y - agent->y, nearest_enemy_x - agent->x);
            flank_angle += (exploration_factor - 0.5f) * M_PI/2; // Add some randomness
            target_x = agent->x + cosf(flank_angle) * 120;
            target_y = agent->y + sinf(flank_angle) * 120;
        }
    } else {
        // Basic tanks: adaptive behavior
        if (nearest_enemy_dist < 280 && aggression_score > 0.25f) {
            target_x = nearest_enemy_x;
            target_y = nearest_enemy_y;
            should_shoot = true;
        } else if (nearest_shape_dist < 160) {
            target_x = nearest_shape_x;
            target_y = nearest_shape_y;
            should_shoot = true;
        } else {
            // Resource gathering with exploration
            if (nearest_shape_dist < 400) {
                target_x = nearest_shape_x;
                target_y = nearest_shape_y;
            } else {
                float explore_angle = agent->angle + exploration_factor * M_PI;
                target_x = agent->x + cosf(explore_angle) * 180;
                target_y = agent->y + sinf(explore_angle) * 180;
            }
        }
    }
    
    // Calculate movement and aiming with MUZE-style continuous control
    float dx = target_x - agent->x;
    float dy = target_y - agent->y;
    float target_angle = atan2f(dy, dx);
    
    // Normalize angle difference for smooth control
    float angle_diff = target_angle - agent->angle;
    while (angle_diff > M_PI) angle_diff -= 2 * M_PI;
    while (angle_diff < -M_PI) angle_diff += 2 * M_PI;
    
    // MUZE-inspired continuous action outputs
    float movement_intensity = fminf(1.0f, aggression_score + exploration_factor);
    agent->ai_action[0] = cosf(target_angle) * agent->stats.speed * movement_intensity; // move_x
    agent->ai_action[1] = sinf(target_angle) * agent->stats.speed * movement_intensity; // move_y
    agent->ai_action[2] = target_angle; // aim_angle
    agent->ai_action[3] = should_shoot ? 1.0f : 0.0f; // shoot
    
    // Add some noise to simulate neural network uncertainty
    if (exploration_factor > 0.8f) {
        agent->ai_action[0] += (rand() % 100 - 50) / 100.0f;
        agent->ai_action[1] += (rand() % 100 - 50) / 100.0f;
    }
}

void execute_ai_action(GameState *game, int agent_index) {
    Agent *agent = &game->agents[agent_index];
    
    // Apply movement
    agent->vx = agent->ai_action[0];
    agent->vy = agent->ai_action[1];
    
    // Apply aiming (smooth rotation)
    float target_angle = agent->ai_action[2];
    float angle_diff = target_angle - agent->angle;
    while (angle_diff > M_PI) angle_diff -= 2 * M_PI;
    while (angle_diff < -M_PI) angle_diff += 2 * M_PI;
    agent->angle += angle_diff * 0.1f;
    
    // Shoot if requested
    if (agent->ai_action[3] > 0.5f) {
        fire_bullet(game, agent_index);
    }
}