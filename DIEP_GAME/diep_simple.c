// Simplified Diep.io-style tank game without AI dependencies
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/select.h>
#include <termios.h>

// Simple console-based game for now
#define SCREEN_WIDTH 80
#define SCREEN_HEIGHT 24
#define TANK_SIZE 2
#define BULLET_SIZE 1
#define BULLET_SPEED 2
#define TANK_SPEED 1

typedef struct {
    int x, y;
    int vx, vy;
    float angle;
    int health;
    bool alive;
    char symbol;
} Tank;

typedef struct {
    int x, y;
    int vx, vy;
    bool active;
    int owner;
} Bullet;

typedef struct {
    Tank player;
    Tank ai_tank;
    Bullet bullets[50];
    int score[2];
    bool game_running;
    char screen[SCREEN_HEIGHT][SCREEN_WIDTH];
} GameState;

void init_game(GameState* game);
void update_game(GameState* game);
void render_game(GameState* game);
void handle_input(GameState* game);
void update_ai(GameState* game);
void fire_bullet(GameState* game, int owner);
void update_bullets(GameState* game);
void check_collisions(GameState* game);
void clear_screen(GameState* game);
void draw_tank(GameState* game, Tank* tank);
void draw_bullet(GameState* game, Bullet* bullet);

int main() {
    GameState game;
    srand(time(NULL));
    
    init_game(&game);
    
    printf("=== DIEP.IO TANK GAME ===\n");
    printf("Controls:\n");
    printf("W/S - Move forward/backward\n");
    printf("A/D - Rotate left/right\n");
    printf("Space - Shoot\n");
    printf("Q - Quit\n");
    printf("\n");
    
    while (game.game_running) {
        clear_screen(&game);
        handle_input(&game);
        update_game(&game);
        render_game(&game);
        
        // Simple delay
        struct timespec ts = {0, 50000000}; // 50ms
        nanosleep(&ts, NULL);
    }
    
    printf("\nGame Over! Final Score - Player: %d, AI: %d\n", 
           game.score[0], game.score[1]);
    
    return 0;
}

void init_game(GameState* game) {
    // Initialize player
    game->player.x = 20;
    game->player.y = SCREEN_HEIGHT / 2;
    game->player.vx = 0;
    game->player.vy = 0;
    game->player.angle = 0;
    game->player.health = 100;
    game->player.alive = true;
    game->player.symbol = 'P';
    
    // Initialize AI
    game->ai_tank.x = 60;
    game->ai_tank.y = SCREEN_HEIGHT / 2;
    game->ai_tank.vx = 0;
    game->ai_tank.vy = 0;
    game->ai_tank.angle = M_PI;
    game->ai_tank.health = 100;
    game->ai_tank.alive = true;
    game->ai_tank.symbol = 'A';
    
    // Initialize bullets
    for (int i = 0; i < 50; i++) {
        game->bullets[i].active = false;
    }
    
    game->score[0] = 0;
    game->score[1] = 0;
    game->game_running = true;
}

void handle_input(GameState* game) {
    // Non-blocking input check (simplified)
    fd_set readfds;
    struct timeval tv = {0, 0};
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);
    
    if (select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv) > 0) {
        char c;
        if (read(STDIN_FILENO, &c, 1) > 0) {
            switch (c) {
                case 'w':
                case 'W':
                    if (game->player.alive) {
                        game->player.vx = (int)(cos(game->player.angle) * TANK_SPEED);
                        game->player.vy = (int)(sin(game->player.angle) * TANK_SPEED);
                    }
                    break;
                case 's':
                case 'S':
                    if (game->player.alive) {
                        game->player.vx = -(int)(cos(game->player.angle) * TANK_SPEED);
                        game->player.vy = -(int)(sin(game->player.angle) * TANK_SPEED);
                    }
                    break;
                case 'a':
                case 'A':
                    if (game->player.alive) {
                        game->player.angle -= 0.2;
                    }
                    break;
                case 'd':
                case 'D':
                    if (game->player.alive) {
                        game->player.angle += 0.2;
                    }
                    break;
                case ' ':
                    fire_bullet(game, 0);
                    break;
                case 'q':
                case 'Q':
                    game->game_running = false;
                    break;
            }
        }
    }
}

void update_game(GameState* game) {
    // Update player
    if (game->player.alive) {
        game->player.x += game->player.vx;
        game->player.y += game->player.vy;
        game->player.vx = 0;
        game->player.vy = 0;
        
        // Keep player in bounds
        if (game->player.x < 2) game->player.x = 2;
        if (game->player.x > SCREEN_WIDTH - 3) game->player.x = SCREEN_WIDTH - 3;
        if (game->player.y < 2) game->player.y = 2;
        if (game->player.y > SCREEN_HEIGHT - 3) game->player.y = SCREEN_HEIGHT - 3;
    }
    
    // Update AI
    if (game->ai_tank.alive) {
        update_ai(game);
        game->ai_tank.x += game->ai_tank.vx;
        game->ai_tank.y += game->ai_tank.vy;
        game->ai_tank.vx = 0;
        game->ai_tank.vy = 0;
        
        // Keep AI in bounds
        if (game->ai_tank.x < 2) game->ai_tank.x = 2;
        if (game->ai_tank.x > SCREEN_WIDTH - 3) game->ai_tank.x = SCREEN_WIDTH - 3;
        if (game->ai_tank.y < 2) game->ai_tank.y = 2;
        if (game->ai_tank.y > SCREEN_HEIGHT - 3) game->ai_tank.y = SCREEN_HEIGHT - 3;
    }
    
    update_bullets(game);
    check_collisions(game);
}

void update_ai(GameState* game) {
    if (!game->ai_tank.alive || !game->player.alive) return;
    
    // Simple AI: aim and move towards player
    int dx = game->player.x - game->ai_tank.x;
    int dy = game->player.y - game->ai_tank.y;
    float target_angle = atan2(dy, dx);
    
    // Rotate towards player
    float angle_diff = target_angle - game->ai_tank.angle;
    while (angle_diff > M_PI) angle_diff -= 2 * M_PI;
    while (angle_diff < -M_PI) angle_diff += 2 * M_PI;
    
    if (fabs(angle_diff) > 0.1) {
        game->ai_tank.angle += (angle_diff > 0) ? 0.1 : -0.1;
    }
    
    // Move towards player if far away
    int distance = sqrt(dx * dx + dy * dy);
    if (distance > 20) {
        game->ai_tank.vx = (int)(cos(game->ai_tank.angle) * TANK_SPEED);
        game->ai_tank.vy = (int)(sin(game->ai_tank.angle) * TANK_SPEED);
    }
    
    // Shoot if aimed well
    if (fabs(angle_diff) < 0.2 && distance < 30) {
        if (rand() % 10 < 3) { // 30% chance to shoot
            fire_bullet(game, 1);
        }
    }
}

void fire_bullet(GameState* game, int owner) {
    Tank* tank = (owner == 0) ? &game->player : &game->ai_tank;
    
    if (!tank->alive) return;
    
    for (int i = 0; i < 50; i++) {
        if (!game->bullets[i].active) {
            game->bullets[i].x = tank->x + (int)(cos(tank->angle) * 3);
            game->bullets[i].y = tank->y + (int)(sin(tank->angle) * 3);
            game->bullets[i].vx = (int)(cos(tank->angle) * BULLET_SPEED);
            game->bullets[i].vy = (int)(sin(tank->angle) * BULLET_SPEED);
            game->bullets[i].active = true;
            game->bullets[i].owner = owner;
            break;
        }
    }
}

void update_bullets(GameState* game) {
    for (int i = 0; i < 50; i++) {
        if (game->bullets[i].active) {
            game->bullets[i].x += game->bullets[i].vx;
            game->bullets[i].y += game->bullets[i].vy;
            
            // Check bounds
            if (game->bullets[i].x < 0 || game->bullets[i].x >= SCREEN_WIDTH ||
                game->bullets[i].y < 0 || game->bullets[i].y >= SCREEN_HEIGHT) {
                game->bullets[i].active = false;
            }
        }
    }
}

void check_collisions(GameState* game) {
    for (int i = 0; i < 50; i++) {
        if (!game->bullets[i].active) continue;
        
        // Check player collision
        if (game->player.alive && game->bullets[i].owner == 1) {
            int dx = game->bullets[i].x - game->player.x;
            int dy = game->bullets[i].y - game->player.y;
            if (dx * dx + dy * dy < 9) { // Hit
                game->player.health -= 20;
                game->bullets[i].active = false;
                
                if (game->player.health <= 0) {
                    game->player.alive = false;
                    game->score[1]++;
                }
            }
        }
        
        // Check AI collision
        if (game->ai_tank.alive && game->bullets[i].owner == 0) {
            int dx = game->bullets[i].x - game->ai_tank.x;
            int dy = game->bullets[i].y - game->ai_tank.y;
            if (dx * dx + dy * dy < 9) { // Hit
                game->ai_tank.health -= 20;
                game->bullets[i].active = false;
                
                if (game->ai_tank.health <= 0) {
                    game->ai_tank.alive = false;
                    game->score[0]++;
                }
            }
        }
    }
    
    // Respawn dead tanks
    if (!game->player.alive) {
        static int respawn_timer = 0;
        respawn_timer++;
        if (respawn_timer > 60) { // 3 seconds at 20 FPS
            game->player.alive = true;
            game->player.health = 100;
            game->player.x = 20;
            game->player.y = SCREEN_HEIGHT / 2;
            respawn_timer = 0;
        }
    }
    
    if (!game->ai_tank.alive) {
        static int respawn_timer = 0;
        respawn_timer++;
        if (respawn_timer > 60) {
            game->ai_tank.alive = true;
            game->ai_tank.health = 100;
            game->ai_tank.x = 60;
            game->ai_tank.y = SCREEN_HEIGHT / 2;
            respawn_timer = 0;
        }
    }
}

void clear_screen(GameState* game) {
    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            game->screen[y][x] = ' ';
        }
    }
    
    // Draw borders
    for (int x = 0; x < SCREEN_WIDTH; x++) {
        game->screen[0][x] = '-';
        game->screen[SCREEN_HEIGHT - 1][x] = '-';
    }
    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        game->screen[y][0] = '|';
        game->screen[y][SCREEN_WIDTH - 1] = '|';
    }
}

void render_game(GameState* game) {
    clear_screen(game);
    
    // Draw tanks
    if (game->player.alive) {
        draw_tank(game, &game->player);
    }
    if (game->ai_tank.alive) {
        draw_tank(game, &game->ai_tank);
    }
    
    // Draw bullets
    for (int i = 0; i < 50; i++) {
        if (game->bullets[i].active) {
            draw_bullet(game, &game->bullets[i]);
        }
    }
    
    // Clear screen and print
    printf("\033[2J\033[H"); // ANSI escape codes to clear screen and move cursor
    printf("=== DIEP.IO TANK GAME ===\n");
    printf("Player: %d HP  |  AI: %d HP  |  Score - P: %d, AI: %d\n", 
           game->player.health, game->ai_tank.health, game->score[0], game->score[1]);
    printf("Controls: W/S=Move, A/D=Rotate, Space=Shoot, Q=Quit\n\n");
    
    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            putchar(game->screen[y][x]);
        }
        putchar('\n');
    }
}

void draw_tank(GameState* game, Tank* tank) {
    if (tank->x >= 0 && tank->x < SCREEN_WIDTH && 
        tank->y >= 0 && tank->y < SCREEN_HEIGHT) {
        game->screen[tank->y][tank->x] = tank->symbol;
        
        // Draw cannon
        int cannon_x = tank->x + (int)(cos(tank->angle) * 2);
        int cannon_y = tank->y + (int)(sin(tank->angle) * 2);
        if (cannon_x >= 0 && cannon_x < SCREEN_WIDTH && 
            cannon_y >= 0 && cannon_y < SCREEN_HEIGHT) {
            game->screen[cannon_y][cannon_x] = '-';
        }
    }
}

void draw_bullet(GameState* game, Bullet* bullet) {
    if (bullet->x >= 0 && bullet->x < SCREEN_WIDTH && 
        bullet->y >= 0 && bullet->y < SCREEN_HEIGHT) {
        game->screen[bullet->y][bullet->x] = '*';
    }
}
