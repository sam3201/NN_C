#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "utils/Raylib/raylib.h"
#include "utils/NN/NN.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

#define PADDLE_WIDTH 100
#define PADDLE_HEIGHT 20
#define PADDLE_SPEED 7

#define BALL_RADIUS 10
#define BALL_SPEED 5

#define BRICK_ROWS 5
#define BRICK_COLUMNS 10
#define BRICK_WIDTH 70
#define BRICK_HEIGHT 20

typedef enum { TITLE_SCREEN = 0, IN_PLAY, GAME_OVER, GAME_WON } GameState;

typedef struct {
    Vector2 position;
    int width;
    int height;
    Color color;
} Paddle;

typedef struct {
    Vector2 position;
    Vector2 speed;
    bool active;
} Ball;

typedef struct {
    Vector2 position;
    int width;
    int height;
    bool active;
    Color color;
} Brick;

typedef struct {
    Paddle paddle;
    Ball ball;
    Brick bricks[BRICK_ROWS][BRICK_COLUMNS];
    int lives;
    int score;
    int bricksRemaining;
    GameState state;
} Game;

typedef struct {
    NN_t *actor;
    NN_t *critic;
    float gamma;
    float epsilon;
    float *old_action_probs;
    long double *rewards;
    size_t step_count;
    size_t max_steps;
} PPOAgent;

void InitPPOAgent(PPOAgent *agent, size_t max_steps) {
    size_t actorLayers[] = {SCREEN_WIDTH * SCREEN_HEIGHT, 128, 3};  
    size_t criticLayers[] = {SCREEN_WIDTH * SCREEN_HEIGHT, 128, 1};

    ActivationFunction activationFunctions[] = {RELU, SIGMOID, SIGMOID};
    ActivationDerivative activationDerivatives[] = {RELU_DERIVATIVE, SIGMOID_DERIVATIVE, RELU_DERIVATIVE};
    LossFunction loss = MSE;
    LossDerivative lossDerivative = MSE_DERIVATIVE;

    agent->actor = NN_init(actorLayers, activationFunctions, activationDerivatives, loss, lossDerivative);
    agent->critic = NN_init(criticLayers, activationFunctions, activationDerivatives, loss, lossDerivative);

    agent->gamma = 0.99;
    agent->epsilon = 0.2;
    agent->old_action_probs = calloc(3, sizeof(float));
    agent->rewards = calloc(max_steps, sizeof(long double));
    agent->step_count = 0;
    agent->max_steps = max_steps;
}

// Fixed ball rendering bounds and memory freeing in OneHotEncodeScreen
int *OneHotEncodeScreen(Game *game) {
    int *encodedScreen = calloc(SCREEN_WIDTH * SCREEN_HEIGHT, sizeof(int));  // Using calloc to initialize to zero

    for (int x = game->paddle.position.x; x < game->paddle.position.x + game->paddle.width && x < SCREEN_WIDTH; x++) {
        for (int y = game->paddle.position.y; y < game->paddle.position.y + PADDLE_HEIGHT && y < SCREEN_HEIGHT; y++) {
            encodedScreen[y * SCREEN_WIDTH + x] = 1;
        }
    }

    for (int x = game->ball.position.x - BALL_RADIUS; x <= game->ball.position.x + BALL_RADIUS && x < SCREEN_WIDTH; x++) {
        for (int y = game->ball.position.y - BALL_RADIUS; y <= game->ball.position.y + BALL_RADIUS && y < SCREEN_HEIGHT; y++) {
            if (x >= 0 && y >= 0)  // Ensure valid indices
                encodedScreen[y * SCREEN_WIDTH + x] = 2;
        }
    }

    for (int row = 0; row < BRICK_ROWS; row++) {
        for (int col = 0; col < BRICK_COLUMNS; col++) {
            if (game->bricks[row][col].active) {
                for (int x = game->bricks[row][col].position.x; x < game->bricks[row][col].position.x + BRICK_WIDTH && x < SCREEN_WIDTH; x++) {
                    for (int y = game->bricks[row][col].position.y; y < game->bricks[row][col].position.y + BRICK_HEIGHT && y < SCREEN_HEIGHT; y++) {
                        if (x >= 0 && y >= 0) 
                            encodedScreen[y * SCREEN_WIDTH + x] = 3;
                    }
                }
            }
        }
    }

    return encodedScreen;
}

void UpdatePaddleWithPPO(Game *game, PPOAgent *agent) {
    int *encodedState = OneHotEncodeScreen(game);

    if (encodedState == NULL) {
        printf("Error: encodedState is NULL\n");
        return;
    }
    
    long double *action_probs = NN_forward(agent->actor, encodedState);

    if (action_probs == NULL) {
        printf("Error: action_probs is NULL\n");
        free(encodedState);
        return;
    }

    int action = 0;
    if (action_probs[1] > action_probs[0] && action_probs[1] > action_probs[2]) {
        action = 1; 
    } else if (action_probs[2] > action_probs[0]) {
        action = -1; 
    }

    game->paddle.position.x += action * PADDLE_SPEED;

    agent->step_count++;
    agent->rewards[agent->step_count] = game->score;

    free(encodedState); 
    free(action_probs);  
}

void UpdateBall(Game *game) {
    game->ball.position.x += game->ball.speed.x;
    game->ball.position.y += game->ball.speed.y;

    if (game->ball.position.x < 0 || game->ball.position.x > SCREEN_WIDTH - BALL_RADIUS) {
        game->ball.speed.x *= -1;
    }

    if (game->ball.position.y < 0) {
        game->ball.speed.y *= -1;
    }

    if (game->ball.position.y > SCREEN_HEIGHT) {
        game->lives--;
        game->ball.position = (Vector2){SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2};
        game->ball.speed = (Vector2){BALL_SPEED, -BALL_SPEED};
    }
}

void CheckBallBrickCollision(Game *game) {
    for (int row = 0; row < BRICK_ROWS; row++) {
        for (int col = 0; col < BRICK_COLUMNS; col++) {
            if (game->bricks[row][col].active) {
                Brick *brick = &game->bricks[row][col];
                if (CheckCollisionCircleRec(game->ball.position, BALL_RADIUS,
                        (Rectangle){brick->position.x, brick->position.y, brick->width, brick->height})) {
                    game->ball.speed.y *= -1;
                    brick->active = false;
                    game->bricksRemaining--;
                    game->score += 100;
                }
            }
        }
    }
}

void TrainPPOAgent(PPOAgent *agent, long double inputs[]) {
    long double G = 0;
    long double critic_value = *NN_forward(agent->critic, inputs);  
    long double advantage = G - critic_value;

    long double *actor_targets = calloc(3, sizeof(long double));
    actor_targets[0] = advantage;
    actor_targets[1] = 0;  
    actor_targets[2] = 0; 

    NN_backprop(agent->actor, inputs, actor_targets, advantage);
    NN_backprop(agent->critic, inputs, &G, critic_value);

    free(actor_targets);
}

void DrawTitleScreen() {
    ClearBackground(BLACK);
    DrawText("Breakout PPO", SCREEN_WIDTH / 2 - 120, SCREEN_HEIGHT / 2 - 80, 40, WHITE);
    DrawText("Press SPACE to Start", SCREEN_WIDTH / 2 - 150, SCREEN_HEIGHT / 2, 30, WHITE);
}

void DrawGame(Game *game) {
    ClearBackground(BLACK);

    DrawRectangleV(game->paddle.position, (Vector2){game->paddle.width, PADDLE_HEIGHT}, game->paddle.color);

    DrawCircleV(game->ball.position, BALL_RADIUS, WHITE);

    for (int row = 0; row < BRICK_ROWS; row++) {
        for (int col = 0; col < BRICK_COLUMNS; col++) {
            if (game->bricks[row][col].active) {
                DrawRectangleV(game->bricks[row][col].position, (Vector2){BRICK_WIDTH, BRICK_HEIGHT}, game->bricks[row][col].color);
            }
        }
    }

    DrawText(TextFormat("Score: %d", game->score), 10, 10, 20, WHITE);
    DrawText(TextFormat("Lives: %d", game->lives), SCREEN_WIDTH - 120, 10, 20, WHITE);

    if (game->state == GAME_OVER) {
        DrawText("Game Over!", SCREEN_WIDTH / 2 - 100, SCREEN_HEIGHT / 2, 40, RED);
    } else if (game->state == GAME_WON) {
        DrawText("You Won!", SCREEN_WIDTH / 2 - 80, SCREEN_HEIGHT / 2, 40, GREEN);
    }
}

void UpdateGame(Game *game, PPOAgent *agent) {
    UpdateBall(game);
    CheckBallBrickCollision(game);

    if (game->lives <= 0) {
        game->state = GAME_OVER;
    } else if (game->bricksRemaining == 0) {
        game->state = GAME_WON;
    } else if (game->state == IN_PLAY) {
        UpdatePaddleWithPPO(game, agent);
    }
}

void InitGame(Game *game) {
    game->paddle = (Paddle){(Vector2){SCREEN_WIDTH / 2 - PADDLE_WIDTH / 2, SCREEN_HEIGHT - PADDLE_HEIGHT - 10}, PADDLE_WIDTH, PADDLE_HEIGHT, BLUE};
    game->ball = (Ball){(Vector2){SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2}, (Vector2){BALL_SPEED, -BALL_SPEED}, true};
    game->lives = 10000000000000;
    game->score = 0;
    game->bricksRemaining = BRICK_ROWS * BRICK_COLUMNS;
    game->state = IN_PLAY;

    for (int row = 0; row < BRICK_ROWS; row++) {
        for (int col = 0; col < BRICK_COLUMNS; col++) {
            game->bricks[row][col] = (Brick){(Vector2){col * BRICK_WIDTH, row * BRICK_HEIGHT}, BRICK_WIDTH, BRICK_HEIGHT, true, RED};
        }
    }
}

int main(void) {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Breakout PPO");
    SetTargetFPS(60);

    Game game;
    InitGame(&game);

    PPOAgent agent;
    InitPPOAgent(&agent, 1000); 

    while (!WindowShouldClose()) {
        if (game.state == TITLE_SCREEN) {
            if (IsKeyDown(KEY_SPACE)) {
                game.state = IN_PLAY;
                InitGame(&game);  
            }
        } else if (game.state == IN_PLAY) {
            if (IsKeyDown(KEY_RIGHT)) {
                game.paddle.position.x += PADDLE_SPEED;
                if (game.paddle.position.x > SCREEN_WIDTH - PADDLE_WIDTH) {
                    game.paddle.position.x = SCREEN_WIDTH - PADDLE_WIDTH;
                }
            } else if (IsKeyDown(KEY_LEFT)) {
                game.paddle.position.x -= PADDLE_SPEED;
                if (game.paddle.position.x < 0) {
                    game.paddle.position.x = 0;
                }
            }

            UpdateGame(&game, &agent);
            TrainPPOAgent(&agent, (long double *)OneHotEncodeScreen(&game)); 
        }

        BeginDrawing();
        if (game.state == TITLE_SCREEN) {
            DrawTitleScreen();
        } else {
            DrawGame(&game);
        }
        EndDrawing();
    }

    CloseWindow();

    free(agent.old_action_probs);
    free(agent.rewards);
    NN_destroy(agent.actor);
    NN_destroy(agent.critic);

    return 0;
}

