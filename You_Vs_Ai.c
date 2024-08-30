#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include "utils/Raylib/raylib.h"
#include "utils/Raylib/raymath.h"
#include "utils/NN/NN.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

#define HIDDEN_SIZE 2 
#define OUTPUT_SIZE 5  

#define BASE_RADIUS 20
#define TURRET_RADIUS 15
#define ARM_LENGTH 30 
#define ARM_THICKNESS 12
#define PLAYER_SPEED 5
#define TURN_SPEED 2
#define BULLET_RADIUS 5
#define BULLET_SPEED 10

const int ACTIONS[5] = { 0, 1, 2, 3, 4 }; 

typedef struct Bullet {
    Vector2 position;
    Vector2 velocity;
    bool active;
    struct Bullet* next;
} Bullet;

typedef struct {
    Vector2 position;
    float angle;  
    Color color;
} Tank;

Tank *create_tank(float x, float y, Color color) {
    Tank *tank = (Tank *)malloc(sizeof(Tank));
    tank->position = (Vector2){x, y};
    tank->angle = 0;  
    tank->color = color;
    return tank;
}

void DrawTank(const Tank *tank) {
    DrawCircleV(tank->position, BASE_RADIUS, tank->color);

    Vector2 armOffset1 = Vector2Rotate((Vector2){BASE_RADIUS, 0}, tank->angle);
    Vector2 direction1 = Vector2Rotate((Vector2){0, -ARM_LENGTH}, tank->angle);
    Vector2 cannonPos1 = Vector2Add(tank->position, armOffset1);
    DrawLineEx(cannonPos1, Vector2Add(cannonPos1, direction1), ARM_THICKNESS, tank->color);

    Vector2 armOffset2 = Vector2Rotate((Vector2){-BASE_RADIUS, 0}, tank->angle);
    Vector2 direction2 = Vector2Rotate((Vector2){0, -ARM_LENGTH}, tank->angle);
    Vector2 cannonPos2 = Vector2Add(tank->position, armOffset2);
    DrawLineEx(cannonPos2, Vector2Add(cannonPos2, direction2), ARM_THICKNESS, tank->color);
}

bool CheckCollisionBulletTank(const Bullet *bullet, const Tank *tank) {
    return CheckCollisionCircles(bullet->position, BULLET_RADIUS, tank->position, BASE_RADIUS);
}

void UpdateBullet(Bullet *bullet, Tank *opponent, int *reward, int *score) {
    if (bullet->active) {
        bullet->position = Vector2Add(bullet->position, bullet->velocity);

        // Check for collision with walls
        if (bullet->position.x < 0 || bullet->position.x > SCREEN_WIDTH ||
            bullet->position.y < 0 || bullet->position.y > SCREEN_HEIGHT) {
            bullet->active = false;
        }

        // Check for collision with the opponent tank
        if (CheckCollisionBulletTank(bullet, opponent)) {
            bullet->active = false;
            *reward = 1; // Agent hit the opponent
            (*score)++;   // Increase the score of the shooter
        }
    }
}

void DrawBullet(const Bullet *bullet) {
    if (bullet->active) {
        DrawCircleV(bullet->position, BULLET_RADIUS, WHITE);
    }
}

void AddBullet(Bullet **bulletList, Vector2 position, Vector2 velocity) {
    Bullet *newBullet = (Bullet *)malloc(sizeof(Bullet));
    newBullet->position = position;
    newBullet->velocity = velocity;
    newBullet->active = true;
    newBullet->next = *bulletList;
    *bulletList = newBullet;
}

void UpdateBullets(Bullet **bulletList, Tank *opponent, int *reward, int *score) {
    Bullet *current = *bulletList;
    Bullet *prev = NULL;

    while (current != NULL) {
        UpdateBullet(current, opponent, reward, score);

        if (!current->active) {
            Bullet *toDelete = current;
            if (prev == NULL) { // If it's the head of the list
                *bulletList = current->next;
            } else {
                prev->next = current->next;
            }
            current = current->next;
            free(toDelete);
        } else {
            prev = current;
            current = current->next;
        }
    }
}

void DrawBullets(Bullet *bulletList) {
    Bullet *current = bulletList;
    while (current != NULL) {
        DrawBullet(current);
        current = current->next;
    }
}

void FreeBullets(Bullet *bulletList) {
    Bullet *current = bulletList;
    while (current != NULL) {
        Bullet *next = current->next;
        free(current);
        current = next;
    }
}

void QLearningUpdate(NN_t *nn, long double *input, int action, long double reward, long double gamma) {
    long double *q_values = NN_forward(nn, input);

    long double *q_values_true = (long double *)malloc(sizeof(long double) * OUTPUT_SIZE);
    memcpy(q_values_true, q_values, sizeof(long double) * OUTPUT_SIZE);

    long double max_future_q = -INFINITY;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (q_values[i] > max_future_q) {
            max_future_q = q_values[i];
        }
    }

    q_values_true[action] = reward + gamma * max_future_q;

    NN_backprop(nn, input, q_values_true);

    free(q_values);
    free(q_values_true);
}

void UpdateTankPosition(Tank *tank) {
    if (tank->position.x < BASE_RADIUS) {
        tank->position.x = BASE_RADIUS;
    } else if (tank->position.x > SCREEN_WIDTH - BASE_RADIUS) {
        tank->position.x = SCREEN_WIDTH - BASE_RADIUS;
    }

    if (tank->position.y < BASE_RADIUS) {
        tank->position.y = BASE_RADIUS;
    } else if (tank->position.y > SCREEN_HEIGHT - BASE_RADIUS) {
        tank->position.y = SCREEN_HEIGHT - BASE_RADIUS;
    }
}

void OneHotEncodeScreen(int screen[SCREEN_HEIGHT][SCREEN_WIDTH], Tank *player, Tank *agent, Bullet *bulletList, int currentPlayerID) {
    memset(screen, 0, sizeof(int) * SCREEN_WIDTH * SCREEN_HEIGHT);

    // Mark player and agent positions
    screen[(int)player->position.y][(int)player->position.x] = 1;
    screen[(int)agent->position.y][(int)agent->position.x] = 2;

    // Mark bullets
    Bullet *current = bulletList;
    while (current != NULL) {
        if (current->active) {
            screen[(int)current->position.y][(int)current->position.x] = 3;
        }
        current = current->next;
    }

    // Indicate which player we are
    screen[0][0] = currentPlayerID;
}

int get_best_action(NN_t *nn, long double *input, int actions[]) {
    long double *q_values = NN_forward(nn, input);

    int best_action = actions[0];
    long double max_value = q_values[0];

    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (q_values[i] > max_value) {
            max_value = q_values[i];
            best_action = actions[i];
        }
    }

    free(q_values);
    return best_action;
}


int EpsilonGreedyAction(NN_t *nn, long double *input, float epsilon, int actions[], int numActions) {
    if ((float)rand() / RAND_MAX < epsilon) {
        return actions[rand() % numActions]; // Random action (exploration)
    } else {
        return get_best_action(nn, input, actions); // Best action (exploitation)
    }
}

void DrawTitleScreen() {
    ClearBackground(BLACK);
    DrawText("Choose Mode:", SCREEN_WIDTH / 2 - 80, SCREEN_HEIGHT / 2 - 60, 20, WHITE);
    DrawText("1. Play as Player", SCREEN_WIDTH / 2 - 80, SCREEN_HEIGHT / 2 - 20, 20, GREEN);
    DrawText("2. Watch AI vs AI", SCREEN_WIDTH / 2 - 80, SCREEN_HEIGHT / 2 + 20, 20, RED);
    DrawText("Press 1 or 2", SCREEN_WIDTH / 2 - 80, SCREEN_HEIGHT / 2 + 60, 20, YELLOW);
}

int main(void) {
    srand(time(NULL));

    SetTraceLogLevel(LOG_ERROR); 
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "You_Vs_Ai");
    SetTargetFPS(60);

    size_t layers[] = {SCREEN_WIDTH * SCREEN_HEIGHT + 1, HIDDEN_SIZE, OUTPUT_SIZE, 0}; // +1 for player ID
    ActivationFunction activationFunctions[] = {SIGMOID, SIGMOID, RELU};
    ActivationDerivative activationDerivatives[] = {SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE, RELU_DERIVATIVE};
    NN_t *nn = NN_init(layers, activationFunctions, activationDerivatives, MSE, MSE_DERIVATIVE);

    Tank *player = create_tank(SCREEN_WIDTH / 4, SCREEN_HEIGHT / 2, GREEN);
    Tank *agent = create_tank(3 * SCREEN_WIDTH / 4, SCREEN_HEIGHT / 2, RED);

    Bullet *bulletList = NULL;

    float epsilon = 1.0; 
    float epsilonDecay = 0.995;
    float minEpsilon = 0.01;
    long double gamma = 0.9;
    int reward = 0;
    int playerScore = 0;
    int agentScore = 0;

    bool running = true;
    int mode = 0; 

    while (!WindowShouldClose()) {
        if (mode == 0) { 
            DrawTitleScreen();

            if (IsKeyPressed(KEY_ONE)) {
                mode = 1; 
            } else if (IsKeyPressed(KEY_TWO)) {
                mode = 2; 
            }

            BeginDrawing();
            EndDrawing();
            continue;
        }

        if (mode == 1) { 
            if (IsKeyDown(KEY_W)) player->position = Vector2Add(player->position, Vector2Rotate((Vector2){0, -PLAYER_SPEED}, player->angle));
            if (IsKeyDown(KEY_S)) player->position = Vector2Add(player->position, Vector2Rotate((Vector2){0, PLAYER_SPEED}, player->angle));
            if (IsKeyDown(KEY_A)) player->angle -= TURN_SPEED * DEG2RAD;
            if (IsKeyDown(KEY_D)) player->angle += TURN_SPEED * DEG2RAD;
            UpdateTankPosition(player);

            if (IsKeyPressed(KEY_SPACE)) {
                Vector2 armOffset = Vector2Rotate((Vector2){BASE_RADIUS, 0}, player->angle);
                Vector2 bulletDirection = Vector2Rotate((Vector2){0, -BULLET_SPEED}, player->angle);
                AddBullet(&bulletList, Vector2Add(player->position, armOffset), bulletDirection);
                UpdateBullets(&bulletList, player, &reward, &agentScore);
            }
        }

        int screen[SCREEN_HEIGHT][SCREEN_WIDTH];
        OneHotEncodeScreen(screen, player, agent, bulletList, 1);

        long double *input = (long double *)malloc(sizeof(long double) * (SCREEN_WIDTH * SCREEN_HEIGHT + 1));

        for (int i = 0; i < SCREEN_HEIGHT; i++) {
            for (int j = 0; j < SCREEN_WIDTH; j++) {
                input[i * SCREEN_WIDTH + j] = (long double)screen[i][j];
            }
        }

        input[SCREEN_WIDTH * SCREEN_HEIGHT] = 2;

        int action;
        if (mode == 1) {
            action = EpsilonGreedyAction(nn, input, epsilon, ACTIONS, OUTPUT_SIZE);
        } else {
            action = get_best_action(nn, input, ACTIONS);
        }

        switch (action) {
            case 0: agent->position = Vector2Add(agent->position, Vector2Rotate((Vector2){0, -PLAYER_SPEED}, agent->angle)); break;
            case 1: agent->position = Vector2Add(agent->position, Vector2Rotate((Vector2){0, PLAYER_SPEED}, agent->angle)); break;
            case 2: agent->angle -= TURN_SPEED * DEG2RAD; break;
            case 3: agent->angle += TURN_SPEED * DEG2RAD; break;
            case 4: {
                Vector2 armOffset = Vector2Rotate((Vector2){BASE_RADIUS, 0}, agent->angle);
                Vector2 bulletDirection = Vector2Rotate((Vector2){0, -BULLET_SPEED}, agent->angle);
                AddBullet(&bulletList, Vector2Add(agent->position, armOffset), bulletDirection);
            } break;
            default: break;
        }

        UpdateTankPosition(agent);
        UpdateBullets(&bulletList, agent, &reward, &playerScore);

        if (mode == 2) {
            QLearningUpdate(nn, input, action, reward, gamma);
        }
        reward = 0;

        free(input);

        BeginDrawing();
        ClearBackground(BLACK);

        DrawTank(player);
        DrawTank(agent);
        DrawBullets(bulletList);
        DrawText(TextFormat("Score: %i", playerScore), 10, 10, 20, WHITE);
        DrawText(TextFormat("Score: %i", agentScore), 10, 40, 20, WHITE);
        DrawFPS(10, 60); 

        EndDrawing();

        if (epsilon > minEpsilon) {
            epsilon *= epsilonDecay; 
        }
    }

    NN_destroy(nn);
    FreeBullets(bulletList);
    free(player);
    free(agent);

    CloseWindow();

    return 0;
}



