
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "utils/Raylib/raylib.h"
#include "utils/Raylib/raymath.h"
#include "utils/NN/NN.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

#define HIDDEN_SIZE 2 
#define OUTPUT_SIZE 5  
#define ACTIONS {0, 1, 2, 3, 4}

#define BASE_RADIUS 20
#define TURRET_RADIUS 15
#define ARM_LENGTH 30 
#define ARM_THICKNESS 12
#define PLAYER_SPEED 5
#define TURN_SPEED 2
#define BULLET_RADIUS 5
#define BULLET_SPEED 10

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

    // First arm
    Vector2 armOffset1 = Vector2Rotate((Vector2){BASE_RADIUS, 0}, tank->angle);
    Vector2 direction1 = Vector2Rotate((Vector2){0, -ARM_LENGTH}, tank->angle);
    Vector2 cannonPos1 = Vector2Add(tank->position, armOffset1);
    DrawLineEx(cannonPos1, Vector2Add(cannonPos1, direction1), ARM_THICKNESS, tank->color);

    // Second arm (opposite side)
    Vector2 armOffset2 = Vector2Rotate((Vector2){-BASE_RADIUS, 0}, tank->angle);
    Vector2 direction2 = Vector2Rotate((Vector2){0, -ARM_LENGTH}, tank->angle);
    Vector2 cannonPos2 = Vector2Add(tank->position, armOffset2);
    DrawLineEx(cannonPos2, Vector2Add(cannonPos2, direction2), ARM_THICKNESS, tank->color);
}

bool CheckCollisionBulletTank(const Bullet *bullet, const Tank *tank) {
    return CheckCollisionCircles(bullet->position, BULLET_RADIUS, tank->position, BASE_RADIUS);
}

void UpdateBullet(Bullet *bullet, Tank *opponent, int *reward) {
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

void UpdateBullets(Bullet **bulletList, Tank *opponent, int *reward) {
    Bullet *current = *bulletList;
    Bullet *prev = NULL;

    while (current != NULL) {
        UpdateBullet(current, opponent, reward);

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

int main(void) {
    SetTraceLogLevel(LOG_ERROR); 
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "You_Vs_Ai");
    SetTargetFPS(60);

    size_t layers[] = {SCREEN_WIDTH * SCREEN_HEIGHT, HIDDEN_SIZE, OUTPUT_SIZE, 0};
    ActivationFunction activationFunctions[] = {SIGMOID, SIGMOID, RELU};
    ActivationDerivative activationDerivatives[] = {SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE, RELU_DERIVATIVE};
    NN_t *nn = NN_init(layers, activationFunctions, activationDerivatives, MSE, MSE_DERIVATIVE);

    Tank *player = create_tank(SCREEN_WIDTH / 2, SCREEN_HEIGHT * 3 / 4, BLUE);
    Tank *agent = create_tank(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 4, RED);

    Bullet *bulletList = NULL;

    long double gamma = 0.99; 
    int reward = 0;

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_W)) player->position = Vector2Add(player->position, Vector2Rotate((Vector2){0, -PLAYER_SPEED}, player->angle));
        if (IsKeyDown(KEY_S)) player->position = Vector2Add(player->position, Vector2Rotate((Vector2){0, PLAYER_SPEED}, player->angle));
        if (IsKeyDown(KEY_A)) player->angle -= TURN_SPEED * DEG2RAD;
        if (IsKeyDown(KEY_D)) player->angle += TURN_SPEED * DEG2RAD;

        UpdateTankPosition(player);

        if (IsKeyPressed(KEY_SPACE)) {
            Vector2 bulletPosition = Vector2Add(player->position, Vector2Rotate((Vector2){0, -BASE_RADIUS - BULLET_RADIUS}, player->angle));
            Vector2 bulletVelocity = Vector2Rotate((Vector2){0, -BULLET_SPEED}, player->angle);
            AddBullet(&bulletList, bulletPosition, bulletVelocity);
        }

        UpdateBullets(&bulletList, agent, &reward);

        long double input[4];
        memset(input, 0, sizeof(input));
        input[0] = player->position.x / SCREEN_WIDTH;
        input[1] = player->position.y / SCREEN_HEIGHT;
        input[2] = agent->position.x / SCREEN_WIDTH;
        input[3] = agent->position.y / SCREEN_HEIGHT;

        long double *q_values = NN_forward(nn, input);
        int action = 0;
        for (int i = 1; i < OUTPUT_SIZE; ++i) {
            if (q_values[i] > q_values[action]) {
                action = i;
            }
        }
        printf("Action: %d\n", action);

        switch (action) {
            case 0: agent->position = Vector2Add(agent->position, Vector2Rotate((Vector2){0, -PLAYER_SPEED}, agent->angle)); break; // Move forward
            case 1: agent->position = Vector2Add(agent->position, Vector2Rotate((Vector2){0, PLAYER_SPEED}, agent->angle)); break; // Move backward
            case 2: agent->angle -= TURN_SPEED * DEG2RAD; break;
            case 3: agent->angle += TURN_SPEED * DEG2RAD; break;
            case 4: {
                Vector2 bulletPosition = Vector2Add(agent->position, Vector2Rotate((Vector2){0, -BASE_RADIUS - BULLET_RADIUS}, agent->angle));
                Vector2 bulletVelocity = Vector2Rotate((Vector2){0, -BULLET_SPEED}, agent->angle);
                AddBullet(&bulletList, bulletPosition, bulletVelocity);
            } break;
        }

        UpdateTankPosition(agent);

        QLearningUpdate(nn, input, action, reward, gamma);

        BeginDrawing();
        ClearBackground(BLACK);
        DrawTank(player);
        DrawTank(agent);
        DrawBullets(bulletList);
        DrawFPS(10, 10);
        EndDrawing();

        free(q_values);
    }

    FreeBullets(bulletList);
    free(player);
    free(agent);
    NN_destroy(nn);
    CloseWindow();

    return 0;
}
