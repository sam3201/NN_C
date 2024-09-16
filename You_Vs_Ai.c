#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "utils/Raylib/raylib.h"
#include "utils/Raylib/raymath.h"
#include "utils/NN/NN.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define GRID_SIZE 30
#define CELL_SIZE (SCREEN_WIDTH / GRID_SIZE)

#define BASE_RADIUS 20
#define TURRET_RADIUS 15
#define ARM_LENGTH 30 
#define ARM_THICKNESS 12
#define PLAYER_SPEED 5
#define TURN_SPEED 0.1f 
#define MAX_COOLDOWN 20
#define BULLET_RADIUS 5
#define BULLET_SPEED 10

#define NUM_ACTIONS 5
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 1 

typedef struct Bullet {
    Vector2 position;
    Vector2 velocity;
    bool active;
    int tank_id; 
    struct Bullet* next;
} Bullet;

typedef struct {
    Vector2 position;
    Vector2 velocity;  // Added velocity field
    float angle;
    Color color;
    int id;
    int score;
    Bullet *bulletList;
} Tank;

Tank* create_tank(int x, int y, Color color, int id) {
    Tank* newTank = (Tank*)malloc(sizeof(Tank));
    newTank->position = (Vector2){x, y};
    newTank->velocity = (Vector2){0, 0};  // Initialize velocity
    newTank->angle = 0.0f;
    newTank->color = color;
    newTank->id = id;
    newTank->score = 0;
    newTank->bulletList = NULL;
    return newTank;
}

void AddBullet(Bullet** bulletList, Vector2 position, Vector2 velocity, int tank_id) {
    Bullet* newBullet = (Bullet*)malloc(sizeof(Bullet));
    newBullet->position = position;
    newBullet->velocity = velocity;
    newBullet->active = true;
    newBullet->tank_id = tank_id;
    newBullet->next = *bulletList;
    *bulletList = newBullet;
}
void OneHotEncodeScreen(long double* screen, Tank* player, Tank* agent, int player_id) {
    memset(screen, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(long double)); 

    int playerIndex = (int)(player->position.y * SCREEN_WIDTH + player->position.x);
    if (playerIndex >= 0 && playerIndex < SCREEN_WIDTH * SCREEN_HEIGHT) {
        screen[playerIndex] = 1.0;  
    }

    int agentIndex = (int)(agent->position.y * SCREEN_WIDTH + agent->position.x);
    if (agentIndex >= 0 && agentIndex < SCREEN_WIDTH * SCREEN_HEIGHT) {
        screen[agentIndex] = (player_id == agent->id) ? -1.0 : 1.0;  
    }
}

void UpdateBullets(Bullet** bulletList, Tank tanks[], int numTanks, int* playerReward) {
    Bullet* current = *bulletList;
    Bullet* prev = NULL;

    while (current != NULL) {
        if (current->active) {
            current->position.x += current->velocity.x;
            current->position.y += current->velocity.y;

            for (int i = 0; i < numTanks; i++) {
                if (CheckCollisionCircles(current->position, BULLET_RADIUS, tanks[i].position, BASE_RADIUS)) {
                    current->active = false; 
                    if (tanks[i].id != current->tank_id) {
                        *playerReward += 10;  
                    }
                    break;
                }
            }

            if (current->position.x < 0 || current->position.x > SCREEN_WIDTH ||
                current->position.y < 0 || current->position.y > SCREEN_HEIGHT) {
                current->active = false;
            }
        }

        if (!current->active) {
            if (prev == NULL) {
                *bulletList = current->next;
                free(current);
                current = *bulletList;
            } else {
                prev->next = current->next;
                free(current);
                current = prev->next;
            }
        } else {
            prev = current;
            current = current->next;
        }
    }
}

void DrawTank(Tank* tank) {
    DrawCircleV(tank->position, BASE_RADIUS, tank->color);
    Vector2 armEnd = Vector2Add(tank->position, Vector2Rotate((Vector2){0, -ARM_LENGTH}, tank->angle));
    DrawLineEx(tank->position, armEnd, ARM_THICKNESS, tank->color);
}
void DrawBullets(Bullet* bulletList) {
    Bullet* current = bulletList;
    while (current != NULL) {
        if (current->active) {
            DrawCircleV(current->position, BULLET_RADIUS, YELLOW);
        }
        current = current->next;
    }
}
void FreeBullets(Bullet* bulletList) {
    Bullet* current = bulletList;
    while (current != NULL) {
        Bullet* temp = current;
        current = current->next;
        free(temp);
    }
}

void UpdateTankPosition(Tank* tank) {
    tank->position.x += tank->velocity.x;
    tank->position.y += tank->velocity.y;

    if (tank->position.x < 0) tank->position.x = 0;
    if (tank->position.x > SCREEN_WIDTH) tank->position.x = SCREEN_WIDTH;
    if (tank->position.y < 0) tank->position.y = 0;
    if (tank->position.y > SCREEN_HEIGHT) tank->position.y = SCREEN_HEIGHT;
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

    NN_backprop(nn, input, q_values_true, action);

    printf("QLearning update: Action %d, Reward %.Lf, Max future Q-value %.Lf\n", action, reward, max_future_q);

    free(q_values);
    free(q_values_true);
}

int SelectActionWithExploration(long double *q_values, float epsilon, int numActions) {
    if ((float)rand() / RAND_MAX < epsilon) {
        return rand() % numActions;
    } else {
        int bestAction = 0;
        for (int i = 1; i < numActions; i++) {
            if (q_values[i] > q_values[bestAction]) {
                bestAction = i;
            }
        }
        return bestAction;
    }
}

void RunSimulation() {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Tank Battle with Q-learning Exploration");
    SetTargetFPS(60);

    Tank *player = create_tank(100, 100, BLUE, 0);
    Tank *agent = create_tank(300, 300, RED, 1);

    Bullet *playerBullets = NULL;
    Bullet *agentBullets = NULL;

    long double *playerAction;
    long double *agentAction; 
    int playerReward = 0;
    int agentReward = 0;

    size_t layers[] = {GRID_SIZE * GRID_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, 0};
    ActivationFunction activations[] = {RELU, SIGMOID, RELU}; 
    ActivationDerivative activations_d[] = {RELU_DERIVATIVE, SIGMOID_DERIVATIVE, RELU_DERIVATIVE};
    LossFunction loss = MSE;
    LossDerivative loss_d = MSE_DERIVATIVE;
    NN_t *playerNN = NN_init(layers, activations, activations_d, loss, loss_d);
    NN_t *agentNN = NN_init(layers, activations, activations_d, loss, loss_d); 

    long double gamma = 0.9;
    
    float epsilon = 1.0f;  
    float epsilonDecay = 0.995f;  
    float epsilonMin = 0.01f;  

    long double screen[SCREEN_WIDTH * SCREEN_HEIGHT] = {0};

    srand(time(NULL));
    while (!WindowShouldClose()) {
        Vector2 move = {0};
        if (IsKeyDown(KEY_RIGHT)) move.x += PLAYER_SPEED;
        if (IsKeyDown(KEY_LEFT)) move.x -= PLAYER_SPEED;
        if (IsKeyDown(KEY_UP)) move.y -= PLAYER_SPEED;
        if (IsKeyDown(KEY_DOWN)) move.y += PLAYER_SPEED;

        player->velocity = move;
        UpdateTankPosition(player);

        Bullet *newBullet = NULL;
        if (IsKeyPressed(KEY_SPACE)) {
            Vector2 bulletPos = Vector2Add(player->position, Vector2Rotate((Vector2){0, -ARM_LENGTH}, player->angle));
            Vector2 bulletVel = Vector2Rotate((Vector2){0, -BULLET_SPEED}, player->angle);
            AddBullet(&playerBullets, bulletPos, bulletVel, player->id);
        }

        OneHotEncodeScreen(screen, player, agent, player->id);
        playerAction = NN_forward(playerNN, screen);

        int action = SelectActionWithExploration(playerAction, epsilon, NUM_ACTIONS);

        Vector2 actionDir = {0};
        switch (action) {
            case 0: actionDir.x = PLAYER_SPEED; break;
            case 1: actionDir.x = -PLAYER_SPEED; break;
            case 2: actionDir.y = PLAYER_SPEED; break;
            case 3: actionDir.y = -PLAYER_SPEED; break;
            case 4:
                if (newBullet == NULL) {
                    Vector2 bulletPos = Vector2Add(player->position, Vector2Rotate((Vector2){0, -ARM_LENGTH}, player->angle));
                    Vector2 bulletVel = Vector2Rotate((Vector2){0, -BULLET_SPEED}, player->angle);
                    AddBullet(&playerBullets, bulletPos, bulletVel, player->id);
                }
                break;
        }
        player->velocity = actionDir;
        UpdateTankPosition(player);

        UpdateBullets(&playerBullets, (Tank[]){*player, *agent}, 2, &playerReward);

        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }

        BeginDrawing();
        ClearBackground(BLACK);

        DrawTank(player);
        DrawTank(agent);
        DrawBullets(playerBullets);
        DrawBullets(agentBullets);

        EndDrawing();
    }

    FreeBullets(playerBullets);
    FreeBullets(agentBullets);
    CloseWindow();
}
int main() {
    RunSimulation();  
    return 0;
}

