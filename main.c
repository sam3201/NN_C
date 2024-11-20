#include "utils/Raylib/raylib.h"
#include "utils/NN/NN.h"
#include "utils/NN/NEAT.h"
#include "utils/VISUALIZER/client.h"
#include "utils/VISUALIZER/NN_visualizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

// Window Constants
#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

// Game Physics Constants
#define GRAVITY 0.5f
#define JUMP_FORCE -12.0f
#define MOVEMENT_SPEED 4.0f

// Population Constants
#define POPULATION_SIZE 50
#define AGENT_SIZE 20
#define PLATFORM_COUNT 5
#define ENEMY_COUNT 5
#define COIN_COUNT 10

// Neural Network Constants
#define INPUT_COUNT 2
#define OUTPUT_COUNT 1
#define MUTATION_RATE 0.1f
#define CROSSOVER_RATE 0.7f

// Game Object Colors
#define AGENT_COLOR BLUE
#define PLATFORM_COLOR GREEN
#define ENEMY_COLOR RED
#define COIN_COLOR GOLD
#define BACKGROUND_COLOR SKYBLUE

// Fitness Rewards
#define COIN_REWARD 100.0f
#define DISTANCE_REWARD 1.0f
#define SURVIVAL_REWARD 0.5f

// Constants
#define PLATFORM_WIDTH 200
#define PLATFORM_HEIGHT 20
#define ENEMY_SIZE 20
#define COIN_SIZE 10

// Object Encoding Values
#define ENCODE_EMPTY 0
#define ENCODE_AGENT 1
#define ENCODE_PLATFORM 2
#define ENCODE_ENEMY 3
#define ENCODE_COIN 4
#define ENCODE_WIDTH_START 10   // Start of width encoding range
#define ENCODE_HEIGHT_START 20  // Start of height encoding range

// Level Generation Constants
#define CHUNK_WIDTH 800
#define CHUNK_COUNT 3
#define MIN_GROUND_SEGMENT 100
#define MAX_HOLE_WIDTH 120
#define GROUND_SEGMENT_HEIGHT 20

// Structures
typedef struct {
    Vector2 position;
    Vector2 velocity;
    bool isJumping;
    bool isDead;
    float fitness;
    Color color;
    Rectangle bounds;
    NEAT_t *brain;
} Agent;

typedef struct {
    Vector2 position;
    Rectangle bounds;
    Color color;
} Platform;

typedef struct {
    Vector2 position;
    Vector2 velocity;
    Rectangle bounds;
    Color color;
    float direction;
} Enemy;

typedef struct {
    Vector2 position;
    float radius;
    bool collected;
    Color color;
} Coin;

typedef struct {
    Rectangle bounds;
    bool isHole;
} GroundSegment;

typedef struct {
    GroundSegment* segments;
    int segmentCount;
    float startX;
} LevelChunk;

typedef struct {
    LevelChunk chunks[CHUNK_COUNT];
    int currentChunk;
    float worldOffset;
    Enemy* enemies;
    int enemyCount;
    Coin* coins;
    int coinCount;
} GameWorld;

// Forward declarations
void InitGame(void);
void UpdateGame(void);
void DrawGame(void);
void UnloadGame(void);
void CloseGame(void);
void StartNewGeneration(void);
void UpdateGameCamera(void);
float LerpValue(float start, float end, float amount);
int* EncodeScreen(void);

// Function declarations
void GenerateChunk(LevelChunk* chunk, float startX);
void UpdateChunks(void);
void DrawChunks(void);
float NEAT_feedforward(NEAT_t* neat, float* inputs);

// Global variables
static Agent *agents;
static GameWorld gameWorld;
static Camera2D camera = { 0 };
static int generation = 1;
static float bestFitness = 0;
static bool gameStarted = false;
static int bestAgentIndex = 0;
static int frameCount = 0;

void PrintEncodedScreen(void) {
    int* screen = EncodeScreen();
    
    // Clear terminal
    printf("\033[H\033[J");
    
    // Print legend
    printf("ASCII Legend (with size encoding):\n");
    printf("' ' : Empty\n");
    printf("'@' : Agent (size: %d)\n", AGENT_SIZE);
    printf("'#' : Platform (w: %d, h: %d)\n", PLATFORM_WIDTH, PLATFORM_HEIGHT);
    printf("'X' : Enemy (size: %d)\n", ENEMY_SIZE);
    printf("'O' : Coin (size: %d)\n", COIN_SIZE);
    printf("'=' : Ground\n\n");
    
    // Print the screen matrix
    for (int y = 0; y < SCREEN_HEIGHT / 8; y++) {
        for (int x = 0; x < SCREEN_WIDTH / 8; x++) {
            int value = screen[y * (SCREEN_WIDTH / 8) + x];
            char c = ' ';
            
            // Extract base type (removing size information)
            int baseType = value;
            if (value >= ENCODE_HEIGHT_START) {
                baseType = value % ENCODE_HEIGHT_START;
            }
            if (baseType >= ENCODE_WIDTH_START) {
                baseType = baseType % ENCODE_WIDTH_START;
            }
            
            switch(baseType) {
                case ENCODE_EMPTY:
                    c = ' ';
                    break;
                case ENCODE_AGENT:
                    c = '@';
                    break;
                case ENCODE_PLATFORM:
                    c = '#';
                    break;
                case ENCODE_ENEMY:
                    c = 'X';
                    break;
                case ENCODE_COIN:
                    c = 'O';
                    break;
                default:
                    c = ' ';
            }
            printf("%c", c);
        }
        printf("\n");
    }
    
    printf("\nEncoded dimensions: %d x %d\n", SCREEN_WIDTH / 8, SCREEN_HEIGHT / 8);
    printf("\nPress ENTER to start the game...\n");
    free(screen);
}

void GenerateChunk(LevelChunk* chunk, float startX) {
    chunk->startX = startX;
    chunk->segmentCount = PLATFORM_COUNT;
    chunk->segments = (GroundSegment*)malloc(PLATFORM_COUNT * sizeof(GroundSegment));
    
    float platformSpacing = CHUNK_WIDTH / (PLATFORM_COUNT + 1);
    
    for (int i = 0; i < PLATFORM_COUNT; i++) {
        float xPos = startX + platformSpacing * (i + 1) - PLATFORM_WIDTH/2;
        float yPos = SCREEN_HEIGHT - PLATFORM_HEIGHT - 20;  // 20 pixels from bottom
        
        chunk->segments[i].bounds = (Rectangle){
            xPos,
            yPos,
            PLATFORM_WIDTH,
            PLATFORM_HEIGHT
        };
        chunk->segments[i].isHole = false;
        
        // Add coins above platforms (30% chance)
        if (gameWorld.coinCount < COIN_COUNT && (rand() % 3 == 0)) {
            float coinX = xPos + PLATFORM_WIDTH/2;
            gameWorld.coins[gameWorld.coinCount].position = 
                (Vector2){ coinX, yPos - COIN_SIZE - 40 };  // Place coin above platform
            gameWorld.coins[gameWorld.coinCount].radius = COIN_SIZE;
            gameWorld.coins[gameWorld.coinCount].collected = false;
            gameWorld.coins[gameWorld.coinCount].color = COIN_COLOR;
            gameWorld.coinCount++;
        }
        
        // Add enemies on platforms (20% chance)
        if (gameWorld.enemyCount < ENEMY_COUNT && (rand() % 5 == 0)) {
            float enemyX = xPos + PLATFORM_WIDTH/2;
            gameWorld.enemies[gameWorld.enemyCount].position = (Vector2){ enemyX, yPos - ENEMY_SIZE };
            gameWorld.enemies[gameWorld.enemyCount].velocity = (Vector2){ 2, 0 };
            gameWorld.enemies[gameWorld.enemyCount].bounds = 
                (Rectangle){ enemyX, yPos - ENEMY_SIZE, ENEMY_SIZE, ENEMY_SIZE };
            gameWorld.enemies[gameWorld.enemyCount].color = ENEMY_COLOR;
            gameWorld.enemies[gameWorld.enemyCount].direction = 1;
            gameWorld.enemyCount++;
        }
    }
}

void UpdateChunks(void) {
    // Get the rightmost agent position
    float maxX = -INFINITY;
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (!agents[i].isDead && agents[i].position.x > maxX) {
            maxX = agents[i].position.x;
        }
    }
    
    // Check if we need to generate new chunks
    float screenRight = maxX + SCREEN_WIDTH/2;
    if (screenRight > gameWorld.chunks[gameWorld.currentChunk].startX + CHUNK_WIDTH - SCREEN_WIDTH) {
        // Generate next chunk
        int nextChunk = (gameWorld.currentChunk + 1) % CHUNK_COUNT;
        float nextStartX = gameWorld.chunks[gameWorld.currentChunk].startX + CHUNK_WIDTH;
        
        // Free old chunk's segments
        free(gameWorld.chunks[nextChunk].segments);
        
        // Generate new chunk
        GenerateChunk(&gameWorld.chunks[nextChunk], nextStartX);
        gameWorld.currentChunk = nextChunk;
    }
}

void DrawChunks(void) {
    // Draw all chunks
    for (int i = 0; i < CHUNK_COUNT; i++) {
        LevelChunk* chunk = &gameWorld.chunks[i];
        for (int j = 0; j < chunk->segmentCount; j++) {
            Rectangle bounds = chunk->segments[j].bounds;
            DrawRectangleRec(bounds, PLATFORM_COLOR);
        }
    }
}

void UpdateGameCamera(void) {
    // Find the best performing agent
    int bestAgentIndex = 0;
    float bestFitness = -1;
    
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (!agents[i].isDead) {
            float fitness = agents[i].fitness;
            if (fitness > bestFitness) {
                bestFitness = fitness;
                bestAgentIndex = i;
            }
        }
    }
    
    // Update camera to follow best agent
    if (bestFitness > -1) {  // If we found a living agent
        float targetX = agents[bestAgentIndex].position.x - SCREEN_WIDTH/4;  
        camera.target.x = LerpValue(camera.target.x, targetX, 0.1f);
    }
}

// Helper function for linear interpolation
float LerpValue(float start, float end, float amount) {
    return start + amount * (end - start);
}

void CheckCollisions(Agent* agent) {
    bool onGround = false;
    
    // Check platform collision
    for (int c = 0; c < CHUNK_COUNT; c++) {
        LevelChunk* chunk = &gameWorld.chunks[c];
        for (int s = 0; s < chunk->segmentCount; s++) {
            if (CheckCollisionRecs(agent->bounds, chunk->segments[s].bounds)) {
                // Only collide if falling
                if (agent->velocity.y > 0) {
                    agent->position.y = chunk->segments[s].bounds.y - AGENT_SIZE/2;
                    agent->velocity.y = 0;
                    agent->isJumping = false;
                    onGround = true;
                }
            }
        }
    }
    
    // Check enemy collisions
    for (int e = 0; e < gameWorld.enemyCount; e++) {
        if (gameWorld.enemies[e].direction != 0 && 
            CheckCollisionRecs(agent->bounds, gameWorld.enemies[e].bounds)) {
            agent->isDead = true;
        }
    }
    
    // Check coin collisions
    for (int c = 0; c < gameWorld.coinCount; c++) {
        if (!gameWorld.coins[c].collected) {
            if (CheckCollisionCircleRec(
                    gameWorld.coins[c].position,
                    gameWorld.coins[c].radius,
                    agent->bounds)) {
                gameWorld.coins[c].collected = true;
                agent->fitness += COIN_REWARD;
            }
        }
    }
    
    // Check if agent fell off the screen
    if (agent->position.y > SCREEN_HEIGHT + 100) {
        agent->isDead = true;
    }
}

int main(void) {
    // Initialize random seed
    srand(time(NULL));
    
    // Initialize game window
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Balance Game");
    SetTargetFPS(60);

    // Initialize game state
    InitGame();

    // Initialize both visualizers
    InitializeVisualizer();  // Neural network visualizer window
    if (InitializeVisualizerClient() < 0) {
        printf("Warning: Could not connect to visualizer server\n");
    }

    // Main game loop
    while (!WindowShouldClose()) {
        // Update game state
        UpdateGame();
        
        // Draw game window
        BeginDrawing();
        ClearBackground(RAYWHITE);
        DrawGame();
        EndDrawing();

        // Draw neural network visualization for best agent
        if (agents && agents[0].brain && agents[0].brain->nodes && agents[0].brain->nodes[0]) {
            DrawNeuralNetwork(agents[0].brain->nodes[0]->nn);
            
            // Also send to remote visualizer if connected
            if (frameCount % 30 == 0) {  // Update every 30 frames
                SendNetworkToVisualizer(agents[0].brain->nodes[0]->nn);
            }
        }
    }

    // Cleanup
    CloseVisualizerClient();
    CloseVisualizer();
    CloseGame();
    CloseWindow();

    return 0;
}

void InitGame(void) {
    // Initialize camera
    camera.target = (Vector2){ 0.0f, 0.0f };
    camera.offset = (Vector2){ SCREEN_WIDTH/2.0f, SCREEN_HEIGHT/2.0f };
    camera.rotation = 0.0f;
    camera.zoom = 1.0f;

    // Initialize agents
    agents = (Agent*)malloc(POPULATION_SIZE * sizeof(Agent));
    if (agents == NULL) {
        printf("Failed to allocate memory for agents\n");
        exit(1);
    }
    for (int i = 0; i < POPULATION_SIZE; i++) {
        agents[i].position = (Vector2){ 100, SCREEN_HEIGHT - AGENT_SIZE };
        agents[i].velocity = (Vector2){ 0, 0 };
        agents[i].isJumping = false;
        agents[i].isDead = false;
        agents[i].fitness = 0;
        agents[i].color = AGENT_COLOR;
        agents[i].bounds = (Rectangle){ agents[i].position.x, agents[i].position.y, AGENT_SIZE, AGENT_SIZE };
        agents[i].brain = NEAT_init(INPUT_COUNT, OUTPUT_COUNT);  
        if (agents[i].brain == NULL) {
            printf("Failed to initialize brain for agent %d\n", i);
            exit(1);
        }
    }

    // Initialize game world
    gameWorld.enemies = (Enemy*)malloc(ENEMY_COUNT * sizeof(Enemy));
    if (gameWorld.enemies == NULL) {
        printf("Failed to allocate memory for enemies\n");
        exit(1);
    }
    gameWorld.coins = (Coin*)malloc(COIN_COUNT * sizeof(Coin));
    if (gameWorld.coins == NULL) {
        printf("Failed to allocate memory for coins\n");
        exit(1);
    }
    gameWorld.enemyCount = 0;
    gameWorld.coinCount = 0;

    // Initialize level chunks
    for (int i = 0; i < CHUNK_COUNT; i++) {
        GenerateChunk(&gameWorld.chunks[i], i * CHUNK_WIDTH);
    }
    gameWorld.currentChunk = 0;
    
    // Initialize first generation
    StartNewGeneration();
}

void UpdateGame(void) {
    if (!gameStarted) return;

    // Update each agent
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (!agents[i].isDead) {
            // Get inputs for neural network
            float inputs[2];
            inputs[0] = agents[i].velocity.x;
            inputs[1] = agents[i].position.y;

            // Get output from neural network
            float output = NEAT_feedforward(agents[i].brain, inputs);

            // Apply force based on neural network output
            if (output > 0.5f) {
                agents[i].velocity.y = -JUMP_FORCE;
            }

            // Update agent physics
            agents[i].velocity.y += GRAVITY * GetFrameTime();
            agents[i].position.x += agents[i].velocity.x * GetFrameTime();
            agents[i].position.y += agents[i].velocity.y * GetFrameTime();

            // Check collisions
            CheckCollisions(&agents[i]);

            // Send the best performing agent's network to visualizer
            if (i == bestAgentIndex && frameCount % 30 == 0) {  // Update every 30 frames
                if (agents[i].brain && agents[i].brain->nodes && agents[i].brain->nodes[0]) {
                    SendNetworkToVisualizer(agents[i].brain->nodes[0]->nn);
                }
            }

            // Update fitness
            agents[i].fitness = agents[i].position.x;
        }
    }

    // Update camera to follow best performing agent
    UpdateGameCamera();

    // Update level chunks
    UpdateChunks();

    // Check if all agents are dead
    bool allDead = true;
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (!agents[i].isDead) {
            allDead = false;
            break;
        }
    }

    if (allDead) {
        StartNewGeneration();
    }

    frameCount++;
}

void DrawGame(void) {
    ClearBackground(BACKGROUND_COLOR);

    if (!gameStarted) {
        // Draw start screen
        const char* message = "Press ENTER to Start";
        int textWidth = MeasureText(message, 40);
        DrawText(message, SCREEN_WIDTH/2 - textWidth/2, SCREEN_HEIGHT/2, 40, BLACK);
        return;
    }

    BeginMode2D(camera);

    // Draw coins
    for (int i = 0; i < gameWorld.coinCount; i++) {
        if (!gameWorld.coins[i].collected) {
            DrawCircle(gameWorld.coins[i].position.x, gameWorld.coins[i].position.y, 
                      gameWorld.coins[i].radius, COIN_COLOR);
        }
    }

    // Draw enemies
    for (int i = 0; i < gameWorld.enemyCount; i++) {
        if (gameWorld.enemies[i].direction != 0) {
            DrawRectangleRec(gameWorld.enemies[i].bounds, ENEMY_COLOR);
        }
    }

    // Draw agents
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (!agents[i].isDead) {
            DrawRectangleRec(agents[i].bounds, AGENT_COLOR);
        }
    }

    // Draw level chunks
    DrawChunks();

    EndMode2D();

    // Draw UI
    DrawText(TextFormat("Generation: %d", generation), 10, 10, 20, BLACK);
    DrawText(TextFormat("Best Fitness: %.2f", bestFitness), 10, 40, 20, BLACK);
}

void StartNewGeneration(void) {
    generation++;

    // Find best fitness
    float maxFitness = 0;
    int bestAgentIndex = 0;
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (agents[i].fitness > maxFitness) {
            maxFitness = agents[i].fitness;
            bestAgentIndex = i;
        }
    }
    if (maxFitness > bestFitness) bestFitness = maxFitness;

    // Store the best performing agent's brain
    NEAT_t* bestBrain = agents[bestAgentIndex].brain;

    // Reset game world state
    gameWorld.enemyCount = 0;
    gameWorld.coinCount = 0;
    gameWorld.currentChunk = 0;

    // Free and regenerate all chunks
    for (int i = 0; i < CHUNK_COUNT; i++) {
        if (gameWorld.chunks[i].segments != NULL) {
            free(gameWorld.chunks[i].segments);
        }
        GenerateChunk(&gameWorld.chunks[i], i * CHUNK_WIDTH);
    }

    // Reset and evolve agents
    for (int i = 0; i < POPULATION_SIZE; i++) {
        // Reset agent position and state
        agents[i].position = (Vector2){ 100, SCREEN_HEIGHT - AGENT_SIZE - PLATFORM_HEIGHT - 40 };
        agents[i].velocity = (Vector2){ 0, 0 };
        agents[i].isJumping = false;
        agents[i].isDead = false;
        agents[i].fitness = 0;
        agents[i].bounds = (Rectangle){ agents[i].position.x, agents[i].position.y, AGENT_SIZE, AGENT_SIZE };

        if (i != bestAgentIndex) {
            // Free the old brain
            if (agents[i].brain != NULL) {
                NEAT_destroy(agents[i].brain);
            }
            // Create a new brain based on the best brain
            agents[i].brain = NEAT_init(INPUT_COUNT, OUTPUT_COUNT);  // 2 inputs, 1 output
            NEAT_evolve(agents[i].brain);
        }
    }

    // Reset camera to focus on first agent
    camera.target = (Vector2){ agents[0].position.x + 400, agents[0].position.y };
}

int* EncodeScreen(void) {
    int* screen = (int*)calloc(SCREEN_WIDTH / 8 * SCREEN_HEIGHT / 8, sizeof(int));
    if (screen == NULL) {
        printf("Failed to allocate memory for screen encoding\n");
        exit(1);
    }
    // Encode platforms with size information
    for (int c = 0; c < CHUNK_COUNT; c++) {
        LevelChunk* chunk = &gameWorld.chunks[c];
        for (int s = 0; s < chunk->segmentCount; s++) {
            int startX = chunk->segments[s].bounds.x / 8;
            int endX = (chunk->segments[s].bounds.x + chunk->segments[s].bounds.width) / 8;
            int startY = chunk->segments[s].bounds.y / 8;
            int endY = (chunk->segments[s].bounds.y + chunk->segments[s].bounds.height) / 8;
            
            for (int x = startX; x < endX; x++) {
                for (int y = startY; y < endY; y++) {
                    if (x >= 0 && x < SCREEN_WIDTH / 8 && y >= 0 && y < SCREEN_HEIGHT / 8) {
                        screen[y * (SCREEN_WIDTH / 8) + x] = ENCODE_PLATFORM + 
                            ENCODE_WIDTH_START + (int)chunk->segments[s].bounds.width +
                            ENCODE_HEIGHT_START + PLATFORM_HEIGHT;
                    }
                }
            }
        }
    }
    
    // Encode enemies with size information
    for (int i = 0; i < gameWorld.enemyCount; i++) {
        if (gameWorld.enemies[i].direction != 0) {
            int startX = (gameWorld.enemies[i].position.x - ENEMY_SIZE/2) / 8;
            int startY = (gameWorld.enemies[i].position.y - ENEMY_SIZE/2) / 8;
            int endX = (gameWorld.enemies[i].position.x + ENEMY_SIZE/2) / 8;
            int endY = (gameWorld.enemies[i].position.y + ENEMY_SIZE/2) / 8;
            
            for (int x = startX; x < endX; x++) {
                for (int y = startY; y < endY; y++) {
                    if (x >= 0 && x < SCREEN_WIDTH / 8 && y >= 0 && y < SCREEN_HEIGHT / 8) {
                        screen[y * (SCREEN_WIDTH / 8) + x] = ENCODE_ENEMY + 
                            ENCODE_WIDTH_START + ENEMY_SIZE +
                            ENCODE_HEIGHT_START + ENEMY_SIZE;
                    }
                }
            }
        }
    }
    
    // Encode coins with size information
    for (int i = 0; i < gameWorld.coinCount; i++) {
        if (!gameWorld.coins[i].collected) {
            int startX = (gameWorld.coins[i].position.x - COIN_SIZE/2) / 8;
            int startY = (gameWorld.coins[i].position.y - COIN_SIZE/2) / 8;
            int endX = (gameWorld.coins[i].position.x + COIN_SIZE/2) / 8;
            int endY = (gameWorld.coins[i].position.y + COIN_SIZE/2) / 8;
            
            for (int x = startX; x < endX; x++) {
                for (int y = startY; y < endY; y++) {
                    if (x >= 0 && x < SCREEN_WIDTH / 8 && y >= 0 && y < SCREEN_HEIGHT / 8) {
                        screen[y * (SCREEN_WIDTH / 8) + x] = ENCODE_COIN + 
                            ENCODE_WIDTH_START + COIN_SIZE +
                            ENCODE_HEIGHT_START + COIN_SIZE;
                    }
                }
            }
        }
    }
    
    // Encode agents with size information
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (!agents[i].isDead) {
            int startX = (agents[i].position.x - AGENT_SIZE/2) / 8;
            int startY = (agents[i].position.y - AGENT_SIZE/2) / 8;
            int endX = (agents[i].position.x + AGENT_SIZE/2) / 8;
            int endY = (agents[i].position.y + AGENT_SIZE/2) / 8;
            
            for (int x = startX; x < endX; x++) {
                for (int y = startY; y < endY; y++) {
                    if (x >= 0 && x < SCREEN_WIDTH / 8 && y >= 0 && y < SCREEN_HEIGHT / 8) {
                        screen[y * (SCREEN_WIDTH / 8) + x] = ENCODE_AGENT + 
                            ENCODE_WIDTH_START + AGENT_SIZE +
                            ENCODE_HEIGHT_START + AGENT_SIZE;
                    }
                }
            }
        }
    }
    
    return screen;
}

float NEAT_feedforward(NEAT_t* neat, float* inputs) {
    // Convert inputs to long double
    long double longInputs[2] = {(long double)inputs[0], (long double)inputs[1]};
    
    // Forward pass through network
    long double* output = NEAT_forward(neat, longInputs);
    if (output == NULL) {
        printf("Failed to get neural network output\n");
        exit(1);
    }
    float result = (float)output[0];
    
    // Cleanup
    free(output);   
    
    return result; 
}

void CloseGame(void) {
    // Free memory
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (agents[i].brain) {
            NEAT_destroy(agents[i].brain);
        }
    }
    free(agents);
    free(gameWorld.enemies);
    free(gameWorld.coins);
    
    // Close visualization client
    CloseVisualizer();
    
    // Close window
    CloseWindow();
}
