#include "../utils/NN/MEMORY/memory.h"
#include "../utils/NN/MUZE/all.h"
#include "../utils/NN/NEAT.h"
#include "../utils/NN/NN.h"
#include "../utils/Raylib/src/raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define FRAME_RATE 60

#define MEMORY_CAPACITY infinity

#define CHUNK_SIZE 10
#define MAX_RESOURCES 10
#define MAX_MOBS 10

typedef struct {
  Vector2 position;
  float value;
  int type;
  bool visited;
} Resource;

typedef struct {
  Vector2 position;
  float value;
  int type;
  bool visited;
} Mob;

typedef struct {
  int biome_type;
  Vector3 terrain[CHUNK_SIZE][CHUNK_SIZE];
  Resource resources[MAX_RESOURCES];
  Mob mobs[MAX_MOBS];
  bool visited;
} Chunk;

typedef enum {
  ACTION_NONE = 0,
  ACTION_MOVE_LEFT,
  ACTION_MOVE_RIGHT,
  ACTION_MOVE_UP,
  ACTION_MOVE_DOWN,
  ACTION_COUNT
} Action;

typedef struct {
  Vector2 position;
  unsigned int radius;
  unsigned int size;
  float time_alive;
  int agent_id;
  float breeding_timer;
  Color color;
  NEAT_t *brain;
  Memory memory;
  size_t input_size;
} Agent;

// ---------------- MuZero AI -----------------
MuModel *ai_model = NULL;

void init_ai() {
  MuConfig cfg = {.obs_dim = 4, .latent_dim = 16, .action_count = 5};
  ai_model = mu_model_create(&cfg);
}

void free_ai() {
  if (ai_model) {
    mu_model_free(ai_model);
    ai_model = NULL;
  }
}

int ai_choose_action(Agent *ai) {
  if (!ai_model)
    return 0;

  float obs[4] = {ai->position.x, ai->position.y, player->position.x,
                  player->position.y};

  MCTSParams params = {.num_simulations = 25,
                       .c_puct = 1.0f,
                       .max_depth = 10,
                       .dirichlet_alpha = 0.3f,
                       .dirichlet_eps = 0.25f,
                       .temperature = 1.0f,
                       .discount = 0.99f};

  MCTSResult res = mcts_run(ai_model, obs, &params);
  return res.chosen_action;
}

float calculate_fitness(Agent *agent) {
  return (agent->total_xp + agent->num_offsprings * XP_FROM_OFFSPRING +
          agent->num_eaten * XP_FROM_AGENT + agent->exploration_score);
}

void title_screen(int *topAI, int *bottomAI) {
  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);
    DrawText("Tank NN_C Testbed", 220, 50, 40, LIGHTGRAY);
    DrawText("Top Tank:", 100, 200, 20, LIGHTGRAY);
    DrawText(*topAI ? "AI" : "PLAYER", 250, 200, 20, *topAI ? RED : GREEN);
    DrawText("Press 1 to toggle", 400, 200, 20, GRAY);
    DrawText("Bottom Tank:", 100, 300, 20, LIGHTGRAY);
    DrawText(*bottomAI ? "AI" : "PLAYER", 250, 300, 20,
             *bottomAI ? RED : GREEN);
    DrawText("Press 2 to toggle", 400, 300, 20, GRAY);
    DrawText("Press ENTER to start", 200, 450, 30, YELLOW);
    EndDrawing();

    if (IsKeyPressed(KEY_ONE))
      *topAI = !(*topAI);
    if (IsKeyPressed(KEY_TWO))
      *bottomAI = !(*bottomAI);
    if (IsKeyPressed(KEY_ENTER))
      break;
  }
}

// ---------------- Main -----------------
int main() {
  InitWindow(800, 600, "Tank Game");
  SetTargetFPS(60);

  title_screen();

initialize_world()
initialize_agents()
while simulation_running:
    for each agent:
        encode_state()
        predicted_action = MuZero_forward(state)
        execute_action(action)
        observe_next_state_and_reward()
        store_transition(state, action, reward)
    update_world()
    render_world()
    if generation_end:
        evaluate_fitness()
        evolve_agents()
        optionally scale world difficulty

  free_ai();
CloseWindow();
return 0;
}
