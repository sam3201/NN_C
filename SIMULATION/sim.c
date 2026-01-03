#include "../utils/NN/MEMORY/memory.h"
#include "../utils/NN/MUZE/mu_model.h"
#include "../utils/NN/NEAT.h"
#include "../utils/NN/NN.h"
#include "../utils/Raylib/src/raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define POPULATION_SIZE 20
#define MAX_FOOD 100
#define MAX_GROUNDSKEEPERS (POPULATION_SIZE / 2)
#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define FRAME_RATE 60
#define XP_PER_LEVEL 100
#define XP_FROM_FOOD 1
#define XP_FROM_AGENT 25
#define XP_FROM_OFFSPRING 50
#define BREEDING_DURATION 2.0f
#define INITIAL_AGENT_SIZE 1
#define MOVEMENT_SPEED 2.0f
#define FOOD_SIZE 5
#define FOOD_SPAWN_CHANCE 0.1f
#define LABEL_SIZE 10
#define MEMORY_CAPACITY infinity

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
  Rectangle rect;
  unsigned int size;
  int level;
  int total_xp;
  float time_alive;
  int agent_id;
  int parent_id;
  int num_offsprings;
  int num_eaten;
  bool is_breeding;
  float breeding_timer;
  Color color;
  MuModel *brain;
  Memory memory;
  size_t input_size;
} Agent;

typedef struct {
  Vector2 position;
  Rectangle rect;
} Food;

typedef struct {
  Agent agents[POPULATION_SIZE];
  Food food[MAX_FOOD];
  Action last_actions[POPULATION_SIZE];
  bool over;
  bool paused;
  float evolution_timer;
  unsigned int current_generation;
  long double *vision_inputs;
  int next_agent_id;
  unsigned int num_active_players;
} GameState;

// --- DECLARATIONS ---
void init_agent(Agent *agent, int id);
void init_game(GameState *game);
void update_game(GameState *game);
void encode_vision(GameState *game, int agent_idx, long double *vision_output);
Action get_action_from_output(long double *outputs);
void execute_action(GameState *game, int agent_idx, Action action);
void check_collisions(GameState *game, int agent_idx);
void handle_agent_collision(GameState *game, int agent1_idx, int agent2_idx);
void eat_agent(Agent *predator, Agent *prey);
void kill_agent(GameState *game, int agent_idx);
void start_breeding(Agent *agent1, Agent *agent2);
void handle_breeding(GameState *game, int agent_idx);
void eat_food(Agent *agent);
void transfer_weights_with_mutation(NN_t *old_nn, NN_t *new_nn, int level);
void update_agent_state(GameState *game, int agent_idx);
void level_up(Agent *agent);
float calculate_fitness(Agent *agent);
void evolve_agent(Agent *agent);
void spawn_food(Food *food);
bool can_move_to(GameState *game, Agent *agent, Vector2 new_pos);

// --- MEMORY ---
size_t get_total_input_size() { return (SCREEN_WIDTH * SCREEN_HEIGHT) + 7; }

void store_experience(Agent *agent, long double *inputs, int action,
                      float reward) {
  store_memory(&agent->memory, inputs, action, reward, 0.0f);
}

// --- INIT ---
// --- INIT AGENT ---
void init_agent(Agent *agent, int id) {
  agent->level = 0;
  agent->total_xp = 0;
  agent->size = INITIAL_AGENT_SIZE;
  agent->time_alive = 0;
  agent->agent_id = id;
  agent->parent_id = -1;
  agent->num_offsprings = 0;
  agent->num_eaten = 0;
  agent->is_breeding = false;
  agent->breeding_timer = 0;
  agent->color = WHITE;

  agent->position.x = (float)(rand() % (SCREEN_WIDTH - 10));
  agent->position.y = (float)(rand() % (SCREEN_HEIGHT - 10));
  agent->rect = (Rectangle){agent->position.x, agent->position.y,
                            (float)agent->size, (float)agent->size};

  agent->input_size = get_total_input_size();

  // 1. Initialize MuZero Brain
  MuConfig mu_cfg = {.obs_dim = (int)agent->input_size,
                     .latent_dim = 32, // Starting IQ
                     .action_count = ACTION_COUNT};
  agent->brain = mu_model_create(&mu_cfg);

  // 2. Initialize Infinite Memory
  init_memory(&agent->memory, 100, (int)agent->input_size);
}

// --- UPDATE AGENT (Thinking & Moving) ---
void update_agent_state(GameState *game, int agent_idx) {
  Agent *agent = &game->agents[agent_idx];
  agent->time_alive += GetFrameTime();

  // 1. Vision Setup
  encode_vision(game, agent_idx, game->vision_inputs);

  // Convert vision to float for MUZE
  float obs[agent->input_size];
  for (int i = 0; i < agent->input_size; i++)
    obs[i] = (float)game->vision_inputs[i];

  // 2. MCTS Planning (Thinking before moving)
  MCTSParams mcts_cfg = {.num_simulations = 40,
                         .c_puct = 1.2f,
                         .discount = 0.95f,
                         .temperature = 1.0f};

  MCTSResult res = mcts_run(agent->brain, obs, &mcts_cfg);
  Action action = (Action)res.chosen_action;

  // 3. Act & Remember
  execute_action(game, agent_idx, action);
  game->last_actions[agent_idx] = action;

  // Use current XP as reward signal
  float reward = (float)agent->total_xp;
  store_memory(&agent->memory, game->vision_inputs, (int)action, reward,
               res.root_value);

  mcts_result_free(&res);

  if (agent->is_breeding) {
    agent->breeding_timer += GetFrameTime();
    if (agent->breeding_timer >= BREEDING_DURATION)
      handle_breeding(game, agent_idx);
  }
}

// --- LEVEL UP (Growth) ---
void level_up(Agent *agent) {
  while (agent->total_xp >= (agent->level + 1) * XP_PER_LEVEL) {
    agent->total_xp -= (agent->level + 1) * XP_PER_LEVEL;
    agent->level++;

    // Physical Growth
    agent->size = agent->level + 1;
    agent->rect.width = agent->rect.height = (float)agent->size;

    // Neural Growth: Expand Latent Dimensions
    // We add 8 more "concepts" the agent can imagine per level
    mu_model_grow_latent(agent->brain, agent->brain->cfg.latent_dim + 8);

    printf("Agent %d leveled up to %d! Latent Dims: %d\n", agent->agent_id,
           agent->level, agent->brain->cfg.latent_dim);
  }
}

// --- VISION ---
void encode_vision(GameState *game, int agent_idx, long double *vision_output) {
  for (int i = 0; i < get_total_input_size(); i++)
    vision_output[i] = 0.0L;

  vision_output[0] = 1.0L; // mark self
}

// --- ACTION ---
Action get_action_from_output(long double *outputs) {
  int max_idx = 0;
  long double max_val = outputs[0];
  for (int i = 1; i < ACTION_COUNT; i++) {
    if (outputs[i] > max_val) {
      max_val = outputs[i];
      max_idx = i;
    }
  }
  return (Action)max_idx;
}

bool can_move_to(GameState *game, Agent *agent, Vector2 new_pos) {
  Rectangle new_rect = {new_pos.x, new_pos.y, agent->rect.width,
                        agent->rect.height};
  if (new_pos.x < 0 || new_pos.x + agent->size > SCREEN_WIDTH ||
      new_pos.y < 0 || new_pos.y + agent->size > SCREEN_HEIGHT)
    return false;

  for (int i = 0; i < POPULATION_SIZE; i++) {
    if (&game->agents[i] != agent && game->agents[i].level >= 0) {
      if (CheckCollisionRecs(new_rect, game->agents[i].rect))
        return false;
    }
  }
  return true;
}

void move_agent(GameState *game, Agent *agent, Action action) {
  Vector2 new_pos = agent->position;
  switch (action) {
  case ACTION_MOVE_LEFT:
    new_pos.x -= MOVEMENT_SPEED;
    break;
  case ACTION_MOVE_RIGHT:
    new_pos.x += MOVEMENT_SPEED;
    break;
  case ACTION_MOVE_UP:
    new_pos.y -= MOVEMENT_SPEED;
    break;
  case ACTION_MOVE_DOWN:
    new_pos.y += MOVEMENT_SPEED;
    break;
  default:
    break;
  }
  if (can_move_to(game, agent, new_pos)) {
    agent->position = new_pos;
    agent->rect.x = new_pos.x;
    agent->rect.y = new_pos.y;
  }
}

void execute_action(GameState *game, int agent_idx, Action action) {
  Agent *agent = &game->agents[agent_idx];
  if (action != ACTION_NONE) {
    move_agent(game, agent, action);
    check_collisions(game, agent_idx);
  }
}

// --- GAME UPDATE ---
void update_agent_state(GameState *game, int agent_idx) {
  Agent *agent = &game->agents[agent_idx];
  agent->time_alive += GetFrameTime();

  encode_vision(game, agent_idx, game->vision_inputs);
  long double *outputs = NEAT_forward(agent->brain, game->vision_inputs);
  Action action = get_action_from_output(outputs);
  execute_action(game, agent_idx, action);
  game->last_actions[agent_idx] = action;

  float reward = (float)agent->total_xp;
  store_experience(agent, game->vision_inputs, (int)action, reward);

  if (agent->is_breeding) {
    agent->breeding_timer += GetFrameTime();
    if (agent->breeding_timer >= BREEDING_DURATION)
      handle_breeding(game, agent_idx);
  }
}

void update_game(GameState *game) {
  for (int i = 0; i < MAX_FOOD; i++) {
    if (game->food[i].rect.width == 0 &&
        ((float)rand() / RAND_MAX) < FOOD_SPAWN_CHANCE)
      spawn_food(&game->food[i]);
  }

  for (int i = 0; i < POPULATION_SIZE; i++) {
    if (game->agents[i].level >= 0)
      update_agent_state(game, i);
  }
}

// --- FOOD ---
void spawn_food(Food *food) {
  food->position.x = (float)(rand() % (SCREEN_WIDTH - FOOD_SIZE));
  food->position.y = (float)(rand() % (SCREEN_HEIGHT - FOOD_SIZE));
  food->rect =
      (Rectangle){food->position.x, food->position.y, FOOD_SIZE, FOOD_SIZE};
}

void eat_food(Agent *agent) {
  agent->total_xp += XP_FROM_FOOD;
  if (agent->total_xp >= (agent->level + 1) * XP_PER_LEVEL)
    level_up(agent);
}

// --- COLLISIONS ---
void check_collisions(GameState *game, int agent_idx) {
  Agent *agent = &game->agents[agent_idx];
  for (int i = 0; i < MAX_FOOD; i++) {
    if (CheckCollisionRecs(agent->rect, game->food[i].rect)) {
      eat_food(agent);
      game->food[i].rect.width = 0;
    }
  }

  for (int i = 0; i < POPULATION_SIZE; i++) {
    if (i != agent_idx && game->agents[i].level >= 0 &&
        CheckCollisionRecs(agent->rect, game->agents[i].rect))
      handle_agent_collision(game, agent_idx, i);
  }
}

// --- AGENT INTERACTIONS ---
void handle_agent_collision(GameState *game, int agent1_idx, int agent2_idx) {
  Agent *a1 = &game->agents[agent1_idx], *a2 = &game->agents[agent2_idx];
  if (a1->size > a2->size) {
    eat_agent(a1, a2);
    kill_agent(game, agent2_idx);
  } else if (a2->size > a1->size) {
    eat_agent(a2, a1);
    kill_agent(game, agent1_idx);
  } else if (!a1->is_breeding && !a2->is_breeding)
    start_breeding(a1, a2);
}

void eat_agent(Agent *predator, Agent *prey) {
  predator->total_xp += XP_FROM_AGENT;
  predator->num_eaten++;
  if (predator->total_xp >= (predator->level + 1) * XP_PER_LEVEL)
    level_up(predator);
}

void kill_agent(GameState *game, int agent_idx) {
  game->agents[agent_idx].level = -1;
  game->agents[agent_idx].total_xp = 0;
  game->num_active_players--;
}

// --- BREEDING ---
void start_breeding(Agent *agent1, Agent *agent2) {
  agent1->is_breeding = agent2->is_breeding = true;
  agent1->breeding_timer = agent2->breeding_timer = 0;
}

void handle_breeding(GameState *game, int agent_idx) {
  Agent *parent = &game->agents[agent_idx];
  parent->is_breeding = false;
  parent->breeding_timer = 0;
  parent->num_offsprings++;

  for (int i = 0; i < POPULATION_SIZE; i++) {
    if (game->agents[i].level < 0) {
      init_agent(&game->agents[i], game->next_agent_id++);
      game->agents[i].parent_id = parent->agent_id;
      game->num_active_players++;
      break;
    }
  }
}

// --- LEVELING ---
void level_up(Agent *agent) {
  while (agent->total_xp >= (agent->level + 1) * XP_PER_LEVEL) {
    agent->total_xp -= (agent->level + 1) * XP_PER_LEVEL;
    agent->level++;
    agent->size = agent->level + 1;
    agent->rect.width = agent->rect.height = agent->size;
    evolve_agent(agent);
    mu_model_grow_latent(agent->mu_brain, agent->mu_brain->cfg.latent_dim + 1);
  }
}

// --- EVOLUTION ---
float calculate_fitness(Agent *agent) {
  return (float)(agent->total_xp + agent->num_offsprings + agent->num_eaten +
                 agent->level);
}

void evolve_agent(Agent *agent) {
  if (!agent->brain || !agent->brain->nodes[0])
    return;
  Perceptron_t *old_brain = agent->brain->nodes[0];
  NEAT_destroy(agent->brain); // Destroy old brain
  agent->brain = NEAT_init(get_total_input_size(), ACTION_COUNT, 1);
  agent->brain->nodes[0] = old_brain; // Reattach old perceptron
}

// --- GAME INIT ---
void init_game(GameState *game) {
  game->current_generation = 0;
  game->num_active_players = POPULATION_SIZE;
  game->next_agent_id = 0;
  game->paused = false;
  game->vision_inputs = malloc(get_total_input_size() * sizeof(long double));
  for (int i = 0; i < POPULATION_SIZE; i++)
    init_agent(&game->agents[i], game->next_agent_id++);
  for (int i = 0; i < MAX_FOOD; i++)
    spawn_food(&game->food[i]);
}

// --- MAIN ---
int main(void) {
  InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Evolution Simulator");
  SetTargetFPS(FRAME_RATE);
  srand(time(NULL));

  GameState game = {0};
  init_game(&game);

  system("clear");

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_SPACE))
      game.paused = !game.paused;
    if (!game.paused)
      update_game(&game);

    BeginDrawing();
    ClearBackground(BLACK);

    for (int i = 0; i < MAX_FOOD; i++)
      DrawRectangleRec(game.food[i].rect, GREEN);

    for (int i = 0; i < POPULATION_SIZE; i++) {
      Agent *a = &game.agents[i];
      if (a->level >= 0) {
        DrawRectangleRec(a->rect, a->color);
        if (a->is_breeding)
          DrawRectangleLinesEx(a->rect, 2, PINK);
        const char *label = (i < MAX_GROUNDSKEEPERS) ? "GK" : "AG";
        DrawText(label, a->rect.x, a->rect.y - LABEL_SIZE - 2, LABEL_SIZE,
                 (i < MAX_GROUNDSKEEPERS) ? RED : WHITE);
        DrawText(TextFormat("%d", a->level), a->rect.x + a->rect.width + 2,
                 a->rect.y, LABEL_SIZE, WHITE);
      }
    }

    DrawText(TextFormat("Generation: %d", game.current_generation), 10, 10, 20,
             WHITE);
    DrawText(TextFormat("Active Agents: %d", game.num_active_players), 10, 35,
             20, WHITE);
    if (game.paused)
      DrawText("PAUSED", SCREEN_WIDTH / 2 - 50, 10, 20, WHITE);

    EndDrawing();
  }

  free(game.vision_inputs);
  CloseWindow();
  return 0;
}
