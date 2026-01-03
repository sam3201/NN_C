#include "../utils/NN/MEMORY/memory.h"
#include "../utils/NN/MUZE/all.h"
#include "../utils/NN/NEAT.h"
#include "../utils/NN/NN.h"
#include "../utils/Raylib/src/raylib.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define POPULATION_SIZE 10
#define MAX_FOOD 100
#define MAX_GROUNDSKEEPERS 3
#define GROUNDSKEEPER_SPEED 3.0f
#define MOVEMENT_SPEED 2.0f
#define XP_LEECH_RATE 1.0f
#define PUNISHMENT_COOLDOWN 3.0f
#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define FRAME_RATE 60
#define XP_PER_LEVEL 100
#define XP_FROM_FOOD 1
#define XP_FROM_AGENT 25
#define XP_FROM_OFFSPRING 50
#define BREEDING_DURATION 2.0f
#define INITIAL_AGENT_SIZE 1
#define FOOD_SIZE 5
#define FOOD_SPAWN_CHANCE 0.1f
#define LABEL_SIZE 10

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
  float punishment_timer;
  Color color;
} Groundkeeper;

typedef struct {
  Vector2 position;
  Rectangle rect;
} Food;

typedef struct {
  Agent agents[POPULATION_SIZE - MAX_GROUNDSKEEPERS];
  Groundkeeper gks[MAX_GROUNDSKEEPERS];
  Food food[MAX_FOOD];
  Action last_actions[POPULATION_SIZE];
  bool over;
  bool paused;
  float evolution_timer;
  unsigned int current_generation;
  long double *vision_inputs;
  int next_agent_id;
  unsigned int num_active_agents;
} GameState;

// --- MEMORY ---
size_t get_total_input_size() {
  return (SCREEN_WIDTH * SCREEN_HEIGHT) + 7 +
         4; // 4 extra: time_alive, punishment_timer, xp_stolen, relative_size
}

void store_experience(Agent *agent, long double *inputs, int action,
                      float reward) {
  store_memory(&agent->memory, inputs, action, reward, 0.0f);
}

// --- AGENT FUNCTIONS ---
void update_agent_color(Agent *agent) {
  int red = (int)((agent->total_xp % XP_PER_LEVEL) / (float)XP_PER_LEVEL * 255);
  int green =
      (int)((agent->total_xp / (float)((agent->level + 1) * XP_PER_LEVEL)) *
            255);
  int blue = fmin(agent->size * 50, 255); // scale for visibility
  agent->color = (Color){red, green, blue, 255};
}

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
  agent->position.x = (float)(rand() % (SCREEN_WIDTH - 10));
  agent->position.y = (float)(rand() % (SCREEN_HEIGHT - 10));
  agent->rect = (Rectangle){agent->position.x, agent->position.y,
                            (float)agent->size, (float)agent->size};
  agent->input_size = get_total_input_size();
  update_agent_color(agent);

  MuConfig cfg = {.obs_dim = (int)agent->input_size,
                  .latent_dim = 32,
                  .action_count = ACTION_COUNT};
  agent->brain = mu_model_create(&cfg);
  init_memory(&agent->memory, 100, (int)agent->input_size);
}

void encode_vision(GameState *game, int agent_idx, long double *vision_output) {
  for (int i = 0; i < get_total_input_size(); i++)
    vision_output[i] = 0.0L;
  vision_output[0] = 1.0L; // self

  Agent *agent = &game->agents[agent_idx];
  vision_output[1] = agent->time_alive; // additional input
  // further vision encoding: food, offspring, other agents, GKs etc.
  // Simplified for example, can be expanded with grid scanning
}

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

bool can_move_to_agent(GameState *game, Agent *agent, Vector2 new_pos) {
  Rectangle new_rect = {new_pos.x, new_pos.y, agent->rect.width,
                        agent->rect.height};
  if (new_pos.x < 0 || new_pos.x + agent->size > SCREEN_WIDTH ||
      new_pos.y < 0 || new_pos.y + agent->size > SCREEN_HEIGHT)
    return false;

  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++) {
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
  if (can_move_to_agent(game, agent, new_pos)) {
    agent->position = new_pos;
    agent->rect.x = new_pos.x;
    agent->rect.y = new_pos.y;
  }
}

void eat_food(Agent *agent) {
  agent->total_xp += XP_FROM_FOOD;
  if (agent->total_xp >= (agent->level + 1) * XP_PER_LEVEL) {
    agent->total_xp -= (agent->level + 1) * XP_PER_LEVEL;
    agent->level++;
    agent->size = agent->level + 1;
    agent->rect.width = agent->rect.height = (float)agent->size;
  }
  update_agent_color(agent);
}

// --- AGENT EATS OTHER AGENT ---
void eat_agent(Agent *predator, Agent *prey) {
  if (prey->level < 0)
    return;                             // already dead
  predator->total_xp += prey->total_xp; // gain all XP
  predator->num_eaten += 1;

  // Reset prey
  prey->total_xp = 0;
  prey->level = 0;
  prey->size = INITIAL_AGENT_SIZE;
  prey->rect.width = prey->rect.height = (float)prey->size;
  prey->position.x = (float)(rand() % (SCREEN_WIDTH - 10));
  prey->position.y = (float)(rand() % (SCREEN_HEIGHT - 10));
  prey->time_alive = 0;
  prey->num_offsprings = 0;
  prey->num_eaten = 0;
  prey->is_breeding = false;
  update_agent_color(prey);

  // Update predator color/level
  if (predator->total_xp >= (predator->level + 1) * XP_PER_LEVEL) {
    predator->total_xp -= (predator->level + 1) * XP_PER_LEVEL;
    predator->level++;
    predator->size = predator->level + 1;
    predator->rect.width = predator->rect.height = (float)predator->size;
  }
  update_agent_color(predator);
}

void execute_agent_action(GameState *game, int agent_idx, Action action) {
  Agent *agent = &game->agents[agent_idx];
  move_agent(game, agent, action);
}

void update_agent(GameState *game, int agent_idx) {
  Agent *agent = &game->agents[agent_idx];
  agent->time_alive += GetFrameTime();

  encode_vision(game, agent_idx, game->vision_inputs);
  float obs[agent->input_size];
  for (int i = 0; i < agent->input_size; i++)
    obs[i] = (float)game->vision_inputs[i];

  MCTSParams cfg = {.num_simulations = 40,
                    .c_puct = 1.2f,
                    .discount = 0.95f,
                    .temperature = 1.0f};
  MCTSResult res = mcts_run(agent->brain, obs, &cfg);
  Action action = (Action)res.chosen_action;

  execute_agent_action(game, agent_idx, action);
  store_experience(agent, game->vision_inputs, (int)action, agent->total_xp);
  mcts_result_free(&res);
}

// --- GROUNDSKEEPER FUNCTIONS ---
void init_groundkeeper(Groundkeeper *gk) {
  gk->punishment_timer = 0;
  gk->color = RED;
  gk->position.x = (float)(rand() % (SCREEN_WIDTH - 10));
  gk->position.y = (float)(rand() % (SCREEN_HEIGHT - 10));
  gk->rect = (Rectangle){gk->position.x, gk->position.y, INITIAL_AGENT_SIZE,
                         INITIAL_AGENT_SIZE};
}

bool can_move_to_gk(GameState *game, Groundkeeper *gk, Vector2 new_pos) {
  Rectangle new_rect = {new_pos.x, new_pos.y, gk->rect.width, gk->rect.height};
  if (new_pos.x < 0 || new_pos.x + gk->rect.width > SCREEN_WIDTH ||
      new_pos.y < 0 || new_pos.y + gk->rect.height > SCREEN_HEIGHT)
    return false;
  return true;
}

void update_groundkeeper(GameState *game, int idx) {
  Groundkeeper *gk = &game->gks[idx];
  if (gk->punishment_timer > 0)
    gk->punishment_timer -= GetFrameTime();

  Vector2 new_pos = gk->position;
  switch (rand() % 4) {
  case 0:
    new_pos.x -= GROUNDSKEEPER_SPEED;
    break;
  case 1:
    new_pos.x += GROUNDSKEEPER_SPEED;
    break;
  case 2:
    new_pos.y -= GROUNDSKEEPER_SPEED;
    break;
  case 3:
    new_pos.y += GROUNDSKEEPER_SPEED;
    break;
  }

  if (can_move_to_gk(game, gk, new_pos)) {
    gk->position = new_pos;
    gk->rect.x = new_pos.x;
    gk->rect.y = new_pos.y;
  }

  // Leech XP from agents
  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++) {
    Agent *a = &game->agents[i];
    if (CheckCollisionRecs(a->rect, gk->rect)) {
      float leech = XP_LEECH_RATE * GetFrameTime();
      a->total_xp -= (int)leech;
      if (gk->punishment_timer <= 0) {
        a->level = fmax(a->level - 1, 0);
        gk->punishment_timer = PUNISHMENT_COOLDOWN;
      }
    }
  }
}

// --- FOOD ---
void spawn_food(Food *food) {
  food->position.x = (float)(rand() % (SCREEN_WIDTH - FOOD_SIZE));
  food->position.y = (float)(rand() % (SCREEN_HEIGHT - FOOD_SIZE));
  food->rect =
      (Rectangle){food->position.x, food->position.y, FOOD_SIZE, FOOD_SIZE};
}

// --- GAME ---
void update_game(GameState *game) {
  for (int i = 0; i < MAX_FOOD; i++)
    if (game->food[i].rect.width == 0 &&
        ((float)rand() / RAND_MAX) < FOOD_SPAWN_CHANCE)
      spawn_food(&game->food[i]);

  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++)
    update_agent(game, i);

  for (int i = 0; i < MAX_GROUNDSKEEPERS; i++)
    update_groundkeeper(game, i);
}

void init_game(GameState *game) {
  game->current_generation = 0;
  game->num_active_agents = POPULATION_SIZE - MAX_GROUNDSKEEPERS;
  game->next_agent_id = 0;
  game->paused = false;
  game->vision_inputs = malloc(get_total_input_size() * sizeof(long double));

  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++)
    init_agent(&game->agents[i], game->next_agent_id++);

  for (int i = 0; i < MAX_GROUNDSKEEPERS; i++)
    init_groundkeeper(&game->gks[i]);

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

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_SPACE))
      game.paused = !game.paused;
    if (!game.paused)
      update_game(&game);

    BeginDrawing();
    ClearBackground(BLACK);

    for (int i = 0; i < MAX_FOOD; i++)
      DrawRectangleRec(game.food[i].rect, GREEN);

    for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++) {
      Agent *a = &game.agents[i];
      DrawRectangleRec(a->rect, a->color);
      if (a->is_breeding)
        DrawRectangleLinesEx(a->rect, 2, PINK);
      DrawText("AG", a->rect.x, a->rect.y - LABEL_SIZE - 2, LABEL_SIZE, WHITE);
    }

    for (int i = 0; i < MAX_GROUNDSKEEPERS; i++) {
      Groundkeeper *gk = &game.gks[i];
      DrawRectangleRec(gk->rect, gk->color);
      DrawText("GK", gk->rect.x, gk->rect.y - LABEL_SIZE - 2, LABEL_SIZE, RED);
    }

    DrawText(TextFormat("Generation: %d", game.current_generation), 10, 10, 20,
             WHITE);
    DrawText(TextFormat("Active Agents: %d", game.num_active_agents), 10, 35,
             20, WHITE);
    if (game.paused)
      DrawText("PAUSED", SCREEN_WIDTH / 2 - 50, 10, 20, WHITE);

    EndDrawing();
  }

  free(game.vision_inputs);
  CloseWindow();
  return 0;
}
