#include "../utils/NN/MEMORY/memory.h"
#include "../utils/NN/MUZE/all.h"
#include "../utils/NN/NEAT.h"
#include "../utils/NN/NN.h"
#include "../utils/Raylib/src/raylib.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

#define LATENT_MAX 64

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
  float latent[LATENT_MAX];
  bool has_latent;
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
  size_t size = 1;                                        // self
  size += 1;                                              // time_alive
  size += MAX_GROUNDSKEEPERS;                             // punishment timers
  size += 1;                                              // xp_stolen
  size += 1;                                              // relative size
  size += MAX_FOOD;                                       // food presence
  size += (POPULATION_SIZE - MAX_GROUNDSKEEPERS - 1) * 4; // other agents
  size += MAX_GROUNDSKEEPERS * 3;                         // groundkeepers
  return size;
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
  agent->has_latent = false;
}

// --- VISION ENCODING ---
/*
void encode_vision(GameState *game, int agent_idx, long double *vision_output) {
  Agent *self = &game->agents[agent_idx];

  // Clear vision
  for (int i = 0; i < get_total_input_size(); i++)
    vision_output[i] = 0.0L;

  int idx = 0;

  // Self indicator
  vision_output[idx++] = 1.0L;

  // Time alive (normalized)
  vision_output[idx++] = self->time_alive / 100.0L; // adjust scaling as needed

  // Punishment timer of nearby gks (max of 3)
  for (int i = 0; i < MAX_GROUNDSKEEPERS; i++) {
    vision_output[idx++] = game->gks[i].punishment_timer / PUNISHMENT_COOLDOWN;
  }

  // XP stolen recently (sum of XP leeched by GKs)
  long double xp_stolen = 0.0L;
  for (int i = 0; i < MAX_GROUNDSKEEPERS; i++) {
    if (CheckCollisionRecs(self->rect, game->gks[i].rect)) {
      xp_stolen += XP_LEECH_RATE * GetFrameTime();
    }
  }
  vision_output[idx++] = xp_stolen;

  // Relative size (normalized)
  vision_output[idx++] =
      (long double)self->size / 10.0L; // assuming max 10 size

  // Vision grid for food
  for (int i = 0; i < MAX_FOOD; i++) {
    Food *f = &game->food[i];
    vision_output[idx++] = (f->rect.width > 0) ? 1.0L : 0.0L; // food present
    // optionally could encode distance or direction here
  }

  // Vision of other agents
  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++) {
    if (i == agent_idx)
      continue;
    Agent *other = &game->agents[i];
    vision_output[idx++] =
        (long double)other->level / 10.0L;                   // normalized level
    vision_output[idx++] = (long double)other->size / 10.0L; // normalized size
    vision_output[idx++] =
        CheckCollisionRecs(self->rect, other->rect) ? 1.0L : 0.0L; // touching
    vision_output[idx++] = other->is_breeding ? 1.0L : 0.0L;
  }

  // Vision of groundkeepers
  for (int i = 0; i < MAX_GROUNDSKEEPERS; i++) {
    Groundkeeper *gk = &game->gks[i];
    vision_output[idx++] = (gk->rect.width > 0) ? 1.0L : 0.0L;         // exists
    vision_output[idx++] = gk->punishment_timer / PUNISHMENT_COOLDOWN; // timer
    vision_output[idx++] =
        CheckCollisionRecs(self->rect, gk->rect) ? 1.0L : 0.0L; // touching
  }
}
*/

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
  mu_model_end_episode(prey->brain, -5.0f);
  prey->has_latent = false;
  mu_model_reset_episode(prey->brain);
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

// --- AGENT BREEDING ---
bool can_breed(Agent *a1, Agent *a2) {
  return !a1->is_breeding && !a2->is_breeding &&
         CheckCollisionRecs(a1->rect, a2->rect);
}

Agent spawn_offspring(Agent *parent1, Agent *parent2, int new_id) {
  Agent child;
  child.level = 0;
  child.total_xp = XP_FROM_OFFSPRING;
  child.size = INITIAL_AGENT_SIZE;
  child.time_alive = 0;
  child.agent_id = new_id;
  child.parent_id = parent1->agent_id;
  child.num_offsprings = 0;
  child.num_eaten = 0;
  child.is_breeding = false;
  child.breeding_timer = 0;
  child.position.x = (parent1->position.x + parent2->position.x) / 2;
  child.position.y = (parent1->position.y + parent2->position.y) / 2;
  child.rect = (Rectangle){child.position.x, child.position.y,
                           (float)child.size, (float)child.size};
  child.input_size = get_total_input_size();
  update_agent_color(&child);
  child.has_latent = false;

  MuConfig cfg = {.obs_dim = (int)child.input_size,
                  .latent_dim = 32,
                  .action_count = ACTION_COUNT};
  child.brain = mu_model_create(&cfg);
  init_memory(&child.memory, 100, (int)child.input_size);

  parent1->num_offsprings++;
  parent2->num_offsprings++;

  return child;
}

// Call this inside update_agent or update_game to handle breeding
void handle_breeding(GameState *game) {
  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++) {
    for (int j = i + 1; j < POPULATION_SIZE - MAX_GROUNDSKEEPERS; j++) {
      Agent *a1 = &game->agents[i];
      Agent *a2 = &game->agents[j];
      if (can_breed(a1, a2)) {
        a1->is_breeding = a2->is_breeding = true;
        a1->breeding_timer = a2->breeding_timer = BREEDING_DURATION;

        // Replace first inactive agent with offspring
        for (int k = 0; k < POPULATION_SIZE - MAX_GROUNDSKEEPERS; k++) {
          if (game->agents[k].level == 0 && game->agents[k].time_alive == 0) {
            game->agents[k] = spawn_offspring(a1, a2, game->next_agent_id++);
            break;
          }
        }
      }
    }
  }
}
// --- EVOLUTION CHECK ---
void check_evolution(GameState *game) {
  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++) {
    Agent *a = &game->agents[i];
    if (a->total_xp >= (a->level + 1) * XP_PER_LEVEL) {
      a->level++;
      a->size = a->level + 1;
      a->rect.width = a->rect.height = (float)a->size;
      update_agent_color(a);
    }
  }
  // Groundskeeper evolution (simplified example)
  for (int i = 0; i < MAX_GROUNDSKEEPERS; i++) {
    Groundkeeper *gk = &game->gks[i];
    if (gk->punishment_timer == 0) {
      // Could increase speed, leech rate, or other parameters
      gk->punishment_timer = PUNISHMENT_COOLDOWN;
    }
  }
}

void execute_agent_action(GameState *game, int agent_idx, Action action) {
  Agent *agent = &game->agents[agent_idx];
  move_agent(game, agent, action);
}

void gather_agent_inputs(GameState *state, Agent *agent, long double *inputs) {
  int idx = 0;

  // Self
  inputs[idx++] = agent->size;
  inputs[idx++] = agent->time_alive;

  // Punishment timers of all groundkeepers
  for (int i = 0; i < MAX_GROUNDSKEEPERS; i++)
    inputs[idx++] = state->gks[i].punishment_timer;

  // XP stolen (placeholder: could be computed dynamically)
  inputs[idx++] = 0;

  // Relative size (normalized)
  inputs[idx++] = agent->size / 10.0;

  // Food presence (binary)
  for (int i = 0; i < MAX_FOOD; i++)
    inputs[idx++] = (state->food[i].rect.width > 0) ? 1.0L : 0.0L;

  // Other agents
  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++) {
    if (state->agents[i].agent_id == agent->agent_id)
      continue;
    inputs[idx++] = state->agents[i].position.x - agent->position.x;
    inputs[idx++] = state->agents[i].position.y - agent->position.y;
    inputs[idx++] = state->agents[i].size;
    inputs[idx++] = state->agents[i].total_xp;
  }

  // Groundkeepers
  for (int i = 0; i < MAX_GROUNDSKEEPERS; i++) {
    inputs[idx++] = state->gks[i].position.x - agent->position.x;
    inputs[idx++] = state->gks[i].position.y - agent->position.y;
    inputs[idx++] = state->gks[i].punishment_timer;
  }
}

float compute_reward(Agent *a, int old_xp, int old_level) {
  float r = 0.0f;

  if (a->total_xp > old_xp)
    r += 0.1f * (a->total_xp - old_xp);

  if (a->level > old_level)
    r += 1.0f;

  if (a->num_eaten > 0)
    r += 2.0f;

  if (a->total_xp < 0)
    r -= 1.0f;

  return r;
}

int decide_action(Agent *agent, long double *inputs) {
  MuModel *brain = agent->brain;

  int obs_dim = brain->cfg.obs_dim;
  float obs[obs_dim];
  for (int i = 0; i < obs_dim; i++)
    obs[i] = (float)inputs[i];

  if (!agent->has_latent) {
    mu_model_repr(brain, obs, agent->latent);
    agent->has_latent = true;
  }

  MCTSParams mcts = {.num_simulations = 25,
                     .c_puct = 1.2f,
                     .discount = 0.99f,
                     .temperature = 1.0f};

  MCTSResult res = mcts_run_latent(brain, agent->latent, &mcts);
  int action = res.chosen_action;
  mcts_result_free(&res);

  return action;
}

void update_latent_after_step(Agent *agent, long double *obs, int action,
                              float reward) {
  float obs_f[agent->input_size];
  for (int i = 0; i < agent->input_size; i++)
    obs_f[i] = (float)obs[i];

  float next_latent[LATENT_MAX];
  float predicted_reward;

  mu_model_dynamics(agent->brain, agent->latent, action, next_latent,
                    &predicted_reward);

  int L = agent->brain->cfg.latent_dim;
  memcpy(agent->latent, next_latent, sizeof(float) * L);
}

void step_agent(GameState *state, Agent *agent, int action) {
  Vector2 old_pos = agent->position;

  switch (action) {
  case ACTION_MOVE_LEFT:
    agent->position.x -= MOVEMENT_SPEED;
    break;
  case ACTION_MOVE_RIGHT:
    agent->position.x += MOVEMENT_SPEED;
    break;
  case ACTION_MOVE_UP:
    agent->position.y -= MOVEMENT_SPEED;
    break;
  case ACTION_MOVE_DOWN:
    agent->position.y += MOVEMENT_SPEED;
    break;
  default:
    break;
  }

  // Clamp to screen
  if (agent->position.x < 0)
    agent->position.x = 0;
  if (agent->position.x > SCREEN_WIDTH)
    agent->position.x = SCREEN_WIDTH;
  if (agent->position.y < 0)
    agent->position.y = 0;
  if (agent->position.y > SCREEN_HEIGHT)
    agent->position.y = SCREEN_HEIGHT;

  agent->rect.x = agent->position.x;
  agent->rect.y = agent->position.y;

  // Food collection
  for (int i = 0; i < MAX_FOOD; i++) {
    if (CheckCollisionRecs(agent->rect, state->food[i].rect)) {
      eat_food(agent);
      agent->num_eaten++;

      // move food to random position
      state->food[i].position =
          (Vector2){rand() % SCREEN_WIDTH, rand() % SCREEN_HEIGHT};
      state->food[i].rect.x = state->food[i].position.x;
      state->food[i].rect.y = state->food[i].position.y;
    }
  }
}

/*
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
*/

void update_agents(GameState *state, float dt) {
  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++) {
    Agent *agent = &state->agents[i];
    agent->time_alive += dt;

    // Gather inputs
    long double inputs[get_total_input_size()];
    gather_agent_inputs(state, agent, inputs);

    int old_xp = agent->total_xp;
    int old_level = agent->level;

    // Decide
    int action = decide_action(agent, inputs);

    // Step environment
    step_agent(state, agent, action);

    // Compute reward
    // Compute reward
    float reward = compute_reward(agent, old_xp, old_level);

    // TERMINAL CONDITION
    if (agent->total_xp < 0) {
      mu_model_end_episode(agent->brain, reward - 2.0f);
      agent->has_latent = false;
      mu_model_reset_episode(agent->brain);
      continue; // skip latent update
    }

    // Update latent (non-terminal transition)
    update_latent_after_step(agent, inputs, action, reward);

    // Optional logging memory (not required for MuZE)
    store_experience(agent, inputs, action, reward);

    // Update color based on XP
    update_agent_color(agent);
  }
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
      a->total_xp -= leech;
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

  update_agents(game, GetFrameTime());

  for (int i = 0; i < MAX_GROUNDSKEEPERS; i++)
    update_groundkeeper(game, i);

  // Handle breeding
  handle_breeding(game);

  // Check evolution & level up
  check_evolution(game);

  // Handle agent collisions (eat each other)
  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++) {
    for (int j = i + 1; j < POPULATION_SIZE - MAX_GROUNDSKEEPERS; j++) {
      if (CheckCollisionRecs(game->agents[i].rect, game->agents[j].rect)) {
        eat_agent(&game->agents[i], &game->agents[j]);
      }
    }
  }
}

void init_game(GameState *state) {
  state->over = false;
  state->paused = false;
  state->evolution_timer = 0;
  state->current_generation = 0;
  state->next_agent_id = 0;
  state->num_active_agents = POPULATION_SIZE - MAX_GROUNDSKEEPERS;

  // Initialize agents
  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++) {
    init_agent(&state->agents[i], state->next_agent_id++);
  }

  // Initialize groundkeepers
  for (int i = 0; i < MAX_GROUNDSKEEPERS; i++) {
    state->gks[i].position =
        (Vector2){rand() % SCREEN_WIDTH, rand() % SCREEN_HEIGHT};
    state->gks[i].rect =
        (Rectangle){state->gks[i].position.x, state->gks[i].position.y, 20, 20};
    state->gks[i].punishment_timer = 0;
    state->gks[i].color = RED;
  }

  // Initialize food
  for (int i = 0; i < MAX_FOOD; i++) {
    state->food[i].position =
        (Vector2){rand() % SCREEN_WIDTH, rand() % SCREEN_HEIGHT};
    state->food[i].rect =
        (Rectangle){state->food[i].position.x, state->food[i].position.y,
                    FOOD_SIZE, FOOD_SIZE};
  }

  // Clear last actions
  for (int i = 0; i < POPULATION_SIZE; i++)
    state->last_actions[i] = ACTION_NONE;

  state->vision_inputs = malloc(sizeof(long double) * get_total_input_size());
}

void free_game(GameState *game) {
  free(game->vision_inputs);
  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++) {
    mu_model_free(game->agents[i].brain);
    free(game->agents[i].memory.buffer);
  }
}

void save_game(GameState *game, const char *filename) {
  FILE *file = fopen(filename, "wb");
  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++)
    NN_save(game->agents[i].brain, file);

  for (int i = 0; i < MAX_FOOD; i++)
    fwrite(&game->food[i].position, sizeof(Vector2), 1, file);

  for (int i = 0; i < MAX_GROUNDSKEEPERS; i++)
    fwrite(&game->gks[i].position, sizeof(Vector2), 1, file);

  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++)
    fwrite(&game->agents[i].position, sizeof(Vector2), 1, file);

  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++)
    fwrite(&game->agents[i].total_xp, sizeof(long double), 1, file);

  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++)
    fwrite(&game->agents[i].level, sizeof(long double), 1, file);

  fwrite(&game->current_generation, sizeof(int), 1, file);
  fwrite(&game->next_agent_id, sizeof(int), 1, file);
  fwrite(&game->num_active_agents, sizeof(int), 1, file);

  fwrite(&game->evolution_timer, sizeof(long double), 1, file);
  fwrite(&game->paused, sizeof(bool), 1, file);
  fwrite(&game->over, sizeof(bool), 1, file);

  fwrite(game->last_actions, sizeof(Action), POPULATION_SIZE, file);

  fwrite(game, sizeof(GameState), 1, file);

  fclose(file);
}

void load_game(GameState *game, const char *filename) {
  FILE *file = fopen(filename, "rb");
  fread(game, sizeof(GameState), 1, file);
  fclose(file);
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

  free_game(&game);
  CloseWindow();
  return 0;
}
