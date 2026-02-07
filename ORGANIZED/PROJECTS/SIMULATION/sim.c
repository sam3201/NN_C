#include "../utils/NN/MEMORY/MEMORY.h"
#include "../utils/NN/MUZE/all.h"
#include "../utils/NN/NEAT/NEAT.h"
#include "../utils/NN/NN/NN.h"
#include "../utils/Raylib/src/raylib.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

typedef struct {
  uint32_t s;
} SimRng;

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
  int last_num_eaten;
  bool is_breeding;
  float breeding_timer;
  Color color;
  MuModel *brain;
  Memory memory;
  size_t input_size;
  float latent[LATENT_MAX];
  bool has_latent;
  SimRng rng_state;
  MCTSRng rng;

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
  MuzeConfig muze_cfg;
  MuModel *muze_model;
  ReplayBuffer *replay;
} SimulationState;

static pthread_mutex_t sim_model_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t sim_rb_mtx = PTHREAD_MUTEX_INITIALIZER;
static MuzeLoopThread sim_loop;
static uint32_t sim_loop_rng_state = 0;

static float sim_rng01(void *ctx) {
  SimRng *r = (SimRng *)ctx;
  uint32_t x = r->s ? r->s : 0x12345678u;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  r->s = x;
  return (float)(x & 0xFFFFFFu) / (float)0x1000000u;
}

static float loop_rng01(void *ctx) {
  uint32_t *s = (uint32_t *)ctx;
  uint32_t x = *s ? *s : 0x12345678u;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return (float)(x & 0xFFFFFFu) / (float)0x1000000u;
}

static void init_muze_config(MuzeConfig *cfg, int obs_dim) {
  if (!cfg)
    return;
  muze_config_init_defaults(cfg, obs_dim, ACTION_COUNT);

  cfg->model.latent_dim = 32;
  cfg->nn.hidden_repr = 64;
  cfg->nn.hidden_dyn = 64;
  cfg->nn.hidden_pred = 64;
  cfg->nn.hidden_vprefix = 64;
  cfg->nn.hidden_reward = 64;
  cfg->nn.action_embed_dim = 32;

  cfg->mcts.num_simulations = 25;
  cfg->mcts.batch_simulations = 0;
  cfg->mcts.c_puct = 1.2f;
  cfg->mcts.max_depth = 16;
  cfg->mcts.dirichlet_alpha = 0.3f;
  cfg->mcts.dirichlet_eps = 0.25f;
  cfg->mcts.temperature = 1.0f;
  cfg->mcts.discount = 0.99f;

  cfg->trainer.batch_size = 64;
  cfg->trainer.train_steps = 50;
  cfg->trainer.min_replay_size = 1024;

  cfg->loop.iterations = 0;
  cfg->loop.selfplay_episodes_per_iter = 0;
  cfg->loop.selfplay_disable = 1;
  cfg->loop.train_calls_per_iter = 1;
  cfg->loop.use_reanalyze = 0;
  cfg->loop.eval_interval = 0;
  cfg->loop.checkpoint_interval = 0;
  cfg->loop.selfplay_actor_count = 2;
  cfg->loop.selfplay_use_threads = 1;
}

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

void init_agent(Agent *agent, int id, const MuzeConfig *cfg,
                MuModel *shared_model) {
  agent->level = 0;
  agent->total_xp = 0;
  agent->size = INITIAL_AGENT_SIZE;
  agent->time_alive = 0;
  agent->agent_id = id;
  agent->parent_id = -1;
  agent->num_offsprings = 0;
  agent->num_eaten = 0;
  agent->last_num_eaten = -1;
  agent->is_breeding = false;
  agent->breeding_timer = 0;
  agent->position.x = (float)(rand() % (SCREEN_WIDTH - 10));
  agent->position.y = (float)(rand() % (SCREEN_HEIGHT - 10));
  agent->rect = (Rectangle){agent->position.x, agent->position.y,
                            (float)agent->size, (float)agent->size};
  agent->input_size = get_total_input_size();
  update_agent_color(agent);

  agent->brain = shared_model;
  init_memory(&agent->memory, 100, (int)agent->input_size);
  agent->has_latent = false;
  assert(cfg->model.latent_dim <= LATENT_MAX);

  agent->rng_state.s = (uint32_t)time(NULL) ^ (0x9e3779b9u * (uint32_t)id);
  agent->rng.ctx = &agent->rng_state;
  agent->rng.rand01 = sim_rng01;
}

// --- VISION ENCODING ---
/*
void encode_vision(SimulationState *game, int agent_idx, long double
*vision_output) { Agent *self = &game->agents[agent_idx];

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

bool can_move_to_agent(SimulationState *game, Agent *agent, Vector2 new_pos) {
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

void move_agent(SimulationState *game, Agent *agent, Action action) {
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

Agent spawn_offspring(Agent *parent1, Agent *parent2, int new_id,
                      const MuzeConfig *cfg) {
  Agent child;
  child.level = 0;
  child.total_xp = XP_FROM_OFFSPRING;
  child.size = INITIAL_AGENT_SIZE;
  child.time_alive = 0;
  child.agent_id = new_id;
  child.parent_id = parent1->agent_id;
  child.num_offsprings = 0;
  child.num_eaten = 0;
  child.last_num_eaten = 0;
  child.is_breeding = false;
  child.breeding_timer = 0;
  child.position.x = (parent1->position.x + parent2->position.x) / 2;
  child.position.y = (parent1->position.y + parent2->position.y) / 2;
  child.rect = (Rectangle){child.position.x, child.position.y,
                           (float)child.size, (float)child.size};
  child.input_size = get_total_input_size();
  update_agent_color(&child);
  child.has_latent = false;

  MuConfig model_cfg = cfg->model;
  model_cfg.obs_dim = (int)child.input_size;
  model_cfg.action_count = ACTION_COUNT;
  child.brain = mu_model_create_nn_with_cfg(&model_cfg, &cfg->nn);
  init_memory(&child.memory, 100, (int)child.input_size);

  parent1->num_offsprings++;
  parent2->num_offsprings++;

  return child;
}

// Call this inside update_agent or update_game to handle breeding
void handle_breeding(SimulationState *game) {
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
            game->agents[k] =
                spawn_offspring(a1, a2, game->next_agent_id++, &game->muze_cfg);
            break;
          }
        }
      }
    }
  }
}
// --- EVOLUTION CHECK ---
void check_evolution(SimulationState *game) {
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

void execute_agent_action(SimulationState *game, int agent_idx, Action action) {
  Agent *agent = &game->agents[agent_idx];
  move_agent(game, agent, action);
}

void gather_agent_inputs(SimulationState *state, Agent *agent,
                         long double *inputs) {
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

  int delta = a->num_eaten - a->last_num_eaten;
  if (delta > 0)
    r += 2.0f * delta;

  a->last_num_eaten = a->num_eaten;

  if (a->total_xp < 0)
    r -= 1.0f;

  return r;
}

int decide_action(Agent *agent, long double *inputs, const MCTSParams *mcts,
                  int fallback_action, float *out_pi) {
  MuModel *brain = agent->brain;

  int obs_dim = brain->cfg.obs_dim;
  float obs[obs_dim];
  for (int i = 0; i < obs_dim; i++)
    obs[i] = (float)inputs[i];

  if (pthread_mutex_trylock(&sim_model_mtx) != 0) {
    if (out_pi) {
      float u = 1.0f / (float)ACTION_COUNT;
      for (int i = 0; i < ACTION_COUNT; i++)
        out_pi[i] = u;
    }
    if (fallback_action >= 0 && fallback_action < ACTION_COUNT)
      return fallback_action;
    return rand() % ACTION_COUNT;
  }
  if (!agent->has_latent) {
    mu_model_repr(brain, obs, agent->latent);
    agent->has_latent = true;
  }

  const MCTSParams *mp = mcts ? mcts : NULL;
  MCTSResult res = mcts_run(brain, obs, mp, &agent->rng);
  pthread_mutex_unlock(&sim_model_mtx);
  int action = res.chosen_action;
  if (out_pi) {
    for (int i = 0; i < ACTION_COUNT; i++)
      out_pi[i] = res.pi[i];
  }
  mcts_result_free(&res);
  if (action < 0)
    action = rand() % ACTION_COUNT;
  return action;
}

void update_latent_after_step(Agent *agent, long double *obs, int action,
                              float reward) {
  float obs_f[agent->input_size];
  for (int i = 0; i < agent->input_size; i++)
    obs_f[i] = (float)obs[i];

  float next_latent[LATENT_MAX];
  float predicted_reward;

  if (pthread_mutex_trylock(&sim_model_mtx) != 0)
    return;
  mu_model_dynamics(agent->brain, agent->latent, action, next_latent,
                    &predicted_reward);
  pthread_mutex_unlock(&sim_model_mtx);

  int L = agent->brain->cfg.latent_dim;
  memcpy(agent->latent, next_latent, sizeof(float) * L);
}

void step_agent(SimulationState *state, Agent *agent, int action) {
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
void update_agent(SimulationState *game, int agent_idx) {
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

void update_agents(SimulationState *state, float dt) {
  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++) {
    Agent *agent = &state->agents[i];
    agent->time_alive += dt;

    // Gather inputs
    long double inputs[get_total_input_size()];
    gather_agent_inputs(state, agent, inputs);

    float obs_f[agent->input_size];
    for (int k = 0; k < (int)agent->input_size; k++)
      obs_f[k] = (float)inputs[k];

    int old_xp = agent->total_xp;
    int old_level = agent->level;

    // Decide
    float pi[ACTION_COUNT];
    int action = decide_action(agent, inputs, &state->muze_cfg.mcts,
                               state->last_actions[i], pi);
    state->last_actions[i] = action;

    // Step environment
    step_agent(state, agent, action);

    // Compute reward
    // Compute reward
    float reward = compute_reward(agent, old_xp, old_level);

    int done_flag = (agent->total_xp < 0);

    // Update latent (non-terminal transition)
    if (!done_flag)
      update_latent_after_step(agent, inputs, action, reward);
    else
      agent->has_latent = false;

    // Optional logging memory (not required for MuZE)
    store_experience(agent, inputs, action, reward);

    long double next_inputs[get_total_input_size()];
    gather_agent_inputs(state, agent, next_inputs);
    float next_obs_f[agent->input_size];
    for (int k = 0; k < (int)agent->input_size; k++)
      next_obs_f[k] = (float)next_inputs[k];

    if (state->replay) {
      pthread_mutex_lock(&sim_rb_mtx);
      size_t idx = rb_push_full(state->replay, obs_f, pi, reward, action,
                                reward, next_obs_f, done_flag);
      rb_set_value_prefix(state->replay, idx, reward);
      pthread_mutex_unlock(&sim_rb_mtx);
    }

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

bool can_move_to_gk(SimulationState *game, Groundkeeper *gk, Vector2 new_pos) {
  Rectangle new_rect = {new_pos.x, new_pos.y, gk->rect.width, gk->rect.height};
  if (new_pos.x < 0 || new_pos.x + gk->rect.width > SCREEN_WIDTH ||
      new_pos.y < 0 || new_pos.y + gk->rect.height > SCREEN_HEIGHT)
    return false;
  return true;
}

void update_groundkeeper(SimulationState *game, int idx) {
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
void update_game(SimulationState *game) {
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

void init_game(SimulationState *state) {
  state->over = false;
  state->paused = false;
  state->evolution_timer = 0;
  state->current_generation = 0;
  state->next_agent_id = 0;
  state->num_active_agents = POPULATION_SIZE - MAX_GROUNDSKEEPERS;
  init_muze_config(&state->muze_cfg, (int)get_total_input_size());
  state->muze_model =
      mu_model_create_nn_with_cfg(&state->muze_cfg.model, &state->muze_cfg.nn);
  state->replay = rb_create(200000, state->muze_cfg.model.obs_dim,
                            state->muze_cfg.model.action_count);

  memset(&sim_loop, 0, sizeof(sim_loop));
  sim_loop.model = state->muze_model;
  sim_loop.rb = state->replay;
  sim_loop.gr = NULL;
  sim_loop.mcts = state->muze_cfg.mcts;
  sim_loop.selfplay = state->muze_cfg.selfplay;
  sim_loop.loop = state->muze_cfg.loop;
  sim_loop.env = muze_env_make_stub();
  sim_loop.use_multi = 1;
  sim_loop.model_mutex = &sim_model_mtx;
  sim_loop.rb_mutex = &sim_rb_mtx;
  sim_loop.gr_mutex = NULL;
  sim_loop_rng_state = (uint32_t)time(NULL);
  sim_loop.rng.ctx = &sim_loop_rng_state;
  sim_loop.rng.rand01 = loop_rng01;

  // Initialize agents
  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++) {
    init_agent(&state->agents[i], state->next_agent_id++, &state->muze_cfg,
               state->muze_model);
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

void free_game(SimulationState *game) {
  free(game->vision_inputs);
  for (int i = 0; i < POPULATION_SIZE - MAX_GROUNDSKEEPERS; i++)
    free(game->agents[i].memory.buffer);
  rb_free(game->replay);
  mu_model_free(game->muze_model);
}

void save_game(SimulationState *game, const char *filename) {
  FILE *file = fopen(filename, "wb");
  if (game->muze_model)
    NN_save(game->muze_model, file);

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

  fwrite(game, sizeof(SimulationState), 1, file);

  fclose(file);
}

void load_game(SimulationState *game, const char *filename) {
  FILE *file = fopen(filename, "rb");
  fread(game, sizeof(SimulationState), 1, file);
  fclose(file);
}

// --- MAIN ---
int main(int argc, char *argv[]) {
  // Parse command line arguments for verbose control
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--quiet") == 0 || strcmp(argv[i], "-q") == 0) {
      muze_set_verbose(0);
    } else if (strcmp(argv[i], "--verbose") == 0 ||
               strcmp(argv[i], "-v") == 0) {
      muze_set_verbose(1);
    }
  }

  InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Evolution Simulator");
  SetTargetFPS(FRAME_RATE);
  srand(time(NULL));

  SimulationState game = {0};
  init_game(&game);
  muze_loop_thread_start(&sim_loop);

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
      float r = 3.0f;
      DrawCircle((int)a->position.x, (int)a->position.y, r, a->color);
      if (a->is_breeding)
        DrawCircleLines((int)a->position.x, (int)a->position.y, r + 2.0f, PINK);
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

  muze_loop_thread_stop(&sim_loop);
  free_game(&game);
  CloseWindow();
  return 0;
}
