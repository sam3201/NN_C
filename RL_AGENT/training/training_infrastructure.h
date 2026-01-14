#ifndef TRAINING_INFRASTRUCTURE_H
#define TRAINING_INFRASTRUCTURE_H

#include "../encoder/observation_encoder.h"
#include "../policy/policy_value_network.h"
#include "../world_model/world_model.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// Training configuration
#define NUM_PARALLEL_ENVIRONMENTS 8
#define MAX_EPISODE_LENGTH 1000
#define BATCH_SIZE 64
#define PPO_EPOCHS 4
#define GAMMA 0.99f
#define LAMBDA 0.95f
#define CLIP_EPSILON 0.2f
#define VALUE_LOSS_COEF 0.5f
#define ENTROPY_COEF 0.01f
#define LEARNING_RATE 3e-4f

// Experience replay buffer
#define REPLAY_BUFFER_SIZE 100000

// Experience structure
typedef struct {
  float state[LATENT_DIM];
  float action[NUM_DISCRETE_MOVEMENTS + NUM_DISCRETE_ACTIONS +
               NUM_CONTINUOUS_ACTIONS];
  float reward;
  float next_state[LATENT_DIM];
  float value;
  float advantage;
  float log_prob;
  int done;
} Experience;

// Replay buffer
typedef struct {
  Experience *experiences;
  int size;
  int capacity;
  int head;
  int tail;
  pthread_mutex_t mutex;
} ReplayBuffer;

// Training statistics
typedef struct {
  float total_reward;
  float episode_length;
  float win_rate;
  float loss;
  int episodes_completed;
  int total_steps;
  time_t start_time;
} TrainingStats;

// Agent pool for self-play
typedef struct {
  PolicyValueNetwork *agents[10]; // Support up to 10 different agent versions
  int num_agents;
  int current_main_agent;
  pthread_mutex_t mutex;
} AgentPool;

// Training thread data
typedef struct {
  int thread_id;
  ObservationEncoder *encoder;
  PolicyValueNetwork *policy;
  WorldModel *world_model;
  ReplayBuffer *replay_buffer;
  AgentPool *agent_pool;
  TrainingStats *stats;
  volatile int *training_active;
  int episodes_completed;
} TrainingThreadData;

// Training environment interface
typedef struct {
  int id;
  GridObservation current_obs;
  float reward;
  int done;
  int episode_step;
  float total_reward;
} Environment;

// Function declarations
ReplayBuffer *replay_buffer_create(int capacity);
void replay_buffer_destroy(ReplayBuffer *buffer);
void replay_buffer_add(ReplayBuffer *buffer, Experience *exp);
int replay_buffer_sample(ReplayBuffer *buffer, Experience *batch,
                         int batch_size);
void replay_buffer_clear(ReplayBuffer *buffer);

AgentPool *agent_pool_create();
void agent_pool_destroy(AgentPool *pool);
PolicyValueNetwork *agent_pool_get_opponent(AgentPool *pool);
void agent_pool_add_agent(AgentPool *pool, PolicyValueNetwork *agent);

TrainingStats *training_stats_create();
void training_stats_destroy(TrainingStats *stats);
void training_stats_update(TrainingStats *stats, float reward,
                           int episode_length, int won);
void training_stats_print(TrainingStats *stats);

void *training_worker_thread(void *arg);
void training_loop(TrainingThreadData *data);
void update_policy(PolicyValueNetwork *policy, ReplayBuffer *replay_buffer,
                   WorldModel *world_model);
float compute_gae(Experience *trajectory, int length, float gamma,
                  float lambda);

// Curriculum learning
typedef enum {
  CURRICULUM_STAGE_0, // Environment only, no enemies
  CURRICULUM_STAGE_1, // Passive enemies only
  CURRICULUM_STAGE_2, // Weak enemies
  CURRICULUM_STAGE_3, // Full self-play
  CURRICULUM_STAGE_COUNT
} CurriculumStage;

CurriculumStage get_curriculum_stage(int episodes_completed);
Environment create_environment(CurriculumStage stage);

#endif // TRAINING_INFRASTRUCTURE_H
