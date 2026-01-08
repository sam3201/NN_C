#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float *obs;      // O
  float *next_obs; // O
  int action;
  float reward;
  int done;

  // optional MuZero targets (can be filled later):
  float *pi;    // A  (policy target, e.g. from MCTS)
  float z;      // scalar return/target value
  float v_pred; // value estimate at time of storage (optional)
} Transition;

typedef struct ReplayBuffer ReplayBuffer;

ReplayBuffer *rb_create(size_t capacity, int obs_dim, int action_count);
void rb_free(ReplayBuffer *rb);
void rb_push(ReplayBuffer *rb, const float *obs, const float *pi, float z);
void rb_push_transition(ReplayBuffer *rb, const float *obs, int action,
                        float reward, const float *next_obs, int done);
int rb_sample_transition(ReplayBuffer *rb, int batch, float *obs_batch,
                         int *a_batch, float *r_batch, float *next_obs_batch,
                         int *done_batch);
int rb_sample(ReplayBuffer *rb, int batch, float *obs_batch, float *pi_batch,
              float *z_batch);
size_t rb_size(ReplayBuffer *rb);

#ifdef __cplusplus
}
#endif

#endif
