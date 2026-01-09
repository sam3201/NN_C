// replay_buffer.h

#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

#include "game_env.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t capacity;
  size_t size;
  size_t write_idx;

  int obs_dim;
  int action_count;

  // MuZero tuples
  float *obs_buf; /* capacity * obs_dim */
  float *pi_buf;  /* capacity * action_count */
  float *z_buf;   /* capacity */

  // Transition tuples
  int *a_buf;          /* capacity */
  float *r_buf;        /* capacity */
  float *next_obs_buf; /* capacity * obs_dim */
  int *done_buf;       /* capacity */
} ReplayBuffer;

ReplayBuffer *rb_create(size_t capacity, int obs_dim, int action_count);
void rb_free(ReplayBuffer *rb);

// Push full sample into *all* buffers and return the index written (for later
// overwrite). Stores: (obs, pi, z) for policy/value (action, reward, next_obs,
// done) for dynamics
size_t rb_push_full(ReplayBuffer *rb, const float *obs, const float *pi,
                    float z, int action, float reward, const float *next_obs,
                    int done);

void rb_push(ReplayBuffer *rb, const float *obs, const float *pi, float z);
void rb_push_transition(ReplayBuffer *rb, const float *obs, int action,
                        float reward, const float *next_obs, int done);
int rb_sample(ReplayBuffer *rb, int batch, float *obs_batch, float *pi_batch,
              float *z_batch);

int rb_sample_transition(ReplayBuffer *rb, int batch, float *obs_batch,
                         int *a_batch, float *r_batch, float *next_obs_batch,
                         int *done_batch);

size_t rb_size(ReplayBuffer *rb);
void rb_set_z(ReplayBuffer *rb, size_t idx, float z);

#ifdef __cplusplus
}
#endif

#endif
