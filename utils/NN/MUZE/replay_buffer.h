#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ReplayBuffer {
  size_t capacity;
  size_t size;
  size_t write_idx;
  int obs_dim;
  int action_count;
  float *obs_buf; /* capacity * obs_dim */
  float *pi_buf;  /* capacity * action_count */
  float *z_buf;   /* capacity */
};

ReplayBuffer *rb_create(size_t capacity, int obs_dim, int action_count);
void rb_free(ReplayBuffer *rb);
void rb_push(ReplayBuffer *rb, const float *obs, const float *pi, float z);
int rb_sample(ReplayBuffer *rb, int batch, float *obs_batch, float *pi_batch,
              float *z_batch);
size_t rb_size(ReplayBuffer *rb);

#ifdef __cplusplus
}
#endif

#endif
