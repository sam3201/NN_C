// replay_buffer.c

#include "replay_buffer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ReplayBuffer *rb_create(size_t capacity, int obs_dim, int action_count) {
  ReplayBuffer *rb = (ReplayBuffer *)malloc(sizeof(ReplayBuffer));
  if (!rb)
    return NULL;

  rb->capacity = capacity;
  rb->size = 0;
  rb->write_idx = 0;
  rb->obs_dim = obs_dim;
  rb->action_count = action_count;

  rb->obs_buf = (float *)malloc(sizeof(float) * capacity * (size_t)obs_dim);
  rb->pi_buf = (float *)malloc(sizeof(float) * capacity * (size_t)action_count);
  rb->z_buf = (float *)malloc(sizeof(float) * capacity);

  rb->a_buf = (int *)malloc(sizeof(int) * capacity);
  rb->r_buf = (float *)malloc(sizeof(float) * capacity);
  rb->next_obs_buf =
      (float *)malloc(sizeof(float) * capacity * (size_t)obs_dim);
  rb->done_buf = (int *)malloc(sizeof(int) * capacity);

  if (!rb->obs_buf || !rb->pi_buf || !rb->z_buf || !rb->a_buf || !rb->r_buf ||
      !rb->next_obs_buf || !rb->done_buf) {
    rb_free(rb);
    return NULL;
  }
  return rb;
}

void rb_free(ReplayBuffer *rb) {
  if (!rb)
    return;

  free(rb->obs_buf);
  free(rb->pi_buf);
  free(rb->z_buf);

  free(rb->a_buf);
  free(rb->r_buf);
  free(rb->next_obs_buf);
  free(rb->done_buf);

  free(rb);
}

void rb_push(ReplayBuffer *rb, const float *obs, const float *pi, float z) {
  if (!rb)
    return;

  size_t idx = rb->write_idx;
  memcpy(rb->obs_buf + idx * (size_t)rb->obs_dim, obs,
         sizeof(float) * (size_t)rb->obs_dim);
  memcpy(rb->pi_buf + idx * (size_t)rb->action_count, pi,
         sizeof(float) * (size_t)rb->action_count);
  rb->z_buf[idx] = z;

  rb->write_idx = (rb->write_idx + 1) % rb->capacity;
  if (rb->size < rb->capacity)
    rb->size++;
}

void rb_push_transition(ReplayBuffer *rb, const float *obs, int action,
                        float reward, const float *next_obs, int done) {
  if (!rb || !obs || !next_obs)
    return;

  size_t idx = rb->write_idx;

  memcpy(rb->obs_buf + idx * (size_t)rb->obs_dim, obs,
         sizeof(float) * (size_t)rb->obs_dim);
  rb->a_buf[idx] = action;
  rb->r_buf[idx] = reward;
  memcpy(rb->next_obs_buf + idx * (size_t)rb->obs_dim, next_obs,
         sizeof(float) * (size_t)rb->obs_dim);
  rb->done_buf[idx] = done ? 1 : 0;

  // Keep MuZero tuple slots valid too (optional, but avoids junk reads)
  // Default pi = one-hot(action), z = immediate reward
  if (rb->pi_buf) {
    float *pi = rb->pi_buf + idx * (size_t)rb->action_count;
    for (int i = 0; i < rb->action_count; i++)
      pi[i] = 0.0f;
    if (action >= 0 && action < rb->action_count)
      pi[action] = 1.0f;
  }
  if (rb->z_buf)
    rb->z_buf[idx] = reward;

  rb->write_idx = (rb->write_idx + 1) % rb->capacity;
  if (rb->size < rb->capacity)
    rb->size++;
}

static int rand_int(int n) {
  return (int)((double)rand() / ((double)RAND_MAX + 1.0) * n);
}

int rb_sample(ReplayBuffer *rb, int batch, float *obs_batch, float *pi_batch,
              float *z_batch) {
  if (!rb || rb->size == 0)
    return 0;
  if (!obs_batch || !pi_batch || !z_batch)
    return 0;

  int actual = batch;
  if ((size_t)batch > rb->size)
    actual = (int)rb->size;

  for (int i = 0; i < actual; i++) {
    int idx = rand_int((int)rb->size);

    memcpy(obs_batch + i * rb->obs_dim,
           rb->obs_buf + (size_t)idx * (size_t)rb->obs_dim,
           sizeof(float) * (size_t)rb->obs_dim);

    memcpy(pi_batch + i * rb->action_count,
           rb->pi_buf + (size_t)idx * (size_t)rb->action_count,
           sizeof(float) * (size_t)rb->action_count);

    z_batch[i] = rb->z_buf[idx];
  }
  return actual;
}

int rb_sample_transition(ReplayBuffer *rb, int batch, float *obs_batch,
                         int *a_batch, float *r_batch, float *next_obs_batch,
                         int *done_batch) {
  if (!rb || rb->size == 0)
    return 0;
  if (!obs_batch || !a_batch || !r_batch || !next_obs_batch || !done_batch)
    return 0;

  int actual = batch;
  if ((size_t)batch > rb->size)
    actual = (int)rb->size;

  for (int i = 0; i < actual; i++) {
    int idx = rand_int((int)rb->size);

    memcpy(obs_batch + i * rb->obs_dim,
           rb->obs_buf + (size_t)idx * (size_t)rb->obs_dim,
           sizeof(float) * (size_t)rb->obs_dim);

    a_batch[i] = rb->a_buf[idx];
    r_batch[i] = rb->r_buf[idx];

    memcpy(next_obs_batch + i * rb->obs_dim,
           rb->next_obs_buf + (size_t)idx * (size_t)rb->obs_dim,
           sizeof(float) * (size_t)rb->obs_dim);

    done_batch[i] = rb->done_buf[idx];
  }

  return actual;
}

// rb_sample(...) stays exactly as you already have it (samples obs,pi,z)

size_t rb_size(ReplayBuffer *rb) { return rb ? rb->size : 0; }
