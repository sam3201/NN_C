// replay_buffer.c

#include "replay_buffer.h"
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

  // Optional: zero init to reduce chance of NaNs if something goes wrong
  memset(rb->obs_buf, 0, sizeof(float) * capacity * (size_t)obs_dim);
  memset(rb->pi_buf, 0, sizeof(float) * capacity * (size_t)action_count);
  memset(rb->z_buf, 0, sizeof(float) * capacity);
  memset(rb->a_buf, 0, sizeof(int) * capacity);
  memset(rb->r_buf, 0, sizeof(float) * capacity);
  memset(rb->next_obs_buf, 0, sizeof(float) * capacity * (size_t)obs_dim);
  memset(rb->done_buf, 0, sizeof(int) * capacity);

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

void rb_set_z(ReplayBuffer *rb, size_t idx, float z) {
  if (!rb)
    return;
  if (idx >= rb->capacity)
    return;
  rb->z_buf[idx] = z;
}

void rb_push(ReplayBuffer *rb, const float *obs, const float *pi, float z) {
  if (!rb || !obs || !pi)
    return;

  // We MUST make transition fields valid too; use next_obs = obs as safe
  // default. (Trainer dynamics can still learn something; or you can skip
  // done=0.)
  rb_push_full(rb, obs, pi, z, /*action*/ 0, /*reward*/ 0.0f, obs, /*done*/ 0);
}

-- -a / replay_buffer.c++ +
    b / replay_buffer.c @ @-size_t rb_push_full(ReplayBuffer *rb,
                                                const float *obs,
                                                const float *pi, -float z) {
  size_t rb_push_full(ReplayBuffer * rb, +const float *obs, +const float *pi,
                      +float z, +int action, +float reward,
                      +const float *next_obs, +int done) {
    if (!rb)
      return 0;

    size_t idx = rb->write_idx;
    // --- policy/value tuples ---
    memcpy(rb->obs_buf + idx * rb->obs_dim, obs, sizeof(float) * rb->obs_dim);
    memcpy(rb->pi_buf + idx * rb->action_count, pi,
           sizeof(float) * rb->action_count);
    rb->z_buf[idx] = z;

    // --- transition tuples ---
    rb->a_buf[idx] = action;
    rb->r_buf[idx] = reward;
    memcpy(rb->next_obs_buf + idx * rb->obs_dim, next_obs,
           +sizeof(float) * rb->obs_dim);
    rb->done_buf[idx] = done ? 1 : 0;

    // advance head / size
    rb->write_idx = (rb->write_idx + 1) % rb->capacity;
    if (rb->size < rb->capacity)
      rb->size++;

    return idx;
  }
  void rb_push(ReplayBuffer * rb, const float *obs, const float *pi, float z) {
    if (!rb || !obs || !pi)
      return;

    // Make transition fields valid too; safest default: next_obs = obs.
    rb_push_full(rb, obs, pi, z, /*action*/ 0, /*reward*/ 0.0f,
                 +/*next_obs*/ obs, /*done*/ 0);
  }
  void rb_push_transition(ReplayBuffer * rb, const float *obs, int action,
                          float reward, const float *next_obs, int done) {
    if (!rb || !obs || !next_obs)
      return;

    Default pi = one - hot(action),
            z = reward(1 - step) float *tmp_pi =
                (float *)malloc(sizeof(float) * (size_t)rb->action_count);
    if (!tmp_pi)
      return;
    if (action >= 0 && action < rb->action_count)
      tmp_pi[action] = 1.0f;

    rb_push_full(rb, obs, tmp_pi, /*z*/ reward, +action, reward, next_obs,
                 done);

    free(tmp_pi);
  }

  void rb_push_transition(ReplayBuffer * rb, const float *obs, int action,
                          float reward, const float *next_obs, int done) {
    if (!rb || !obs || !next_obs)
      return;

    // Default pi = one-hot(action), z = reward (1-step)
    float *tmp_pi = (float *)malloc(sizeof(float) * (size_t)rb->action_count);
    if (!tmp_pi)
      return;

    for (int i = 0; i < rb->action_count; i++)
      tmp_pi[i] = 0.0f;
    if (action >= 0 && action < rb->action_count)
      tmp_pi[action] = 1.0f;

    rb_push_full(rb, obs, tmp_pi, /*z*/ reward, action, reward, next_obs, done);

    free(tmp_pi);
  }

  static int rand_int(int n) {
    return (int)((double)rand() / ((double)RAND_MAX + 1.0) * n);
  }

  int rb_sample(ReplayBuffer * rb, int batch, float *obs_batch, float *pi_batch,
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

  int rb_sample_transition(ReplayBuffer * rb, int batch, float *obs_batch,
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

  size_t rb_size(ReplayBuffer * rb) { return rb ? rb->size : 0; }
