// replay_buffer.c

#include "replay_buffer.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
  RB_MAGIC = 0x52425631u, /* "RBV1" */
  RB_VERSION = 3u
};

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
  rb->vprefix_buf = (float *)malloc(sizeof(float) * capacity);
  rb->prio_buf = (float *)malloc(sizeof(float) * capacity);

  rb->a_buf = (int *)malloc(sizeof(int) * capacity);
  rb->r_buf = (float *)malloc(sizeof(float) * capacity);
  rb->next_obs_buf =
      (float *)malloc(sizeof(float) * capacity * (size_t)obs_dim);
  rb->done_buf = (int *)malloc(sizeof(int) * capacity);

  if (!rb->obs_buf || !rb->pi_buf || !rb->z_buf || !rb->vprefix_buf ||
      !rb->prio_buf || !rb->a_buf || !rb->r_buf || !rb->next_obs_buf ||
      !rb->done_buf) {
    rb_free(rb);
    return NULL;
  }

  // Optional: zero init to reduce chance of NaNs if something goes wrong
  memset(rb->obs_buf, 0, sizeof(float) * capacity * (size_t)obs_dim);
  memset(rb->pi_buf, 0, sizeof(float) * capacity * (size_t)action_count);
  memset(rb->z_buf, 0, sizeof(float) * capacity);
  memset(rb->vprefix_buf, 0, sizeof(float) * capacity);
  memset(rb->prio_buf, 1, sizeof(float) * capacity);
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
  free(rb->vprefix_buf);
  free(rb->prio_buf);
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

void rb_set_value_prefix(ReplayBuffer *rb, size_t idx, float vprefix) {
  if (!rb)
    return;
  if (idx >= rb->capacity)
    return;
  rb->vprefix_buf[idx] = vprefix;
}

void rb_set_priority(ReplayBuffer *rb, size_t idx, float prio) {
  if (!rb)
    return;
  if (idx >= rb->capacity)
    return;
  rb->prio_buf[idx] = (prio > 0.0f) ? prio : 1e-6f;
}

size_t rb_push_full(ReplayBuffer *rb, const float *obs, const float *pi,
                    float z, int action, float reward, const float *next_obs,
                    int done) {
  if (!rb)
    return 0;
  if (!obs || !pi || !next_obs)
    return 0;

  size_t idx = rb->write_idx;

  // policy/value training fields
  memcpy(rb->obs_buf + idx * (size_t)rb->obs_dim, obs,
         sizeof(float) * (size_t)rb->obs_dim);
  memcpy(rb->pi_buf + idx * (size_t)rb->action_count, pi,
         sizeof(float) * (size_t)rb->action_count);
  rb->z_buf[idx] = z;
  rb->vprefix_buf[idx] = 0.0f;
  rb->prio_buf[idx] = 1.0f;

  // transition fields (for dynamics/reward training)
  rb->a_buf[idx] = action;
  rb->r_buf[idx] = reward;
  memcpy(rb->next_obs_buf + idx * (size_t)rb->obs_dim, next_obs,
         sizeof(float) * (size_t)rb->obs_dim);
  rb->done_buf[idx] = done ? 1 : 0;

  // advance head / size
  rb->write_idx = (rb->write_idx + 1) % rb->capacity;
  if (rb->size < rb->capacity)
    rb->size++;

  return idx;
}

void rb_push_transition(ReplayBuffer *rb, const float *obs, int action,
                        float reward, const float *next_obs, int done) {
  if (!rb || !obs || !next_obs)
    return;

  float *tmp_pi = (float *)malloc(sizeof(float) * (size_t)rb->action_count);
  if (!tmp_pi)
    return;

  // IMPORTANT: zero init
  for (int i = 0; i < rb->action_count; i++)
    tmp_pi[i] = 0.0f;

  if (action >= 0 && action < rb->action_count)
    tmp_pi[action] = 1.0f;

  rb_push_full(rb, obs, tmp_pi, reward, action, reward, next_obs, done);

  free(tmp_pi);
}

void rb_push(ReplayBuffer *rb, const float *obs, const float *pi, float z) {
  if (!rb || !obs || !pi)
    return;

  // Keep transition fields valid: next_obs = obs, action=0, reward=0,
  // done=0
  rb_push_full(rb, obs, pi, z, 0, 0.0f, obs, 0);
}

static int rand_int(int n) {
  return (int)((double)rand() / ((double)RAND_MAX + 1.0) * n);
}

static size_t rb_logical_to_physical(const ReplayBuffer *rb, size_t logical) {
  if (!rb)
    return 0;
  if (rb->size < rb->capacity)
    return logical;
  return (rb->write_idx + logical) % rb->capacity;
}

static float prio_weight(const ReplayBuffer *rb, size_t idx, float alpha) {
  float p = rb->prio_buf[idx];
  if (p < 1e-6f)
    p = 1e-6f;
  return powf(p, alpha);
}

static int sample_index_per(const ReplayBuffer *rb, float alpha, size_t *idx_out,
                            float *prob_out) {
  if (!rb || rb->size == 0)
    return 0;
  if (!idx_out || !prob_out)
    return 0;

  double sum = 0.0;
  for (size_t i = 0; i < rb->size; i++) {
    size_t idx = rb_logical_to_physical(rb, i);
    sum += (double)prio_weight(rb, idx, alpha);
  }
  if (sum <= 0.0) {
    *idx_out = rb_logical_to_physical(rb, 0);
    *prob_out = 1.0f / (float)rb->size;
    return 1;
  }

  double r = (double)rand() / ((double)RAND_MAX + 1.0) * sum;
  double acc = 0.0;
  for (size_t i = 0; i < rb->size; i++) {
    size_t idx = rb_logical_to_physical(rb, i);
    double w = (double)prio_weight(rb, idx, alpha);
    acc += w;
    if (r <= acc) {
      *idx_out = idx;
      *prob_out = (float)(w / sum);
      return 1;
    }
  }
  {
    size_t idx = rb_logical_to_physical(rb, rb->size - 1);
    *idx_out = idx;
    *prob_out = (float)(prio_weight(rb, idx, alpha) / sum);
  }
  return 1;
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

int rb_sample_sequence(ReplayBuffer *rb, int batch, int unroll_steps,
                       float *obs_seq, float *pi_seq, float *z_seq,
                       int *a_seq, float *r_seq, int *done_seq) {
  if (!rb || rb->size == 0)
    return 0;
  if (!obs_seq || !pi_seq || !z_seq || !a_seq || !r_seq || !done_seq)
    return 0;
  if (batch <= 0 || unroll_steps < 0)
    return 0;

  size_t need = (size_t)unroll_steps + 1;
  if (rb->size < need)
    return 0;

  int actual = batch;
  size_t max_start = rb->size - need;

  int O = rb->obs_dim;
  int A = rb->action_count;

  for (int b = 0; b < actual; b++) {
    size_t start_logical =
        (max_start > 0) ? (size_t)rand_int((int)max_start + 1) : 0;

    for (size_t k = 0; k < need; k++) {
      size_t logical = start_logical + k;
      size_t idx = rb_logical_to_physical(rb, logical);

      memcpy(obs_seq + ((size_t)b * need + k) * (size_t)O,
             rb->obs_buf + idx * (size_t)O, sizeof(float) * (size_t)O);
      memcpy(pi_seq + ((size_t)b * need + k) * (size_t)A,
             rb->pi_buf + idx * (size_t)A, sizeof(float) * (size_t)A);
      z_seq[(size_t)b * need + k] = rb->z_buf[idx];

      if (k < (size_t)unroll_steps) {
        a_seq[(size_t)b * (size_t)unroll_steps + k] = rb->a_buf[idx];
        r_seq[(size_t)b * (size_t)unroll_steps + k] = rb->r_buf[idx];
        done_seq[(size_t)b * (size_t)unroll_steps + k] = rb->done_buf[idx];
      }
    }
  }

  return actual;
}

int rb_sample_sequence_vprefix(ReplayBuffer *rb, int batch, int unroll_steps,
                               float *obs_seq, float *pi_seq, float *z_seq,
                               float *vprefix_seq, int *a_seq, float *r_seq,
                               int *done_seq) {
  if (!rb || rb->size == 0)
    return 0;
  if (!obs_seq || !pi_seq || !z_seq || !vprefix_seq || !a_seq || !r_seq ||
      !done_seq)
    return 0;
  if (batch <= 0 || unroll_steps < 0)
    return 0;

  size_t need = (size_t)unroll_steps + 1;
  if (rb->size < need)
    return 0;

  int actual = batch;
  size_t max_start = rb->size - need;

  int O = rb->obs_dim;
  int A = rb->action_count;

  for (int b = 0; b < actual; b++) {
    size_t start_logical =
        (max_start > 0) ? (size_t)rand_int((int)max_start + 1) : 0;

    for (size_t k = 0; k < need; k++) {
      size_t logical = start_logical + k;
      size_t idx = rb_logical_to_physical(rb, logical);

      memcpy(obs_seq + ((size_t)b * need + k) * (size_t)O,
             rb->obs_buf + idx * (size_t)O, sizeof(float) * (size_t)O);
      memcpy(pi_seq + ((size_t)b * need + k) * (size_t)A,
             rb->pi_buf + idx * (size_t)A, sizeof(float) * (size_t)A);
      z_seq[(size_t)b * need + k] = rb->z_buf[idx];
      vprefix_seq[(size_t)b * need + k] = rb->vprefix_buf[idx];

      if (k < (size_t)unroll_steps) {
        a_seq[(size_t)b * (size_t)unroll_steps + k] = rb->a_buf[idx];
        r_seq[(size_t)b * (size_t)unroll_steps + k] = rb->r_buf[idx];
        done_seq[(size_t)b * (size_t)unroll_steps + k] = rb->done_buf[idx];
      }
    }
  }

  return actual;
}

int rb_sample_per(ReplayBuffer *rb, int batch, float alpha, float *obs_batch,
                  float *pi_batch, float *z_batch, size_t *idx_out,
                  float *prob_out) {
  if (!rb || rb->size == 0)
    return 0;
  if (!obs_batch || !pi_batch || !z_batch || !idx_out || !prob_out)
    return 0;

  int actual = batch;
  int O = rb->obs_dim;
  int A = rb->action_count;

  for (int i = 0; i < actual; i++) {
    size_t idx = 0;
    float p = 0.0f;
    if (!sample_index_per(rb, alpha, &idx, &p))
      return 0;
    idx_out[i] = idx;
    prob_out[i] = p;

    memcpy(obs_batch + i * O, rb->obs_buf + idx * (size_t)O,
           sizeof(float) * (size_t)O);
    memcpy(pi_batch + i * A, rb->pi_buf + idx * (size_t)A,
           sizeof(float) * (size_t)A);
    z_batch[i] = rb->z_buf[idx];
  }

  return actual;
}

int rb_sample_sequence_per(ReplayBuffer *rb, int batch, int unroll_steps,
                           float alpha, float *obs_seq, float *pi_seq,
                           float *z_seq, float *vprefix_seq, int *a_seq,
                           float *r_seq, int *done_seq, size_t *idx_out,
                           float *prob_out) {
  if (!rb || rb->size == 0)
    return 0;
  if (!obs_seq || !pi_seq || !z_seq || !vprefix_seq || !a_seq || !r_seq ||
      !done_seq || !idx_out || !prob_out)
    return 0;
  if (batch <= 0 || unroll_steps < 0)
    return 0;

  size_t need = (size_t)unroll_steps + 1;
  if (rb->size < need)
    return 0;

  int actual = batch;
  int O = rb->obs_dim;
  int A = rb->action_count;

  for (int b = 0; b < actual; b++) {
    size_t start = 0;
    float p = 0.0f;
    if (!sample_index_per(rb, alpha, &start, &p))
      return 0;
    size_t start_logical = 0;
    if (rb->size < rb->capacity) {
      start_logical = start;
    } else {
      start_logical =
          (start + rb->capacity - rb->write_idx) % rb->capacity;
      if (start_logical > rb->size - need)
        start_logical = rb->size - need;
    }
    idx_out[b] = rb_logical_to_physical(rb, start_logical);
    prob_out[b] = p;

    for (size_t k = 0; k < need; k++) {
      size_t logical = start_logical + k;
      size_t idx = rb_logical_to_physical(rb, logical);

      memcpy(obs_seq + ((size_t)b * need + k) * (size_t)O,
             rb->obs_buf + idx * (size_t)O, sizeof(float) * (size_t)O);
      memcpy(pi_seq + ((size_t)b * need + k) * (size_t)A,
             rb->pi_buf + idx * (size_t)A, sizeof(float) * (size_t)A);
      z_seq[(size_t)b * need + k] = rb->z_buf[idx];
      vprefix_seq[(size_t)b * need + k] = rb->vprefix_buf[idx];

      if (k < (size_t)unroll_steps) {
        a_seq[(size_t)b * (size_t)unroll_steps + k] = rb->a_buf[idx];
        r_seq[(size_t)b * (size_t)unroll_steps + k] = rb->r_buf[idx];
        done_seq[(size_t)b * (size_t)unroll_steps + k] = rb->done_buf[idx];
      }
    }
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

size_t rb_size(ReplayBuffer *rb) { return rb ? rb->size : 0; }

int rb_save(ReplayBuffer *rb, const char *filename) {
  if (!rb || !filename)
    return 0;

  FILE *f = fopen(filename, "wb");
  if (!f)
    return 0;

  uint32_t magic = RB_MAGIC;
  uint32_t version = RB_VERSION;
  if (fwrite(&magic, sizeof(magic), 1, f) != 1 ||
      fwrite(&version, sizeof(version), 1, f) != 1) {
    fclose(f);
    return 0;
  }

  if (fwrite(&rb->capacity, sizeof(size_t), 1, f) != 1 ||
      fwrite(&rb->size, sizeof(size_t), 1, f) != 1 ||
      fwrite(&rb->write_idx, sizeof(size_t), 1, f) != 1 ||
      fwrite(&rb->obs_dim, sizeof(int), 1, f) != 1 ||
      fwrite(&rb->action_count, sizeof(int), 1, f) != 1) {
    fclose(f);
    return 0;
  }

  size_t obs_bytes = sizeof(float) * rb->capacity * (size_t)rb->obs_dim;
  size_t pi_bytes =
      sizeof(float) * rb->capacity * (size_t)rb->action_count;
  size_t z_bytes = sizeof(float) * rb->capacity;
  size_t vprefix_bytes = sizeof(float) * rb->capacity;
  size_t prio_bytes = sizeof(float) * rb->capacity;
  size_t a_bytes = sizeof(int) * rb->capacity;
  size_t r_bytes = sizeof(float) * rb->capacity;
  size_t next_obs_bytes = sizeof(float) * rb->capacity * (size_t)rb->obs_dim;
  size_t done_bytes = sizeof(int) * rb->capacity;

  if (fwrite(rb->obs_buf, 1, obs_bytes, f) != obs_bytes ||
      fwrite(rb->pi_buf, 1, pi_bytes, f) != pi_bytes ||
      fwrite(rb->z_buf, 1, z_bytes, f) != z_bytes ||
      fwrite(rb->vprefix_buf, 1, vprefix_bytes, f) != vprefix_bytes ||
      fwrite(rb->prio_buf, 1, prio_bytes, f) != prio_bytes ||
      fwrite(rb->a_buf, 1, a_bytes, f) != a_bytes ||
      fwrite(rb->r_buf, 1, r_bytes, f) != r_bytes ||
      fwrite(rb->next_obs_buf, 1, next_obs_bytes, f) != next_obs_bytes ||
      fwrite(rb->done_buf, 1, done_bytes, f) != done_bytes) {
    fclose(f);
    return 0;
  }

  fclose(f);
  return 1;
}

ReplayBuffer *rb_load(const char *filename) {
  if (!filename)
    return NULL;

  FILE *f = fopen(filename, "rb");
  if (!f)
    return NULL;

  uint32_t magic = 0;
  uint32_t version = 0;
  if (fread(&magic, sizeof(magic), 1, f) != 1 ||
      fread(&version, sizeof(version), 1, f) != 1) {
    fclose(f);
    return NULL;
  }

  if (magic != RB_MAGIC || version != RB_VERSION) {
    if (magic != RB_MAGIC || (version != 1u && version != 2u)) {
      fclose(f);
      return NULL;
    }
  }

  size_t capacity = 0;
  size_t size = 0;
  size_t write_idx = 0;
  int obs_dim = 0;
  int action_count = 0;

  if (fread(&capacity, sizeof(size_t), 1, f) != 1 ||
      fread(&size, sizeof(size_t), 1, f) != 1 ||
      fread(&write_idx, sizeof(size_t), 1, f) != 1 ||
      fread(&obs_dim, sizeof(int), 1, f) != 1 ||
      fread(&action_count, sizeof(int), 1, f) != 1) {
    fclose(f);
    return NULL;
  }

  if (capacity == 0 || obs_dim <= 0 || action_count <= 0 ||
      size > capacity || write_idx >= capacity) {
    fclose(f);
    return NULL;
  }

  ReplayBuffer *rb = rb_create(capacity, obs_dim, action_count);
  if (!rb) {
    fclose(f);
    return NULL;
  }

  rb->size = size;
  rb->write_idx = write_idx;

  size_t obs_bytes = sizeof(float) * capacity * (size_t)obs_dim;
  size_t pi_bytes = sizeof(float) * capacity * (size_t)action_count;
  size_t z_bytes = sizeof(float) * capacity;
  size_t vprefix_bytes = sizeof(float) * capacity;
  size_t prio_bytes = sizeof(float) * capacity;
  size_t a_bytes = sizeof(int) * capacity;
  size_t r_bytes = sizeof(float) * capacity;
  size_t next_obs_bytes = sizeof(float) * capacity * (size_t)obs_dim;
  size_t done_bytes = sizeof(int) * capacity;

  if (fread(rb->obs_buf, 1, obs_bytes, f) != obs_bytes ||
      fread(rb->pi_buf, 1, pi_bytes, f) != pi_bytes ||
      fread(rb->z_buf, 1, z_bytes, f) != z_bytes) {
    rb_free(rb);
    fclose(f);
    return NULL;
  }

  if (version == RB_VERSION) {
    if (fread(rb->vprefix_buf, 1, vprefix_bytes, f) != vprefix_bytes) {
      rb_free(rb);
      fclose(f);
      return NULL;
    }
    if (fread(rb->prio_buf, 1, prio_bytes, f) != prio_bytes) {
      rb_free(rb);
      fclose(f);
      return NULL;
    }
  } else if (version == 2u) {
    if (fread(rb->vprefix_buf, 1, vprefix_bytes, f) != vprefix_bytes) {
      rb_free(rb);
      fclose(f);
      return NULL;
    }
    memset(rb->prio_buf, 1, prio_bytes);
  } else {
    memset(rb->vprefix_buf, 0, vprefix_bytes);
    memset(rb->prio_buf, 1, prio_bytes);
  }

  if (fread(rb->a_buf, 1, a_bytes, f) != a_bytes ||
      fread(rb->r_buf, 1, r_bytes, f) != r_bytes ||
      fread(rb->next_obs_buf, 1, next_obs_bytes, f) != next_obs_bytes ||
      fread(rb->done_buf, 1, done_bytes, f) != done_bytes) {
    rb_free(rb);
    fclose(f);
    return NULL;
  }

  fclose(f);
  return rb;
}
