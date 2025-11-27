#include "replay_buffer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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

ReplayBuffer *rb_create(size_t capacity, int obs_dim, int action_count) {
    ReplayBuffer *rb = (ReplayBuffer*)malloc(sizeof(ReplayBuffer));
    if (!rb) return NULL;
    rb->capacity = capacity;
    rb->size = 0;
    rb->write_idx = 0;
    rb->obs_dim = obs_dim;
    rb->action_count = action_count;
    rb->obs_buf = (float*)malloc(sizeof(float) * capacity * obs_dim);
    rb->pi_buf  = (float*)malloc(sizeof(float) * capacity * action_count);
    rb->z_buf   = (float*)malloc(sizeof(float) * capacity);
    if (!rb->obs_buf || !rb->pi_buf || !rb->z_buf) { rb_free(rb); return NULL; }
    return rb;
}

void rb_free(ReplayBuffer *rb) {
    if (!rb) return;
    if (rb->obs_buf) free(rb->obs_buf);
    if (rb->pi_buf) free(rb->pi_buf);
    if (rb->z_buf) free(rb->z_buf);
    free(rb);
}

void rb_push(ReplayBuffer *rb, const float *obs, const float *pi, float z) {
    if (!rb) return;
    size_t idx = rb->write_idx;
    memcpy(rb->obs_buf + idx * rb->obs_dim, obs, sizeof(float) * rb->obs_dim);
    memcpy(rb->pi_buf  + idx * rb->action_count, pi,  sizeof(float) * rb->action_count);
    rb->z_buf[idx] = z;
    rb->write_idx = (rb->write_idx + 1) % rb->capacity;
    if (rb->size < rb->capacity) rb->size++;
}

static int rand_int(int n) {
    return (int)((double)rand() / ((double)RAND_MAX + 1.0) * n);
}

int rb_sample(ReplayBuffer *rb, int batch, float *obs_batch, float *pi_batch, float *z_batch) {
    if (!rb || rb->size == 0) return 0;
    int actual = batch;
    if ((size_t)batch > rb->size) actual = (int)rb->size;
    for (int i=0;i<actual;i++) {
        int idx = rand_int((int)rb->size);
        memcpy(obs_batch + i*rb->obs_dim, rb->obs_buf + idx*rb->obs_dim, sizeof(float)*rb->obs_dim);
        memcpy(pi_batch  + i*rb->action_count, rb->pi_buf + idx*rb->action_count, sizeof(float)*rb->action_count);
        z_batch[i] = rb->z_buf[idx];
    }
    return actual;
}

size_t rb_size(ReplayBuffer *rb) { return rb ? rb->size : 0; }

