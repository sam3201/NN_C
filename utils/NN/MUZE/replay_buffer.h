#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ReplayBuffer ReplayBuffer;

/* Create buffer. obs_dim = length of observation vector; action_count = length of pi vectors. */
ReplayBuffer *rb_create(size_t capacity, int obs_dim, int action_count);
void rb_free(ReplayBuffer *rb);

/* Push a single training sample: obs (len obs_dim), pi (len action_count), value z (scalar) */
void rb_push(ReplayBuffer *rb, const float *obs, const float *pi, float z);

/* Sample a minibatch uniformly. Caller provides preallocated arrays:
   obs_batch: [batch][obs_dim] flattened
   pi_batch:  [batch][action_count] flattened
   z_batch:   [batch] floats
   returns actual sampled size (<= batch) */
int rb_sample(ReplayBuffer *rb, int batch, float *obs_batch, float *pi_batch, float *z_batch);

/* Get size */
size_t rb_size(ReplayBuffer *rb);

#ifdef __cplusplus
}
#endif
#endif

