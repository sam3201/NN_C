#ifndef TOY_ENV_H
#define TOY_ENV_H

#include <stddef.h>

typedef struct {
  int pos;
  size_t size;
} ToyEnvState;

typedef void (*toy_env_reset_fn)(void *state, float *obs);
typedef int (*toy_env_step_fn)(void *state, int action, float *obs,
                               float *reward, int *done);

void toy_env_reset(void *state_ptr, float *obs);
int toy_env_step(void *state_ptr, int action, float *obs, float *reward,
                 int *done);

#endif
