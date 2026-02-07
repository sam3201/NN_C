#include "toy_env.h"
#include <string.h>

void toy_env_reset(void *state_ptr, float *obs) {
  ToyEnvState *state = (ToyEnvState *)state_ptr;
  state->pos = 0;
  memset(obs, 0, sizeof(float) * state->size);
  obs[state->pos] = 1.0f; // one-hot position
}

// changed return type from void -> int
int toy_env_step(void *state_ptr, int action, float *obs, float *reward,
                 int *done) {
  ToyEnvState *state = (ToyEnvState *)state_ptr;

  if (action == 0 && state->pos > 0)
    state->pos--;
  if (action == 1 && state->pos < state->size - 1)
    state->pos++;

  *reward = (state->pos == state->size - 1) ? 1.0f : 0.0f;
  *done = (state->pos == state->size - 1) ? 1 : 0;

  // update observation (one-hot)
  memset(obs, 0, sizeof(float) * state->size);
  obs[state->pos] = 1.0f;

  return 0; // success
}
