#ifndef TOY_ENV_H
#define TOY_ENV_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int pos;
  int size;
} ToyEnvState;

typedef void (*env_reset_fn)(void *state, float *obs);
typedef void (*env_step_fn)(void *state, int action, float *obs, float *reward,
                            int *done);

// initialize/reset
void toy_env_reset(void *state, float *obs);

// step
int toy_env_step(void *state_ptr, int action, float *obs, float *reward,
                 int *done);
#endif
