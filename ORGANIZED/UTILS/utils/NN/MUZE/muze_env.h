#ifndef MUZE_ENV_H
#define MUZE_ENV_H

#include "self_play.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  void *state;
  selfplay_env_reset_fn reset;
  selfplay_env_step_fn step;
  selfplay_env_clone_fn clone;
  selfplay_env_destroy_fn destroy;
} MuzeEnv;

void muze_env_stub_reset(void *env_state, float *obs_out);
int muze_env_stub_step(void *env_state, int action, float *obs_out,
                       float *reward_out, int *done_out);
void *muze_env_stub_clone(void *env_state);
void muze_env_stub_destroy(void *env_state);

MuzeEnv muze_env_make_stub(void);

#ifdef __cplusplus
}
#endif

#endif
