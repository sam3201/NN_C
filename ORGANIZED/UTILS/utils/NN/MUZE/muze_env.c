#include "muze_env.h"
#include <string.h>

void muze_env_stub_reset(void *env_state, float *obs_out) {
  (void)env_state;
  if (obs_out)
    obs_out[0] = 0.0f;
}

int muze_env_stub_step(void *env_state, int action, float *obs_out,
                       float *reward_out, int *done_out) {
  (void)env_state;
  (void)action;
  if (obs_out)
    obs_out[0] = 0.0f;
  if (reward_out)
    *reward_out = 0.0f;
  if (done_out)
    *done_out = 1;
  return 0;
}

void *muze_env_stub_clone(void *env_state) { return env_state; }

void muze_env_stub_destroy(void *env_state) { (void)env_state; }

MuzeEnv muze_env_make_stub(void) {
  MuzeEnv env;
  memset(&env, 0, sizeof(env));
  env.reset = muze_env_stub_reset;
  env.step = muze_env_stub_step;
  env.clone = muze_env_stub_clone;
  env.destroy = muze_env_stub_destroy;
  env.state = NULL;
  return env;
}
