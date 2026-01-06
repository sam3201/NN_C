#include "../../SAM/SAM.h"
#include "../NN/MUZE/all.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_obs(const float *obs, int n) {
  printf("[");
  for (int i = 0; i < n; i++) {
    printf("%.0f", obs[i]);
    if (i + 1 < n)
      printf(" ");
  }
  printf("]");
}

int main(void) {
  srand(1);

  ToyEnvState env = {.pos = 0, .size = 8};
  size_t obs_dim = env.size;
  const int action_count = 2;
  const int goal_pos = env.size - 1;

  SAM_t *sam = SAM_init((size_t)obs_dim, (size_t)action_count, 2, 0);
  if (!sam) {
    printf("SAM_init failed\n");
    return 1;
  }

  MuCortex *cortex = SAM_as_MUZE(sam);
  if (!cortex) {
    printf("SAM_as_MUZE failed\n");
    SAM_destroy(sam);
    return 1;
  }

  const int episodes = 50;
  const int max_steps = 128;

  for (int ep = 0; ep < episodes; ep++) {
    float obs[obs_dim];
    toy_env_reset(&env, obs);

    float ep_return = 0.0f;

    for (int step = 0; step < max_steps; step++) {
      int action =
          muze_plan(cortex, obs, (size_t)obs_dim, (size_t)action_count);

      float next_obs[obs_dim];
      float env_reward = 0.0f;
      int env_done = 0;

      if (toy_env_step(&env, action, next_obs, &env_reward, &env_done) != 0) {
        printf("env_step error\n");
        break;
      }

      float reward = (env.pos == goal_pos) ? 1.0f : 0.0f;
      int done = (env.pos == goal_pos) ? 1 : 0;

      cortex->learn(cortex->brain, obs, (size_t)obs_dim, action, reward, done);

      ep_return += reward;
      memcpy(obs, next_obs, sizeof(obs));

      if (ep < 3 || done) {
        printf("ep=%d step=%d action=%d reward=%.1f pos=%d done=%d obs=", ep,
               step, action, reward, env.pos, done);
        print_obs(obs, obs_dim);
        printf("\n");
      }

      if (done)
        break;
    }

    printf("episode %d return=%.1f final_pos=%d\n", ep, ep_return, env.pos);
  }

  SAM_MUZE_destroy(cortex);
  SAM_destroy(sam);
  return 0;
}
