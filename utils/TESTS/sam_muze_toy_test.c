#include "../../SAM/SAM.h"
#include "../NN/MUZE/all.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
  srand(1);

  // Toy env: 1D position one-hot observation of size N
  ToyEnvState env = {.pos = 0, .size = 8};
  const int obs_dim = env.size;
  const int action_count = 2; // left/right

  // SAM: model_dim must match obs_dim for now (since you feed obs directly)
  SAM_t *sam = SAM_init((size_t)obs_dim, (size_t)action_count, /*num_heads=*/2,
                        /*context_id=*/0);
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

  float obs[obs_dim];
  toy_env_reset(&env, obs);

  for (int step = 0; step < 256; step++) {
    if (step == 255)
      done = 1;
    int action = muze_plan(cortex, obs, (size_t)obs_dim, (size_t)action_count);

    float reward = 0.0f;
    int done = 0;
    float next_obs[obs_dim];

    if (toy_env_step(&env, action, next_obs, &reward, &done) != 0) {
      printf("env_step error\n");
      break;
    }

    /* ---------------------------
       Reward shaping + terminal
       Goal: reach env.size - 1
       --------------------------- */
    if (env.pos == (int)env.size - 1) {
      reward = 1.0f;
      done = 1;
    } else {
      // small penalty every step to encourage reaching the goal quickly
      reward = -0.01f;
    }
  }

  // Tell SAM/MUZE about reward + terminal
  cortex->learn(cortex->brain, reward, done);

  memcpy(obs, next_obs, sizeof(obs));

  printf("step=%d action=%d reward=%.3f pos=%d done=%d\n", step, action, reward,
         env.pos, done);

  if (done)
    break;
}

// cleanup
free(cortex->brain);
free(cortex);
SAM_destroy(sam);

return 0;
}
