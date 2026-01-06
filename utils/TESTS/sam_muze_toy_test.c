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

  // Toy env: 1D position one-hot observation of size N
  ToyEnvState env = {.pos = 0, .size = 8};
  const int obs_dim = env.size;
  const int action_count = 2; // left/right

  // Goal: reach far right
  const int goal_pos = env.size - 1;

  // SAM: model_dim must match obs_dim for now
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

  const int episodes = 50;
  const int max_steps_per_ep = 128;

  for (int ep = 0; ep < episodes; ep++) {
    float obs[obs_dim];
    toy_env_reset(&env, obs);

    float ep_return = 0.0f;
    int done = 0;

    for (int step = 0; step < max_steps_per_ep; step++) {
      int action =
          muze_plan(cortex, obs, (size_t)obs_dim, (size_t)action_count);

      float next_obs[obs_dim];
      float reward_from_env = 0.0f;
      int done_from_env = 0;

      if (toy_env_step(&env, action, next_obs, &reward_from_env,
                       &done_from_env) != 0) {
        printf("env_step error\n");
        done = 1;
      }

      // ---------------------------
      // Sparse reward + terminal
      // Reward only when reaching goal
      // ---------------------------
      float reward = 0.0f;
      if (env.pos == goal_pos) {
        reward = 1.0f;
        done = 1;
      } else if (step == max_steps_per_ep - 1) {
        // timeout terminal (no reward)
        done = 1;
      } else {
        done = 0;
      }

      // Tell SAM/MUZE about reward + terminal
      cortex->learn(cortex->brain, reward, done);

      ep_return += reward;

      memcpy(obs, next_obs, sizeof(obs));

      // Debug print occasionally
      if ((ep < 5) || (ep % 10 == 0 && step < 20) || done) {
        printf("ep=%d step=%d action=%d reward=%.3f pos=%d done=%d obs=", ep,
               step, action, reward, env.pos, done);
        print_obs(obs, obs_dim);
        printf("\n");
      }

      if (done)
        break;
    }

    printf("episode %d return=%.3f final_pos=%d\n", ep, ep_return, env.pos);
  }

  // cleanup
  SAM_MUZE_destroy(cortex);
  SAM_destroy(sam);

  return 0;
}
