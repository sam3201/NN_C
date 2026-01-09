#include "../NN/MUZE/all.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

typedef struct {
  ToyEnvState toy;
} MyEnv;

static void env_reset_cb(void *st, float *obs_out) {
  MyEnv *E = (MyEnv *)st;
  toy_env_reset(&E->toy, obs_out);
}

static int env_step_cb(void *st, int action, float *obs_out, float *reward_out,
                       int *done_out) {
  MyEnv *E = (MyEnv *)st;
  return toy_env_step(&E->toy, action, obs_out, reward_out, done_out);
}

/* optional rng hook */
typedef struct {
  unsigned int s;
} SimpleRng;
static float rng01(void *ctx) {
  SimpleRng *r = (SimpleRng *)ctx;
  // xorshift32
  unsigned int x = r->s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  r->s = x;
  return (x / (float)UINT_MAX);
}

int main(void) {
  srand((unsigned int)time(NULL));

  // --- env ---
  MyEnv env;
  env.toy.size = 9;
  env.toy.pos = 0;

  // --- model ---
  // Use your trainable linear-ish model OR toy model.
  // For a true learnable run, use mu_model_create(cfg).
  // For sanity testing, you can start with the toy model:
  MuModel *model = mu_model_create_toy((int)env.toy.size, 2);

  // Replay buffer
  ReplayBuffer *rb = rb_create(4096, (int)env.toy.size, 2);

  // MCTS params
  MCTSParams mp = {
      .num_simulations = 50,
      .c_puct = 1.25f,
      .max_depth = 16,
      .dirichlet_alpha = 0.3f,
      .dirichlet_eps = 0.25f,
      .temperature = 1.0f,
      .discount = 0.997f,
  };

  // Self-play params
  SelfPlayParams sp = {
      .max_steps = 64,
      .gamma = 0.997f,
      .temp_start = 1.0f,
      .temp_end = 0.25f,
      .temp_decay_episodes = 200,
      .dirichlet_alpha = 0.3f,
      .dirichlet_eps = 0.25f,
      .total_episodes = 50,
      .log_every = 10,
  };

  // Loop config
  MuLoopConfig lc = {
      .iterations = 10,
      .selfplay_episodes_per_iter = 50,
      .train_calls_per_iter = 2,
      .train_cfg =
          (TrainerConfig){
              .batch_size = 32,
              .train_steps = 200,
              .min_replay_size = 256,
              .lr = 0.05f,
          },
      .use_reanalyze = 1,
      .reanalyze_samples_per_iter = 256,
      .reanalyze_gamma = 0.997f,
  };

  // rng
  SimpleRng sr = {.s = 123456789u};
  MCTSRng rng = {.ctx = &sr, .rand01 = rng01};

  muze_run_loop(model, &env, env_reset_cb, env_step_cb, rb, &mp, &sp, &lc,
                &rng);

  rb_free(rb);
  mu_model_free_toy(model);
  return 0;
}
