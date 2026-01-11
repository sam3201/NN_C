#include "../NN/MUZE/all.h"
#include <limits.h>
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
  MuConfig cfg = {
      .obs_dim = (int)env.toy.size,
      .latent_dim = 16,
      .action_count = 2,
  };
  MuModel *model = mu_model_create_nn(&cfg);

  // Replay buffer
  ReplayBuffer *rb = rb_create(4096, (int)env.toy.size, 2);
  GameReplay *gr = gr_create(128, 64, (int)env.toy.size, 2);

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
              .unroll_steps = 5,
              .bootstrap_steps = 5,
              .discount = 0.997f,
              .use_per = 1,
              .per_alpha = 0.6f,
              .per_beta = 0.4f,
              .per_beta_start = 0.4f,
              .per_beta_end = 1.0f,
              .per_beta_anneal_steps = 2000,
              .per_eps = 1e-3f,
              .train_reward_head = 0,
              .reward_target_is_vprefix = 1,
              .lr = 0.05f,
          },
      .use_reanalyze = 1,
      .reanalyze_samples_per_iter = 256,
      .reanalyze_gamma = 0.997f,
      .reanalyze_full_games = 1,
      .eval_interval = 0,
      .eval_episodes = 10,
      .eval_max_steps = 64,
      .checkpoint_interval = 0,
      .checkpoint_prefix = "logs/muzero_ckpt",
      .checkpoint_save_replay = 1,
      .checkpoint_save_games = 1,
  };

  // rng
  SimpleRng sr = {.s = 123456789u};
  MCTSRng rng = {.ctx = &sr, .rand01 = rng01};

  muze_run_loop(model, &env, env_reset_cb, env_step_cb, rb, gr, &mp, &sp, &lc,
                &rng);

  rb_free(rb);
  gr_free(gr);
  mu_model_free(model);
  return 0;
}
