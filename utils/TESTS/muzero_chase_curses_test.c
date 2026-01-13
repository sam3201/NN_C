#include "../NN/MUZE/game_env.h"
#include "../NN/MUZE/game_replay.h"
#include "../NN/MUZE/mcts.h"
#include "../NN/MUZE/muze_config.h"
#include "../NN/MUZE/muzero_model.h"
#include "../NN/MUZE/replay_buffer.h"
#include "../NN/MUZE/runtime.h"
#include "../NN/MUZE/trainer.h"
#include "../NN/NN.h"

#include <curses.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
  uint32_t s;
} XorShift32;

typedef struct {
  int w;
  int h;

  int px;
  int py;

  int ex;
  int ey;

  int gx;
  int gy;

  int step;
  int max_steps;

  GameState gs;
  XorShift32 rng;
} ChaseEnv;

static int iabs_i(int x) { return (x < 0) ? -x : x; }

static int clamp_i(int x, int lo, int hi) {
  if (x < lo)
    return lo;
  if (x > hi)
    return hi;
  return x;
}

static int idx_of(int x, int y, int w) { return y * w + x; }

static void encode_obs(float *obs, int w, int h, int px, int py, int ex, int ey,
                       int gx, int gy) {
  int n = w * h;
  memset(obs, 0, sizeof(float) * (size_t)n);
  // Single-grid encoding:
  //   0 = empty
  //   1 = self/player
  //   2 = goal
  //   3 = enemy
  obs[idx_of(px, py, w)] = 1.0f;
  obs[idx_of(gx, gy, w)] = 2.0f;
  obs[idx_of(ex, ey, w)] = 3.0f;
}

static void apply_action(int w, int h, int action, int *x, int *y) {
  int nx = *x;
  int ny = *y;
  switch (action) {
  case 0:
    ny -= 1;
    break;
  case 1:
    ny += 1;
    break;
  case 2:
    nx -= 1;
    break;
  case 3:
    nx += 1;
    break;
  default:
    break;
  }
  nx = clamp_i(nx, 0, w - 1);
  ny = clamp_i(ny, 0, h - 1);
  *x = nx;
  *y = ny;
}

static void enemy_chase_step(int w, int h, int px, int py, int *ex, int *ey) {
  int dx = px - *ex;
  int dy = py - *ey;

  int ax = iabs_i(dx);
  int ay = iabs_i(dy);

  int nx = *ex;
  int ny = *ey;

  if (ax >= ay) {
    if (dx > 0)
      nx++;
    else if (dx < 0)
      nx--;
    else if (dy > 0)
      ny++;
    else if (dy < 0)
      ny--;
  } else {
    if (dy > 0)
      ny++;
    else if (dy < 0)
      ny--;
    else if (dx > 0)
      nx++;
    else if (dx < 0)
      nx--;
  }

  nx = clamp_i(nx, 0, w - 1);
  ny = clamp_i(ny, 0, h - 1);
  *ex = nx;
  *ey = ny;
}

static float rng01(void *ctx) {
  XorShift32 *r = (XorShift32 *)ctx;
  uint32_t x = r->s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  r->s = x;
  return (float)((double)x / (double)UINT32_MAX);
}

static int rng_int(XorShift32 *r, int n) {
  if (n <= 0)
    return 0;
  uint32_t x = r->s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  r->s = x;
  return (int)(x % (uint32_t)n);
}

static void chase_env_reset(ChaseEnv *E, XorShift32 *rng) {
  E->px = 0;
  E->py = 0;

  // Reset goal EVERY episode (randomized). Ensure no overlap.
  do {
    E->gx = rng_int(rng, E->w);
    E->gy = rng_int(rng, E->h);
  } while ((E->gx == E->px && E->gy == E->py));

  // Enemy starts random too, not overlapping player or goal.
  do {
    E->ex = rng_int(rng, E->w);
    E->ey = rng_int(rng, E->h);
  } while ((E->ex == E->px && E->ey == E->py) ||
           (E->ex == E->gx && E->ey == E->gy));

  E->step = 0;
  encode_obs(E->gs.obs, E->w, E->h, E->px, E->py, E->ex, E->ey, E->gx, E->gy);
}

static int chase_env_step(ChaseEnv *E, int action, float *reward_out,
                          int *done_out) {
  if (!E || !reward_out || !done_out)
    return -1;

  apply_action(E->w, E->h, action, &E->px, &E->py);

  int done = 0;
  float r = -0.01f;

  if (E->px == E->gx && E->py == E->gy) {
    done = 1;
    r = 1.0f;
  } else {
    enemy_chase_step(E->w, E->h, E->px, E->py, &E->ex, &E->ey);
    if (E->px == E->ex && E->py == E->ey) {
      done = 1;
      r = -1.0f;
    }
  }

  encode_obs(E->gs.obs, E->w, E->h, E->px, E->py, E->ex, E->ey, E->gx, E->gy);

  E->step++;
  if (E->max_steps > 0 && E->step >= E->max_steps)
    done = 1;

  *reward_out = r;
  *done_out = done;
  return 0;
}

static void chase_env_reset_fn(void *state, float *obs_out) {
  ChaseEnv *E = (ChaseEnv *)state;
  if (!E || !obs_out)
    return;
  chase_env_reset(E, &E->rng);
  memcpy(obs_out, E->gs.obs, sizeof(float) * (size_t)E->gs.obs_size);
}

static int chase_env_step_fn(void *state, int action, float *obs_out,
                             float *reward_out, int *done_out) {
  ChaseEnv *E = (ChaseEnv *)state;
  if (!E || !obs_out)
    return -1;
  int ret = chase_env_step(E, action, reward_out, done_out);
  if (ret == 0)
    memcpy(obs_out, E->gs.obs, sizeof(float) * (size_t)E->gs.obs_size);
  return ret;
}

static void *chase_env_clone(void *state) {
  ChaseEnv *src = (ChaseEnv *)state;
  if (!src)
    return NULL;
  ChaseEnv *dst = (ChaseEnv *)calloc(1, sizeof(ChaseEnv));
  if (!dst)
    return NULL;
  *dst = *src;
  if (!game_state_init(&dst->gs, src->gs.obs_size)) {
    free(dst);
    return NULL;
  }
  memcpy(dst->gs.obs, src->gs.obs, sizeof(float) * src->gs.obs_size);
  dst->rng.s = (uint32_t)rand();
  return dst;
}

static void chase_env_destroy(void *state) {
  ChaseEnv *E = (ChaseEnv *)state;
  if (!E)
    return;
  game_state_destroy(&E->gs);
  free(E);
}

int main(void) {
  int headless = 0;

  if (getenv("MUZE_HEADLESS"))
    headless = 1;

  if (!headless) {
    initscr();
    cbreak();
    noecho();
    curs_set(0);
    keypad(stdscr, TRUE);
    nodelay(stdscr, TRUE);
  }

  const int w = 30;
  const int h = 20;

  ChaseEnv E;
  memset(&E, 0, sizeof(E));
  E.w = w;
  E.h = h;
  E.max_steps = 200;

  int obs_dim = w * h;
  if (!game_state_init(&E.gs, (size_t)obs_dim)) {
    if (!headless)
      endwin();
    return 1;
  }

  uint32_t seed = 0x12345678u;
  const char *seed_env = getenv("MUZE_SEED");
  if (seed_env && *seed_env) {
    seed = (uint32_t)strtoul(seed_env, NULL, 10);
  }
  muze_seed(seed);
  E.rng.s = seed;
  MCTSRng rng = {.ctx = &E.rng, .rand01 = rng01};

  MuzeConfig cfg = {
      .model =
          (MuConfig){
              .obs_dim = obs_dim,
              .latent_dim = 64,
              .action_count = 5,
          },
      .nn =
          (MuNNConfig){
              .opt_repr = ADAM,
              .opt_dyn = ADAM,
              .opt_pred = ADAM,
              .opt_vprefix = ADAM,
              .opt_reward = ADAM,
              .loss_repr = MSE,
              .loss_dyn = MSE,
              .loss_pred = MSE,
              .loss_vprefix = MSE,
              .loss_reward = MSE,
              .lossd_repr = MSE_DERIVATIVE,
              .lossd_dyn = MSE_DERIVATIVE,
              .lossd_pred = MSE_DERIVATIVE,
              .lossd_vprefix = MSE_DERIVATIVE,
              .lossd_reward = MSE_DERIVATIVE,
              .lr_repr = 0.001L,
              .lr_dyn = 0.001L,
              .lr_pred = 0.001L,
              .lr_vprefix = 0.001L,
              .lr_reward = 0.001L,
              .lr_mult_repr_start = 1.0L,
              .lr_mult_repr_end = 0.2L,
              .lr_mult_repr_steps = 2000,
              .lr_mult_dyn_start = 1.0L,
              .lr_mult_dyn_end = 0.2L,
              .lr_mult_dyn_steps = 2000,
              .lr_mult_pred_start = 1.0L,
              .lr_mult_pred_end = 0.2L,
              .lr_mult_pred_steps = 2000,
              .lr_mult_vprefix_start = 1.0L,
              .lr_mult_vprefix_end = 0.2L,
              .lr_mult_vprefix_steps = 2000,
              .lr_mult_reward_start = 1.0L,
              .lr_mult_reward_end = 0.2L,
              .lr_mult_reward_steps = 2000,
              .hidden_repr = 128,
              .hidden_dyn = 128,
              .hidden_pred = 128,
              .hidden_vprefix = 128,
              .hidden_reward = 128,
              .use_value_support = 1,
              .use_reward_support = 1,
              .support_size = 21,
              .support_min = -2.0f,
              .support_max = 2.0f,
              .action_embed_dim = 64,
              .w_policy = 1.0f,
              .w_value = 1.0f,
              .w_vprefix = 1.0f,
              .w_latent = 1.0f,
              .w_reward = 1.0f,
              .grad_clip = 5.0f,
              .global_grad_clip = 1.0f,
          },
      .mcts =
          (MCTSParams){
              .num_simulations = 100,
              .batch_simulations = 8,
              .c_puct = 1.25f,
              .max_depth = 24,
              .dirichlet_alpha = 0.3f,
              .dirichlet_eps = 0.25f,
              .temperature = 1.0f,
              .discount = 0.99f,
          },
      .selfplay =
          (SelfPlayParams){
              .max_steps = E.max_steps,
              .gamma = 0.99f,
              .temp_start = 1.0f,
              .temp_end = 0.25f,
              .temp_decay_episodes = 200,
              .dirichlet_alpha = 0.3f,
              .dirichlet_eps = 0.25f,
              .total_episodes = 0,
              .log_every = 10,
          },
      .trainer =
          (TrainerConfig){
              .batch_size = 32,
              .train_steps = 250,
              .min_replay_size = 256,
              .reward_target_is_vprefix = 1,
              .lr = 0.05f,
          },
      .loop =
          (MuLoopConfig){
              .iterations = 200,
              .selfplay_episodes_per_iter = 2,
              .train_calls_per_iter = 1,
              .train_cfg = {0},
              .use_reanalyze = 1,
              .reanalyze_samples_per_iter = 256,
              .reanalyze_gamma = 0.99f,
              .reanalyze_full_games = 1,
              .eval_interval = 20,
              .eval_episodes = 20,
              .eval_max_steps = E.max_steps,
              .checkpoint_interval = 50,
              .checkpoint_prefix = "logs/chase_ckpt",
              .checkpoint_save_replay = 1,
              .checkpoint_save_games = 1,
              .checkpoint_keep_last = 3,
              .selfplay_actor_count = 2,
              .selfplay_use_threads = 1,
              .reanalyze_interval = 1,
              .reanalyze_fraction = 0.0f,
              .reanalyze_min_replay = 256,
              .replay_shard_interval = 50,
              .replay_shard_keep_last = 3,
              .replay_shard_max_entries = 4096,
              .replay_shard_prefix = "logs/chase_shard",
              .replay_shard_save_games = 1,
              .eval_best_model = 1,
              .best_checkpoint_prefix = "logs/chase_best",
              .best_save_replay = 1,
              .best_save_games = 1,
          },
      .actors =
          (MuzeActorConfig){
              .actor_count = 2,
              .use_threads = 1,
          },
      .replay_shards =
          (MuzeReplayShardConfig){
              .shard_interval = 50,
              .shard_keep_last = 3,
              .shard_max_entries = 4096,
              .shard_prefix = "logs/chase_shard",
              .shard_save_games = 1,
          },
      .reanalyze =
          (MuzeReanalyzeSchedule){
              .interval = 1,
              .fraction = 0.0f,
              .min_replay_size = 256,
          },
      .best =
          (MuzeBestModelConfig){
              .eval_best_model = 1,
              .best_checkpoint_prefix = "logs/chase_best",
              .best_save_replay = 1,
              .best_save_games = 1,
          },
      .seed =
          (MuzeSeedConfig){
              .seed = seed,
          },
  };

  cfg.loop.train_cfg = cfg.trainer;

  MuModel *model = mu_model_create_nn_with_cfg(&cfg.model, &cfg.nn);
  if (!model) {
    game_state_destroy(&E.gs);
    if (!headless)
      endwin();
    return 1;
  }

  ReplayBuffer *rb = rb_create(8192, cfg.model.obs_dim, cfg.model.action_count);
  if (rb && cfg.nn.use_value_support && cfg.nn.support_size > 1)
    rb_enable_value_support(rb, cfg.nn.support_size);
  GameReplay *gr =
      gr_create(256, E.max_steps, cfg.model.obs_dim, cfg.model.action_count);
  if (!rb || !gr) {
    if (rb)
      rb_free(rb);
    if (gr)
      gr_free(gr);
    mu_model_free(model);
    game_state_destroy(&E.gs);
    if (!headless)
      endwin();
    return 1;
  }

  if (!headless)
    clear();
  muze_run_loop_multi(model, &E, chase_env_reset_fn, chase_env_step_fn,
                      chase_env_clone, chase_env_destroy, rb, gr, &cfg.mcts,
                      &cfg.selfplay, &cfg.loop, &rng);
  rb_free(rb);
  gr_free(gr);
  mu_model_free(model);
  game_state_destroy(&E.gs);
  if (!headless)
    endwin();
  return 0;
}
