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

static void draw_env(const ChaseEnv *E, int top, int left) {
  int w = E->w;
  int h = E->h;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      char ch = '.';
      if (x == E->gx && y == E->gy)
        ch = 'G';
      if (x == E->ex && y == E->ey)
        ch = 'E';
      if (x == E->px && y == E->py)
        ch = 'P';
      mvaddch(top + y, left + x, ch);
    }
  }
}

static int sample_from_probs_rng(const float *p, int n, MCTSRng *rng) {
  if (!p || n <= 0)
    return 0;

  float r = 0.0f;
  if (rng && rng->rand01)
    r = rng->rand01(rng->ctx);
  else
    r = (float)rand() / (float)RAND_MAX;

  float c = 0.0f;
  for (int i = 0; i < n; i++) {
    c += p[i];
    if (r <= c)
      return i;
  }
  return n - 1;
}

static int select_action_mcts(MuModel *model, const float *obs,
                              const MCTSParams *params, MCTSRng *rng,
                              float *pi_out) {
  if (!model || !obs || !params || !pi_out)
    return 0;

  MCTSResult mr = mcts_run(model, obs, params, rng);
  int action = sample_from_probs_rng(mr.pi, mr.action_count, rng);
  memcpy(pi_out, mr.pi, sizeof(float) * (size_t)mr.action_count);
  mcts_result_free(&mr);
  return action;
}

typedef struct {
  MuModel *train_model;
  MuModel *actor_model;
  ReplayBuffer *rb;
  GameReplay *gr;
  TrainerConfig tc;
  pthread_mutex_t *rb_mutex;
  pthread_mutex_t *model_mutex;
  pthread_mutex_t *gr_mutex;
  int snapshot_interval_ms;
  int *running;
} TrainThreadCtx;

static ReplayBuffer *rb_clone(const ReplayBuffer *src) {
  if (!src)
    return NULL;
  ReplayBuffer *dst = rb_create(src->capacity, src->obs_dim, src->action_count);
  if (!dst)
    return NULL;
  if (src->support_size > 1)
    rb_enable_value_support(dst, src->support_size);
  dst->size = src->size;
  dst->write_idx = src->write_idx;

  size_t obs_bytes = sizeof(float) * src->capacity * (size_t)src->obs_dim;
  size_t pi_bytes = sizeof(float) * src->capacity * (size_t)src->action_count;
  size_t z_bytes = sizeof(float) * src->capacity;
  size_t vprefix_bytes = sizeof(float) * src->capacity;
  size_t prio_bytes = sizeof(float) * src->capacity;
  size_t a_bytes = sizeof(int) * src->capacity;
  size_t r_bytes = sizeof(float) * src->capacity;
  size_t next_obs_bytes = sizeof(float) * src->capacity * (size_t)src->obs_dim;
  size_t done_bytes = sizeof(int) * src->capacity;

  memcpy(dst->obs_buf, src->obs_buf, obs_bytes);
  memcpy(dst->pi_buf, src->pi_buf, pi_bytes);
  memcpy(dst->z_buf, src->z_buf, z_bytes);
  memcpy(dst->vprefix_buf, src->vprefix_buf, vprefix_bytes);
  memcpy(dst->prio_buf, src->prio_buf, prio_bytes);
  if (src->value_dist_buf && dst->value_dist_buf) {
    size_t dist_bytes =
        sizeof(float) * src->capacity * (size_t)src->support_size;
    memcpy(dst->value_dist_buf, src->value_dist_buf, dist_bytes);
  }
  memcpy(dst->a_buf, src->a_buf, a_bytes);
  memcpy(dst->r_buf, src->r_buf, r_bytes);
  memcpy(dst->next_obs_buf, src->next_obs_buf, next_obs_bytes);
  memcpy(dst->done_buf, src->done_buf, done_bytes);

  return dst;
}

static GameReplay *gr_clone(const GameReplay *src) {
  if (!src)
    return NULL;
  GameReplay *dst = gr_create(src->max_games, src->max_steps, src->obs_dim,
                              src->action_count);
  if (!dst)
    return NULL;
  dst->game_count = src->game_count;
  dst->next_game = src->next_game;
  dst->cur_game = src->cur_game;
  dst->cur_step = src->cur_step;
  dst->in_episode = src->in_episode;

  size_t games = (size_t)src->max_games;
  size_t steps = (size_t)src->max_steps;
  size_t obs_bytes = sizeof(float) * games * steps * (size_t)src->obs_dim;
  size_t pi_bytes = sizeof(float) * games * steps * (size_t)src->action_count;
  size_t a_bytes = sizeof(int) * games * steps;
  size_t r_bytes = sizeof(float) * games * steps;
  size_t done_bytes = sizeof(int) * games * steps;
  size_t idx_bytes = sizeof(size_t) * games * steps;

  memcpy(dst->lengths, src->lengths, sizeof(int) * games);
  memcpy(dst->obs_buf, src->obs_buf, obs_bytes);
  memcpy(dst->pi_buf, src->pi_buf, pi_bytes);
  memcpy(dst->a_buf, src->a_buf, a_bytes);
  memcpy(dst->r_buf, src->r_buf, r_bytes);
  memcpy(dst->done_buf, src->done_buf, done_bytes);
  memcpy(dst->rb_idx_buf, src->rb_idx_buf, idx_bytes);

  return dst;
}

static void *train_thread_main(void *arg) {
  TrainThreadCtx *ctx = (TrainThreadCtx *)arg;
  if (!ctx || !ctx->train_model || !ctx->actor_model || !ctx->rb)
    return NULL;

  while (*ctx->running) {
    pthread_mutex_lock(ctx->rb_mutex);
    size_t n = rb_size(ctx->rb);
    pthread_mutex_unlock(ctx->rb_mutex);

    if ((int)n >= ctx->tc.min_replay_size) {
      if (ctx->snapshot_interval_ms > 0)
        usleep((useconds_t)ctx->snapshot_interval_ms * 1000u);

      pthread_mutex_lock(ctx->rb_mutex);
      ReplayBuffer *rb_snap = rb_clone(ctx->rb);
      pthread_mutex_unlock(ctx->rb_mutex);
      if (!rb_snap) {
        usleep(2000);
        continue;
      }

      GameReplay *gr_snap = NULL;
      if (ctx->gr && ctx->tc.unroll_steps > 0) {
        pthread_mutex_lock(ctx->gr_mutex);
        gr_snap = gr_clone(ctx->gr);
        pthread_mutex_unlock(ctx->gr_mutex);
        if (gr_snap)
          gr_save(gr_snap, "logs/gr_snapshot.bin");
      }

      if (!gr_snap && ctx->tc.unroll_steps > 0) {
        gr_snap = gr_load("logs/gr_snapshot.bin");
      }
      if (ctx->tc.unroll_steps > 0 && gr_snap) {
        trainer_train_from_replay_games(ctx->train_model, rb_snap, gr_snap,
                                        &ctx->tc);
      } else {
        trainer_train_from_replay(ctx->train_model, rb_snap, &ctx->tc);
      }

      if (ctx->tc.unroll_steps <= 0)
        trainer_train_dynamics(ctx->train_model, rb_snap, &ctx->tc);

      pthread_mutex_lock(ctx->model_mutex);
      mu_model_copy_weights(ctx->actor_model, ctx->train_model);
      pthread_mutex_unlock(ctx->model_mutex);

      rb_free(rb_snap);
      if (gr_snap)
        gr_free(gr_snap);
    }

    usleep(2000);
  }
  return NULL;
}

static float run_eval(ChaseEnv *E, MuModel *model, const MCTSParams *mp,
                      int episodes, int max_steps, float gamma, MCTSRng *rng,
                      float *out_win_rate, float *out_mean_steps) {
  if (!E || !model || !mp || episodes <= 0)
    return 0.0f;

  float sum_return = 0.0f;
  float sum_steps = 0.0f;
  int wins = 0;

  int A = model->cfg.action_count;
  float *pi = (float *)malloc(sizeof(float) * (size_t)A);
  if (!pi)
    return 0.0f;

  for (int ep = 0; ep < episodes; ep++) {
    chase_env_reset(E, (XorShift32 *)rng->ctx);
    float *obs = E->gs.obs;

    int done = 0;
    float ep_ret = 0.0f;
    int steps = 0;
    while (!done && steps < max_steps) {
      MCTSParams mp_eval = *mp;
      mp_eval.dirichlet_alpha = 0.0f;
      mp_eval.dirichlet_eps = 0.0f;
      mp_eval.temperature = 0.0f;
      int action = select_action_mcts(model, obs, &mp_eval, rng, pi);
      if (action < 0)
        action = 0;

      float r = 0.0f;
      if (chase_env_step(E, action, &r, &done) != 0)
        break;
      ep_ret += r;
      steps++;
      if (done && r > 0.0f)
        wins++;
    }

    sum_return += ep_ret;
    sum_steps += (float)steps;
  }

  free(pi);

  if (out_win_rate)
    *out_win_rate = (episodes > 0) ? ((float)wins / (float)episodes) : 0.0f;
  if (out_mean_steps)
    *out_mean_steps = (episodes > 0) ? (sum_steps / (float)episodes) : 0.0f;
  return (episodes > 0) ? (sum_return / (float)episodes) : 0.0f;
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
