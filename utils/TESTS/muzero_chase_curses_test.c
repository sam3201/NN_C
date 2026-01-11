#include "../NN/MUZE/game_env.h"
#include "../NN/MUZE/muzero_model.h"
#include "../NN/MUZE/replay_buffer.h"
#include "../NN/MUZE/runtime.h"
#include "../NN/MUZE/trainer.h"
#include "../NN/NN.h"

#include <curses.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

typedef struct {
  uint32_t s;
} XorShift32;

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

int main(void) {
  initscr();
  cbreak();
  noecho();
  curs_set(0);
  keypad(stdscr, TRUE);
  nodelay(stdscr, TRUE);

  const int w = 12;
  const int h = 10;

  ChaseEnv E;
  memset(&E, 0, sizeof(E));
  E.w = w;
  E.h = h;
  E.max_steps = 200;

  int obs_dim = w * h;
  if (!game_state_init(&E.gs, (size_t)obs_dim)) {
    endwin();
    return 1;
  }

  XorShift32 xr = {.s = 0x12345678u};
  MCTSRng rng = {.ctx = &xr, .rand01 = rng01};

  MuConfig cfg = {
      .obs_dim = obs_dim,
      .latent_dim = 64,
      .action_count = 5,
  };
  MuNNConfig nn_cfg = {
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
      .hidden_repr = 128,
      .hidden_dyn = 128,
      .hidden_pred = 128,
      .hidden_vprefix = 128,
      .hidden_reward = 128,
  };
  MuModel *model = mu_model_create_nn_with_cfg(&cfg, &nn_cfg);
  if (!model) {
    game_state_destroy(&E.gs);
    endwin();
    return 1;
  }

  ReplayBuffer *rb = rb_create(8192, cfg.obs_dim, cfg.action_count);
  if (!rb) {
    mu_model_free(model);
    game_state_destroy(&E.gs);
    endwin();
    return 1;
  }

  TrainerConfig tc = {
      .batch_size = 32,
      .train_steps = 250,
      .min_replay_size = 256,
      .reward_target_is_vprefix = 1,
      .lr = 0.05f,
  };

  MuCortex cortex;
  memset(&cortex, 0, sizeof(cortex));
  cortex.use_mcts = true;
  cortex.mcts_model = model;
  cortex.mcts_params.num_simulations = 100;
  cortex.mcts_params.c_puct = 1.25f;
  cortex.mcts_params.max_depth = 24;
  cortex.mcts_params.dirichlet_alpha = 0.3f;
  cortex.mcts_params.dirichlet_eps = 0.25f;
  cortex.mcts_params.temperature = 1.0f;
  cortex.mcts_params.discount = 0.99f;
  cortex.policy_temperature = 1.0f;
  cortex.policy_epsilon = 0.05f;

  int delay_ms = 40;

  const int episodes = 200;
  const float gamma = 0.99f;

  float *obs = (float *)malloc(sizeof(float) * (size_t)obs_dim);
  float *next_obs = (float *)malloc(sizeof(float) * (size_t)obs_dim);
  float *pi = (float *)malloc(sizeof(float) * (size_t)cfg.action_count);

  int *ep_actions = (int *)malloc(sizeof(int) * (size_t)E.max_steps);
  float *ep_rewards = (float *)malloc(sizeof(float) * (size_t)E.max_steps);
  size_t *ep_rb_idx = (size_t *)malloc(sizeof(size_t) * (size_t)E.max_steps);

  if (!obs || !next_obs || !pi || !ep_actions || !ep_rewards || !ep_rb_idx) {
    free(obs);
    free(next_obs);
    free(pi);
    free(ep_actions);
    free(ep_rewards);
    free(ep_rb_idx);
    rb_free(rb);
    mu_model_free(model);
    game_state_destroy(&E.gs);
    endwin();
    return 1;
  }

  for (int ep = 0; ep < episodes; ep++) {
    chase_env_reset(&E, &xr);
    memcpy(obs, E.gs.obs, sizeof(float) * (size_t)obs_dim);

    float ep_ret = 0.0f;
    int done = 0;
    int t = 0;

    while (!done && t < E.max_steps) {
      int ch = getch();
      if (ch == 'q' || ch == 'Q') {
        free(obs);
        free(next_obs);
        free(pi);
        free(ep_actions);
        free(ep_rewards);
        free(ep_rb_idx);
        rb_free(rb);
        mu_model_free(model);
        game_state_destroy(&E.gs);
        endwin();
        return 0;
      }
      if (ch == 'f' || ch == 'F')
        delay_ms = (delay_ms > 5) ? (delay_ms - 5) : delay_ms;
      if (ch == 's' || ch == 'S')
        delay_ms = (delay_ms < 500) ? (delay_ms + 5) : delay_ms;

      int action = muze_select_action(&cortex, obs, (size_t)obs_dim, pi,
                                      (size_t)cfg.action_count, &rng);
      if (action < 0)
        action = 0;

      float r = 0.0f;
      if (chase_env_step(&E, action, &r, &done) != 0)
        break;

      memcpy(next_obs, E.gs.obs, sizeof(float) * (size_t)obs_dim);

      size_t idx =
          rb_push_full(rb, obs, pi, /*z*/ r, action, r, next_obs, done);

      ep_actions[t] = action;
      ep_rewards[t] = r;
      ep_rb_idx[t] = idx;

      ep_ret += r;

      clear();
      mvprintw(0, 0, "MuZero chase (learned)  q=quit  f=faster  s=slower");
      mvprintw(1, 0,
               "ep=%d/%d step=%d/%d  action=%d  r=%.2f  return=%.2f  "
               "replay=%zu  delay=%dms",
               ep + 1, episodes, t + 1, E.max_steps, action, r, ep_ret,
               rb_size(rb), delay_ms);
      mvprintw(2, 0, "P=(%d,%d)  E=(%d,%d)  G=(%d,%d)", E.px, E.py, E.ex, E.ey,
               E.gx, E.gy);
      draw_env(&E, 4, 0);
      refresh();
      usleep((useconds_t)delay_ms * 1000u);

      memcpy(obs, next_obs, sizeof(float) * (size_t)obs_dim);
      t++;
    }

    float G = 0.0f;
    for (int i = t - 1; i >= 0; i--) {
      G = ep_rewards[i] + gamma * G;
      rb_set_z(rb, ep_rb_idx[i], G);
    }

    trainer_train_from_replay(model, rb, &tc);
    trainer_train_dynamics(model, rb, &tc);

    usleep(40 * 1000);
  }

  free(obs);
  free(next_obs);
  free(pi);
  free(ep_actions);
  free(ep_rewards);
  free(ep_rb_idx);

  rb_free(rb);
  mu_model_free(model);
  game_state_destroy(&E.gs);
  endwin();
  return 0;
}
