#include "muze_loop.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------- helpers -------- */

static float clampf(float x, float lo, float hi) {
  if (x < lo)
    return lo;
  if (x > hi)
    return hi;
  return x;
}

static void normalize_probs(float *p, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    if (!isfinite(p[i]) || p[i] < 0.0f)
      p[i] = 0.0f;
    sum += p[i];
  }
  if (sum <= 1e-12f) {
    float u = (n > 0) ? (1.0f / (float)n) : 0.0f;
    for (int i = 0; i < n; i++)
      p[i] = u;
    return;
  }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++)
    p[i] *= inv;
}

/*
  Reanalyze a single observation:
    - run MCTS with latest model
    - set pi_target = visit-count policy
    - set z_target = discounted return estimate:
        Here we can only do a lightweight version because ReplayBuffer stores
        per-step z already (from selfplay). So we refresh pi; optionally we can
        keep z as-is.
    - If you want true MuZero bootstrap targets, see notes below.
*/
static void reanalyze_one(MuModel *model, const float *obs, float *pi_out,
                          float *out_root_v, const MCTSParams *mp,
                          MCTSRng *rng) {

  MCTSResult mr = mcts_run(model, obs, mp, rng);
  memcpy(pi_out, mr.pi, sizeof(float) * (size_t)mr.action_count);
  normalize_probs(pi_out, mr.action_count);

  if (out_root_v)
    *out_root_v = mr.root_value;

  mcts_result_free(&mr);
}

/*
  Because your ReplayBuffer is a ring buffer of *independent* samples,
  we reanalyze by:
    - sampling a random batch of observations from rb->obs_buf
    - running MCTS on each
    - overwriting rb->pi_buf for those slots (and optionally rb->z_buf)
*/
static void reanalyze_replay(MuModel *model, ReplayBuffer *rb,
                             const MCTSParams *mp, int samples, MCTSRng *rng) {

  if (!model || !rb || rb->size == 0 || samples <= 0)
    return;

  const int O = rb->obs_dim;
  const int A = rb->action_count;

  float *pi_tmp = (float *)malloc(sizeof(float) * (size_t)A);
  if (!pi_tmp)
    return;

  double avg_v = 0.0;
  int cnt = 0;

  // random indices in [0, rb->size)
  for (int s = 0; s < samples; s++) {
    int idx;
    if (rng && rng->rand01) {
      idx = (int)(rng->rand01(rng->ctx) * (float)rb->size);
    } else {
      idx = (int)((double)rand() / ((double)RAND_MAX + 1.0) * (int)rb->size);
    }
    if (idx < 0)
      idx = 0;
    if ((size_t)idx >= rb->size)
      idx = (int)rb->size - 1;

    float root_v = 0.0f;
    const float *obs = rb->obs_buf + (size_t)idx * (size_t)O;

    reanalyze_one(model, obs, pi_tmp, &root_v, mp, rng);

    // overwrite stored pi target
    memcpy(rb->pi_buf + (size_t)idx * (size_t)A, pi_tmp,
           sizeof(float) * (size_t)A);

    avg_v += root_v;
    cnt++;
  }

  if (cnt > 0) {
    avg_v /= (double)cnt;
    printf("[reanalyze] samples=%d avg_root_v=%.4f\n", cnt, (float)avg_v);
  }

  free(pi_tmp);
}

/* -------- public loop -------- */

void muze_run_loop(MuModel *model, void *env_state,
                   selfplay_env_reset_fn env_reset,
                   selfplay_env_step_fn env_step, ReplayBuffer *rb,
                   const MCTSParams *base_mcts_params,
                   const SelfPlayParams *base_sp_params,
                   const MuLoopConfig *loop_cfg, MCTSRng *rng) {

  if (!model || !rb || !env_reset || !env_step || !base_mcts_params ||
      !base_sp_params || !loop_cfg)
    return;

  MCTSParams mp0 = *base_mcts_params;
  SelfPlayParams sp0 = *base_sp_params;

  int iters = loop_cfg->iterations > 0 ? loop_cfg->iterations : 1;
  int eps_per = loop_cfg->selfplay_episodes_per_iter > 0
                    ? loop_cfg->selfplay_episodes_per_iter
                    : 10;
  int train_calls =
      loop_cfg->train_calls_per_iter > 0 ? loop_cfg->train_calls_per_iter : 1;

  TrainerConfig tc = loop_cfg->train_cfg;

  // If user didn’t set these, keep sane defaults
  if (tc.batch_size <= 0)
    tc.batch_size = 32;
  if (tc.train_steps <= 0)
    tc.train_steps = 200;
  if (tc.min_replay_size <= 0)
    tc.min_replay_size = 1024;
  if (tc.lr <= 0.0f)
    tc.lr = 0.05f;

  for (int it = 0; it < iters; it++) {
    // ---- self-play ----
    SelfPlayParams sp = sp0;
    sp.total_episodes = eps_per;

    printf("\n=== [loop] iter=%d/%d selfplay_episodes=%d ===\n", it + 1, iters,
           eps_per);
    selfplay_run(model, env_state, env_reset, env_step, &mp0, &sp, rb, rng);

    // ---- reanalyze (optional) ----
    if (loop_cfg->use_reanalyze) {
      int samples = loop_cfg->reanalyze_samples_per_iter;
      if (samples <= 0)
        samples = 256;

      // Use a “training-time” MCTS setup: usually no root dirichlet during
      // reanalyze, and lower temperature. But you can keep it.
      MCTSParams mp = mp0;
      mp.dirichlet_alpha = 0.0f;
      mp.dirichlet_eps = 0.0f;
      mp.temperature =
          1.0f; // store full visit distribution; don’t harden too much

      reanalyze_replay(model, rb, &mp, samples, rng);
    }

    // ---- training ----
    printf("=== [loop] train_calls=%d ===\n", train_calls);
    for (int k = 0; k < train_calls; k++) {
      trainer_train_from_replay(model, rb, &tc);
      trainer_train_dynamics(model, rb, &tc);
    }

    printf("=== [loop] iter=%d done replay=%zu ===\n", it + 1, rb_size(rb));
  }
}
