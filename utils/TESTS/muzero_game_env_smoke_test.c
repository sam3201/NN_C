// utils/TESTS/muzero_game_env_smoke_test.c
//
// Smoke/integration test for full (non-toy) MuZero stack using game_env.
// This is meant to:
//   - step a real env (game_env) for a few episodes with random actions
//   - store transitions into replay buffer
//   - run a tiny training loop (a few gradient steps)
//   - sanity-check for NaNs / crashes / shape mismatches
//
// You MUST fill in the ADAPTER SECTION to match your actual MUZE env API.

#include <assert.h>
#include <curses.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Pull in MUZE
#include "../NN/MUZE/all.h"      // adjust include path if needed
#include "../NN/MUZE/game_env.h" // real env header

// ===========================
// ADAPTER SECTION (EDIT THIS)
// ===========================
//
// The idea: normalize your env API into the small set of calls this test uses.
// If your env API already matches, you can delete adapter wrappers and call
// directly.

#ifndef MUZE_TEST_MAX_OBS
#define MUZE_TEST_MAX_OBS 4096
#endif

#ifndef MUZE_TEST_MAX_ACTIONS
#define MUZE_TEST_MAX_ACTIONS 64
#endif

// ---- Types you likely already have ----
// Replace these with your actual types if they differ.
typedef struct {
  float *data;
  int len;
} ObsVec;

// A single step outcome (replace with your actual step result type).
typedef struct {
  float reward;
  int done;
} StepOut;

// ---- Adapter: env handle type ----
// If your game_env is a struct, use that type here.
// If it’s an opaque pointer, use void*.
typedef GameEnv Env; // <-- CHANGE if your env type is named differently

// ---- Adapter functions (map to your real ones) ----

// Create an environment
static Env *env_create(unsigned int seed) {
  // CHANGE these to your actual constructors:
  // Examples you might have:
  //   Env *e = game_env_create(seed);
  //   Env *e = GAME_ENV_init(seed);
  //   Env *e = game_env_init_default(seed);
  Env *e = {0};
  e = Game_env_init(e, seed);
  return e;
}

// Destroy/free environment
static void env_destroy(Env *e) {
  // e.g. game_env_destroy(e);
  game_env_destroy(e); // <-- EDIT
}

// Reset environment and return initial observation into out_obs.
// Return observation length.
static int env_reset(Env *e, float *out_obs, int max_obs) {
  // Examples:
  //   int n = game_env_reset(e, out_obs, max_obs);
  //   Obs o = game_env_reset(e); memcpy(out_obs, o.data, ...)

  int n = game_env_reset(e, out_obs, max_obs); // <-- EDIT
  return n;
}

// Step environment with action. Writes next obs into out_obs.
// Returns (reward, done) and obs length through *out_nobs.
static StepOut env_step(Env *e, int action, float *out_obs, int max_obs,
                        int *out_nobs) {
  StepOut r = {0};

  // Examples:
  //   r.reward = game_env_step(e, action, out_obs, max_obs, out_nobs, &r.done);
  //   StepResult sr = game_env_step(e, action); ...
  r.reward = 0.0f;
  r.done = 0;

  // A common pattern:
  //   float reward; int done; int nobs;
  //   reward = game_env_step(e, action, out_obs, max_obs, &nobs, &done);
  //   r.reward = reward; r.done = done; *out_nobs = nobs;

  float reward = 0.0f;
  int done = 0;
  int nobs = 0;

  reward = game_env_step(e, action, out_obs, max_obs, &nobs, &done); // <-- EDIT
  r.reward = reward;
  r.done = done;
  *out_nobs = nobs;

  return r;
}

// Number of discrete actions in this env
static int env_num_actions(Env *e) {
  // e.g. return game_env_action_space(e);
  return game_env_num_actions(e); // <-- EDIT
}

// ===========================
// END ADAPTER SECTION
// ===========================

// ---------------------------
// Small helpers
// ---------------------------
static int rand_int(int lo, int hi_inclusive) {
  return lo + (rand() % (hi_inclusive - lo + 1));
}

static int is_finite_array(const float *x, int n) {
  for (int i = 0; i < n; i++) {
    if (!isfinite((double)x[i]))
      return 0;
  }
  return 1;
}

static void banner(const char *msg) {
  printf("\n====================\n%s\n====================\n", msg);
}

// ---------------------------
// MAIN TEST
// ---------------------------
int main(void) {
  srand((unsigned)time(NULL));

  banner("MUZE non-toy MuZero smoke test (game_env)");

  // ---- Hyperparameters (small, fast smoke) ----
  const int episodes = 3;
  const int max_steps_per_episode = 200;

  const int train_steps = 25; // small sanity training
  const int batch_size = 16;

  // ---- Create env ----
  unsigned int seed = (unsigned int)time(NULL);
  Env *env = env_create(seed);
  if (!env) {
    fprintf(stderr, "ERROR: env_create failed\n");
    return 1;
  }

  const int action_n = env_num_actions(env);
  if (action_n <= 1 || action_n > MUZE_TEST_MAX_ACTIONS) {
    fprintf(stderr, "ERROR: suspicious action count: %d\n", action_n);
    env_destroy(env);
    return 1;
  }
  printf("Action space size: %d\n", action_n);

  // ---- Build MuZero model/config ----
  //
  // NOTE: These names may differ in your codebase.
  // The intention: create model + replay buffer + trainer/runtime.

  // If you already have a single "runtime" object, prefer that.
  // Otherwise use your existing components.

  // --- Replay Buffer ---
  replay_buffer_t *rb =
      replay_buffer_create(/*capacity=*/50000); // <-- EDIT if name differs
  if (!rb) {
    fprintf(stderr, "ERROR: replay_buffer_create failed\n");
    env_destroy(env);
    return 1;
  }

  // --- MuZero Model ---
  // You likely have a config struct; keep it minimal for smoke test.
  muzero_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));

  // Fill reasonable defaults; EDIT these fields to match your struct.
  // Common fields:
  // cfg.action_space_size = action_n;
  // cfg.observation_size  = ??? (if fixed); else model reads obs length
  // dynamically. cfg.num_unroll_steps  = ... cfg.discount          = 0.997f;
  // cfg.hidden_dim        = 128;
  // cfg.learning_rate     = 1e-3f;

  cfg.action_space_size = action_n; // <-- EDIT if field differs
  cfg.discount = 0.997f;            // <-- EDIT if field differs
  cfg.num_unroll_steps = 5;         // <-- EDIT if field differs
  cfg.hidden_dim = 128;             // <-- EDIT if field differs
  cfg.learning_rate = 1e-3f;        // <-- EDIT if field differs

  muzero_model_t *model = muzero_model_create(&cfg); // <-- EDIT if name differs
  if (!model) {
    fprintf(stderr, "ERROR: muzero_model_create failed\n");
    replay_buffer_destroy(rb);
    env_destroy(env);
    return 1;
  }

  // --- Trainer ---
  trainer_t *trainer = trainer_create(model, rb); // <-- EDIT if name differs
  if (!trainer) {
    fprintf(stderr, "ERROR: trainer_create failed\n");
    muzero_model_destroy(model);
    replay_buffer_destroy(rb);
    env_destroy(env);
    return 1;
  }

  // -------------------------
  // Collect experience
  // -------------------------
  banner("Collecting experience (random policy, real env)");

  float obs[MUZE_TEST_MAX_OBS];
  float next_obs[MUZE_TEST_MAX_OBS];

  long total_steps = 0;

  for (int ep = 0; ep < episodes; ep++) {
    int obs_n = env_reset(env, obs, MUZE_TEST_MAX_OBS);
    if (obs_n <= 0 || obs_n > MUZE_TEST_MAX_OBS) {
      fprintf(stderr, "ERROR: env_reset returned obs_n=%d\n", obs_n);
      break;
    }
    if (!is_finite_array(obs, obs_n)) {
      fprintf(stderr, "ERROR: env_reset produced non-finite obs\n");
      break;
    }

    float ep_return = 0.0f;

    for (int t = 0; t < max_steps_per_episode; t++) {
      int a = rand_int(0, action_n - 1);

      int next_n = 0;
      StepOut so = env_step(env, a, next_obs, MUZE_TEST_MAX_OBS, &next_n);

      if (next_n <= 0 || next_n > MUZE_TEST_MAX_OBS) {
        fprintf(stderr, "ERROR: env_step returned next_n=%d\n", next_n);
        so.done = 1;
      }
      if (!is_finite_array(next_obs, next_n)) {
        fprintf(stderr, "ERROR: env_step produced non-finite next_obs\n");
        so.done = 1;
      }
      if (!isfinite((double)so.reward)) {
        fprintf(stderr, "ERROR: env_step produced non-finite reward\n");
        so.done = 1;
      }

      // Push to replay buffer (EDIT to match your transition format)
      //
      // Typical fields:
      //  - observation (s)
      //  - action (a)
      //  - reward (r)
      //  - done (terminal)
      //  - next observation (s')
      //
      // If your buffer stores trajectories, you may want:
      //    replay_buffer_begin_episode(...)
      //    replay_buffer_add_step(...)
      //    replay_buffer_end_episode(...)
      //
      replay_buffer_add_transition(rb, obs, obs_n, a, so.reward, next_obs,
                                   next_n,
                                   so.done); // <-- EDIT if signature differs

      // Advance
      memcpy(obs, next_obs, (size_t)next_n * sizeof(float));
      obs_n = next_n;

      ep_return += so.reward;
      total_steps++;

      if (so.done)
        break;
    }

    printf("Episode %d return: %.3f\n", ep, ep_return);
  }

  printf("Collected total steps: %ld\n", total_steps);

  // -------------------------
  // Train a tiny amount
  // -------------------------
  banner("Training smoke (few steps)");

  for (int i = 0; i < train_steps; i++) {
    // If you have a function that samples internally:
    //   trainer_step(trainer, batch_size);
    // If you need to pass config, pass cfg.

    int ok =
        trainer_step(trainer, batch_size); // <-- EDIT if name/signature differs
    if (!ok) {
      fprintf(stderr, "ERROR: trainer_step failed at i=%d\n", i);
      break;
    }

    if ((i % 5) == 0) {
      // Optional: print loss stats if you have them
      // printf("step %d: loss=%f\n", i, trainer->last_loss);
      printf("train step %d ok\n", i);
    }
  }

  // -------------------------
  // Optional: quick inference sanity
  // -------------------------
  banner("Inference sanity");

  int obs_n = env_reset(env, obs, MUZE_TEST_MAX_OBS);
  if (obs_n > 0) {
    // Typical: model_initial_inference(model, obs, obs_n, &out)
    // Where out contains policy logits & value.
    muzero_infer_out_t out;
    memset(&out, 0, sizeof(out));

    int ok =
        muzero_model_initial_inference(model, obs, obs_n, &out); // <-- EDIT
    if (!ok) {
      fprintf(stderr, "WARN: initial_inference failed (check adapter)\n");
    } else {
      // Basic checks
      if (!isfinite((double)out.value)) {
        fprintf(stderr, "ERROR: value is non-finite\n");
      } else {
        printf("value: %f\n", (double)out.value);
      }
    }
  }

  // -------------------------
  // Cleanup
  // -------------------------
  banner("Cleanup");

  trainer_destroy(trainer);    // <-- EDIT if name differs
  muzero_model_destroy(model); // <-- EDIT if name differs
  replay_buffer_destroy(rb);   // <-- EDIT if name differs
  env_destroy(env);

  printf("DONE\n");
  return 0;
}
