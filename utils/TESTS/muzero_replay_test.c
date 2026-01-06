#include "../NN/MUZE/all.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int argmax_policy_from_model(MuModel *m, const float *obs) {
  int A = m->cfg.action_count;
  int L = m->cfg.latent_dim;

  float *latent = (float *)calloc((size_t)L, sizeof(float));
  float *logits = (float *)calloc((size_t)A, sizeof(float));
  float value = 0.0f;

  mu_model_repr(m, obs, latent);
  mu_model_predict(m, latent, logits, &value);

  int best = 0;
  float bestv = logits[0];
  for (int a = 1; a < A; a++) {
    if (logits[a] > bestv) {
      bestv = logits[a];
      best = a;
    }
  }

  free(latent);
  free(logits);
  return best;
}

static float eval_success_rate(MuModel *m, int episodes, int max_steps) {
  ToyEnvState env = {.pos = 0, .size = 8};
  const int obs_dim = env.size;
  const int goal = env.size - 1;

  int success = 0;

  for (int ep = 0; ep < episodes; ep++) {
    float obs[obs_dim];
    toy_env_reset(&env, obs);

    int done = 0;
    for (int t = 0; t < max_steps; t++) {
      int a = argmax_policy_from_model(m, obs);

      float next_obs[obs_dim];
      float r = 0.0f;
      int d = 0;
      toy_env_step(&env, a, next_obs, &r, &d);

      memcpy(obs, next_obs, sizeof(obs));
      done = d;
      if (done)
        break;
    }

    if (env.pos == goal)
      success++;
  }

  return (float)success / (float)episodes;
}

int main(void) {
  srand(1);

  // ---- model config ----
  MuConfig cfg = {
      .obs_dim = 8,
      .latent_dim = 16,
      .action_count = 2,
  };

  MuModel *model = mu_model_create(&cfg);
  if (!model) {
    printf("mu_model_create failed\n");
    return 1;
  }

  // ---- replay ----
  ReplayBuffer *rb = rb_create(4096, cfg.obs_dim, cfg.action_count);
  if (!rb) {
    printf("rb_create failed\n");
    mu_model_free(model);
    return 1;
  }

  // ---- selfplay params ----
  MCTSParams mcts = {
      .num_simulations = 64,
      .c_puct = 1.25f,
      .max_depth = 32,
      .dirichlet_alpha = 0.3f,
      .dirichlet_eps = 0.25f,
      .temperature = 1.0f,
      .discount = 0.99f,
  };

  SelfPlayParams sp = {
      .max_steps = 128,
      .gamma = 0.99f,
      .temperature = 1.0f,
      .total_episodes = 200,
  };

  // ---- trainer params ----
  TrainerConfig tc = {
      .batch_size = 32,
      .train_steps = 1000,
      .min_replay_size = 128,
      .lr = 0.05f, // tune later
  };

  // ---- eval BEFORE ----
  float before = eval_success_rate(model, 100, 128);
  printf("Eval before training: success=%.2f\n", before);

  // ---- generate data ----
  ToyEnvState env = {.pos = 0, .size = 8};
  printf("Running selfplay to fill replay...\n");
  selfplay_run(model, &env, toy_env_reset, toy_env_step, &mcts, &sp, rb);
  printf("Replay size after selfplay: %zu\n", rb_size(rb));

  // ---- train ----
  printf("Training from replay...\n");
  time_t start = time(NULL);
  trainer_train_from_replay(model, rb, &tc);
  time_t end = time(NULL);
  printf("Training took %ld seconds\n", end - start);

  // ---- eval AFTER ----
  float after = eval_success_rate(model, 100, 128);
  printf("Eval after training:  success=%.2f\n", after);

  rb_free(rb);
  mu_model_free(model);
  return 0;
}
