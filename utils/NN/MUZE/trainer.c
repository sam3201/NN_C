#include "trainer.h"
#include "muzero_model.h"
#include "replay_buffer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float mse(float a, float b) {
  float d = a - b;
  return d * d;
}

// pi_target: [A], p_pred: [A] probabilities (or softmaxed logits)
static float cross_entropy(const float *pi_target, const float *p_pred, int A) {
  float loss = 0.0f;
  const float eps = 1e-8f;
  for (int i = 0; i < A; i++) {
    float p = p_pred[i];
    if (p < eps)
      p = eps;
    loss -= pi_target[i] * logf(p);
  }
  return loss;
}

void trainer_train_from_replay(MuModel *model, ReplayBuffer *rb,
                               const TrainerConfig *cfg) {
  if (!model || !rb || !cfg)
    return;

  size_t n = rb_size(rb);
  if ((int)n < cfg->min_replay_size) {
    // not enough data yet
    return;
  }

  int B = cfg->batch_size;
  int obs_dim = muzero_model_obs_dim(model);
  int A = muzero_model_action_count(model);

  float *obs_batch = (float *)malloc(sizeof(float) * B * obs_dim);
  float *pi_batch = (float *)malloc(sizeof(float) * B * A);
  float *z_batch = (float *)malloc(sizeof(float) * B);

  // outputs
  float *p_pred = (float *)malloc(sizeof(float) * B * A);
  float *v_pred = (float *)malloc(sizeof(float) * B);

  if (!obs_batch || !pi_batch || !z_batch || !p_pred || !v_pred) {
    free(obs_batch);
    free(pi_batch);
    free(z_batch);
    free(p_pred);
    free(v_pred);
    return;
  }

  for (int step = 0; step < cfg->train_steps; step++) {
    int actual = rb_sample(rb, B, obs_batch, pi_batch, z_batch);
    if (actual <= 0)
      break;

    // forward pass
    // Must exist (or you implement): fills p_pred (probs) and v_pred (scalar)
    muzero_model_forward_batch(model, obs_batch, actual, p_pred, v_pred);

    // compute loss (for logging)
    float pol_loss = 0.0f;
    float val_loss = 0.0f;
    for (int i = 0; i < actual; i++) {
      pol_loss += cross_entropy(&pi_batch[i * A], &p_pred[i * A], A);
      val_loss += mse(z_batch[i], v_pred[i]);
    }
    pol_loss /= (float)actual;
    val_loss /= (float)actual;

    // backward/update
    // Must exist (or you implement): does SGD step using targets
    muzero_model_train_batch(model, obs_batch, pi_batch, z_batch, actual,
                             cfg->lr);

    if ((step % 50) == 0) {
      printf("[train] step=%d pol=%.4f val=%.4f replay=%zu\n", step, pol_loss,
             val_loss, rb_size(rb));
    }
  }

  free(obs_batch);
  free(pi_batch);
  free(z_batch);
  free(p_pred);
  free(v_pred);
}
