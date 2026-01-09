#include "trainer.h"
#include "muzero_model.h"
#include "replay_buffer.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static float mse(float a, float b) {
  float d = a - b;
  return d * d;
}

static float cross_entropy_logits(const float *pi_target, const float *logits,
                                  int A) {
  // find max logit for stability
  float m = logits[0];
  for (int i = 1; i < A; i++)
    if (logits[i] > m)
      m = logits[i];

  // logsumexp
  float sum = 0.0f;
  for (int i = 0; i < A; i++)
    sum += expf(logits[i] - m);
  const float logZ = logf(sum) + m;

  float loss = 0.0f;
  for (int i = 0; i < A; i++) {
    // log_softmax = logits - logZ
    loss -= pi_target[i] * (logits[i] - logZ);
  }
  return loss;
}

static bool is_finite_float(float x) { return isfinite((double)x); }

void trainer_train_dynamics(MuModel *model, ReplayBuffer *rb,
                            const TrainerConfig *cfg) {
  if (!model || !rb || !cfg)
    return;

  -size_t n = rb_size(rb);
  -if ((int)n < cfg->min_replay_size) + size_t n0 = rb_size(rb);
  +if ((int)n0 < cfg->min_replay_size) return;

  int B = cfg->batch_size;
  int O = muzero_model_obs_dim(model);
@@ -69,7 +92,7 @@ void trainer_train_dynamics(MuModel *model, ReplayBuffer *rb,
     float lat_mse = 0.0f, rew_mse = 0.0f;

     if (model->train_dynamics) {
    // hook matches muzero_model.h: includes mse out params
    model->train_dynamics(model, obs, a, r, obs2, actual, lr, &lat_mse,
                          &rew_mse);
     } else {
    muzero_model_train_dynamics_batch(model, obs, a, r, obs2, actual, lr,
                                      &lat_mse, &rew_mse);
     }

+    if (!is_finite_float(lat_mse) || !is_finite_float(rew_mse)) {
    +printf("[dyn] NaN/Inf detected at step=%d (lat_mse=%f rew_mse=%f)\n",
            +step, lat_mse, rew_mse);
    +break;
    +    }
+
     if ((step % 50) == 0) {
    printf("[dyn] step=%d lat_mse=%.6f rew_mse=%.6f replay=%zu\n", step,
           lat_mse, rew_mse, rb_size(rb));
     }
}
@@ -105,12 +128,12 @@ void trainer_train_from_replay(MuModel *model, ReplayBuffer *rb,
   float *pi_batch = (float *)malloc(sizeof(float) * (size_t)B * (size_t)A);
   float *z_batch = (float *)malloc(sizeof(float) * (size_t)B);

-  float *p_pred = (float *)malloc(sizeof(float) * (size_t)B * (size_t)A);
+  float *logits_pred = (float *)malloc(sizeof(float) * (size_t)B * (size_t)A);
   float *v_pred = (float *)malloc(sizeof(float) * (size_t)B);

-  if (!obs_batch || !pi_batch || !z_batch || !p_pred || !v_pred) {
  +if (!obs_batch || !pi_batch || !z_batch || !logits_pred || !v_pred) {
    free(obs_batch);
    free(pi_batch);
    free(z_batch);
    -free(p_pred);
    +free(logits_pred);
    free(v_pred);
    return;
  }
  void trainer_train_from_replay(MuModel * model, ReplayBuffer * rb,
                                 const TrainerConfig *cfg) {
    if (!model || !rb || !cfg)
      return;

    size_t n = rb_size(rb);
    if (n < (size_t)cfg->min_replay_size) {
      printf("[train] waiting: replay=%zu < min=%d\n", n, cfg->min_replay_size);
      return;
    }

    const int B = cfg->batch_size;
    const int obs_dim = muzero_model_obs_dim(model);
    const int A = muzero_model_action_count(model);

    // pick your actual config field name here:
    float lr = cfg->lr;
    if (!(lr > 0.0f))
      lr = 0.05f;

    float *obs_batch =
        (float *)malloc(sizeof(float) * (size_t)B * (size_t)obs_dim);
    float *pi_batch = (float *)malloc(sizeof(float) * (size_t)B * (size_t)A);
    float *z_batch = (float *)malloc(sizeof(float) * (size_t)B);

    float *p_pred = (float *)malloc(sizeof(float) * (size_t)B * (size_t)A);
    float *v_pred = (float *)malloc(sizeof(float) * (size_t)B);

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
      muzero_model_forward_batch(model, obs_batch, actual, p_pred, v_pred);

      // compute loss (logging)
      float pol_loss = 0.0f, val_loss = 0.0f;
      for (int i = 0; i < actual; i++) {
        pol_loss += cross_entropy(&pi_batch[i * A], &p_pred[i * A], A);
        val_loss += mse(z_batch[i], v_pred[i]);
      }
      pol_loss /= (float)actual;
      val_loss /= (float)actual;

      // backward/update  <-- use actual, not B
      if (model->train_policy_value) {
        model->train_policy_value(model, obs_batch, pi_batch, z_batch, actual,
                                  lr);
      } else {
        muzero_model_train_batch(model, obs_batch, pi_batch, z_batch, actual,
                                 lr);
      }

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
