#include "muzero_model.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int onehot_argmax(const float *x, int n) {
  int best = 0;
  float bestv = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] > bestv) {
      bestv = x[i];
      best = i;
    }
  }
  return best;
}

static void onehot_set(float *x, int n, int k) {
  for (int i = 0; i < n; i++)
    x[i] = 0.0f;
  if (k >= 0 && k < n)
    x[k] = 1.0f;
}

MuModel *mu_model_create_toy(int size /* e.g. 8 */, int action_count /*2*/) {
  MuConfig cfg = {0};
  cfg.obs_dim = size;
  cfg.latent_dim = size;
  cfg.action_count = action_count;

  // You can reuse the normal allocator if it just stores cfg + buffers.
  // If your existing mu_model_create allocates NN weights you don't want,
  // make a dedicated minimal model struct or add a flag.
  MuModel *m = (MuModel *)calloc(1, sizeof(MuModel));
  if (!m)
    return NULL;
  m->cfg = cfg;

  return m;
}

void mu_model_free_toy(MuModel *m) { free(m); }

/* repr: obs -> latent */
void mu_model_repr_toy(MuModel *m, const float *obs, float *latent_out) {
  if (!m || !obs || !latent_out)
    return;
  memcpy(latent_out, obs, sizeof(float) * m->cfg.latent_dim);
}

/* prediction: latent -> (policy logits, value) */
void mu_model_predict_toy(MuModel *m, const float *latent, float *policy_logits,
                          float *value_out) {
  if (!m || !latent || !policy_logits || !value_out)
    return;

  const int L = m->cfg.latent_dim;
  const int A = m->cfg.action_count;

  int pos = onehot_argmax(latent, L);
  int goal = L - 1;

  // simple policy: prefer right unless already at goal
  // action 0 = left, action 1 = right
  for (int a = 0; a < A; a++)
    policy_logits[a] = 0.0f;

  if (A >= 2) {
    policy_logits[0] = (pos > 0) ? 0.25f : -0.25f;
    policy_logits[1] = (pos < goal) ? 1.25f : -0.25f;
  }

  // value: normalized distance-to-goal (1 at goal, ~0 at start)
  float dist = (float)(goal - pos);
  float denom = (goal > 0) ? (float)goal : 1.0f;
  float v = 1.0f - (dist / denom);
  if (v < 0.0f)
    v = 0.0f;
  if (v > 1.0f)
    v = 1.0f;
  *value_out = v;
}

/* dynamics: (latent, action) -> (latent2, reward) */
void mu_model_dynamics_toy(MuModel *m, const float *latent, int action,
                           float *latent2_out, float *reward_out) {
  if (!m || !latent || !latent2_out || !reward_out)
    return;

  const int L = m->cfg.latent_dim;
  int pos = onehot_argmax(latent, L);
  int goal = L - 1;

  // action 0=left, 1=right
  if (action == 0 && pos > 0)
    pos--;
  else if (action == 1 && pos < goal)
    pos++;

  onehot_set(latent2_out, L, pos);
  *reward_out = (pos == goal) ? 1.0f : 0.0f;
}
