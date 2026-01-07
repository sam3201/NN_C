#include "SAM_MUZE_BRIDGE.h"
#include "../../utils/NN/MUZE/all.h"
#include "../SAM.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAM_MUZE_HISTORY 8

typedef struct {
  SAM_t *sam;

  size_t obs_dim;

  size_t hist_cap;
  size_t hist_len;
  size_t write_idx;

  long double *hist_data; /* hist_cap * obs_dim */
  long double **seq_ptrs; /* hist_cap pointers into hist_data */

  /* last decision cache for REINFORCE update */
  int has_last;
  size_t last_action;
  size_t last_action_count;
  float *last_probs;
  long double **last_seq_ptrs;
  size_t last_seq_len;

  int episode_counter;

  /* epsilon-greedy exploration */
  float epsilon;
  float epsilon_min;
  float epsilon_decay;

  /* reward baseline */
  float reward_baseline;
} SAMMuAdapter;

/* ---------- helpers ---------- */

static size_t argmaxf(const float *x, size_t n) {
  size_t best = 0;
  float bestv = x[0];
  for (size_t i = 1; i < n; i++) {
    if (x[i] > bestv) {
      bestv = x[i];
      best = i;
    }
  }
  return best;
}

static void softmaxf_inplace(float *x, size_t n) {
  if (n == 0)
    return;

  float maxv = x[0];
  for (size_t i = 1; i < n; i++)
    if (x[i] > maxv)
      maxv = x[i];

  float sum = 0.0f;
  for (size_t i = 0; i < n; i++) {
    x[i] = expf(x[i] - maxv);
    sum += x[i];
  }

  if (sum <= 0.0f) {
    float u = 1.0f / (float)n;
    for (size_t i = 0; i < n; i++)
      x[i] = u;
    return;
  }

  float inv = 1.0f / sum;
  for (size_t i = 0; i < n; i++)
    x[i] *= inv;
}

static size_t sample_from_probs(const float *p, size_t n) {
  float r = (float)rand() / (float)RAND_MAX;
  float c = 0.0f;
  for (size_t i = 0; i < n; i++) {
    c += p[i];
    if (r <= c)
      return i;
  }
  return n ? (n - 1) : 0;
}

/* ---------- cortex hooks ---------- */

static void sam_encode(void *brain, float *obs, size_t obs_dim,
                       long double ***latent_seq, size_t *seq_len) {
  SAMMuAdapter *ad = (SAMMuAdapter *)brain;
  if (!ad || !latent_seq || !seq_len || !obs)
    return;

  /* if obs_dim changes, reset buffers */
  if (ad->obs_dim != obs_dim) {
    ad->obs_dim = obs_dim;
    ad->hist_len = 0;
    ad->write_idx = 0;

    free(ad->hist_data);
    free(ad->seq_ptrs);
    ad->hist_data = NULL;
    ad->seq_ptrs = NULL;

    ad->hist_data =
        (long double *)calloc(ad->hist_cap * ad->obs_dim, sizeof(long double));
    ad->seq_ptrs = (long double **)calloc(ad->hist_cap, sizeof(long double *));
    if (!ad->hist_data || !ad->seq_ptrs) {
      *latent_seq = NULL;
      *seq_len = 0;
      return;
    }

    ad->has_last = 0;
    ad->last_seq_ptrs = NULL;
    ad->last_seq_len = 0;
  }

  long double *dst = ad->hist_data + (ad->write_idx * ad->obs_dim);
  for (size_t i = 0; i < ad->obs_dim; i++)
    dst[i] = (long double)obs[i];

  ad->write_idx = (ad->write_idx + 1) % ad->hist_cap;
  if (ad->hist_len < ad->hist_cap)
    ad->hist_len++;

  size_t start = (ad->write_idx + ad->hist_cap - ad->hist_len) % ad->hist_cap;
  for (size_t i = 0; i < ad->hist_len; i++) {
    size_t idx = (start + i) % ad->hist_cap;
    ad->seq_ptrs[i] = ad->hist_data + idx * ad->obs_dim;
  }

  *latent_seq = ad->seq_ptrs;
  *seq_len = ad->hist_len;
}

static void sam_policy(void *brain, long double **latent_seq, size_t seq_len,
                       float *action_probs, size_t action_count) {
  SAMMuAdapter *ad = (SAMMuAdapter *)brain;
  if (!ad || !ad->sam || !latent_seq || seq_len == 0 || !action_probs ||
      action_count == 0)
    return;

  long double *logits = SAM_forward(ad->sam, latent_seq, seq_len);
  if (!logits)
    return;

  size_t sam_out_dim = ad->sam->layer_sizes[ad->sam->num_layers - 1];
  size_t n = action_count < sam_out_dim ? action_count : sam_out_dim;

  for (size_t i = 0; i < n; i++)
    action_probs[i] = (float)logits[i];
  for (size_t i = n; i < action_count; i++)
    action_probs[i] = 0.0f;

  softmaxf_inplace(action_probs, action_count);

  /* epsilon-greedy (real exploration) */
  if (((float)rand() / (float)RAND_MAX) < ad->epsilon) {
    ad->last_action = (size_t)(rand() % (int)action_count);
  } else {
    ad->last_action = sample_from_probs(action_probs, action_count);
  }

  /* cache probs for REINFORCE */
  if (ad->last_action_count != action_count) {
    free(ad->last_probs);
    ad->last_probs = (float *)calloc(action_count, sizeof(float));
    ad->last_action_count = action_count;
  }

  if (ad->last_probs) {
    memcpy(ad->last_probs, action_probs, action_count * sizeof(float));
    ad->last_seq_ptrs = latent_seq;
    ad->last_seq_len = seq_len;
    ad->has_last = 1;
  } else {
    ad->has_last = 0;
  }

  /* IMPORTANT: do NOT one-hot here. MUZE will select action itself. */
  free(logits);
}

static void sam_learn(void *brain, const float *obs, size_t obs_dim, int action,
                      float reward, int terminal) {
  (void)obs;
  (void)obs_dim;

  SAMMuAdapter *ad = (SAMMuAdapter *)brain;
  if (!ad || !ad->sam)
    return;

  /* baseline update */
  ad->reward_baseline = 0.95f * ad->reward_baseline + 0.05f * reward;

  SAM_update_context(ad->sam, (long double)reward);

  if (ad->has_last && ad->last_probs && ad->last_seq_ptrs &&
      ad->last_seq_len > 0 && ad->last_action_count > 0) {

    size_t A = ad->last_action_count;
    long double *grad = (long double *)calloc(A, sizeof(long double));
    if (grad) {
      long double adv = (long double)(reward - ad->reward_baseline);

      for (size_t i = 0; i < A; i++) {
        long double p = (long double)ad->last_probs[i];
        long double oh = (i == (size_t)action) ? 1.0L : 0.0L;
        grad[i] = adv * (p - oh);
      }

      SAM_backprop(ad->sam, ad->last_seq_ptrs, ad->last_seq_len, grad);
      free(grad);
    }
  }

  if (terminal) {
    ad->episode_counter++;

    SAM_generalize(ad->sam);

    if (ad->episode_counter % 10 == 0) {
      SAM_transfuse(ad->sam);
    }

    if (ad->epsilon > ad->epsilon_min) {
      ad->epsilon *= ad->epsilon_decay;
      if (ad->epsilon < ad->epsilon_min)
        ad->epsilon = ad->epsilon_min;
    }

    ad->has_last = 0;
  }
}

static void sam_free_latent_seq(void *brain, long double **latent_seq,
                                size_t seq_len) {
  (void)brain;
  (void)latent_seq;
  (void)seq_len;
  /* We use a ring buffer. Nothing to free per-step. */
}

/* ---------- public API ---------- */

MuCortex *SAM_as_MUZE(SAM_t *sam) {
  if (!sam)
    return NULL;

  MuCortex *c = (MuCortex *)calloc(1, sizeof(MuCortex));
  if (!c)
    return NULL;

  SAMMuAdapter *ad = (SAMMuAdapter *)calloc(1, sizeof(SAMMuAdapter));
  if (!ad) {
    free(c);
    return NULL;
  }

#include "sam_muze_adapter.h"
#include <stdlib.h>
#include <string.h>

  // The adapter stores function pointers MUZE will call.
  typedef struct {
    SAM_t *sam;
    MuCortex cortex; // must match MUZE expectations
  } SAMMuAdapter;

  static int adapter_plan(void *brain, const float *obs, size_t obs_dim,
                          size_t action_count) {
    SAMMuAdapter *a = (SAMMuAdapter *)brain;
    // call whatever SAM uses to pick actions
    return SAM_choose_action(a->sam, obs, obs_dim, (int)action_count);
  }

  static void adapter_learn(void *brain, const float *obs, size_t obs_dim,
                            int action, float reward, int terminal) {
    SAMMuAdapter *a = (SAMMuAdapter *)brain;
    SAM_learn(a->sam, obs, obs_dim, action, reward, terminal);
  }

  MuCortex *SAM_as_MUZE(SAM_t * sam) {
    SAMMuAdapter *a = (SAMMuAdapter *)calloc(1, sizeof(*a));
    if (!a)
      return NULL;

    a->sam = sam;
    a->cortex.brain = a;           // MUZE calls cortex->plan/learn with brain
    a->cortex.plan = adapter_plan; // or whatever MUZE names it
    a->cortex.learn = adapter_learn;

    return &a->cortex;
  }

  void SAM_as_MUZE_free(MuCortex * cortex) {
    if (!cortex)
      return;
    SAMMuAdapter *a = (SAMMuAdapter *)cortex->brain;
    free(a);
  }

  ad->sam = sam;

  ad->obs_dim = 0;
  ad->hist_cap = SAM_MUZE_HISTORY;
  ad->hist_len = 0;
  ad->write_idx = 0;

  ad->hist_data = NULL;
  ad->seq_ptrs = NULL;

  ad->has_last = 0;
  ad->last_action = 0;
  ad->last_action_count = 0;
  ad->last_probs = NULL;
  ad->last_seq_ptrs = NULL;
  ad->last_seq_len = 0;

  ad->episode_counter = 0;

  ad->epsilon = 0.30f;
  ad->epsilon_min = 0.02f;
  ad->epsilon_decay = 0.95f;

  ad->reward_baseline = 0.0f;

  c->brain = ad;
  c->encode = sam_encode;
  c->policy = sam_policy;
  c->learn = sam_learn;
  c->free_latent_seq = sam_free_latent_seq;

  /* Default: no MCTS unless you enable it */
  c->use_mcts = false;
  c->mcts_model = NULL;

  /* Defaults */
  c->mcts_params.num_simulations = 50;
  c->mcts_params.c_puct = 1.25f;
  c->mcts_params.max_depth = 16;
  c->mcts_params.dirichlet_alpha = 0.3f;
  c->mcts_params.dirichlet_eps = 0.25f;
  c->mcts_params.temperature = 1.0f;
  c->mcts_params.discount = 0.997f;

  return c;
}

void SAM_MUZE_destroy(MuCortex *cortex) {
  if (!cortex)
    return;

  SAMMuAdapter *ad = (SAMMuAdapter *)cortex->brain;
  if (ad) {
    free(ad->hist_data);
    free(ad->seq_ptrs);
    free(ad->last_probs);
    free(ad);
  }
  free(cortex);
}

#include "sam_muze_adapter.h"
#include <stdlib.h>
#include <string.h>

// The adapter stores function pointers MUZE will call.
typedef struct {
  SAM_t *sam;
  MuCortex cortex; // must match MUZE expectations
} SAMMuAdapter;

static int adapter_plan(void *brain, const float *obs, size_t obs_dim,
                        size_t action_count) {
  SAMMuAdapter *a = (SAMMuAdapter *)brain;
  // call whatever SAM uses to pick actions
  return SAM_choose_action(a->sam, obs, obs_dim, (int)action_count);
}

static void adapter_learn(void *brain, const float *obs, size_t obs_dim,
                          int action, float reward, int terminal) {
  SAMMuAdapter *a = (SAMMuAdapter *)brain;
  SAM_learn(a->sam, obs, obs_dim, action, reward, terminal);
}

void SAM_as_MUZE_destroy(MuCortex *cortex) {
  if (!cortex)
    return;
  SAMMuAdapter *a = (SAMMuAdapter *)cortex->brain;
  free(a);
}
