#include "../utils/NN/MUZE/all.h"
#include "SAM.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SAM_MUZE_HISTORY 8 /* tweak: 4, 8, 16... */

typedef struct {
  SAM_t *sam;

  size_t obs_dim;

  size_t hist_cap;
  size_t hist_len;
  size_t write_idx;

  long double *hist_data;
  long double **seq_ptrs;

  // ---- training cache for REINFORCE-like update ----
  size_t last_seq_len;
  long double **last_seq_ptrs; // points into hist_data (do NOT free)

  size_t last_action_count;
  float *last_probs;  // malloc'd, length = last_action_count
  size_t last_action; // argmax action we forced MUZE to take
  int has_last;       // 0/1
} SAMMuAdapter;

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

static void onehotf(float *x, size_t n, size_t k) {
  for (size_t i = 0; i < n; i++)
    x[i] = 0.0f;
  if (k < n)
    x[k] = 1.0f;
}

/* softmax for action logits -> probs */
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

  for (size_t i = 0; i < n; i++)
    x[i] /= sum;
}

static void sam_encode(void *brain, float *obs, size_t obs_dim,
                       long double ***latent_seq, size_t *seq_len) {
  SAMMuAdapter *ad = (SAMMuAdapter *)brain;
  if (!ad || !latent_seq || !seq_len || !obs)
    return;

  /* If obs_dim changes, reinit (rare, but safe) */
  if (ad->obs_dim != obs_dim) {
    ad->obs_dim = obs_dim;
    ad->hist_len = 0;
    ad->write_idx = 0;

    free(ad->hist_data);
    free(ad->seq_ptrs);

    ad->hist_data =
        (long double *)calloc(ad->hist_cap * ad->obs_dim, sizeof(long double));
    ad->seq_ptrs = (long double **)calloc(ad->hist_cap, sizeof(long double *));
    if (!ad->hist_data || !ad->seq_ptrs) {
      *latent_seq = NULL;
      *seq_len = 0;
      return;
    }
  }

  /* write obs into ring buffer */
  long double *dst = ad->hist_data + (ad->write_idx * ad->obs_dim);
  for (size_t i = 0; i < ad->obs_dim; i++)
    dst[i] = (long double)obs[i];

  ad->write_idx = (ad->write_idx + 1) % ad->hist_cap;
  if (ad->hist_len < ad->hist_cap)
    ad->hist_len++;

  /* build ordered sequence pointers (oldest -> newest) */
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

  // Copy logits into float buffer
  size_t sam_out_dim = ad->sam->layer_sizes[ad->sam->num_layers - 1];
  size_t n = action_count < sam_out_dim ? action_count : sam_out_dim;

  for (size_t i = 0; i < n; i++)
    action_probs[i] = (float)logits[i];
  for (size_t i = n; i < action_count; i++)
    action_probs[i] = 0.0f;

  // Softmax -> probs
  softmaxf_inplace(action_probs, action_count);

  // Ensure ad->last_probs buffer exists
  if (ad->last_action_count != action_count) {
    free(ad->last_probs);
    ad->last_probs = (float *)calloc(action_count, sizeof(float));
    ad->last_action_count = action_count;
  }

  // Store last probs + last seq pointers/len
  if (ad->last_probs) {
    memcpy(ad->last_probs, action_probs, action_count * sizeof(float));
    ad->last_seq_ptrs = latent_seq; // points into hist_data, do not free
    ad->last_seq_len = seq_len;
    ad->has_last = 1;
  }

  // Force deterministic action so learn() knows which action happened
  ad->last_action = argmaxf(action_probs, action_count);
  onehotf(action_probs, action_count, ad->last_action);

  free(logits);
}

static void sam_learn(void *brain, float reward, int terminal) {
  SAMMuAdapter *ad = (SAMMuAdapter *)brain;
  if (!ad || !ad->sam)
    return;

  SAM_update_context(ad->sam, (long double)reward);

  // If we have a cached (state, action, probs), do REINFORCE-style update.
  if (ad->has_last && ad->last_probs && ad->last_seq_ptrs &&
      ad->last_seq_len > 0 && ad->last_action_count > 0) {

    size_t A = ad->last_action_count;

    // grad_logits[i] = reward * (p[i] - 1(i==a))
    long double *grad = (long double *)calloc(A, sizeof(long double));
    if (grad) {
      for (size_t i = 0; i < A; i++) {
        long double p = (long double)ad->last_probs[i];
        long double oh = (i == ad->last_action) ? 1.0L : 0.0L;
        grad[i] = (long double)reward * (p - oh);
      }

      // Train SAM (head + transformer)
      SAM_backprop(ad->sam, ad->last_seq_ptrs, ad->last_seq_len, grad);

      free(grad);
    }
  }

  if (terminal) {
    SAM_generalize(ad->sam);
    SAM_transfuse(ad->sam);
  }
}

/* Because encode returns internal pointers, MUZE must NOT free them */
static void sam_free_latent_seq(void *brain, long double **latent_seq,
                                size_t seq_len) {
  (void)brain;
  (void)latent_seq;
  (void)seq_len;
  /* no-op */
}

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

  ad->sam = sam;
  ad->obs_dim = 0; /* set on first encode() */
  ad->hist_cap = SAM_MUZE_HISTORY;
  ad->hist_len = 0;
  ad->write_idx = 0;
  ad->hist_data = NULL;
  ad->seq_ptrs = NULL;

  ad->last_seq_len = 0;
  ad->last_seq_ptrs = NULL;
  ad->last_action_count = 0;
  ad->last_probs = NULL;
  ad->last_action = 0;
  ad->has_last = 0;

  c->brain = ad;
  c->encode = sam_encode;
  c->policy = sam_policy;
  c->learn = sam_learn;
  c->free_latent_seq = sam_free_latent_seq;

  return c;
}
