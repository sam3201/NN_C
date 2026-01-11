#include "mcts.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* =========================
   Debug visualization hooks
   Compile with: -DMCTS_DEBUG
   ========================= */
#ifdef MCTS_DEBUG
#include <stdio.h>
static void dbg_print_root(const char *tag, const struct Node *root);
#else
#define dbg_print_root(tag, root) ((void)0)
#endif

/* =========================
   Node structure
   ========================= */
typedef struct Node {
  float *W; /* sum of backed-up returns per action at THIS node */
  float *Q; /* mean value per action (W/N) */
  int *N;   /* visit counts */
  float *P; /* prior probs */
  struct Node **children;

  float *latent; /* latent state stored at node */
  int action_count;
  int support_size;
  float *dist_sum; /* per-action support distribution sum */
  int expanded;
} Node;

/* =========================
   Helpers
   ========================= */
static Node *node_create(int action_count, int latent_dim, int support_size) {
  Node *n = (Node *)calloc(1, sizeof(Node));
  if (!n)
    return NULL;

  n->action_count = action_count;
  n->support_size = support_size;
  n->W = (float *)calloc((size_t)action_count, sizeof(float));
  n->Q = (float *)calloc((size_t)action_count, sizeof(float));
  n->N = (int *)calloc((size_t)action_count, sizeof(int));
  n->P = (float *)calloc((size_t)action_count, sizeof(float));
  n->children = (Node **)calloc((size_t)action_count, sizeof(Node *));
  n->latent = (float *)calloc((size_t)latent_dim, sizeof(float));
  if (support_size > 1) {
    n->dist_sum = (float *)calloc((size_t)action_count * (size_t)support_size,
                                  sizeof(float));
  }
  n->expanded = 0;

  if (!n->W || !n->Q || !n->N || !n->P || !n->children || !n->latent ||
      (support_size > 1 && !n->dist_sum)) {
    free(n->W);
    free(n->Q);
    free(n->N);
    free(n->P);
    free(n->children);
    free(n->latent);
    free(n->dist_sum);
    free(n);
    return NULL;
  }

  return n;
}

static void node_free(Node *n) {
  if (!n)
    return;

  if (n->children) {
    for (int i = 0; i < n->action_count; i++)
      node_free(n->children[i]);
  }

  free(n->W);
  free(n->Q);
  free(n->N);
  free(n->P);
  free(n->children);
  free(n->latent);
  free(n->dist_sum);
  free(n);
}

static int node_Nsum(const Node *n) {
  int sum = 0;
  for (int i = 0; i < n->action_count; i++)
    sum += n->N[i];
  return sum;
}

static void softmax(const float *logits, int len, float *out) {
  float maxv = -INFINITY;
  for (int i = 0; i < len; i++)
    if (logits[i] > maxv)
      maxv = logits[i];

  float sum = 0.0f;
  for (int i = 0; i < len; i++) {
    out[i] = expf(logits[i] - maxv);
    sum += out[i];
  }

  if (sum > 0.0f) {
    for (int i = 0; i < len; i++)
      out[i] /= sum;
  } else {
    float u = 1.0f / (float)len;
    for (int i = 0; i < len; i++)
      out[i] = u;
  }
}

static int select_puct(const Node *n, float c_puct) {
  int best = 0;
  float best_score = -FLT_MAX;
  float sqrt_N = sqrtf((float)(node_Nsum(n) + 1));
  float q_min = FLT_MAX;
  float q_max = -FLT_MAX;

  for (int a = 0; a < n->action_count; a++) {
    float q = n->Q[a];
    if (q < q_min)
      q_min = q;
    if (q > q_max)
      q_max = q;
  }
  float q_range = q_max - q_min;

  for (int a = 0; a < n->action_count; a++) {
    float u = c_puct * n->P[a] * (sqrt_N / (1.0f + (float)n->N[a]));
    float q = n->Q[a];
    float q_norm = (q_range > 1e-6f) ? ((q - q_min) / q_range) : 0.5f;
    float score = q_norm + u;
    if (score > best_score) {
      best_score = score;
      best = a;
    }
  }
  return best;
}

/* Not a true Dirichlet sampler; good enough for exploration noise at root. */
static void add_dirichlet_noise(Node *root, float alpha, float eps,
                                MCTSRng *rng) {
  if (!root || alpha <= 0.0f || eps <= 0.0f)
    return;

  int A = root->action_count;
  float *g = (float *)malloc(sizeof(float) * (size_t)A);
  if (!g)
    return;

  float sum = 0.0f;
  for (int i = 0; i < A; i++) {
    float u = rng ? rng->rand01(rng->ctx) : ((float)rand() / (float)RAND_MAX);
    if (u <= 0.0f)
      u = 1e-6f;

    g[i] = -logf(u); /* exponential(1) */
    sum += g[i];
  }
  if (sum <= 0.0f)
    sum = 1.0f;

  for (int i = 0; i < A; i++) {
    float d = g[i] / sum;
    root->P[i] = (1.0f - eps) * root->P[i] + eps * d;
  }

  free(g);
}

static float expand_node(Node *node, MuModel *model, float *value_dist,
                         int bins) {
  int A = node->action_count;
  float value = 0.0f;

  float *logits = (float *)malloc(sizeof(float) * (size_t)A);
  float *pri = (float *)malloc(sizeof(float) * (size_t)A);
  if (!logits || !pri) {
    free(logits);
    free(pri);
    return 0.0f;
  }

  mu_model_predict(model, node->latent, logits, &value);
  if (model->use_value_support && bins > 0) {
    float *v_probs = (float *)malloc(sizeof(float) * (size_t)bins);
    if (v_probs) {
      int out_bins =
          mu_model_predict_value_support(model, node->latent, v_probs, bins);
      if (out_bins > 0) {
        if (value_dist) {
          memcpy(value_dist, v_probs, sizeof(float) * (size_t)out_bins);
        }
        value = mu_model_support_expected(model, v_probs, out_bins);
      } else {
        value = mu_model_denorm_value(model, value);
      }
      free(v_probs);
    } else {
      value = mu_model_denorm_value(model, value);
    }
  } else {
    value = mu_model_denorm_value(model, value);
  }
  softmax(logits, A, pri);
  memcpy(node->P, pri, sizeof(float) * (size_t)A);
  node->expanded = 1;

  free(pri);
  free(logits);
  return value;
}

static void project_value_dist(const float *in, int bins, float vmin, float vmax,
                               float reward, float gamma, float *out) {
  if (!in || !out || bins <= 0)
    return;
  memset(out, 0, sizeof(float) * (size_t)bins);
  if (bins == 1) {
    out[0] = 1.0f;
    return;
  }

  float delta = (vmax - vmin) / (float)(bins - 1);
  for (int i = 0; i < bins; i++) {
    float z = vmin + delta * (float)i;
    float tz = reward + gamma * z;
    if (tz <= vmin) {
      out[0] += in[i];
    } else if (tz >= vmax) {
      out[bins - 1] += in[i];
    } else {
      float b = (tz - vmin) / delta;
      int l = (int)floorf(b);
      int u = l + 1;
      float frac = b - (float)l;
      if (l < 0)
        l = 0;
      if (u >= bins)
        u = bins - 1;
      out[l] += in[i] * (1.0f - frac);
      out[u] += in[i] * frac;
    }
  }
}

static float expected_from_sum(const float *sum, int bins, float vmin,
                               float vmax, int n) {
  if (!sum || bins <= 0 || n <= 0)
    return 0.0f;
  if (bins == 1)
    return vmin;
  float delta = (vmax - vmin) / (float)(bins - 1);
  double acc = 0.0;
  for (int i = 0; i < bins; i++) {
    float z = vmin + delta * (float)i;
    acc += (double)sum[i] * (double)z;
  }
  return (float)(acc / (double)n);
}

/* Backup the SAME total return to each node along the path, on the chosen
   action. total = r0 + gamma*r1 + ... + gamma^(d-1)*r(d-1) + gamma^d *
   leaf_value
*/
static void backup_with_discount(Node **path, const int *actions,
                                 const float *rewards, int depth,
                                 float leaf_value, float gamma) {
  float total = leaf_value;
  for (int i = depth - 1; i >= 0; i--)
    total = rewards[i] + gamma * total;

  for (int i = 0; i < depth; i++) {
    Node *n = path[i];
    int a = actions[i];
    n->W[a] += total;
    n->N[a] += 1;
    n->Q[a] = n->W[a] / (float)n->N[a];
  }
}

static void backup_with_discount_dist(Node **path, const int *actions,
                                      const float *rewards, int depth,
                                      const float *leaf_dist, int bins,
                                      float vmin, float vmax, float gamma,
                                      float *dist_buf, float *proj_buf) {
  if (!path || !actions || !rewards || !leaf_dist || !dist_buf || !proj_buf)
    return;
  if (depth <= 0 || bins <= 0)
    return;

  memcpy(dist_buf, leaf_dist, sizeof(float) * (size_t)bins);
  for (int i = depth - 1; i >= 0; i--) {
    project_value_dist(dist_buf, bins, vmin, vmax, rewards[i], gamma, proj_buf);

    Node *n = path[i];
    int a = actions[i];
    float *sum = n->dist_sum + (size_t)a * (size_t)bins;
    for (int k = 0; k < bins; k++)
      sum[k] += proj_buf[k];
    n->N[a] += 1;
    n->Q[a] = expected_from_sum(sum, bins, vmin, vmax, n->N[a]);
    n->W[a] = n->Q[a] * (float)n->N[a];

    memcpy(dist_buf, proj_buf, sizeof(float) * (size_t)bins);
  }
}
static void visits_to_pi(const Node *root, float temperature, float *pi_out) {
  int A = root->action_count;
  if (temperature <= 0.0f)
    temperature = 1e-6f;

  double sum = 0.0;
  for (int a = 0; a < A; a++) {
    double v = pow((double)root->N[a], 1.0 / (double)temperature);
    if (!isfinite(v))
      v = 0.0;
    pi_out[a] = (float)v;
    sum += v;
  }

  if (sum > 0.0) {
    for (int a = 0; a < A; a++)
      pi_out[a] /= (float)sum;
  } else {
    float u = 1.0f / (float)A;
    for (int a = 0; a < A; a++)
      pi_out[a] = u;
  }
}

/* =========================
   Public API
   ========================= */
MCTSResult mcts_run_latent(MuModel *model, const float *latent,
                           const MCTSParams *params, MCTSRng *rng) {
  MCTSResult res;
  memset(&res, 0, sizeof(res));

  if (!model || !latent || !params)
    return res;

  const int A = model->cfg.action_count;
  const int L = model->cfg.latent_dim;

  const int bins =
      (model->use_value_support && model->support_size > 1)
          ? model->support_size
          : 0;

  Node *root = node_create(A, L, bins);
  if (!root)
    return res;

  memcpy(root->latent, latent, sizeof(float) * (size_t)L);

  float root_value = expand_node(root, model, NULL, bins);
  dbg_print_root("after expand(root)", root);

  if (params->dirichlet_alpha > 0.0f && params->dirichlet_eps > 0.0f) {
    add_dirichlet_noise(root, params->dirichlet_alpha, params->dirichlet_eps,
                        rng);
    dbg_print_root("after dirichlet(root)", root);
  }

  int max_depth = (params->max_depth > 0) ? params->max_depth : 64;

  int *actions = (int *)malloc(sizeof(int) * (size_t)max_depth);
  float *rewards = (float *)malloc(sizeof(float) * (size_t)max_depth);
  float *h_cur = (float *)malloc(sizeof(float) * (size_t)L);
  Node **path = (Node **)malloc(sizeof(Node *) * (size_t)(max_depth + 1));
  float *leaf_dist =
      (bins > 0) ? (float *)malloc(sizeof(float) * (size_t)bins) : NULL;
  float *dist_buf =
      (bins > 0) ? (float *)malloc(sizeof(float) * (size_t)bins) : NULL;
  float *proj_buf =
      (bins > 0) ? (float *)malloc(sizeof(float) * (size_t)bins) : NULL;

  if (!actions || !rewards || !h_cur || !path ||
      (bins > 0 && (!leaf_dist || !dist_buf || !proj_buf))) {
    free(actions);
    free(rewards);
    free(h_cur);
    free(path);
    free(leaf_dist);
    free(dist_buf);
    free(proj_buf);
    node_free(root);
    return res;
  }

  for (int sim = 0; sim < params->num_simulations; sim++) {
    Node *node = root;
    int depth = 0;

    memcpy(h_cur, root->latent, sizeof(float) * (size_t)L);
    path[0] = root;

    while (node->expanded && depth < max_depth) {
      int a = select_puct(node, params->c_puct);
      actions[depth] = a;

      if (!node->children[a]) {
        Node *child = node_create(A, L, bins);
        if (!child)
          break;
        node->children[a] = child;

        float r = 0.0f;
        mu_model_dynamics(model, h_cur, a, child->latent, &r);
        rewards[depth] = r;
        path[depth + 1] = child;

        float leaf_value = expand_node(child, model, leaf_dist, bins);
        if (bins > 0 && leaf_dist) {
          backup_with_discount_dist(path, actions, rewards, depth + 1,
                                    leaf_dist, bins, model->support_min,
                                    model->support_max, params->discount,
                                    dist_buf, proj_buf);
        } else {
          backup_with_discount(path, actions, rewards, depth + 1, leaf_value,
                               params->discount);
        }
        break;
      } else {
        Node *child = node->children[a];
        float r = 0.0f;
        mu_model_dynamics(model, h_cur, a, child->latent, &r);
        rewards[depth] = r;
        path[depth + 1] = child;

        memcpy(h_cur, child->latent, sizeof(float) * (size_t)L);
        node = child;
        depth++;
      }
    }
  }

  dbg_print_root("after simulations(root)", root);

  if (bins > 0) {
    float *root_dist = (float *)malloc(sizeof(float) * (size_t)bins);
    if (root_dist) {
      int total = node_Nsum(root);
      if (total > 0 && root->dist_sum) {
        memset(root_dist, 0, sizeof(float) * (size_t)bins);
        for (int a = 0; a < A; a++) {
          float *sum = root->dist_sum + (size_t)a * (size_t)bins;
          for (int k = 0; k < bins; k++)
            root_dist[k] += sum[k];
        }
        float inv = 1.0f / (float)total;
        for (int k = 0; k < bins; k++)
          root_dist[k] *= inv;
        root_value =
            expected_from_sum(root_dist, bins, model->support_min,
                              model->support_max, 1);
      } else {
        int out_bins = mu_model_predict_value_support(model, root->latent,
                                                      root_dist, bins);
        if (out_bins > 0) {
          root_value = mu_model_support_expected(model, root_dist, out_bins);
        }
      }
      res.root_value_dist = root_dist;
      res.root_value_bins = bins;
    }
  }

  float *pi = (float *)malloc(sizeof(float) * (size_t)A);
  if (!pi) {
    free(actions);
    free(rewards);
    free(h_cur);
    node_free(root);
    return res;
  }

  visits_to_pi(root, params->temperature, pi);

  /* Choose action: best Q among visited; fallback to best prior. */
  int best_a = 0;
  float best_q = -INFINITY;
  for (int a = 0; a < A; a++) {
    if (root->N[a] > 0 && root->Q[a] > best_q) {
      best_q = root->Q[a];
      best_a = a;
    }
  }
  if (best_q == -INFINITY) {
    float best_p = -INFINITY;
    for (int a = 0; a < A; a++) {
      if (root->P[a] > best_p) {
        best_p = root->P[a];
        best_a = a;
      }
    }
  }

  res.action_count = A;
  res.pi = pi;
  res.chosen_action = best_a;
  res.root_value = root_value;

  free(actions);
  free(rewards);
  free(h_cur);
  free(path);
  free(leaf_dist);
  free(dist_buf);
  free(proj_buf);
  node_free(root);
  return res;
}

MCTSResult mcts_run(MuModel *model, const float *obs, const MCTSParams *params,
                    MCTSRng *rng) {
  MCTSResult res;
  memset(&res, 0, sizeof(res));

  if (!model || !obs || !params)
    return res;

  const int L = model->cfg.latent_dim;

  float *latent = (float *)malloc(sizeof(float) * (size_t)L);
  if (!latent)
    return res;

  mu_model_repr(model, obs, latent);

  // PASS rng through (fix)
  res = mcts_run_latent(model, latent, params, rng);

  free(latent);
  return res;
}

void mcts_result_free(MCTSResult *res) {
  if (!res)
    return;
  free(res->pi);
  free(res->root_value_dist);
  res->pi = NULL;
  res->root_value_dist = NULL;
  res->action_count = 0;
  res->chosen_action = 0;
  res->root_value = 0.0f;
  res->root_value_bins = 0;
}

#ifdef MCTS_DEBUG
static void dbg_print_root(const char *tag, const struct Node *root_) {
  const Node *root = (const Node *)root_;
  printf("\n[MCTS_DEBUG] %s\n", tag);
  printf("  a |    P        N        Q        W\n");
  for (int a = 0; a < root->action_count; a++) {
    printf(" %2d | % .5f   %4d   % .5f   % .5f\n", a, root->P[a], root->N[a],
           root->Q[a], root->W[a]);
  }
}
#endif
