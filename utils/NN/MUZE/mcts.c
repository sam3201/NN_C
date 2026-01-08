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
  int expanded;
} Node;

/* =========================
   Helpers
   ========================= */
static Node *node_create(int action_count, int latent_dim) {
  Node *n = (Node *)calloc(1, sizeof(Node));
  if (!n)
    return NULL;

  n->action_count = action_count;
  n->W = (float *)calloc((size_t)action_count, sizeof(float));
  n->Q = (float *)calloc((size_t)action_count, sizeof(float));
  n->N = (int *)calloc((size_t)action_count, sizeof(int));
  n->P = (float *)calloc((size_t)action_count, sizeof(float));
  n->children = (Node **)calloc((size_t)action_count, sizeof(Node *));
  n->latent = (float *)calloc((size_t)latent_dim, sizeof(float));
  n->expanded = 0;

  if (!n->W || !n->Q || !n->N || !n->P || !n->children || !n->latent) {
    free(n->W);
    free(n->Q);
    free(n->N);
    free(n->P);
    free(n->children);
    free(n->latent);
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

  for (int a = 0; a < n->action_count; a++) {
    float u = c_puct * n->P[a] * (sqrt_N / (1.0f + (float)n->N[a]));
    float score = n->Q[a] + u;
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

static float expand_node(Node *node, MuModel *model) {
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
  softmax(logits, A, pri);
  memcpy(node->P, pri, sizeof(float) * (size_t)A);
  node->expanded = 1;

  free(pri);
  free(logits);
  return value;
}

/* Backup the SAME total return to each node along the path, on the chosen
   action. total = r0 + gamma*r1 + ... + gamma^(d-1)*r(d-1) + gamma^d *
   leaf_value
*/
static void backup_with_discount(Node *root, const int *actions,
                                 const float *rewards, int depth,
                                 float leaf_value, float gamma) {
  float total = leaf_value;
  for (int i = depth - 1; i >= 0; i--)
    total = rewards[i] + gamma * total;

  Node *n = root;
  for (int i = 0; i < depth; i++) {
    int a = actions[i];
    n->W[a] += total;
    n->N[a] += 1;
    n->Q[a] = n->W[a] / (float)n->N[a];
    if (!n->children[a])
      break;
    n = n->children[a];
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
MCTSResult mcts_run(MuModel *model, const float *latent,
                    const MCTSParams *params) {
  MCTSResult res;
  memset(&res, 0, sizeof(res));

  if (!model || !latent || !params)
    return res;

  const int A = model->cfg.action_count;
  const int L = model->cfg.latent_dim;

  Node *root = node_create(A, L);
  if (!root)
    return res;

  memcpy(root->latent, latent, sizeof(float) * (size_t)L);

  float root_value = expand_node(root, model);
  dbg_print_root("after expand(root)", root);

  if (params->dirichlet_alpha > 0.0f && params->dirichlet_eps > 0.0f) {
    add_dirichlet_noise(root, params->dirichlet_alpha, params->dirichlet_eps);
    dbg_print_root("after dirichlet(root)", root);
  }

  int max_depth = (params->max_depth > 0) ? params->max_depth : 64;

  int *actions = (int *)malloc(sizeof(int) * (size_t)max_depth);
  float *rewards = (float *)malloc(sizeof(float) * (size_t)max_depth);
  float *h_cur = (float *)malloc(sizeof(float) * (size_t)L);

  if (!actions || !rewards || !h_cur) {
    free(actions);
    free(rewards);
    free(h_cur);
    node_free(root);
    return res;
  }

  for (int sim = 0; sim < params->num_simulations; sim++) {
    Node *node = root;
    int depth = 0;

    memcpy(h_cur, root->latent, sizeof(float) * (size_t)L);

    while (node->expanded && depth < max_depth) {
      int a = select_puct(node, params->c_puct);
      actions[depth] = a;

      if (!node->children[a]) {
        Node *child = node_create(A, L);
        if (!child)
          break;
        node->children[a] = child;

        float r = 0.0f;
        mu_model_dynamics(model, h_cur, a, child->latent, &r);
        rewards[depth] = r;

        float leaf_value = expand_node(child, model);
        backup_with_discount(root, actions, rewards, depth + 1, leaf_value,
                             params->discount);
        break;
      } else {
        Node *child = node->children[a];
        float r = 0.0f;
        mu_model_dynamics(model, h_cur, a, child->latent, &r);
        rewards[depth] = r;

        memcpy(h_cur, child->latent, sizeof(float) * (size_t)L);
        node = child;
        depth++;
      }
    }
  }

  dbg_print_root("after simulations(root)", root);

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
  node_free(root);
  return res;
}

MCTSResult mcts_run(MuModel *model, const float *obs,
                    const MCTSParams *params) {
  MCTSResult res;
  memset(&res, 0, sizeof(res));

  if (!model || !obs || !params)
    return res;

  const int L = model->cfg.latent_dim;

  float *latent = (float *)malloc(sizeof(float) * (size_t)L);
  if (!latent)
    return res;

  mu_model_repr(model, obs, latent);
  res = mcts_run_latent(model, latent, params);

  free(latent);
  return res;
}

void mcts_result_free(MCTSResult *res) {
  if (!res)
    return;
  free(res->pi);
  res->pi = NULL;
  res->action_count = 0;
  res->chosen_action = 0;
  res->root_value = 0.0f;
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
