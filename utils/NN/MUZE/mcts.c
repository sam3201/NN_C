#include "mcts.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------
   Internal Node
----------------------------- */

typedef struct Node {
  float *W; /* sum of returns per action */
  float *Q; /* mean return per action */
  int *N;   /* visit count per action */
  float *P; /* prior prob per action */

  struct Node **children; /* child pointers per action */
  float *latent;          /* stored latent at this node */

  int action_count;
  int expanded;
} Node;

static Node *node_create(int action_count, int latent_dim) {
  if (action_count <= 0 || latent_dim <= 0)
    return NULL;

  Node *n = (Node *)calloc(1, sizeof(Node));
  if (!n)
    return NULL;

  n->action_count = action_count;
  n->expanded = 0;

  n->W = (float *)calloc((size_t)action_count, sizeof(float));
  n->Q = (float *)calloc((size_t)action_count, sizeof(float));
  n->N = (int *)calloc((size_t)action_count, sizeof(int));
  n->P = (float *)calloc((size_t)action_count, sizeof(float));
  n->children = (Node **)calloc((size_t)action_count, sizeof(Node *));
  n->latent = (float *)calloc((size_t)latent_dim, sizeof(float));

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
    for (int a = 0; a < n->action_count; a++)
      node_free(n->children[a]);
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
  for (int a = 0; a < n->action_count; a++)
    sum += n->N[a];
  return sum;
}

/* -----------------------------
   Utils
----------------------------- */

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

/* PUCT selection: Q + c * P * sqrt(Nsum)/(1+N) */
static int select_puct(const Node *n, float c_puct) {
  int best = 0;
  float best_score = -FLT_MAX;

  float sqrtN = sqrtf((float)(node_Nsum(n) + 1));

  for (int a = 0; a < n->action_count; a++) {
    float u = c_puct * n->P[a] * (sqrtN / (1.0f + (float)n->N[a]));
    float score = n->Q[a] + u;

    if (score > best_score) {
      best_score = score;
      best = a;
    }
  }
  return best;
}

/* NOTE: this is a simple "Dirichlet-ish" noise, not a true Gamma(alpha) sampler
 */
static void add_dirichlet_noise(Node *root, float alpha, float eps) {
  (void)alpha;
  if (!root || eps <= 0.0f)
    return;

  int A = root->action_count;

  float *g = (float *)malloc(sizeof(float) * (size_t)A);
  if (!g)
    return;

  float sum = 0.0f;
  for (int i = 0; i < A; i++) {
    float u = (rand() + 1.0f) / (RAND_MAX + 1.0f);
    g[i] = -logf(u);
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

/* Expand: fill priors and return predicted value */
static float expand_node(Node *node, MuModel *model) {
  int A = node->action_count;

  float *logits = (float *)malloc(sizeof(float) * (size_t)A);
  float *pri = (float *)malloc(sizeof(float) * (size_t)A);
  if (!logits || !pri) {
    free(logits);
    free(pri);
    return 0.0f;
  }

  float value = 0.0f;
  mu_model_predict(model, node->latent, logits, &value);

  softmax(logits, A, pri);
  memcpy(node->P, pri, sizeof(float) * (size_t)A);

  node->expanded = 1;

  free(pri);
  free(logits);
  return value;
}

/* Convert visit counts to pi with temperature */
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

/* Backup: apply a single return to each visited edge along the path.
   We compute a discounted return from rewards + leaf_value, then add it to each
   edge.
*/
static void backup_path(Node *root, const int *actions, const float *rewards,
                        int depth, float leaf_value, float discount) {
  float G = leaf_value;
  for (int i = depth - 1; i >= 0; i--)
    G = rewards[i] + discount * G;

  Node *n = root;
  for (int i = 0; i < depth; i++) {
    int a = actions[i];
    n->W[a] += G;
    n->N[a] += 1;
    n->Q[a] = n->W[a] / (float)n->N[a];

    if (!n->children[a])
      break;
    n = n->children[a];
  }
}

/* -----------------------------
   Public API
----------------------------- */

MCTSResult mcts_run_latent(MuModel *model, const float *latent,
                           const MCTSParams *params) {
  MCTSResult res;
  memset(&res, 0, sizeof(res));

  if (!model || !latent || !params)
    return res;

  const int A = model->cfg.action_count;
  const int L = model->cfg.latent_dim;
  if (A <= 0 || L <= 0)
    return res;

  Node *root = node_create(A, L);
  if (!root)
    return res;

  memcpy(root->latent, latent, sizeof(float) * (size_t)L);

  float root_value = expand_node(root, model);

  if (params->dirichlet_alpha > 0.0f && params->dirichlet_eps > 0.0f)
    add_dirichlet_noise(root, params->dirichlet_alpha, params->dirichlet_eps);

  const int max_depth = (params->max_depth > 0) ? params->max_depth : 64;

  int *actions = (int *)malloc(sizeof(int) * (size_t)max_depth);
  float *rewards = (float *)malloc(sizeof(float) * (size_t)max_depth);
  if (!actions || !rewards) {
    free(actions);
    free(rewards);
    node_free(root);
    return res;
  }

  for (int sim = 0; sim < params->num_simulations; sim++) {
    Node *node = root;
    int depth = 0;

    float *h_cur = (float *)malloc(sizeof(float) * (size_t)L);
    if (!h_cur)
      break;
    memcpy(h_cur, root->latent, sizeof(float) * (size_t)L);

    while (node->expanded && depth < max_depth) {
      int a = select_puct(node, params->c_puct);
      actions[depth] = a;

      /* Create child if needed */
      if (!node->children[a]) {
        Node *child = node_create(A, L);
        if (!child) {
          free(h_cur);
          h_cur = NULL;
          break;
        }
        node->children[a] = child;

        float r = 0.0f;
        mu_model_dynamics(model, h_cur, a, child->latent, &r);
        rewards[depth] = r;

        float leaf = expand_node(child, model);
        backup_path(root, actions, rewards, depth + 1, leaf, params->discount);
        break;
      }

      /* Traverse existing child */
      {
        Node *child = node->children[a];
        float r = 0.0f;
        mu_model_dynamics(model, h_cur, a, child->latent, &r);
        rewards[depth] = r;

        memcpy(h_cur, child->latent, sizeof(float) * (size_t)L);
        node = child;
        depth++;
      }
    }

    free(h_cur);
  }

  float *pi = (float *)malloc(sizeof(float) * (size_t)A);
  if (!pi) {
    free(actions);
    free(rewards);
    node_free(root);
    return res;
  }

  visits_to_pi(root, params->temperature, pi);

  int best_a = 0;
  float best_q = -INFINITY;
  for (int a = 0; a < A; a++) {
    if (root->N[a] > 0 && root->Q[a] > best_q) {
      best_q = root->Q[a];
      best_a = a;
    }
  }

  res.action_count = A;
  res.pi = pi;
  res.chosen_action = best_a;
  res.root_value = root_value;

  free(actions);
  free(rewards);
  node_free(root);
  return res;
}

MCTSResult mcts_run(MuModel *model, const float *obs,
                    const MCTSParams *params) {
  MCTSResult res;
  memset(&res, 0, sizeof(res));

  if (!model || !obs || !params)
    return res;

  const int A = model->cfg.action_count;
  const int L = model->cfg.latent_dim;
  if (A <= 0 || L <= 0)
    return res;

  Node *root = node_create(A, L);
  if (!root)
    return res;

  mu_model_repr(model, obs, root->latent);
  float root_value = expand_node(root, model);

  if (params->dirichlet_alpha > 0.0f && params->dirichlet_eps > 0.0f)
    add_dirichlet_noise(root, params->dirichlet_alpha, params->dirichlet_eps);

  const int max_depth = (params->max_depth > 0) ? params->max_depth : 64;

  int *actions = (int *)malloc(sizeof(int) * (size_t)max_depth);
  float *rewards = (float *)malloc(sizeof(float) * (size_t)max_depth);
  if (!actions || !rewards) {
    free(actions);
    free(rewards);
    node_free(root);
    return res;
  }

  for (int sim = 0; sim < params->num_simulations; sim++) {
    Node *node = root;
    int depth = 0;

    float *h_cur = (float *)malloc(sizeof(float) * (size_t)L);
    if (!h_cur)
      break;
    memcpy(h_cur, root->latent, sizeof(float) * (size_t)L);

    while (node->expanded && depth < max_depth) {
      int a = select_puct(node, params->c_puct);
      actions[depth] = a;

      if (!node->children[a]) {
        Node *child = node_create(A, L);
        if (!child) {
          free(h_cur);
          h_cur = NULL;
          break;
        }
        node->children[a] = child;

        float r = 0.0f;
        mu_model_dynamics(model, h_cur, a, child->latent, &r);
        rewards[depth] = r;

        float leaf = expand_node(child, model);
        backup_path(root, actions, rewards, depth + 1, leaf, params->discount);
        break;
      }

      {
        Node *child = node->children[a];
        float r = 0.0f;
        mu_model_dynamics(model, h_cur, a, child->latent, &r);
        rewards[depth] = r;

        memcpy(h_cur, child->latent, sizeof(float) * (size_t)L);
        node = child;
        depth++;
      }
    }

    free(h_cur);
  }

  float *pi = (float *)malloc(sizeof(float) * (size_t)A);
  if (!pi) {
    free(actions);
    free(rewards);
    node_free(root);
    return res;
  }

  visits_to_pi(root, params->temperature, pi);

  /* Choose action: best Q if visited, else best prior */
  int best_a = 0;
  float best_q = -INFINITY;

  for (int a = 0; a < A; a++) {
    float q = (root->N[a] > 0) ? root->Q[a] : -INFINITY;
    if (q > best_q) {
      best_q = q;
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
  node_free(root);
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
