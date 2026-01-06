#include "mcts.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Node structure */
typedef struct Node {
  float *W; /* sum of values per action */
  float *Q; /* mean value per action */
  int *N;   /* visit counts */
  float *P; /* prior probs */
  struct Node **children;
  float *latent; /* latent state stored at node */
  int action_count;
  int expanded;
} Node;

/* allocate node */
static Node *node_create(int action_count, int latent_dim) {
  Node *n = (Node *)calloc(1, sizeof(Node));
  if (!n)
    return NULL;

  n->action_count = action_count;

  n->W = (float *)calloc(action_count, sizeof(float));
  n->Q = (float *)calloc(action_count, sizeof(float));
  n->N = (int *)calloc(action_count, sizeof(int));
  n->P = (float *)calloc(action_count, sizeof(float));
  n->children = (Node **)calloc(action_count, sizeof(Node *));
  n->latent = (float *)calloc(latent_dim, sizeof(float));

  if (!n->W || !n->Q || !n->N || !n->P || !n->children || !n->latent) {
    // free partial
    free(n->W);
    free(n->Q);
    free(n->N);
    free(n->P);
    free(n->children);
    free(n->latent);
    free(n);
    return NULL;
  }

  n->expanded = 0;
  return n;
}

/* free node recursively */
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

/* sum of visits */
static int node_Nsum(Node *n) {
  int sum = 0;
  for (int i = 0; i < n->action_count; i++)
    sum += n->N[i];
  return sum;
}

/* simple softmax */
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
    for (int i = 0; i < len; i++)
      out[i] = 1.0f / (float)len;
  }
}

/* PUCT selection */
static int select_puct(Node *n, float c_puct) {
  int best = 0;
  float best_score = -FLT_MAX;
  float sqrt_N = sqrtf((float)(node_Nsum(n) + 1));

  for (int a = 0; a < n->action_count; a++) {
    float score = n->Q[a] + c_puct * n->P[a] * (sqrt_N / (1.0f + n->N[a]));
    if (score > best_score) {
      best_score = score;
      best = a;
    }
  }
  return best;
}

/* Add Dirichlet-like noise to root priors (simple approx) */
static void add_dirichlet_noise(Node *root, float alpha, float eps) {
  (void)alpha; // alpha not used in this simple sampler
  if (!root || eps <= 0.0f)
    return;

  int A = root->action_count;
  float *g = (float *)malloc(sizeof(float) * A);
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

/* Expand node: fill priors and return predicted value */
static float expand_node(Node *node, MuModel *model) {
  int A = node->action_count;

  float *logits = (float *)malloc(sizeof(float) * A);
  float *pri = (float *)malloc(sizeof(float) * A);
  float value = 0.0f;

  if (!logits || !pri) {
    free(logits);
    free(pri);
    return 0.0f;
  }

  mu_model_predict(model, node->latent, logits, &value);
  softmax(logits, A, pri);
  memcpy(node->P, pri, sizeof(float) * A);
  node->expanded = 1;

  free(pri);
  free(logits);
  return value;
}

/* Backup rewards + leaf value with discount */
static void backup_with_discount(Node *root, int *actions, float *rewards,
                                 int depth, float leaf_value, float gamma) {
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

/* convert visit counts to policy */
static void visits_to_pi(Node *root, float temperature, float *pi_out) {
  int A = root->action_count;
  if (temperature <= 0.0f)
    temperature = 1e-6f;

  double sum = 0.0;
  for (int a = 0; a < A; a++) {
    double val = pow((double)root->N[a], 1.0 / (double)temperature);
    if (!isfinite(val))
      val = 0.0;
    pi_out[a] = (float)val;
    sum += val;
  }

  if (sum > 0.0) {
    for (int a = 0; a < A; a++)
      pi_out[a] /= (float)sum;
  } else {
    for (int a = 0; a < A; a++)
      pi_out[a] = 1.0f / (float)A;
  }
}

/* allocate node */ static Node *node_create(int action_count, int latent_dim) {
  Node *n = (Node *)calloc(1, sizeof(Node));
  if (!n)
    return NULL;
  n->action_count = action_count;
  n->W = (float *)calloc(action_count, sizeof(float));
  n->Q = (float *)calloc(action_count, sizeof(float));
  n->N = (int *)calloc(action_count, sizeof(int));
  n->P = (float *)calloc(action_count, sizeof(float));
  n->children = (Node **)calloc(action_count, sizeof(Node *));
  n->latent = (float *)calloc(latent_dim, sizeof(float));
  n->expanded = 0;
  return n;
} /* free node recursively */
static void node_free(Node *n) {
  if (!n)
    return;
  free(n->W);
  free(n->Q);
  free(n->N);
  free(n->P);
  free(n->latent);
  if (n->children) {
    for (int i = 0; i < n->action_count; i++)
      node_free(n->children[i]);
    free(n->children);
  }
  free(n);
} /* sum of visits */
static int node_Nsum(Node *n) {
  int sum = 0;
  for (int i = 0; i < n->action_count; i++)
    sum += n->N[i];
  return sum;
} /* simple softmax */
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
    for (int i = 0; i < len; i++)
      out[i] = 1.0f / (float)len;
  }
} /* PUCT selection */
static int select_puct(Node *n, float c_puct) {
  int best = 0;
  float best_score = -FLT_MAX;
  float sqrt_N = sqrtf((float)(node_Nsum(n) + 1));
  for (int a = 0; a < n->action_count; a++) {
    float score = n->Q[a] + c_puct * n->P[a] * (sqrt_N / (1.0f + n->N[a]));
    if (score > best_score) {
      best_score = score;
      best = a;
    }
  }
  return best;
} /* Add Dirichlet noise to root priors */
static void add_dirichlet_noise(Node *root, float alpha, float eps) {
  if (!root || alpha <= 0.0f || eps <= 0.0f)
    return;
  int A = root->action_count;
  float *g = (float *)malloc(sizeof(float) * A);
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
} /* Expand node: fill priors and return predicted value */
static float expand_node(Node *node, MuModel *model) {
  int A = node->action_count;
  float *logits = (float *)malloc(sizeof(float) * A);
  float value = 0.0f;
  mu_model_predict(model, node->latent, logits, &value);
  float *pri = (float *)malloc(sizeof(float) * A);
  softmax(logits, A, pri);
  memcpy(node->P, pri, sizeof(float) * A);
  node->expanded = 1;
  free(pri);
  free(logits);
  return value;
} /* Backup rewards + leaf value with discount */
static void backup_with_discount(Node *root, int *actions, float *rewards,
                                 int depth, float leaf_value, float gamma) {
  float total = leaf_value;
  for (int i = depth - 1; i >= 0; i--) {
    total = rewards[i] + gamma * total;
  }
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
} /* convert visit counts to policy */
static void visits_to_pi(Node *root, float temperature, float *pi_out) {
  int A = root->action_count;
  if (temperature <= 0.0f)
    temperature = 1e-6f;
  double sum = 0.0;
  for (int a = 0; a < A; a++) {
    double val = pow((double)root->N[a], 1.0 / (double)temperature);
    if (!isfinite(val))
      val = 0.0;
    pi_out[a] = (float)val;
    sum += val;
  }
  if (sum > 0.0) {
    for (int a = 0; a < A; a++)
      pi_out[a] /= (float)sum;
  } else {
    for (int a = 0; a < A; a++)
      pi_out[a] = 1.0f / (float)A;
  }
}
MCTSResult mcts_run_latent(MuModel *model, const float *latent,
                           const MCTSParams *params) {
  MCTSResult res = {0};
  if (!model || !latent || !params)
    return res;
  int A = model->cfg.action_count;
  int L = model->cfg.latent_dim;
  Node *root = node_create(A, L);
  if (!root)
    return res; /* Copy latent directly */
  memcpy(root->latent, latent, sizeof(float) * L);
  float root_value = expand_node(root, model);
  if (params->dirichlet_alpha > 0.0f && params->dirichlet_eps > 0.0f)
    add_dirichlet_noise(root, params->dirichlet_alpha, params->dirichlet_eps);
  int max_depth = params->max_depth > 0 ? params->max_depth : 64;
  int *actions = malloc(sizeof(int) * max_depth);
  float *rewards = malloc(sizeof(float) * max_depth);
  for (int sim = 0; sim < params->num_simulations; sim++) {
    Node *node = root;
    int depth = 0;
    float *h_cur = malloc(sizeof(float) * L);
    memcpy(h_cur, root->latent, sizeof(float) * L);
    while (node->expanded && depth < max_depth) {
      int a = select_puct(node, params->c_puct);
      actions[depth] = a;
      if (!node->children[a]) {
        node->children[a] = node_create(A, L);
        float r = 0.0f;
        mu_model_dynamics(model, h_cur, a, node->children[a]->latent, &r);
        rewards[depth] = r;
        float leaf_value = expand_node(node->children[a], model);
        backup_with_discount(root, actions, rewards, depth + 1, leaf_value,
                             params->discount);
        break;
      } else {
        float r = 0.0f;
        mu_model_dynamics(model, h_cur, a, node->children[a]->latent, &r);
        rewards[depth] = r;
        memcpy(h_cur, node->children[a]->latent, sizeof(float) * L);
        node = node->children[a];
        depth++;
      }
    }
    free(h_cur);
  }
  float *pi = malloc(sizeof(float) * A);
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
} /* Main MCTS run */
MCTSResult mcts_run(MuModel *model, const float *obs,
                    const MCTSParams *params) {
  MCTSResult res = {0};
  if (!model || !obs || !params)
    return res;
  int A = model->cfg.action_count;
  int L = model->cfg.latent_dim;
  Node *root = node_create(A, L);
  if (!root)
    return res;
  mu_model_repr(model, obs, root->latent);
  float root_value = expand_node(root, model);
  if (params->dirichlet_alpha > 0.0f && params->dirichlet_eps > 0.0f)
    add_dirichlet_noise(root, params->dirichlet_alpha, params->dirichlet_eps);
  int max_depth = params->max_depth > 0 ? params->max_depth : 64;
  int *actions = (int *)malloc(sizeof(int) * max_depth);
  float *rewards = (float *)malloc(sizeof(float) * max_depth);
  for (int sim = 0; sim < params->num_simulations; sim++) {
    Node *node = root;
    int depth = 0;
    float *h_cur = (float *)malloc(sizeof(float) * L);
    memcpy(h_cur, root->latent, sizeof(float) * L);
    while (node->expanded && depth < max_depth) {
      int a = select_puct(node, params->c_puct);
      actions[depth] = a;
      float *h_next = NULL;
      if (!node->children[a]) {
        node->children[a] = node_create(A, L);
        h_next = node->children[a]->latent;
        float r = 0.0f;
        mu_model_dynamics(model, h_cur, a, h_next, &r);
        rewards[depth] = r;
        float leaf_value = expand_node(node->children[a], model);
        backup_with_discount(root, actions, rewards, depth + 1, leaf_value,
                             params->discount);
        break; // only break, don't free h_cur here } else { h_next =
               // node->children[a]->latent; float r = 0.0f;
               // mu_model_dynamics(model, h_cur, a, h_next, &r); rewards[depth]
               // = r; node = node->children[a]; memcpy(h_cur, h_next,
               // sizeof(float) * L); depth++; } } free(h_cur); // free exactly
               // once } float *pi = (float *)malloc(sizeof(float) * A);
               // visits_to_pi(root, params->temperature, pi); int best_a = 0;
               // float best_q = -INFINITY; for (int a = 0; a < A; a++) { float
               // q = (root->N[a] > 0) ? root->Q[a] : -INFINITY; if (q > best_q)
               // { best_q = q; best_a = a; } } if (best_q == -INFINITY) { //
               // all N==0 float best_p = -INFINITY; for (int a = 0; a < A; a++)
               // if (root->P[a] > best_p) { best_p = root->P[a]; best_a = a; }
               // } res.action_count = A; res.pi = pi; res.chosen_action =
               // best_a; res.root_value = root_value; free(actions);
               // free(rewards); node_free(root); return res; } void
               // mcts_result_free(MCTSResult *res) { if (!res) return;
               // free(res->pi); res->pi = NULL; res->action_count = 0;
               // res->chosen_action = 0; res->root_value = 0.0f;
      }
