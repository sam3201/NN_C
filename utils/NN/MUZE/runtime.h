(base) samueldasari@Samuels-MacBook-Air MUZE % ls
all.h           mcts.c          replay_buffer.h toy_env.c       util.h
ewc.c           mcts.h          runtime.c       toy_env.h
ewc.h           muzero_model.c  runtime.h       trainer.c
growth.c        muzero_model.h  selfplay.c      trainer.h
growth.h        replay_buffer.c selfplay.h      util.c
(base) samueldasari@Samuels-MacBook-Air MUZE % for file in *;
do
  if
    [-f "$file"]; then
    echo "--- Filename: $file ---"
    cat "$file"
    echo "" # Optional: Add a blank line for readability between files
  fi
done

--- Filename: all.h ---
// MUZE/all.h

#ifndef MUZE_ALL_H
#define MUZE_ALL_H

#include "ewc.h"
#include "growth.h"
#include "mcts.h"
#include "muzero_model.h"
#include "replay_buffer.h"
#include "runtime.h"
#include "selfplay.h"
#include "toy_env.h"
#include "trainer.h"
#include "util.h"

#endif // MUZE_ALL_H
#Optional : Add a blank line for readability between files
--- Filename: ewc.c ---
#Optional : Add a blank line for readability between files
--- Filename: ewc.h ---
#Optional : Add a blank line for readability between files
--- Filename: growth.c ---
#include "growth.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Small random init for new weights */
static float small_rand() {
  return ((float)rand() / (float)RAND_MAX) * 0.002f - 0.001f;
}

int mu_model_grow_latent(MuModel *m, int new_L) {
  int old_L = m->cfg.latent_dim;
  int O = m->cfg.obs_dim;
  int A = m->cfg.action_count;

  if (new_L <= old_L)
    return -1;

  printf("[Growth] Increasing Latent Dims: %d -> %d\n", old_L, new_L);

  /* Representation weights */
  float *new_repr = malloc(sizeof(float) * new_L * O);
  for (int i = 0; i < new_L; i++) {
    for (int j = 0; j < O; j++) {
      new_repr[i * O + j] = (i < old_L) ? m->repr_W[i * O + j] : small_rand();
    }
  }

  /* Dynamics weights */
  float *new_dyn = malloc(sizeof(float) * new_L * new_L);
  for (int i = 0; i < new_L; i++) {
    for (int j = 0; j < new_L; j++) {
      new_dyn[i * new_L + j] =
          (i < old_L && j < old_L) ? m->dyn_W[i * old_L + j] : small_rand();
    }
  }

  /* Prediction weights */
  float *new_pred = malloc(sizeof(float) * (A + 1) * new_L);
  for (int a = 0; a < A + 1; a++) {
    for (int j = 0; j < new_L; j++) {
      new_pred[a * new_L + j] =
          (j < old_L) ? m->pred_W[a * old_L + j] : small_rand();
    }
  }

  free(m->repr_W);
  free(m->dyn_W);
  free(m->pred_W);

  m->repr_W = new_repr;
  m->dyn_W = new_dyn;
  m->pred_W = new_pred;

  m->repr_W_count = new_L * O;
  m->dyn_W_count = new_L * new_L;
  m->pred_W_count = (A + 1) * new_L;

  m->cfg.latent_dim = new_L;

  return 0;
}
#Optional : Add a blank line for readability between files
-- -Filename : growth.h-- -
#ifndef GROWTH_H
#define GROWTH_H

#include "muzero_model.h"

    int mu_model_grow_latent(MuModel *m, int new_latent_dim);

#endif
#Optional : Add a blank line for readability between files
-- -Filename : mcts.c-- -
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
}

/* free node recursively */
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

/* Add Dirichlet noise to root priors */
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
}

/* Expand node: fill priors and return predicted value */
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
}

/* Backup rewards + leaf value with discount */
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

/* Main MCTS run */
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
        break; // only break, don't free h_cur here
      } else {
        h_next = node->children[a]->latent;
        float r = 0.0f;
        mu_model_dynamics(model, h_cur, a, h_next, &r);
        rewards[depth] = r;
        node = node->children[a];
        memcpy(h_cur, h_next, sizeof(float) * L);
        depth++;
      }
    }
    free(h_cur); // free exactly once
  }

  float *pi = (float *)malloc(sizeof(float) * A);
  visits_to_pi(root, params->temperature, pi);

  int best_a = 0;
  float best_q = -INFINITY;
  for (int a = 0; a < A; a++) {
    float q = (root->N[a] > 0) ? root->Q[a] : -INFINITY;
    if (q > best_q) {
      best_q = q;
      best_a = a;
    }
  }
  if (best_q == -INFINITY) { // all N==0
    float best_p = -INFINITY;
    for (int a = 0; a < A; a++)
      if (root->P[a] > best_p) {
        best_p = root->P[a];
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

void mcts_result_free(MCTSResult *res) {
  if (!res)
    return;
  free(res->pi);
  res->pi = NULL;
  res->action_count = 0;
  res->chosen_action = 0;
  res->root_value = 0.0f;
}
#Optional : Add a blank line for readability between files
-- -Filename : mcts.h-- -
#ifndef MCTS_H
#define MCTS_H

#include "muzero_model.h"

#ifdef __cplusplus
    extern "C" {
#endif

  typedef struct {
    int num_simulations;
    float c_puct;
    int max_depth;
    float dirichlet_alpha;
    float dirichlet_eps;
    float temperature;
    float discount;
  } MCTSParams;

  typedef struct {
    int chosen_action;
    int action_count;
    float *pi;        // visit-count policy, caller frees
    float root_value; // estimated root value
  } MCTSResult;

  MCTSResult mcts_run(MuModel * model, const float *obs,
                      const MCTSParams *params);
  void mcts_result_free(MCTSResult * res);

#ifdef __cplusplus
}
#endif
#endif
#Optional : Add a blank line for readability between files
-- -Filename : muzero_model.c-- -
#include "muzero_model.h"
#include "runtime.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

    /* ------------------------
       Create model
       ------------------------ */
    MuModel *mu_model_create(const MuConfig *cfg) {
  MuModel *m = (MuModel *)malloc(sizeof(MuModel));
  m->cfg = *cfg;

  int obs = cfg->obs_dim;
  int lat = cfg->latent_dim;
  int act = cfg->action_count;

  /* allocate fake/placeholder weights */
  m->repr_W_count = obs * lat;
  m->dyn_W_count = (lat + 1) * lat;  // +1 for action embedding
  m->pred_W_count = lat * (act + 1); // policy + value head

  m->repr_W = (float *)malloc(sizeof(float) * m->repr_W_count);
  m->dyn_W = (float *)malloc(sizeof(float) * m->dyn_W_count);
  m->pred_W = (float *)malloc(sizeof(float) * m->pred_W_count);

  /* simple initialization */
  for (int i = 0; i < m->repr_W_count; i++)
    m->repr_W[i] = 0.01f;
  for (int i = 0; i < m->dyn_W_count; i++)
    m->dyn_W[i] = 0.01f;
  for (int i = 0; i < m->pred_W_count; i++)
    m->pred_W[i] = 0.01f;

  m->runtime = mu_runtime_create(m, 4096, 0.95f);

  return m;
}

/* ------------------------
   Free model
   ------------------------ */
void mu_model_free(MuModel *m) {
  if (!m)
    return;
  free(m->repr_W);
  free(m->dyn_W);
  free(m->pred_W);
  free(m);
  mu_runtime_free(m->runtime);
}

/* ------------------------
   Representation function
   obs → latent
   (Dummy linear layer)
   ------------------------ */
void mu_model_repr(MuModel *m, const float *obs, float *latent_out) {
  int O = m->cfg.obs_dim;
  int L = m->cfg.latent_dim;

  for (int i = 0; i < L; i++) {
    float sum = 0.f;
    for (int j = 0; j < O; j++) {
      sum += obs[j] * m->repr_W[i * O + j];
    }
    latent_out[i] = tanhf(sum);
  }
}

/* ------------------------
   Dynamics function
   latent + action → latent' + reward
   ------------------------ */
void mu_model_dynamics(MuModel *m, const float *latent_in, int action,
                       float *latent_out, float *reward_out) {
  int L = m->cfg.latent_dim;

  /* simple deterministic dynamics */
  for (int i = 0; i < L; i++) {
    float sum = 0.f;
    for (int j = 0; j < L; j++) {
      sum += latent_in[j] * m->dyn_W[i * L + j];
    }
    sum += 0.1f * action;
    latent_out[i] = tanhf(sum);
  }

  *reward_out = 0.01f * action; // placeholder
}

/* ------------------------
   Prediction function
   latent → (policy_logits, value)
   ------------------------ */
void mu_model_predict(MuModel *m, const float *latent_in,
                      float *policy_logits_out, float *value_out) {
  int L = m->cfg.latent_dim;
  int A = m->cfg.action_count;

  /* policy */
  for (int a = 0; a < A; a++) {
    float sum = 0.f;
    for (int j = 0; j < L; j++) {
      sum += latent_in[j] * m->pred_W[a * L + j];
    }
    policy_logits_out[a] = sum;
  }

  /* value head */
  float sum = 0.f;
  for (int j = 0; j < L; j++) {
    sum += latent_in[j] * m->pred_W[(A * L) + j];
  }
  *value_out = tanhf(sum);
}

void mu_model_step(MuModel *m, const float *obs, int action, float reward) {
  mu_runtime_step((MuRuntime *)m->runtime, m, obs, action, reward);
}

void mu_model_end_episode(MuModel *m, float terminal_reward) {
  mu_runtime_end_episode((MuRuntime *)m->runtime, m, terminal_reward);
}

void mu_model_reset_episode(MuModel *m) {
  mu_runtime_reset_episode((MuRuntime *)m->runtime);
}

void mu_model_train(MuModel *m) {
  mu_runtime_train((MuRuntime *)m->runtime, m);
}
#Optional : Add a blank line for readability between files
-- -Filename : muzero_model.h-- -
#ifndef MUZERO_MODEL_H
#define MUZERO_MODEL_H

#ifdef __cplusplus
    extern "C" {
#endif

  typedef struct {
    int obs_dim;
    int latent_dim;
    int action_count;
  } MuConfig;

  typedef struct {
    MuConfig cfg;
    float *repr_W;
    float *dyn_W;
    float *pred_W;
    int repr_W_count;
    int dyn_W_count;
    int pred_W_count;

    void *runtime;
  } MuModel;

  MuModel *mu_model_create(const MuConfig *cfg);
  void mu_model_free(MuModel * m);

  void mu_model_repr(MuModel * m, const float *obs, float *latent_out);
  void mu_model_dynamics(MuModel * m, const float *latent_in, int action,
                         float *latent_out, float *reward_out);
  void mu_model_predict(MuModel * m, const float *latent_in,
                        float *policy_logits_out, float *value_out);

  void mu_model_step(MuModel * m, const float *obs, int action, float reward);
  void mu_model_end_episode(MuModel * m, float terminal_reward);
  void mu_model_reset_episode(MuModel * m);
  void mu_model_train(MuModel * m);

#ifdef __cplusplus
}
#endif
#endif
#Optional : Add a blank line for readability between files
-- -Filename : replay_buffer.c-- -
#include "replay_buffer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

    struct ReplayBuffer {
  size_t capacity;
  size_t size;
  size_t write_idx;
  int obs_dim;
  int action_count;
  float *obs_buf; /* capacity * obs_dim */
  float *pi_buf;  /* capacity * action_count */
  float *z_buf;   /* capacity */
};

ReplayBuffer *rb_create(size_t capacity, int obs_dim, int action_count) {
  ReplayBuffer *rb = (ReplayBuffer *)malloc(sizeof(ReplayBuffer));
  if (!rb)
    return NULL;
  rb->capacity = capacity;
  rb->size = 0;
  rb->write_idx = 0;
  rb->obs_dim = obs_dim;
  rb->action_count = action_count;
  rb->obs_buf = (float *)malloc(sizeof(float) * capacity * obs_dim);
  rb->pi_buf = (float *)malloc(sizeof(float) * capacity * action_count);
  rb->z_buf = (float *)malloc(sizeof(float) * capacity);
  if (!rb->obs_buf || !rb->pi_buf || !rb->z_buf) {
    rb_free(rb);
    return NULL;
  }
  return rb;
}

void rb_free(ReplayBuffer *rb) {
  if (!rb)
    return;
  if (rb->obs_buf)
    free(rb->obs_buf);
  if (rb->pi_buf)
    free(rb->pi_buf);
  if (rb->z_buf)
    free(rb->z_buf);
  free(rb);
}

void rb_push(ReplayBuffer *rb, const float *obs, const float *pi, float z) {
  if (!rb)
    return;
  size_t idx = rb->write_idx;
  memcpy(rb->obs_buf + idx * rb->obs_dim, obs, sizeof(float) * rb->obs_dim);
  memcpy(rb->pi_buf + idx * rb->action_count, pi,
         sizeof(float) * rb->action_count);
  rb->z_buf[idx] = z;
  rb->write_idx = (rb->write_idx + 1) % rb->capacity;
  if (rb->size < rb->capacity)
    rb->size++;
}

static int rand_int(int n) {
  return (int)((double)rand() / ((double)RAND_MAX + 1.0) * n);
}

int rb_sample(ReplayBuffer *rb, int batch, float *obs_batch, float *pi_batch,
              float *z_batch) {
  if (!rb || rb->size == 0)
    return 0;
  int actual = batch;
  if ((size_t)batch > rb->size)
    actual = (int)rb->size;
  for (int i = 0; i < actual; i++) {
    int idx = rand_int((int)rb->size);
    memcpy(obs_batch + i * rb->obs_dim, rb->obs_buf + idx * rb->obs_dim,
           sizeof(float) * rb->obs_dim);
    memcpy(pi_batch + i * rb->action_count, rb->pi_buf + idx * rb->action_count,
           sizeof(float) * rb->action_count);
    z_batch[i] = rb->z_buf[idx];
  }
  return actual;
}

size_t rb_size(ReplayBuffer *rb) { return rb ? rb->size : 0; }

#Optional : Add a blank line for readability between files
-- -Filename : replay_buffer.h-- -
#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

#include <stddef.h>

#ifdef __cplusplus
    extern "C" {
#endif

  typedef struct ReplayBuffer ReplayBuffer;

  ReplayBuffer *rb_create(size_t capacity, int obs_dim, int action_count);
  void rb_free(ReplayBuffer * rb);
  void rb_push(ReplayBuffer * rb, const float *obs, const float *pi, float z);
  int rb_sample(ReplayBuffer * rb, int batch, float *obs_batch, float *pi_batch,
                float *z_batch);
  size_t rb_size(ReplayBuffer * rb);

#ifdef __cplusplus
}
#endif

#endif
#Optional : Add a blank line for readability between files
-- -Filename : runtime.c-- -
#include "runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

    MuRuntime *mu_runtime_create(MuModel *model, int capacity, float gamma) {
  MuRuntime *rt = calloc(1, sizeof(MuRuntime));
  rt->rb = rb_create(capacity, model->cfg.obs_dim, model->cfg.action_count);
  rt->gamma = gamma;
  rt->last_obs = malloc(sizeof(float) * model->cfg.obs_dim);
  rt->has_last = 0;
  return rt;
}

void mu_runtime_free(MuRuntime *rt) {
  if (!rt)
    return;
  rb_free(rt->rb);
  free(rt->last_obs);
  free(rt);
}

void mu_runtime_step(MuModel *model, MuRuntime *rt, const float *obs,
                     int action, float reward) {
  if (!rt->has_last) {
    memcpy(rt->last_obs, obs, sizeof(float) * model->cfg.obs_dim);
    rt->last_action = action;
    rt->has_last = 1;
    return;
  }

  /* Single-step target (bootstrap handled by MuZero value head) */
  float z = reward;

  float pi[model->cfg.action_count];
  for (int i = 0; i < model->cfg.action_count; i++)
    pi[i] = (i == action) ? 1.0f : 0.0f;

  rb_push(rt->rb, rt->last_obs, pi, z);

  memcpy(rt->last_obs, obs, sizeof(float) * model->cfg.obs_dim);
  rt->last_action = action;
}

void mu_runtime_end_episode(MuModel *model, MuRuntime *rt,
                            float terminal_reward) {
  if (!rt->has_last)
    return;

  float pi[model->cfg.action_count];
  memset(pi, 0, sizeof(pi));

  rb_push(rt->rb, rt->last_obs, pi, terminal_reward);
  rt->has_last = 0;
}

void mu_runtime_reset_episode(MuRuntime *rt) { rt->has_last = 0; }

void mu_model_train(MuModel *model, MuRuntime *rt) {
  if (rb_size(rt->rb) < 32)
    return;

  /* Placeholder SGD stub — you can upgrade later */
  printf("[MUZE] Training step (samples=%zu)\n", rb_size(rt->rb));
}

#include "runtime.h"

void mu_model_step(MuModel *m, const float *obs, int action, float reward) {
  mu_model_step(m, (MuRuntime *)m->runtime, obs, action, reward);
}

void mu_model_end_episode(MuModel *m, float terminal_reward) {
  mu_model_end_episode(m, (MuRuntime *)m->runtime, terminal_reward);
}

void mu_model_reset_episode(MuModel *m) {
  mu_model_reset_episode((MuRuntime *)m->runtime);
}

void mu_model_train(MuModel *m) { mu_model_train(m, (MuRuntime *)m->runtime); }
#Optional : Add a blank line for readability between files
-- -Filename : runtime.h-- -
#ifndef MUZE_RUNTIME_H
#define MUZE_RUNTIME_H

#include "muzero_model.h"
#include "replay_buffer.h"
#include <stdint.h>

#define TRAIN_WINDOW 256

    typedef struct {
  ReplayBuffer *rb;
  float *last_obs;
  int last_action;
  int has_last;
  float gamma;

  uint64_t total_steps;
} MuRuntime;

/* Runtime lifecycle */
MuRuntime *mu_runtime_create(MuModel *model, int capacity, float gamma);
void mu_runtime_free(MuRuntime *rt);

/* Internal runtime ops */
void mu_runtime_step(MuRuntime *rt, MuModel *model, const float *obs,
                     int action, float reward);

void mu_runtime_end_episode(MuRuntime *rt, MuModel *model,
                            float terminal_reward);
void mu_runtime_reset_episode(MuRuntime *rt);
void mu_runtime_train(MuRuntime *rt, MuModel *model);

#endif
#Optional : Add a blank line for readability between files
-- -Filename : selfplay.c-- -
#include "selfplay.h"
#include "toy_env.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

    /* Helper: compute discounted returns for an episode of length T.
       rewards[0..T-1], gamma -> returns z_t = sum_{k=0..T-1-t} gamma^k *
       rewards[t+k] output z_out must be length T
    */
    static void compute_discounted_returns(const float *rewards, int T,
                                           float gamma, float *z_out) {
  for (int t = 0; t < T; t++) {
    float acc = 0.0f;
    float g = 1.0f;
    for (int k = t; k < T; k++) {
      acc += g * rewards[k];
      g *= gamma;
    }
    z_out[t] = acc;
  }
}

/* Runs self-play episodes and pushes training tuples into replay buffer */
void selfplay_run(MuModel *model, void *env_state, env_reset_fn env_reset,
                  env_step_fn env_step, MCTSParams *mcts_params,
                  SelfPlayParams *sp_params, ReplayBuffer *rb) {
  if (!model || !env_reset || !env_step || !mcts_params || !sp_params || !rb)
    return;

  int obs_dim = model->cfg.obs_dim;
  int A = model->cfg.action_count;
  int max_steps = sp_params->max_steps > 0 ? sp_params->max_steps : 200;

  /* temporary buffers for one episode */
  float *obs_buf = malloc(sizeof(float) * max_steps * obs_dim);
  float *pi_buf = malloc(sizeof(float) * max_steps * A);
  float *reward_buf = malloc(sizeof(float) * max_steps);
  int *act_buf = malloc(sizeof(int) * max_steps);

  for (int ep = 0; ep < sp_params->total_episodes; ep++) {
    /* reset env */
    float *obs0 = malloc(sizeof(float) * obs_dim);
    env_reset(env_state, obs0);

    int step = 0;
    int done = 0;
    float obs_cur[obs_dim];
    memcpy(obs_cur, obs0, sizeof(float) * obs_dim);

    while (!done && step < max_steps) {
      /* run MCTS for current obs */
      MCTSParams mp = *mcts_params;
      mp.temperature = sp_params->temperature > 0.0f ? sp_params->temperature
                                                     : mcts_params->temperature;

      MCTSResult mr = mcts_run(model, obs_cur, &mp);

      /* sample action according to pi (with rng) */
      float r = (float)rand() / (float)RAND_MAX;
      float cum = 0.0f;
      int chosen = 0;
      for (int a = 0; a < A; a++) {
        cum += mr.pi[a];
        if (r <= cum) {
          chosen = a;
          break;
        }
      }

      /* store obs and pi */
      memcpy(obs_buf + step * obs_dim, obs_cur, sizeof(float) * obs_dim);
      memcpy(pi_buf + step * A, mr.pi, sizeof(float) * A);

      /* step env */
      float next_obs[obs_dim];
      float reward = 0.0f;
      int done_flag = 0;
      int ret = env_step(env_state, chosen, next_obs, &reward, &done_flag);
      if (ret != 0) {
        /* env error: stop episode */
        done_flag = 1;
      }
      reward_buf[step] = reward;
      act_buf[step] = chosen;

      /* advance */
      memcpy(obs_cur, next_obs, sizeof(float) * obs_dim);
      step++;

      mcts_result_free(&mr);

      if (done_flag)
        done = 1;
    }

    /* compute discounted returns z_t and push samples to replay buffer */
    float *z = malloc(sizeof(float) * step);
    compute_discounted_returns(reward_buf, step, sp_params->gamma, z);
    for (int t = 0; t < step; t++) {
      rb_push(rb, obs_buf + t * obs_dim, pi_buf + t * A, z[t]);
    }
    free(z);
    free(obs0);
  }

  free(obs_buf);
  free(pi_buf);
  free(reward_buf);
  free(act_buf);
}
#Optional : Add a blank line for readability between files
-- -Filename : selfplay.h-- -
#ifndef SELFPLAY_H
#define SELFPLAY_H

#include "mcts.h"
#include "muzero_model.h"
#include "replay_buffer.h"

#ifdef __cplusplus
    extern "C" {
#endif

  /* Environment callback signatures:
     - reset: should write initial observation into obs_out (length obs_dim)
     - step: given action, perform env step, write next obs to obs_out (length
     obs_dim), set *reward_out and *done_out (0/1). Return 0 on success.
     env_state is an opaque pointer to environment instance. */
  typedef void (*env_reset_fn)(void *env_state, float *obs_out);
  typedef int (*env_step_fn)(void *state, int action, float *obs, float *reward,
                             int *done);

  /* Self-play params */
  typedef struct {
    int max_steps;      /* max steps per episode */
    float gamma;        /* discount for returns */
    float temperature;  /* sampling temperature during self-play (root) */
    int total_episodes; /* how many episodes to run */
  } SelfPlayParams;

  /* Run self-play episodes: each episode uses MCTS to choose actions (with
     MCTSParams) and pushes (obs, pi, z) samples into the replay buffer.
     env_state is user-provided and env callbacks operate on it.
  */
  void selfplay_run(MuModel * model, void *env_state, env_reset_fn env_reset,
                    env_step_fn env_step, MCTSParams *mcts_params,
                    SelfPlayParams *sp_params, ReplayBuffer *rb);

#ifdef __cplusplus
}
#endif
#endif
#Optional : Add a blank line for readability between files
-- -Filename : toy_env.c-- -
#include "toy_env.h"
#include <string.h>

    void toy_env_reset(void *state_ptr, float *obs) {
  ToyEnvState *state = (ToyEnvState *)state_ptr;
  state->pos = 0;
  memset(obs, 0, sizeof(float) * state->size);
  obs[state->pos] = 1.0f; // one-hot position
}

// changed return type from void -> int
int toy_env_step(void *state_ptr, int action, float *obs, float *reward,
                 int *done) {
  ToyEnvState *state = (ToyEnvState *)state_ptr;

  if (action == 0 && state->pos > 0)
    state->pos--;
  if (action == 1 && state->pos < state->size - 1)
    state->pos++;

  *reward = (state->pos == state->size - 1) ? 1.0f : 0.0f;
  *done = (state->pos == state->size - 1) ? 1 : 0;

  // update observation (one-hot)
  memset(obs, 0, sizeof(float) * state->size);
  obs[state->pos] = 1.0f;

  return 0; // success
}
#Optional : Add a blank line for readability between files
-- -Filename : toy_env.h-- -
#ifndef TOY_ENV_H
#define TOY_ENV_H

    typedef struct {
  int pos;
  int size;
} ToyEnvState;

typedef void (*env_reset_fn)(void *state, float *obs);
typedef int (*env_step_fn)(void *state, int action, float *obs, float *reward,
                           int *done);

void toy_env_reset(void *state_ptr, float *obs);
int toy_env_step(void *state_ptr, int action, float *obs, float *reward,
                 int *done);

#endif
#Optional : Add a blank line for readability between files
-- -Filename : trainer.c-- -
#Optional : Add a blank line for readability between files
    -- -Filename : trainer.h-- -
#Optional : Add a blank line for readability between files
    -- -Filename : util.c-- -
#Optional : Add a blank line for readability between files
    -- -Filename : util.h-- -
#Optional : Add a blank line for readability between files
    (base)samueldasari @Samuels - MacBook - Air MUZE %
