#include "ewc.h"
#include "../NN.h"
#include <stdlib.h>
#include <string.h>

struct EWCState {
  size_t layers;
  size_t *wcount;
  size_t *bcount;
  long double **theta_w;
  long double **theta_b;
  long double **fisher_w;
  long double **fisher_b;
  long double lambda;
};

struct MuEWC {
  EWCState *repr;
  EWCState *dyn;
  EWCState *pred;
  EWCState *vprefix;
  EWCState *reward;
};

static EWCState *ewc_state_create(NN_t *nn, long double lambda) {
  if (!nn)
    return NULL;
  EWCState *st = (EWCState *)calloc(1, sizeof(EWCState));
  if (!st)
    return NULL;
  st->layers = nn->numLayers - 1;
  st->lambda = lambda;
  st->wcount = (size_t *)calloc(st->layers, sizeof(size_t));
  st->bcount = (size_t *)calloc(st->layers, sizeof(size_t));
  st->theta_w = (long double **)calloc(st->layers, sizeof(long double *));
  st->theta_b = (long double **)calloc(st->layers, sizeof(long double *));
  st->fisher_w = (long double **)calloc(st->layers, sizeof(long double *));
  st->fisher_b = (long double **)calloc(st->layers, sizeof(long double *));
  if (!st->wcount || !st->bcount || !st->theta_w || !st->theta_b ||
      !st->fisher_w || !st->fisher_b) {
    free(st->wcount);
    free(st->bcount);
    free(st->theta_w);
    free(st->theta_b);
    free(st->fisher_w);
    free(st->fisher_b);
    free(st);
    return NULL;
  }

  for (size_t l = 0; l < st->layers; l++) {
    st->wcount[l] = nn->layers[l] * nn->layers[l + 1];
    st->bcount[l] = nn->layers[l + 1];
    st->theta_w[l] =
        (long double *)calloc(st->wcount[l], sizeof(long double));
    st->theta_b[l] =
        (long double *)calloc(st->bcount[l], sizeof(long double));
    st->fisher_w[l] =
        (long double *)calloc(st->wcount[l], sizeof(long double));
    st->fisher_b[l] =
        (long double *)calloc(st->bcount[l], sizeof(long double));
    if (!st->theta_w[l] || !st->theta_b[l] || !st->fisher_w[l] ||
        !st->fisher_b[l]) {
      for (size_t k = 0; k <= l; k++) {
        free(st->theta_w[k]);
        free(st->theta_b[k]);
        free(st->fisher_w[k]);
        free(st->fisher_b[k]);
      }
      free(st->wcount);
      free(st->bcount);
      free(st->theta_w);
      free(st->theta_b);
      free(st->fisher_w);
      free(st->fisher_b);
      free(st);
      return NULL;
    }
  }

  return st;
}

static void ewc_state_free(EWCState *st) {
  if (!st)
    return;
  for (size_t l = 0; l < st->layers; l++) {
    free(st->theta_w[l]);
    free(st->theta_b[l]);
    free(st->fisher_w[l]);
    free(st->fisher_b[l]);
  }
  free(st->wcount);
  free(st->bcount);
  free(st->theta_w);
  free(st->theta_b);
  free(st->fisher_w);
  free(st->fisher_b);
  free(st);
}

static void ewc_apply_hook(NN_t *nn, void *ctx) {
  EWCState *st = (EWCState *)ctx;
  if (!nn || !st || !(st->lambda > 0.0L))
    return;

  for (size_t l = 0; l < st->layers; l++) {
    for (size_t i = 0; i < st->wcount[l]; i++) {
      long double diff = nn->weights[l][i] - st->theta_w[l][i];
      nn->weights_grad[l][i] += st->lambda * st->fisher_w[l][i] * diff;
    }
    for (size_t j = 0; j < st->bcount[l]; j++) {
      long double diff = nn->biases[l][j] - st->theta_b[l][j];
      nn->biases_grad[l][j] += st->lambda * st->fisher_b[l][j] * diff;
    }
  }
}

static void ewc_snapshot_nn(EWCState *st, NN_t *nn) {
  if (!st || !nn)
    return;
  for (size_t l = 0; l < st->layers; l++) {
    memcpy(st->theta_w[l], nn->weights[l],
           sizeof(long double) * st->wcount[l]);
    memcpy(st->theta_b[l], nn->biases[l],
           sizeof(long double) * st->bcount[l]);
  }
}

static void ewc_zero_fisher_nn(EWCState *st) {
  if (!st)
    return;
  for (size_t l = 0; l < st->layers; l++) {
    memset(st->fisher_w[l], 0, sizeof(long double) * st->wcount[l]);
    memset(st->fisher_b[l], 0, sizeof(long double) * st->bcount[l]);
  }
}

static void ewc_accumulate_nn(EWCState *st, NN_t *nn, long double scale) {
  if (!st || !nn)
    return;
  if (!(scale > 0.0L))
    scale = 1.0L;
  for (size_t l = 0; l < st->layers; l++) {
    for (size_t i = 0; i < st->wcount[l]; i++) {
      long double g = nn->weights_grad[l][i];
      st->fisher_w[l][i] += scale * g * g;
    }
    for (size_t j = 0; j < st->bcount[l]; j++) {
      long double g = nn->biases_grad[l][j];
      st->fisher_b[l][j] += scale * g * g;
    }
  }
}

static void ewc_attach_hook(EWCState *st, NN_t *nn) {
  if (!st || !nn)
    return;
  NN_set_grad_hook(nn, ewc_apply_hook, st);
}

MuEWC *mu_ewc_create(MuModel *m, long double lambda) {
  if (!m || !m->use_nn)
    return NULL;
  MuEWC *ewc = (MuEWC *)calloc(1, sizeof(MuEWC));
  if (!ewc)
    return NULL;

  ewc->repr = ewc_state_create(m->nn_repr, lambda);
  ewc->dyn = ewc_state_create(m->nn_dyn, lambda);
  ewc->pred = ewc_state_create(m->nn_pred, lambda);
  ewc->vprefix = ewc_state_create(m->nn_vprefix, lambda);
  ewc->reward = ewc_state_create(m->nn_reward, lambda);

  if (!ewc->repr || !ewc->dyn || !ewc->pred || !ewc->vprefix ||
      !ewc->reward) {
    mu_ewc_free(ewc);
    return NULL;
  }

  ewc_attach_hook(ewc->repr, m->nn_repr);
  ewc_attach_hook(ewc->dyn, m->nn_dyn);
  ewc_attach_hook(ewc->pred, m->nn_pred);
  ewc_attach_hook(ewc->vprefix, m->nn_vprefix);
  ewc_attach_hook(ewc->reward, m->nn_reward);

  return ewc;
}

void mu_ewc_free(MuEWC *ewc) {
  if (!ewc)
    return;
  ewc_state_free(ewc->repr);
  ewc_state_free(ewc->dyn);
  ewc_state_free(ewc->pred);
  ewc_state_free(ewc->vprefix);
  ewc_state_free(ewc->reward);
  free(ewc);
}

void mu_ewc_set_lambda(MuEWC *ewc, long double lambda) {
  if (!ewc)
    return;
  if (ewc->repr)
    ewc->repr->lambda = lambda;
  if (ewc->dyn)
    ewc->dyn->lambda = lambda;
  if (ewc->pred)
    ewc->pred->lambda = lambda;
  if (ewc->vprefix)
    ewc->vprefix->lambda = lambda;
  if (ewc->reward)
    ewc->reward->lambda = lambda;
}

void mu_ewc_snapshot(MuEWC *ewc, MuModel *m) {
  if (!ewc || !m)
    return;
  ewc_snapshot_nn(ewc->repr, m->nn_repr);
  ewc_snapshot_nn(ewc->dyn, m->nn_dyn);
  ewc_snapshot_nn(ewc->pred, m->nn_pred);
  ewc_snapshot_nn(ewc->vprefix, m->nn_vprefix);
  ewc_snapshot_nn(ewc->reward, m->nn_reward);
}

void mu_ewc_zero_fisher(MuEWC *ewc) {
  if (!ewc)
    return;
  ewc_zero_fisher_nn(ewc->repr);
  ewc_zero_fisher_nn(ewc->dyn);
  ewc_zero_fisher_nn(ewc->pred);
  ewc_zero_fisher_nn(ewc->vprefix);
  ewc_zero_fisher_nn(ewc->reward);
}

void mu_ewc_accumulate(MuEWC *ewc, MuModel *m, long double scale) {
  if (!ewc || !m)
    return;
  ewc_accumulate_nn(ewc->repr, m->nn_repr, scale);
  ewc_accumulate_nn(ewc->dyn, m->nn_dyn, scale);
  ewc_accumulate_nn(ewc->pred, m->nn_pred, scale);
  ewc_accumulate_nn(ewc->vprefix, m->nn_vprefix, scale);
  ewc_accumulate_nn(ewc->reward, m->nn_reward, scale);
}
