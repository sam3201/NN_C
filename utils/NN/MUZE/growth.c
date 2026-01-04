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
