#include "growth.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Small random init for new weights */
static float small_rand() {
  return ((float)rand() / (float)RAND_MAX) * 0.002f - 0.001f;
}

int mu_model_grow_latent(MuModel *m, int new_Latent_dim) {
  int old_L = m->cfg.latent_dim;
  int O = m->cfg.obs_dim;
  int A = m->cfg.action_count;

  if (new_Latent_dim <= old_L)
    return -1;

  /* Representation weights */
  float *new_repr = malloc(sizeof(float) * new_Latent_dim * O);
  for (int i = 0; i < new_Latent_dim; i++) {
    for (int j = 0; j < O; j++) {
      new_repr[i * O + j] = (i < old_L) ? m->repr_W[i * O + j] : small_rand();
    }
  }

  /* Dynamics weights: shape L x (L+1) */
  float *new_dyn =
      malloc(sizeof(float) * new_Latent_dim * (new_Latent_dim + 1));
  for (int i = 0; i < new_Latent_dim; i++) {
    for (int j = 0; j < new_Latent_dim + 1; j++) {

      // if this is an old row and an old latent column
      if (i < old_L && j < old_L) {
        new_dyn[i * (new_Latent_dim + 1) + j] = m->dyn_W[i * (old_L + 1) + j];
      }
      // preserve old action column into new action column
      else if (i < old_L && j == new_Latent_dim) {
        new_dyn[i * (new_Latent_dim + 1) + j] =
            m->dyn_W[i * (old_L + 1) + old_L];
      }
      // everything else is new
      else {
        new_dyn[i * (new_Latent_dim + 1) + j] = small_rand();
      }
    }
  }

  /* Prediction weights */
  float *new_pred = malloc(sizeof(float) * (A + 1) * new_Latent_dim);
  for (int a = 0; a < A + 1; a++) {
    for (int j = 0; j < new_Latent_dim; j++) {
      new_pred[a * new_Latent_dim + j] =
          (j < old_L) ? m->pred_W[a * old_L + j] : small_rand();
    }
  }

  free(m->repr_W);
  free(m->dyn_W);
  free(m->pred_W);

  m->repr_W = new_repr;
  m->dyn_W = new_dyn;
  m->pred_W = new_pred;

  m->repr_W_count = new_Latent_dim * O;
  m->pred_W_count = (A + 1) * new_Latent_dim;

  m->dyn_W_count = new_Latent_dim * (new_Latent_dim + 1);

  m->cfg.latent_dim = new_Latent_dim;

  return 0;
}
