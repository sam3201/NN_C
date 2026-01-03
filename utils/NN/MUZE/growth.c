#include "growth.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper to get a random small float for new weights
static float small_rand() {
  return ((float)rand() / (float)RAND_MAX) * 0.002f - 0.001f;
}

int mu_model_grow_latent(MuModel *m, int new_L) {
  int old_L = m->cfg.latent_dim;
  int O = m->cfg.obs_dim;
  int A = m->cfg.action_count;

  if (new_L <= old_L)
    return -1; // Can't grow to a smaller size

  printf("[Growth] Increasing Latent Dims: %d -> %d\n", old_L, new_L);

  // 1. GROW REPRESENTATION WEIGHTS (Input -> Latent)
  // Old size: old_L * O | New size: new_L * O
  float *new_repr = (float *)malloc(sizeof(float) * new_L * O);
  for (int i = 0; i < new_L; i++) {
    for (int j = 0; j < O; j++) {
      if (i < old_L) {
        new_repr[i * O + j] = m->repr_W[i * O + j]; // Copy old
      } else {
        new_repr[i * O + j] = small_rand(); // New neuron
      }
    }
  }

  // 2. GROW DYNAMICS WEIGHTS (Latent -> Latent)
  // Old size: old_L * old_L | New size: new_L * new_L
  float *new_dyn = (float *)malloc(sizeof(float) * new_L * new_L);
  for (int i = 0; i < new_L; i++) {
    for (int j = 0; j < new_L; j++) {
      if (i < old_L && j < old_L) {
        new_dyn[i * new_L + j] = m->dyn_W[i * old_L + j]; // Copy old
      } else {
        new_dyn[i * new_L + j] = small_rand();
      }
    }
  }

  // 3. GROW PREDICTION WEIGHTS (Latent -> Policy + Value)
  // Policy weights: A * L | Value weights: 1 * L
  // Note: muzero_model.c packs value head at the end
  float *new_pred = (float *)malloc(sizeof(float) * (A + 1) * new_L);
  for (int a = 0; a < (A + 1); a++) {
    for (int j = 0; j < new_L; j++) {
      if (j < old_L) {
        new_pred[a * new_L + j] = m->pred_W[a * old_L + j];
      } else {
        new_pred[a * new_L + j] = small_rand();
      }
    }
  }

  // 4. SWAP AND FREE
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
