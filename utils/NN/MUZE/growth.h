#ifndef GROWTH_H
#define GROWTH_H

#include "muzero_model.h"

// Expands the latent dimension of a MuModel to a new size.
// Returns 0 on success, non-zero on failure.
int mu_model_grow_latent(MuModel *m, int new_latent_dim);

#endif
