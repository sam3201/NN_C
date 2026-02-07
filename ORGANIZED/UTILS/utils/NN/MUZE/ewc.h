#ifndef MUZE_EWC_H
#define MUZE_EWC_H

#include "muzero_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EWCState EWCState;
typedef struct MuEWC MuEWC;

MuEWC *mu_ewc_create(MuModel *m, long double lambda);
void mu_ewc_free(MuEWC *ewc);
void mu_ewc_set_lambda(MuEWC *ewc, long double lambda);
void mu_ewc_snapshot(MuEWC *ewc, MuModel *m);
void mu_ewc_zero_fisher(MuEWC *ewc);
void mu_ewc_accumulate(MuEWC *ewc, MuModel *m, long double scale);

// Net2WiderNet: widen a hidden layer while preserving function.
// layer_index is the hidden layer index in nn->layers (1..numLayers-2).
int mu_ewc_net2wider_nn(NN_t *nn, size_t layer_index, size_t delta,
                        const size_t *src_map, size_t src_len);
// Apply Net2WiderNet to all NN_t heads inside a MuModel (if present).
int mu_ewc_net2wider_model(MuModel *m, size_t layer_index, size_t delta);

#ifdef __cplusplus
}
#endif

#endif
