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

#ifdef __cplusplus
}
#endif

#endif
