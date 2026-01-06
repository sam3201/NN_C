#include "FORERUNNER.h"
#include "../utils/NN/TRANSFORMER.h"
#include <math.h>

// Initialize the ForeRunner model
ForeRunner_t *ForeRunner_init(size_t input_dim, size_t num_heads,
                              unsigned int num_contents) {
  ForeRunner_t *model = (ForeRunner_t *)malloc(sizeof(ForeRunner_t));
  if (!model)
    return NULL;

  model->transformer = TRANSFORMER_init(input_dim, num_contents, num_heads);
  if (!model->transformer) {
    free(model);
    return NULL;
  }

  model->num_contexts = num_contents;

  return model;
}

// Compute context from input data
unsigned int ForeRunner_forward(ForeRunner_t *model,
                                long double **input_sequence,
                                unsigned int seq_length) {
  if (!model || !input_sequence || seq_length == 0)
    return NONE;

  long double *logits =
      TRANSFORMER_forward(model->transformer, input_sequence, seq_length);
  if (!logits)
    return NONE;

  unsigned int C = model->num_contexts;
  if (C == 0) {
    free(logits);
    return NONE;
  }

  /* argmax over first C outputs */
  unsigned int best = 0;
  long double bestv = logits[0];
  for (unsigned int i = 1; i < C; i++) {
    if (logits[i] > bestv) {
      bestv = logits[i];
      best = i;
    }
  }

  free(logits);
  return best;
}

// Backpropagation function with gradient computation
void ForeRunner_backprop(ForeRunner_t *model, long double **input_sequence,
                         unsigned int seq_length, long double *grad_loss) {
  (void)model;
  (void)input_sequence;
  (void)seq_length;
  (void)grad_loss;
  /* TODO: build grad_output[seq_length][model_dim] and call
   * TRANSFORMER_backprop */
}

// Destroy the ForeRunner model
void ForeRunner_destroy(ForeRunner_t *model) {
  if (!model)
    return;

  if (model->transformer) {
    TRANSFORMER_destroy(model->transformer);
  }

  free(model);
}
