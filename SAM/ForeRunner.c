#include "ForeRunner.h"
#include "../utils/NN/TRANSFORMER.h"
#include <math.h>

// Initialize the ForeRunner model
ForeRunner_t *ForeRunner_init(size_t input_dim, size_t num_heads,
                              unsigned int num_contents) {
  ForeRunner_t *model = (ForeRunner_t *)malloc(sizeof(ForeRunner_t));
  if (!model)
    return NULL;

  model->transformer = TRANSFORMER_init(input_dim, num_heads);
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
  if (!model || !input_sequence)
    return NONE;

  long double **features =
      TRANSFORMER_forward(model->transformer, input_sequence, seq_length);
  if (!features)
    return NONE;

  long double context_probs[model->num_contexts];
  for (unsigned int c = 0; c < model->num_contexts; ++c) {
    context_probs[c] = 0.0;
    for (unsigned int i = 0; i < seq_length; ++i) {
      context_probs[c] += features[i][c];
    }
  }

  unsigned int best_context = 0;
  long double max_prob = context_probs[0];
  for (unsigned int c = 1; c < model->num_contexts; ++c) {
    if (context_probs[c] > max_prob) {
      max_prob = context_probs[c];
      best_context = c;
    }
  }

  // Apply softmax to normalize probabilities
  long double sum_exp = 0.0;
  for (unsigned int c = 0; c < model->num_contexts; ++c) {
    context_probs[c] = expl(context_probs[c]);
    sum_exp += context_probs[c];
  }
  for (unsigned int c = 0; c < model->num_contexts; ++c) {
    context_probs[c] /= sum_exp;
  }

  for (unsigned int i = 0; i < seq_length; ++i) {
    free(features[i]);
  }
  free(features);

  return best_context;
}

// Backpropagation function with gradient computation
void ForeRunner_backprop(ForeRunner_t *model, long double **input_sequence,
                         unsigned int seq_length, long double *grad_loss) {
  if (!model || !grad_loss)
    return;

  TRANSFORMER_backprop(model->transformer, input_sequence, seq_length,
                       grad_loss);
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
