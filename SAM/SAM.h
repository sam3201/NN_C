#ifndef SAM_H
#define SAM_H

#include "../utils/NN/NEAT.h"
#include "../utils/NN/TRANSFORMER.h"
#include <stdlib.h>

// Forward declarations
// typedef struct NEAT_t NEAT_t;

// Performance metrics structure
typedef struct PerformanceMetrics {
  long double accuracy;
  long double loss;
  long double fitness;
} PerformanceMetrics;

// Projection matrix for knowledge transfer
typedef struct ProjectionMatrix {
  long double **matrix;
  size_t rows;
  size_t cols;
} ProjectionMatrix;

// Adaptation parameters
typedef struct AdaptationParams {
  long double learning_rate;
  long double momentum;
  long double weight_decay;
  long double learning_rate_transfusion;
  long double learning_rate_generalization;
} AdaptationParams;

// SAM structure
typedef struct SAM_t {
  Transformer_t *transformer;
  NEAT_t **submodels;
  size_t num_submodels;
  long double context;

  // Neural network weights
  long double ***weights; // [layer][row][col]
  size_t num_layers;
  size_t *layer_sizes;
} SAM_t;

// Core functions
SAM_t *SAM_init(size_t input_dim, size_t output_dim, size_t num_heads,
                size_t context_id);
void SAM_destroy(SAM_t *sam);
void SAM_train(SAM_t *sam, long double **input_sequence, size_t seq_length,
               long double *target);
long double *SAM_forward(SAM_t *sam, long double **input_sequence,
                         size_t seq_length);
// Backpropagate through entire SAM model
void SAM_backprop(SAM_t *sam, long double **input_sequence, size_t seq_length,
                  long double *grad_loss);

void SAM_update_context(SAM_t *sam, long double current_performance);

// Adaptation functions
void SAM_adapt(SAM_t *sam, long double **input_sequence, size_t seq_length);
void SAM_project(SAM_t *sam, long double **source_data, size_t data_length);
void SAM_generalize(SAM_t *sam);
void SAM_transfuse(SAM_t *sam);

// Knowledge transfer functions
void SAM_adapt_transfusion(SAM_t *sam, long double context, ProjectionMatrix *P,
                           AdaptationParams *params);
ProjectionMatrix *SAM_create_projection_matrix(SAM_t *sam, long double context);
long double SAM_calculate_gamma(long double context, size_t submodel_index);
long double SAM_calculate_beta(PerformanceMetrics *metrics,
                               long double context);
void SAM_update_transformer(SAM_t *sam, long double **G,
                            long double learning_rate);

// Matrix operations
void SAM_matrix_multiply(long double **A, long double **B, long double **C,
                         size_t m, size_t n, size_t p);
void SAM_matrix_scale(long double **matrix, long double scalar, size_t rows,
                      size_t cols);
void SAM_matrix_add(long double **A, long double **B, long double **C,
                    size_t rows, size_t cols);

// Utility functions
long double SAM_evaluate_fitness(SAM_t *sam, long double *input,
                                 long double *target);
int SAM_save(SAM_t *sam, const char *filename);
SAM_t *SAM_load(const char *filename);

// Internal helper functions
void SAM_train_submodel(NEAT_t *neat, long double learning_rate);
PerformanceMetrics SAM_calculate_metrics(NEAT_t *neat);

MuCortex *SAM_as_MUZE(SAM_t *sam);
void SAM_MUZE_destroy(MuCortex *cortex);

#endif // SAM_H
