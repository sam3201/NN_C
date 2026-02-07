#ifndef SAM_FULL_CONTEXT_H
#define SAM_FULL_CONTEXT_H

#include "NN.h"
#include <stddef.h>

// Full-context batch learning with dominant compression
// Based on: maximize J - βH - λC + ηI

#define FC_BATCH_SIZE 5
#define DOMINANT_BETA 1.0    // Uncertainty weight
#define DOMINANT_LAMBDA 0.1  // Compute cost weight  
#define DOMINANT_ETA 0.01    // Useful memory weight
#define DOMINANT_KAPPA 0.1   // Required return on compute

typedef struct {
    long double *user_query_embedding;
    long double *search_results_embedding;
    long double *augmented_response;
    long double *verified_response;
    long double confidence_score;
    int is_verified;
} VerifiedExample;

typedef struct {
    VerifiedExample *examples;
    size_t count;
    size_t capacity;
    long double avg_confidence;
    long compute_cost;
    long double mutual_info;
} FullContextBatch;

// Batch management
FullContextBatch* fc_batch_create(size_t capacity);
void fc_batch_destroy(FullContextBatch *batch);
int fc_batch_add(FullContextBatch *batch, VerifiedExample *example);
void fc_batch_clear(FullContextBatch *batch);

// Dominant compression objective
long double compute_dominant_objective(FullContextBatch *batch, long double control_performance);
int should_update_weights(FullContextBatch *batch, long double delta_J, long double delta_C);

// Full-context backpropagation
void NN_backprop_full_context(NN_t *nn, FullContextBatch *batch, long double base_learning_rate);
void NN_apply_dominant_compression_update(NN_t *nn, FullContextBatch *batch);

// Utility functions
long double compute_batch_uncertainty(FullContextBatch *batch);
long double compute_mutual_information(FullContextBatch *batch);
long double compute_compute_cost(size_t batch_size, size_t input_dim);

#endif // SAM_FULL_CONTEXT_H
