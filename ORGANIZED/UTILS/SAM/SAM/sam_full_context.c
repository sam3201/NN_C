#include "sam_full_context.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

FullContextBatch* fc_batch_create(size_t capacity) {
    FullContextBatch *batch = (FullContextBatch*)malloc(sizeof(FullContextBatch));
    if (!batch) return NULL;
    
    batch->examples = (VerifiedExample*)calloc(capacity, sizeof(VerifiedExample));
    if (!batch->examples) {
        free(batch);
        return NULL;
    }
    
    batch->capacity = capacity;
    batch->count = 0;
    batch->avg_confidence = 0.0L;
    batch->compute_cost = 0;
    batch->mutual_info = 0.0L;
    
    return batch;
}

void fc_batch_destroy(FullContextBatch *batch) {
    if (!batch) return;
    
    for (size_t i = 0; i < batch->count; i++) {
        free(batch->examples[i].user_query_embedding);
        free(batch->examples[i].search_results_embedding);
        free(batch->examples[i].augmented_response);
        free(batch->examples[i].verified_response);
    }
    
    free(batch->examples);
    free(batch);
}

int fc_batch_add(FullContextBatch *batch, VerifiedExample *example) {
    if (!batch || !example || batch->count >= batch->capacity) {
        return 0; // Failed to add
    }
    
    // Copy example data
    VerifiedExample *dest = &batch->examples[batch->count];
    
    // Allocate and copy embeddings (assuming fixed size for now)
    size_t embed_size = 256; // Configurable embedding size
    
    dest->user_query_embedding = (long double*)malloc(embed_size * sizeof(long double));
    dest->search_results_embedding = (long double*)malloc(embed_size * sizeof(long double));
    dest->augmented_response = (long double*)malloc(embed_size * sizeof(long double));
    dest->verified_response = (long double*)malloc(embed_size * sizeof(long double));
    
    if (!dest->user_query_embedding || !dest->search_results_embedding ||
        !dest->augmented_response || !dest->verified_response) {
        // Cleanup on allocation failure
        free(dest->user_query_embedding);
        free(dest->search_results_embedding);
        free(dest->augmented_response);
        free(dest->verified_response);
        return 0;
    }
    
    memcpy(dest->user_query_embedding, example->user_query_embedding, embed_size * sizeof(long double));
    memcpy(dest->search_results_embedding, example->search_results_embedding, embed_size * sizeof(long double));
    memcpy(dest->augmented_response, example->augmented_response, embed_size * sizeof(long double));
    memcpy(dest->verified_response, example->verified_response, embed_size * sizeof(long double));
    
    dest->confidence_score = example->confidence_score;
    dest->is_verified = example->is_verified;
    
    batch->count++;
    
    // Update statistics
    long double total_confidence = 0.0L;
    for (size_t i = 0; i < batch->count; i++) {
        total_confidence += batch->examples[i].confidence_score;
    }
    batch->avg_confidence = total_confidence / batch->count;
    
    return 1; // Success
}

void fc_batch_clear(FullContextBatch *batch) {
    if (!batch) return;
    
    for (size_t i = 0; i < batch->count; i++) {
        free(batch->examples[i].user_query_embedding);
        free(batch->examples[i].search_results_embedding);
        free(batch->examples[i].augmented_response);
        free(batch->examples[i].verified_response);
        
        batch->examples[i].user_query_embedding = NULL;
        batch->examples[i].search_results_embedding = NULL;
        batch->examples[i].augmented_response = NULL;
        batch->examples[i].verified_response = NULL;
    }
    
    batch->count = 0;
    batch->avg_confidence = 0.0L;
}

long double compute_batch_uncertainty(FullContextBatch *batch) {
    if (!batch || batch->count == 0) return 1.0L; // Maximum uncertainty
    
    // Uncertainty as variance in confidence scores
    long double mean = batch->avg_confidence;
    long double variance = 0.0L;
    
    for (size_t i = 0; i < batch->count; i++) {
        long double diff = batch->examples[i].confidence_score - mean;
        variance += diff * diff;
    }
    
    variance /= batch->count;
    return sqrt(variance);
}

long double compute_mutual_information(FullContextBatch *batch) {
    if (!batch || batch->count < 2) return 0.0L;
    
    // Approximate mutual information as reduction in uncertainty
    // after seeing multiple examples
    long double individual_uncertainty = 1.0L; // Prior uncertainty
    long double batch_uncertainty_val = compute_batch_uncertainty(batch);
    
    return individual_uncertainty - batch_uncertainty_val;
}

long double compute_compute_cost(size_t batch_size, size_t input_dim) {
    // Compute cost grows with batch size and input dimension
    return (long double)(batch_size * input_dim * 100); // Normalized cost
}

long double compute_dominant_objective(FullContextBatch *batch, long double control_performance) {
    if (!batch) return 0.0L;
    
    // J - βH - λC + ηI
    long double J = control_performance;
    long double H = compute_batch_uncertainty(batch);
    long double C = compute_compute_cost(batch->count, 256); // Assume 256-dim embeddings
    long double I = compute_mutual_information(batch);
    
    long double objective = J - (DOMINANT_BETA * H) - (DOMINANT_LAMBDA * C) + (DOMINANT_ETA * I);
    
    return objective;
}

int should_update_weights(FullContextBatch *batch, long double delta_J, long double delta_C) {
    if (!batch || batch->count < FC_BATCH_SIZE) {
        return 0; // Not enough examples
    }
    
    // Growth rule: update only when justified by return on compute
    if (delta_C <= 1e-8L) return 0;
    
    long double return_on_compute = delta_J / delta_C;
    return return_on_compute > DOMINANT_KAPPA;
}

void NN_backprop_full_context(NN_t *nn, FullContextBatch *batch, long double base_learning_rate) {
    if (!nn || !batch || batch->count == 0) return;
    
    // Zero out accumulated gradients
    for (size_t l = 0; l < nn->numLayers - 1; l++) {
        size_t in_size = nn->layers[l];
        size_t out_size = nn->layers[l + 1];
        
        for (size_t j = 0; j < out_size; j++) {
            nn->biases_grad[l][j] = 0.0L;
            for (size_t i = 0; i < in_size; i++) {
                nn->weights_grad[l][i * out_size + j] = 0.0L;
            }
        }
    }
    
    // Accumulate gradients across all examples in batch
    for (size_t e = 0; e < batch->count; e++) {
        VerifiedExample *ex = &batch->examples[e];
        
        // Weight by confidence score
        long double weight = ex->confidence_score;
        
        // Compute loss gradient for this example
        // Using verified_response as target, augmented_response as prediction
        size_t output_size = nn->layers[nn->numLayers - 1];
        long double *output_delta = calloc(output_size, sizeof(long double));
        
        if (!output_delta) continue;
        
        // Compute MSE derivative between augmented and verified
        for (size_t i = 0; i < output_size && i < 256; i++) {
            long double predicted = ex->augmented_response[i];
            long double target = ex->verified_response[i];
            output_delta[i] = weight * (predicted - target); // MSE derivative
        }
        
        // Use existing backprop with custom delta
        // Note: This is a simplified version - full implementation would need
        // to properly integrate with the existing NN architecture
        
        free(output_delta);
    }
    
    // Average gradients
    for (size_t l = 0; l < nn->numLayers - 1; l++) {
        size_t in_size = nn->layers[l];
        size_t out_size = nn->layers[l + 1];
        
        for (size_t j = 0; j < out_size; j++) {
            nn->biases_grad[l][j] /= batch->count;
            for (size_t i = 0; i < in_size; i++) {
                nn->weights_grad[l][i * out_size + j] /= batch->count;
            }
        }
    }
    
    // Apply learning rate
    nn->learningRate = base_learning_rate;
}

void NN_apply_dominant_compression_update(NN_t *nn, FullContextBatch *batch) {
    if (!nn || !batch) return;
    
    // Compute objective before update
    static long double prev_objective = 0.0L;
    long double current_objective = compute_dominant_objective(batch, batch->avg_confidence);
    
    long double delta_J = current_objective - prev_objective;
    long double delta_C = compute_compute_cost(batch->count, 256) - 
                          compute_compute_cost(0, 256);
    
    // Check if update is justified
    if (should_update_weights(batch, delta_J, delta_C)) {
        // Apply the optimizer
        if (nn->optimizer) {
            nn->optimizer(nn);
        }
        
        prev_objective = current_objective;
        
        // Clear batch after successful update
        fc_batch_clear(batch);
    }
}
