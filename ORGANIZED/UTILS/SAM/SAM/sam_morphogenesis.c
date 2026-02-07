#include "sam_morphogenesis.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Utility: compute matrix rank via SVD approximation (simplified)
static int approximate_rank(long double** matrix, size_t n, long double eps) {
    // Simplified rank estimation using diagonal dominance
    // Full SVD would be better but expensive
    int rank = 0;
    for (size_t i = 0; i < n && i < 100; i++) {  // Cap at 100 for efficiency
        long double row_sum = 0.0L;
        for (size_t j = 0; j < n && j < 100; j++) {
            row_sum += fabsl(matrix[i][j]);
        }
        if (row_sum > eps) rank++;
    }
    return rank;
}

// Utility: copy neural network with expanded dimensions
static int copy_nn_expanded(NN_t* src, NN_t* dst, size_t new_input_dim, size_t new_output_dim) {
    // Copy architecture
    dst->numLayers = src->numLayers;
    dst->layers = (size_t*)malloc(dst->numLayers * sizeof(size_t));
    if (!dst->layers) return 0;
    
    memcpy(dst->layers, src->layers, dst->numLayers * sizeof(size_t));
    
    // Modify first layer input dimension
    dst->layers[0] = new_input_dim;
    // Modify last layer output dimension  
    dst->layers[dst->numLayers - 1] = new_output_dim;
    
    // Allocate new weight matrices
    dst->weights = (long double**)malloc((dst->numLayers - 1) * sizeof(long double*));
    dst->biases = (long double**)malloc((dst->numLayers - 1) * sizeof(long double*));
    dst->weights_grad = (long double**)malloc((dst->numLayers - 1) * sizeof(long double*));
    dst->biases_grad = (long double**)malloc((dst->numLayers - 1) * sizeof(long double*));
    
    if (!dst->weights || !dst->biases || !dst->weights_grad || !dst->biases_grad) {
        free(dst->layers);
        free(dst->weights);
        free(dst->biases);
        free(dst->weights_grad);
        free(dst->biases_grad);
        return 0;
    }
    
    for (size_t l = 0; l < dst->numLayers - 1; l++) {
        size_t in_size = dst->layers[l];
        size_t out_size = dst->layers[l + 1];
        
        dst->weights[l] = (long double*)calloc(in_size * out_size, sizeof(long double));
        dst->biases[l] = (long double*)calloc(out_size, sizeof(long double));
        dst->weights_grad[l] = (long double*)calloc(in_size * out_size, sizeof(long double));
        dst->biases_grad[l] = (long double*)calloc(out_size, sizeof(long double));
        
        if (!dst->weights[l] || !dst->biases[l] || !dst->weights_grad[l] || !dst->biases_grad[l]) {
            // Cleanup on failure
            for (size_t k = 0; k <= l; k++) {
                free(dst->weights[k]);
                free(dst->biases[k]);
                free(dst->weights_grad[k]);
                free(dst->biases_grad[k]);
            }
            return 0;
        }
    }
    
    // Copy existing weights (with zero-padding for new dimensions)
    for (size_t l = 0; l < dst->numLayers - 1 && l < src->numLayers - 1; l++) {
        size_t src_in = src->layers[l];
        size_t src_out = src->layers[l + 1];
        size_t dst_in = dst->layers[l];
        size_t dst_out = dst->layers[l + 1];
        
        // Copy weights
        for (size_t i = 0; i < src_in && i < dst_in; i++) {
            for (size_t j = 0; j < src_out && j < dst_out; j++) {
                dst->weights[l][i * dst_out + j] = src->weights[l][i * src_out + j];
            }
        }
        
        // Copy biases
        for (size_t j = 0; j < src_out && j < dst_out; j++) {
            dst->biases[l][j] = src->biases[l][j];
        }
        
        // New dimensions get small random initialization (Xavier-style)
        for (size_t i = src_in; i < dst_in; i++) {
            for (size_t j = 0; j < dst_out; j++) {
                long double scale = sqrtl(2.0L / (dst_in + dst_out));
                dst->weights[l][i * dst_out + j] = ((long double)rand() / RAND_MAX - 0.5L) * 2.0L * scale;
            }
        }
        
        for (size_t j = src_out; j < dst_out; j++) {
            dst->biases[l][j] = 0.0L;
        }
    }
    
    // Copy other parameters
    dst->learningRate = src->learningRate;
    dst->t = src->t;
    // ... copy other optimizer state as needed
    
    return 1;
}

MorphogenesisState* morphogenesis_create(size_t initial_dim, size_t max_dim) {
    MorphogenesisState* mg = (MorphogenesisState*)calloc(1, sizeof(MorphogenesisState));
    if (!mg) return NULL;
    
    mg->current_dim = initial_dim;
    mg->max_dim = max_dim > MAX_LATENT_DIM ? MAX_LATENT_DIM : max_dim;
    
    // Initialize error history (ring buffer)
    mg->error_history_size = 100;
    mg->error_history = (long double*)calloc(mg->error_history_size, sizeof(long double));
    mg->error_history_idx = 0;
    
    // Allocate concept registry
    mg->concept_capacity = 256;
    mg->concepts = (ConceptState*)calloc(mg->concept_capacity, sizeof(ConceptState));
    mg->concept_count = 0;
    
    // Initialize trigger conditions
    mg->error_threshold = MORPHOGENESIS_THRESHOLD;
    mg->min_history_size = 20;
    mg->gamma_structure = GAMMA_STRUCTURE;
    
    // Initialize statistics
    mg->total_concepts_born = 0;
    mg->total_concepts_died = 0;
    mg->cumulative_error = 0.0L;
    
    printf("🧬 Morphogenesis system initialized: dim=%zu, max=%zu\n", initial_dim, mg->max_dim);
    
    return mg;
}

void morphogenesis_destroy(MorphogenesisState* mg) {
    if (!mg) return;
    
    free(mg->error_history);
    
    // Free concept names
    for (size_t i = 0; i < mg->concept_count; i++) {
        free(mg->concepts[i].concept_name);
    }
    free(mg->concepts);
    
    // Free curvature matrix
    if (mg->curvature_matrix) {
        for (size_t i = 0; i < mg->current_dim && i < 100; i++) {
            free(mg->curvature_matrix[i]);
        }
        free(mg->curvature_matrix);
    }
    
    free(mg->error_gradient);
    
    free(mg);
}

void morphogenesis_record_error(MorphogenesisState* mg, long double error) {
    if (!mg) return;
    
    mg->error_history[mg->error_history_idx] = error;
    mg->error_history_idx = (mg->error_history_idx + 1) % mg->error_history_size;
    mg->cumulative_error += error;
}

long double morphogenesis_get_trend(MorphogenesisState* mg) {
    if (!mg || mg->error_history_idx < mg->min_history_size) return 0.0L;
    
    // Simple linear trend over last N errors
    size_t n = mg->min_history_size;
    long double sum_x = 0.0L, sum_y = 0.0L, sum_xy = 0.0L, sum_x2 = 0.0L;
    
    for (size_t i = 0; i < n; i++) {
        size_t idx = (mg->error_history_idx + mg->error_history_size - n + i) % mg->error_history_size;
        long double x = (long double)i;
        long double y = mg->error_history[idx];
        
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }
    
    long double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    return slope;
}

int morphogenesis_check_trigger(MorphogenesisState* mg, long double current_error) {
    if (!mg) return 0;
    
    // Record error
    morphogenesis_record_error(mg, current_error);
    
    // Check minimum history
    if (mg->error_history_idx < mg->min_history_size) return 0;
    
    // Condition 1: Error above threshold (persistent prediction failure)
    if (current_error < mg->error_threshold) return 0;
    
    // Condition 2: Error not decreasing (stuck optimization)
    long double trend = morphogenesis_get_trend(mg);
    if (trend < -0.01L) return 0;  // Error is decreasing, no trigger
    
    // Condition 3: Not at max capacity
    if (mg->current_dim >= mg->max_dim) return 0;
    
    printf("🚨 Morphogenesis trigger: error=%.4f, trend=%.4f, dim=%zu\n", 
           (double)current_error, (double)trend, mg->current_dim);
    
    return 1;  // Trigger!
}

int morphogenesis_compute_rank_deficiency(MorphogenesisState* mg, NN_t* nn, 
                                          FullContextBatch* batch) {
    if (!mg || !nn || !batch) return 0;
    
    // Allocate curvature matrix if needed
    if (!mg->curvature_matrix) {
        mg->curvature_matrix = (long double**)malloc(100 * sizeof(long double*));
        for (int i = 0; i < 100; i++) {
            mg->curvature_matrix[i] = (long double*)calloc(100, sizeof(long double));
        }
    }
    
    // Compute approximate Hessian via outer product of gradients (Gauss-Newton)
    // Simplified: use identity + noise for prototype
    for (int i = 0; i < 100 && i < (int)mg->current_dim; i++) {
        for (int j = 0; j < 100 && j < (int)mg->current_dim; j++) {
            if (i == j) {
                mg->curvature_matrix[i][j] = 1.0L + ((long double)rand() / RAND_MAX) * 0.1L;
            } else {
                mg->curvature_matrix[i][j] = ((long double)rand() / RAND_MAX - 0.5L) * 0.01L;
            }
        }
    }
    
    // Check rank
    int rank = approximate_rank(mg->curvature_matrix, 
                                mg->current_dim > 100 ? 100 : mg->current_dim, 
                                RANK_DEFICIENCY_EPS);
    
    int deficiency = (int)mg->current_dim - rank;
    
    printf("📊 Curvature analysis: rank=%d, dim=%zu, deficiency=%d\n", 
           rank, mg->current_dim, deficiency);
    
    return deficiency > 0 ? deficiency : 0;
}

int morphogenesis_birth_concept(MorphogenesisState* mg, NN_t* nn, const char* concept_name) {
    if (!mg || !nn) return 0;
    
    if (mg->current_dim >= mg->max_dim) {
        printf("❌ Cannot birth concept: at max dimension\n");
        return 0;
    }
    
    if (mg->concept_count >= mg->concept_capacity) {
        printf("❌ Cannot birth concept: concept registry full\n");
        return 0;
    }
    
    // Create new concept
    ConceptState* concept = &mg->concepts[mg->concept_count];
    concept->dim_index = (int)mg->current_dim;
    concept->birth_error = mg->cumulative_error / (mg->error_history_idx + 1);
    concept->utility = 0.0L;
    concept->age = 0;
    concept->is_active = 1;
    concept->concept_name = strdup(concept_name ? concept_name : "unnamed_concept");
    
    mg->concept_count++;
    mg->total_concepts_born++;
    
    // Expand network (simplified - just track dimension for now)
    // Full implementation would call NN_expand_latent_dim
    mg->current_dim++;
    
    printf("🌱 Concept born: '%s' at dim %d (total: %zu)\n", 
           concept->concept_name, concept->dim_index, mg->concept_count);
    
    return 1;
}

int morphogenesis_prune_concept(MorphogenesisState* mg, int concept_idx) {
    if (!mg || concept_idx < 0 || concept_idx >= (int)mg->concept_count) return 0;
    
    ConceptState* concept = &mg->concepts[concept_idx];
    
    // Check minimum lifetime
    if (concept->age < MIN_CONCEPT_LIFETIME) {
        printf("⚠️ Cannot prune concept '%s': too young (age=%d)\n", 
               concept->concept_name, concept->age);
        return 0;
    }
    
    // Check if low utility
    if (concept->utility > 0.01L) {
        printf("⚠️ Cannot prune concept '%s': still useful (utility=%.4f)\n",
               concept->concept_name, (double)concept->utility);
        return 0;
    }
    
    printf("💀 Concept pruned: '%s' (lived %d updates)\n", 
           concept->concept_name, concept->age);
    
    concept->is_active = 0;
    mg->total_concepts_died++;
    
    // Note: We don't actually reduce current_dim to keep indices stable
    // Instead, mark as inactive for potential reuse
    
    return 1;
}

long double morphogenesis_structure_cost(MorphogenesisState* mg) {
    if (!mg) return 0.0L;
    
    // Cost scales with number of active concepts
    int active_count = 0;
    for (size_t i = 0; i < mg->concept_count; i++) {
        if (mg->concepts[i].is_active) active_count++;
    }
    
    return mg->gamma_structure * (long double)active_count;
}

void morphogenesis_update_structure_penalty(MorphogenesisState* mg, NN_t* nn) {
    if (!mg || !nn) return;
    
    // Add structural cost to objective
    long double cost = morphogenesis_structure_cost(mg);
    
    // This would modify the loss function
    // For now, just print it
    if (mg->concept_count % 10 == 0) {
        printf("📐 Structure cost: %.4f (concepts: %zu)\n", 
               (double)cost, mg->concept_count);
    }
}

void morphogenesis_print_summary(MorphogenesisState* mg) {
    if (!mg) return;
    
    printf("\n=== Morphogenesis Summary ===\n");
    printf("Latent dimension: %zu / %zu\n", mg->current_dim, mg->max_dim);
    printf("Concepts: %zu active, %d born, %d died\n",
           mg->concept_count, mg->total_concepts_born, mg->total_concepts_died);
    printf("Avg error: %.4f\n", (double)(mg->cumulative_error / (mg->error_history_idx + 1)));
    
    printf("Active concepts:\n");
    for (size_t i = 0; i < mg->concept_count; i++) {
        ConceptState* c = &mg->concepts[i];
        if (c->is_active) {
            printf("  [%d] '%s' age=%d utility=%.4f\n",
                   c->dim_index, c->concept_name, c->age, (double)c->utility);
        }
    }
    printf("===========================\n\n");
}

ConceptState* morphogenesis_get_active_concepts(MorphogenesisState* mg, size_t* count) {
    if (!mg || !count) return NULL;
    
    // Count active
    *count = 0;
    for (size_t i = 0; i < mg->concept_count; i++) {
        if (mg->concepts[i].is_active) (*count)++;
    }
    
    if (*count == 0) return NULL;
    
    // Allocate and return array
    ConceptState* active = (ConceptState*)malloc(*count * sizeof(ConceptState));
    size_t idx = 0;
    for (size_t i = 0; i < mg->concept_count; i++) {
        if (mg->concepts[i].is_active) {
            active[idx++] = mg->concepts[i];
        }
    }
    
    return active;
}
