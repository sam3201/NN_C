#ifndef SAM_MORPHOGENESIS_H
#define SAM_MORPHOGENESIS_H

#include "NN.h"
#include "sam_full_context.h"
#include <stddef.h>

// Latent-space morphogenesis (concept birth) system
// Creates new latent dimensions when prediction errors show structural patterns

#define MORPHOGENESIS_THRESHOLD 0.15    // Error threshold for trigger
#define RANK_DEFICIENCY_EPS 1e-6        // Numerical tolerance for rank check
#define MAX_LATENT_DIM 4096             // Prevent unbounded growth
#define MIN_CONCEPT_LIFETIME 10         // Minimum updates before concept can die
#define GAMMA_STRUCTURE 0.1             // Cost of creating new dimension

// Concept state tracking
typedef struct {
    int dim_index;                    // Which latent dimension
    long double birth_error;          // Error level when created
    long double utility;              // How much this concept reduces error
    int age;                          // Updates since birth
    int is_active;                    // Whether dimension is used
    char* concept_name;               // Human-readable identifier
} ConceptState;

// Morphogenesis system state
typedef struct {
    // Current latent space
    size_t current_dim;               // Current latent dimensionality
    size_t max_dim;                   // Maximum allowed
    
    // Error tracking for trigger detection
    long double* error_history;       // Ring buffer of prediction errors
    size_t error_history_size;
    size_t error_history_idx;
    
    // Structural analysis
    long double** curvature_matrix;     // ∇²L (approximate Hessian)
    long double* error_gradient;        // ∇L (error gradient)
    
    // Concept registry
    ConceptState* concepts;             // Array of concept states
    size_t concept_count;
    size_t concept_capacity;
    
    // Trigger conditions
    long double error_threshold;        // MORPHOGENESIS_THRESHOLD
    int min_history_size;               // Minimum data before trigger check
    
    // Structural regularizer weight
    long double gamma_structure;
    
    // Statistics
    int total_concepts_born;
    int total_concepts_died;
    long double cumulative_error;
} MorphogenesisState;

// Initialize morphogenesis system
MorphogenesisState* morphogenesis_create(size_t initial_dim, size_t max_dim);
void morphogenesis_destroy(MorphogenesisState* mg);

// Core morphogenesis operations
int morphogenesis_check_trigger(MorphogenesisState* mg, long double current_error);
int morphogenesis_compute_rank_deficiency(MorphogenesisState* mg, NN_t* nn, 
                                          FullContextBatch* batch);
int morphogenesis_birth_concept(MorphogenesisState* mg, NN_t* nn, 
                                const char* concept_name);
int morphogenesis_prune_concept(MorphogenesisState* mg, int concept_idx);

// Error tracking
void morphogenesis_record_error(MorphogenesisState* mg, long double error);
long double morphogenesis_get_trend(MorphogenesisState* mg);

// Structural regularizer
long double morphogenesis_structure_cost(MorphogenesisState* mg);
void morphogenesis_update_structure_penalty(MorphogenesisState* mg, NN_t* nn);

// Network expansion/compression
int NN_expand_latent_dim(NN_t* nn, size_t new_dim);
int NN_compress_latent_dim(NN_t* nn, int* active_dims, size_t new_dim);

// Utility functions
void morphogenesis_print_summary(MorphogenesisState* mg);
ConceptState* morphogenesis_get_active_concepts(MorphogenesisState* mg, size_t* count);

#endif // SAM_MORPHOGENESIS_H
