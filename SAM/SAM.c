#include "SAM.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../utils/NN/TRANSFORMER.h"

// Helper function to initialize weights
static void init_weights(SAM_t* sam) {
    sam->weights = (long double***)malloc(sam->num_layers * sizeof(long double**));
    for (size_t i = 0; i < sam->num_layers - 1; i++) {
        sam->weights[i] = (long double**)malloc(sam->layer_sizes[i] * sizeof(long double*));
        for (size_t j = 0; j < sam->layer_sizes[i]; j++) {
            sam->weights[i][j] = (long double*)malloc(sam->layer_sizes[i + 1] * sizeof(long double));
            for (size_t k = 0; k < sam->layer_sizes[i + 1]; k++) {
                sam->weights[i][j][k] = ((long double)rand() / RAND_MAX * 2.0L - 1.0L) * sqrtl(2.0L / sam->layer_sizes[i]);
            }
        }
    }
}

// Helper function to free weights
static void free_weights(SAM_t* sam) {
    for (size_t i = 0; i < sam->num_layers - 1; i++) {
        for (size_t j = 0; j < sam->layer_sizes[i]; j++) {
            free(sam->weights[i][j]);
        }
        free(sam->weights[i]);
    }
    free(sam->weights);
}

SAM_t* SAM_init(size_t input_dim, size_t output_dim, size_t num_heads, size_t context_id) {
    SAM_t* sam = (SAM_t*)malloc(sizeof(SAM_t));
    if (!sam) return NULL;

    // Initialize dimensions
    sam->num_layers = 3;  // Input, hidden, output
    sam->layer_sizes = (size_t*)malloc(sam->num_layers * sizeof(size_t));
    sam->layer_sizes[0] = input_dim;
    sam->layer_sizes[1] = 256;  // Hidden layer size
    sam->layer_sizes[2] = output_dim;

    // Initialize weights
    init_weights(sam);

    // Initialize transformer and submodels
    sam->transformer = TRANSFORMER_init(input_dim, num_heads);
    sam->num_submodels = 5;  // Fixed number of submodels for now
    sam->submodels = (NEAT_t**)malloc(sam->num_submodels * sizeof(NEAT_t*));
    
    for (size_t i = 0; i < sam->num_submodels; i++) {
        sam->submodels[i] = NEAT_init(input_dim, output_dim, i);
    }

    sam->context = (long double)context_id;
    return sam;
}

void SAM_destroy(SAM_t* sam) {
    if (!sam) return;

    // Free weights
    free_weights(sam);
    free(sam->layer_sizes);

    // Free transformer
    if (sam->transformer) {
        TRANSFORMER_destroy(sam->transformer);
    }

    // Free submodels
    for (size_t i = 0; i < sam->num_submodels; i++) {
        if (sam->submodels[i]) {
            NEAT_destroy(sam->submodels[i]);
        }
    }
    free(sam->submodels);
    free(sam);
}

void SAM_train(SAM_t* sam, long double** input_sequence, size_t seq_length, long double* target) {
    if (!sam || !input_sequence || !target) return;

    // Train transformer
    TRANSFORMER_train(sam->transformer, input_sequence, seq_length, target);

    // Train submodels
    for (size_t i = 0; i < sam->num_submodels; i++) {
        NEAT_train(sam->submodels[i], input_sequence[0], target);
    }
}

void SAM_adapt_transfusion(SAM_t* sam, long double context, ProjectionMatrix* P, AdaptationParams* params) {
    if (!sam || !P || !params) return;

    // Calculate base gamma for first submodel
    long double gamma = SAM_calculate_gamma(sam->context, 0);  // Base gamma for first submodel

    // Perform transfusion for each submodel
    for (size_t i = 0; i < sam->num_submodels; i++) {
        // Train submodel
        SAM_train_submodel(sam->submodels[i], params->learning_rate_transfusion);

        // Update projection matrix
        for (size_t j = 0; j < P->rows; j++) {
            for (size_t k = 0; k < P->cols; k++) {
                P->matrix[j][k] *= gamma;
            }
        }
    }
}

ProjectionMatrix* SAM_create_projection_matrix(SAM_t* sam, long double context) {
    if (!sam) return NULL;

    ProjectionMatrix* P = (ProjectionMatrix*)malloc(sizeof(ProjectionMatrix));
    if (!P) return NULL;

    // Initialize dimensions based on transformer architecture
    P->rows = sam->layer_sizes[0];  // Input dimension
    P->cols = sam->layer_sizes[sam->num_layers - 1];  // Output dimension

    // Allocate matrix
    P->matrix = (long double**)malloc(P->rows * sizeof(long double*));
    for (size_t i = 0; i < P->rows; i++) {
        P->matrix[i] = (long double*)malloc(P->cols * sizeof(long double));
        for (size_t j = 0; j < P->cols; j++) {
            // Initialize with scaled weights
            P->matrix[i][j] = 1.0L / (context + 1.0L) * sam->weights[0][i][j];
        }
    }

    return P;
}

void SAM_update_transformer(SAM_t* sam, long double** G, long double learning_rate) {
    if (!sam || !G) return;

    // Update weights
    for (size_t i = 0; i < sam->layer_sizes[0]; i++) {
        for (size_t j = 0; j < sam->layer_sizes[sam->num_layers - 1]; j++) {
            sam->weights[0][i][j] += learning_rate * G[i][j];
        }
    }
}

long double SAM_calculate_gamma(long double context, size_t submodel_index) {
    // Calculate gamma based on context and submodel index
    return 1.0L / (1.0L + fabsl(context - (long double)submodel_index));
}

long double SAM_calculate_beta(PerformanceMetrics* metrics, long double context) {
    if (!metrics) return 0.0L;
    
    // Calculate beta based on performance metrics and context
    return metrics->fitness / (1.0L + context);
}

int SAM_save(SAM_t* sam, const char* filename) {
    if (!sam || !filename) return 0;

    FILE* file = fopen(filename, "wb");
    if (!file) return 0;

    // Save SAM parameters
    fwrite(&sam->num_submodels, sizeof(size_t), 1, file);
    fwrite(&sam->context, sizeof(long double), 1, file);

    // Save transformer
    int result = TRANSFORMER_save(sam->transformer, file);

    // Save submodels
    for (size_t i = 0; i < sam->num_submodels; i++) {
        result |= NEAT_save(sam->submodels[i], file);
    }

    fclose(file);
    return result;
}

SAM_t* SAM_load(const char* filename) {
    if (!filename) return NULL;

    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;

    SAM_t* sam = (SAM_t*)malloc(sizeof(SAM_t));
    if (!sam) {
        fclose(file);
        return NULL;
    }

    // Load SAM parameters
    fread(&sam->num_submodels, sizeof(size_t), 1, file);
    fread(&sam->context, sizeof(long double), 1, file);

    // Load transformer
    sam->transformer = TRANSFORMER_load(file);
    if (!sam->transformer) {
        free(sam);
        fclose(file);
        return NULL;
    }

    // Load submodels
    sam->submodels = (NEAT_t**)malloc(sam->num_submodels * sizeof(NEAT_t*));
    if (!sam->submodels) {
        TRANSFORMER_destroy(sam->transformer);
        free(sam);
        fclose(file);
        return NULL;
    }

    for (size_t i = 0; i < sam->num_submodels; i++) {
        sam->submodels[i] = NEAT_init(256, 256, 100);
        if (!sam->submodels[i]) {
            // Cleanup on error
            for (size_t j = 0; j < i; j++) {
                NEAT_destroy(sam->submodels[j]);
            }
            free(sam->submodels);
            TRANSFORMER_destroy(sam->transformer);
            free(sam);
            fclose(file);
            return NULL;
        }
        NEAT_load(sam->submodels[i], file);
    }

    fclose(file);
    return sam;
}

// Matrix operations
void SAM_matrix_multiply(long double** A, long double** B, long double** C,
                        size_t m, size_t n, size_t p) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            C[i][j] = 0.0L;
            for (size_t k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void SAM_matrix_scale(long double** matrix, long double scalar,
                     size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            matrix[i][j] *= scalar;
        }
    }
}

void SAM_matrix_add(long double** A, long double** B, long double** C,
                   size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void SAM_train_submodel(NEAT_t* neat, long double learning_rate) {
    if (!neat) return;
    // TODO: Implement submodel training
}

PerformanceMetrics SAM_calculate_metrics(NEAT_t* neat) {
    PerformanceMetrics metrics;
    metrics.accuracy = 0.0L;
    metrics.loss = 0.0L;
    metrics.fitness = 0.0L;

    // Calculate metrics based on NEAT performance
    // TODO: Implement actual metric calculation

    return metrics;
}
