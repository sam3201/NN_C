/*
 * Pure C Consciousness Loss Implementation - Header
 * No external libraries, pure C data structures and functions
 */

#ifndef CONSCIOUSNESS_CORE_H
#define CONSCIOUSNESS_CORE_H

#include <stddef.h>

// ================================
// C DATA STRUCTURES - Pure C
// ================================

typedef struct {
    double *data;
    size_t rows;
    size_t cols;
    size_t size;
} Matrix;

typedef struct {
    Matrix *weights;
    Matrix *biases;
    size_t input_size;
    size_t output_size;
} LinearLayer;

typedef struct {
    LinearLayer **layers;
    size_t num_layers;
} SequentialModel;

typedef struct {
    SequentialModel *world_model;
    SequentialModel *self_model;
    SequentialModel *policy_model;
    SequentialModel *resource_controller;
    double *stats;
    size_t latent_dim;
    size_t action_dim;
} ConsciousnessLossModule;

// ================================
// MATRIX OPERATIONS - Pure C
// ================================

Matrix *matrix_create(size_t rows, size_t cols);
void matrix_free(Matrix *mat);
Matrix *matrix_copy(const Matrix *src);
void matrix_fill(Matrix *mat, double value);
void matrix_random_normal(Matrix *mat, double mean, double std);
Matrix *matrix_multiply(const Matrix *a, const Matrix *b);
Matrix *matrix_add(const Matrix *a, const Matrix *b);
Matrix *matrix_subtract(const Matrix *a, const Matrix *b);
Matrix *matrix_transpose(const Matrix *mat);
double matrix_sum(const Matrix *mat);
void matrix_scale(Matrix *mat, double scalar);

// ================================
// NEURAL NETWORK LAYERS - Pure C
// ================================

LinearLayer *linear_create(size_t input_size, size_t output_size);
void linear_free(LinearLayer *layer);
Matrix *linear_forward(const LinearLayer *layer, const Matrix *input);

// ================================
// SEQUENTIAL MODEL - Pure C
// ================================

SequentialModel *sequential_create(size_t num_layers, size_t *layer_sizes);
void sequential_free(SequentialModel *model);
Matrix *sequential_forward(const SequentialModel *model, const Matrix *input);

// ================================
// CONSCIOUSNESS LOSS MODULE - Pure C
// ================================

ConsciousnessLossModule *consciousness_create(size_t latent_dim, size_t action_dim);
void consciousness_free(ConsciousnessLossModule *module);
Matrix *world_prediction_loss(const ConsciousnessLossModule *module,
                            const Matrix *z_t, const Matrix *a_t, const Matrix *z_next_actual);
Matrix *self_model_loss(const ConsciousnessLossModule *module,
                       const Matrix *z_t, const Matrix *a_t, const Matrix *m_t, const Matrix *z_next_actual);
Matrix *consciousness_loss(const ConsciousnessLossModule *module,
                          const Matrix *z_t, const Matrix *a_t, const Matrix *z_next_actual, const Matrix *m_t);

// ================================
// COMPUTE ALL LOSSES - Pure C
// ================================

Matrix *consciousness_compute_loss_c(ConsciousnessLossModule *module,
                                 Matrix *z_t, Matrix *a_t, Matrix *z_next,
                                 Matrix *m_t, Matrix *reward, int num_params);

#endif // CONSCIOUSNESS_CORE_H
