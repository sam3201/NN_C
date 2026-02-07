/*
 * SAM Consciousness Loss Implementation - Pure C Header
 * No external libraries, pure C data structures and functions
 */

#ifndef CONSCIOUSNESS_C_H
#define CONSCIOUSNESS_C_H

#include <stddef.h>

// ================================
// C DATA STRUCTURES
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
// MATRIX OPERATIONS
// ================================

Matrix *matrix_create(size_t rows, size_t cols);
void matrix_free(Matrix *mat);
Matrix *matrix_copy(const Matrix *src);
void matrix_fill(Matrix *mat, double value);
void matrix_random_normal(Matrix *mat, double mean, double std);
void matrix_random_uniform(Matrix *mat, double min, double max);
Matrix *matrix_multiply(const Matrix *a, const Matrix *b);
Matrix *matrix_add(const Matrix *a, const Matrix *b);
Matrix *matrix_subtract(const Matrix *a, const Matrix *b);
Matrix *matrix_transpose(const Matrix *mat);
void matrix_scale(Matrix *mat, double scalar);
double matrix_sum(const Matrix *mat);
Matrix *matrix_apply_function(const Matrix *mat, double (*func)(double));

// ================================
// NEURAL NETWORK LAYERS
// ================================

LinearLayer *linear_create(size_t input_size, size_t output_size);
void linear_free(LinearLayer *layer);
Matrix *linear_forward(const LinearLayer *layer, const Matrix *input);

// ================================
// SEQUENTIAL MODEL
// ================================

SequentialModel *sequential_create(size_t num_layers, size_t *layer_sizes);
void sequential_free(SequentialModel *model);
Matrix *sequential_forward(const SequentialModel *model, const Matrix *input);

// ================================
// CONSCIOUSNESS LOSS MODULE
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
// UTILITY FUNCTIONS
// ================================

// Activation functions
double relu_activation(double x);
double tanh_activation(double x);
double sigmoid_activation(double x);

// Square function for MSE
double square_activation(double x);

#endif // CONSCIOUSNESS_C_H
