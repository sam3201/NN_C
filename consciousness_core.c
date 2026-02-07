/*
 * Pure C Consciousness Loss Implementation - No Python Dependencies
 * Complete neural network and consciousness logic in pure C
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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

Matrix *matrix_create(size_t rows, size_t cols) {
    Matrix *mat = malloc(sizeof(Matrix));
    if (!mat) return NULL;

    mat->rows = rows;
    mat->cols = cols;
    mat->size = rows * cols;
    mat->data = calloc(mat->size, sizeof(double));

    if (!mat->data) {
        free(mat);
        return NULL;
    }

    return mat;
}

void matrix_free(Matrix *mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

Matrix *matrix_copy(const Matrix *src) {
    if (!src) return NULL;

    Matrix *dst = matrix_create(src->rows, src->cols);
    if (!dst) return NULL;

    memcpy(dst->data, src->data, src->size * sizeof(double));
    return dst;
}

void matrix_fill(Matrix *mat, double value) {
    if (!mat || !mat->data) return;

    for (size_t i = 0; i < mat->size; i++) {
        mat->data[i] = value;
    }
}

void matrix_random_normal(Matrix *mat, double mean, double std) {
    if (!mat || !mat->data) return;

    // Simple random number generation (not cryptographically secure)
    for (size_t i = 0; i < mat->size; i++) {
        // Box-Muller transform for normal distribution
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        mat->data[i] = mean + std * z;
    }
}

Matrix *matrix_multiply(const Matrix *a, const Matrix *b) {
    if (!a || !b || a->cols != b->rows) {
        return NULL;
    }

    Matrix *result = matrix_create(a->rows, b->cols);
    if (!result) return NULL;

    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result->data[i * result->cols + j] = sum;
        }
    }

    return result;
}

Matrix *matrix_add(const Matrix *a, const Matrix *b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) {
        return NULL;
    }

    Matrix *result = matrix_create(a->rows, a->cols);
    if (!result) return NULL;

    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}

Matrix *matrix_subtract(const Matrix *a, const Matrix *b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) {
        return NULL;
    }

    Matrix *result = matrix_create(a->rows, a->cols);
    if (!result) return NULL;

    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }

    return result;
}

Matrix *matrix_transpose(const Matrix *mat) {
    if (!mat) return NULL;

    Matrix *result = matrix_create(mat->cols, mat->rows);
    if (!result) return NULL;

    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            result->data[j * result->cols + i] = mat->data[i * mat->cols + j];
        }
    }

    return result;
}

double matrix_sum(const Matrix *mat) {
    if (!mat || !mat->data) return 0.0;

    double sum = 0.0;
    for (size_t i = 0; i < mat->size; i++) {
        sum += mat->data[i];
    }
    return sum;
}

void matrix_scale(Matrix *mat, double scalar) {
    if (!mat || !mat->data) return;

    for (size_t i = 0; i < mat->size; i++) {
        mat->data[i] *= scalar;
    }
}

// ================================
// NEURAL NETWORK LAYERS - Pure C
// ================================

LinearLayer *linear_create(size_t input_size, size_t output_size) {
    LinearLayer *layer = malloc(sizeof(LinearLayer));
    if (!layer) return NULL;

    layer->input_size = input_size;
    layer->output_size = output_size;

    // Xavier initialization
    double scale = sqrt(2.0 / (input_size + output_size));
    layer->weights = matrix_create(output_size, input_size);
    if (!layer->weights) {
        free(layer);
        return NULL;
    }

    matrix_random_normal(layer->weights, 0.0, scale);

    layer->biases = matrix_create(output_size, 1);
    if (!layer->biases) {
        matrix_free(layer->weights);
        free(layer);
        return NULL;
    }

    matrix_fill(layer->biases, 0.0);

    return layer;
}

void linear_free(LinearLayer *layer) {
    if (layer) {
        matrix_free(layer->weights);
        matrix_free(layer->biases);
        free(layer);
    }
}

Matrix *linear_forward(const LinearLayer *layer, const Matrix *input) {
    if (!layer || !input || input->rows != layer->input_size) {
        return NULL;
    }

    // W * x + b
    Matrix *weights_T = matrix_transpose(layer->weights);
    if (!weights_T) return NULL;

    Matrix *output = matrix_multiply(weights_T, input);
    matrix_free(weights_T);

    if (!output) return NULL;

    // Add bias to each column
    for (size_t i = 0; i < output->size; i++) {
        size_t bias_idx = i % output->rows;
        output->data[i] += layer->biases->data[bias_idx];
    }

    return output;
}

// Helper function to transpose matrix
Matrix *matrix_transpose(const Matrix *mat) {
    if (!mat) return NULL;

    Matrix *result = matrix_create(mat->cols, mat->rows);
    if (!result) return NULL;

    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            result->data[j * result->cols + i] = mat->data[i * mat->cols + j];
        }
    }

    return result;
}

// ================================
// SEQUENTIAL MODEL - Pure C
// ================================

SequentialModel *sequential_create(size_t num_layers, size_t *layer_sizes) {
    if (!layer_sizes || num_layers < 2) return NULL;

    SequentialModel *model = malloc(sizeof(SequentialModel));
    if (!model) return NULL;

    model->num_layers = num_layers - 1; // number of layers = number of transitions
    model->layers = malloc(model->num_layers * sizeof(LinearLayer*));
    if (!model->layers) {
        free(model);
        return NULL;
    }

    for (size_t i = 0; i < model->num_layers; i++) {
        model->layers[i] = linear_create(layer_sizes[i], layer_sizes[i+1]);
        if (!model->layers[i]) {
            // Clean up previously allocated layers
            for (size_t j = 0; j < i; j++) {
                linear_free(model->layers[j]);
            }
            free(model->layers);
            free(model);
            return NULL;
        }
    }

    return model;
}

void sequential_free(SequentialModel *model) {
    if (model) {
        for (size_t i = 0; i < model->num_layers; i++) {
            linear_free(model->layers[i]);
        }
        free(model->layers);
        free(model);
    }
}

Matrix *sequential_forward(const SequentialModel *model, const Matrix *input) {
    if (!model || !input) return NULL;

    Matrix *current = matrix_copy(input);
    if (!current) return NULL;

    for (size_t i = 0; i < model->num_layers; i++) {
        Matrix *next = linear_forward(model->layers[i], current);
        if (!next) {
            matrix_free(current);
            return NULL;
        }

        // Apply ReLU to hidden layers (not output layer)
        if (i < model->num_layers - 1) {
            // Apply ReLU activation
            for (size_t j = 0; j < next->size; j++) {
                if (next->data[j] < 0.0) next->data[j] = 0.0;
            }
        }

        matrix_free(current);
        current = next;
    }

    return current;
}

// ================================
// CONSCIOUSNESS LOSS MODULE - Pure C
// ================================

ConsciousnessLossModule *consciousness_create(size_t latent_dim, size_t action_dim) {
    ConsciousnessLossModule *module = malloc(sizeof(ConsciousnessLossModule));
    if (!module) return NULL;

    module->latent_dim = latent_dim;
    module->action_dim = action_dim;

    // Initialize models with proper layer sizes
    size_t world_layers[] = {latent_dim + action_dim, 128, latent_dim};
    module->world_model = sequential_create(3, world_layers);
    if (!module->world_model) {
        free(module);
        return NULL;
    }

    size_t self_layers[] = {latent_dim + action_dim + latent_dim, 128, latent_dim};
    module->self_model = sequential_create(3, self_layers);
    if (!module->self_model) {
        sequential_free(module->world_model);
        free(module);
        return NULL;
    }

    size_t policy_layers[] = {latent_dim * 2, 128, action_dim};
    module->policy_model = sequential_create(3, policy_layers);
    if (!module->policy_model) {
        sequential_free(module->world_model);
        sequential_free(module->self_model);
        free(module);
        return NULL;
    }

    size_t resource_layers[] = {latent_dim, 64, 3};
    module->resource_controller = sequential_create(3, resource_layers);
    if (!module->resource_controller) {
        sequential_free(module->world_model);
        sequential_free(module->self_model);
        sequential_free(module->policy_model);
        free(module);
        return NULL;
    }

    // Initialize statistics
    module->stats = calloc(10, sizeof(double)); // l_world, l_self, l_cons, l_total, growth, etc.

    return module;
}

void consciousness_free(ConsciousnessLossModule *module) {
    if (module) {
        sequential_free(module->world_model);
        sequential_free(module->self_model);
        sequential_free(module->policy_model);
        sequential_free(module->resource_controller);
        free(module->stats);
        free(module);
    }
}

// Loss computation functions
Matrix *world_prediction_loss(const ConsciousnessLossModule *module,
                            const Matrix *z_t, const Matrix *a_t, const Matrix *z_next_actual) {
    if (!module || !z_t || !a_t || !z_next_actual) return NULL;

    // Concatenate z_t and a_t
    Matrix *combined = matrix_create(z_t->rows + a_t->rows, 1);
    if (!combined) return NULL;

    memcpy(combined->data, z_t->data, z_t->size * sizeof(double));
    memcpy(combined->data + z_t->size, a_t->data, a_t->size * sizeof(double));

    // Forward pass through world model
    Matrix *prediction = sequential_forward(module->world_model, combined);
    matrix_free(combined);

    if (!prediction) return NULL;

    // MSE loss
    Matrix *diff = matrix_subtract(prediction, z_next_actual);
    if (!diff) {
        matrix_free(prediction);
        return NULL;
    }

    // Squared differences
    Matrix *squared = matrix_create(diff->rows, diff->cols);
    if (!squared) {
        matrix_free(diff);
        matrix_free(prediction);
        return NULL;
    }

    for (size_t i = 0; i < diff->size; i++) {
        squared->data[i] = diff->data[i] * diff->data[i];
    }

    Matrix *loss = matrix_create(1, 1);
    if (!loss) {
        matrix_free(squared);
        matrix_free(diff);
        matrix_free(prediction);
        return NULL;
    }

    loss->data[0] = matrix_sum(squared) / squared->size;

    matrix_free(diff);
    matrix_free(squared);
    matrix_free(prediction);

    return loss;
}

Matrix *self_model_loss(const ConsciousnessLossModule *module,
                       const Matrix *z_t, const Matrix *a_t, const Matrix *m_t, const Matrix *z_next_actual) {
    if (!module || !z_t || !a_t || !m_t || !z_next_actual) return NULL;

    // Concatenate z_t, a_t, m_t
    Matrix *combined = matrix_create(z_t->rows + a_t->rows + m_t->rows, 1);
    if (!combined) return NULL;

    memcpy(combined->data, z_t->data, z_t->size * sizeof(double));
    memcpy(combined->data + z_t->size, a_t->data, a_t->size * sizeof(double));
    memcpy(combined->data + z_t->size + a_t->size, m_t->data, m_t->size * sizeof(double));

    // Forward pass through self model
    Matrix *delta_pred = sequential_forward(module->self_model, combined);
    matrix_free(combined);

    if (!delta_pred) return NULL;

    // Actual state change: z_next - z_t
    Matrix *delta_actual = matrix_subtract(z_next_actual, z_t);
    if (!delta_actual) {
        matrix_free(delta_pred);
        return NULL;
    }

    // MSE loss
    Matrix *diff = matrix_subtract(delta_pred, delta_actual);
    if (!diff) {
        matrix_free(delta_pred);
        matrix_free(delta_actual);
        return NULL;
    }

    Matrix *squared = matrix_create(diff->rows, diff->cols);
    if (!squared) {
        matrix_free(diff);
        matrix_free(delta_pred);
        matrix_free(delta_actual);
        return NULL;
    }

    for (size_t i = 0; i < diff->size; i++) {
        squared->data[i] = diff->data[i] * diff->data[i];
    }

    Matrix *loss = matrix_create(1, 1);
    if (!loss) {
        matrix_free(squared);
        matrix_free(diff);
        matrix_free(delta_pred);
        matrix_free(delta_actual);
        return NULL;
    }

    loss->data[0] = matrix_sum(squared) / squared->size;

    matrix_free(delta_pred);
    matrix_free(delta_actual);
    matrix_free(diff);
    matrix_free(squared);

    return loss;
}

Matrix *consciousness_loss(const ConsciousnessLossModule *module,
                          const Matrix *z_t, const Matrix *a_t, const Matrix *z_next_actual, const Matrix *m_t) {
    if (!module || !z_t || !a_t || !z_next_actual || !m_t) return NULL;

    // Get world model prediction
    Matrix *combined_world = matrix_create(z_t->rows + a_t->rows, 1);
    if (!combined_world) return NULL;

    memcpy(combined_world->data, z_t->data, z_t->size * sizeof(double));
    memcpy(combined_world->data + z_t->size, a_t->data, a_t->size * sizeof(double));

    Matrix *z_world = sequential_forward(module->world_model, combined_world);
    matrix_free(combined_world);

    if (!z_world) return NULL;

    // Get self model prediction
    Matrix *combined_self = matrix_create(z_t->rows + a_t->rows + m_t->rows, 1);
    if (!combined_self) {
        matrix_free(z_world);
        return NULL;
    }

    memcpy(combined_self->data, z_t->data, z_t->size * sizeof(double));
    memcpy(combined_self->data + z_t->size, a_t->data, a_t->size * sizeof(double));
    memcpy(combined_self->data + z_t->size + a_t->size, m_t->data, m_t->size * sizeof(double));

    Matrix *delta_self = sequential_forward(module->self_model, combined_self);
    matrix_free(combined_self);

    if (!delta_self) {
        matrix_free(z_world);
        return NULL;
    }

    Matrix *z_self = matrix_add(z_t, delta_self);
    if (!z_self) {
        matrix_free(z_world);
        matrix_free(delta_self);
        return NULL;
    }

    // KL divergence approximation: MSE between world and self predictions
    Matrix *diff = matrix_subtract(z_world, z_self);
    if (!diff) {
        matrix_free(z_world);
        matrix_free(delta_self);
        matrix_free(z_self);
        return NULL;
    }

    Matrix *squared = matrix_create(diff->rows, diff->cols);
    if (!squared) {
        matrix_free(diff);
        matrix_free(z_world);
        matrix_free(delta_self);
        matrix_free(z_self);
        return NULL;
    }

    for (size_t i = 0; i < diff->size; i++) {
        squared->data[i] = diff->data[i] * diff->data[i];
    }

    Matrix *loss = matrix_create(1, 1);
    if (!loss) {
        matrix_free(squared);
        matrix_free(diff);
        matrix_free(z_world);
        matrix_free(delta_self);
        matrix_free(z_self);
        return NULL;
    }

    loss->data[0] = matrix_sum(squared) / squared->size;

    matrix_free(z_world);
    matrix_free(delta_self);
    matrix_free(z_self);
    matrix_free(diff);
    matrix_free(squared);

    return loss;
}

// Helper function to compute all losses and return as matrix
Matrix *consciousness_compute_loss_c(ConsciousnessLossModule *module,
                                 Matrix *z_t, Matrix *a_t, Matrix *z_next,
                                 Matrix *m_t, Matrix *reward, int num_params) {
    if (!module || !z_t || !a_t || !z_next || !m_t || !reward) return NULL;

    // Compute individual losses
    Matrix *l_world_mat = world_prediction_loss(module, z_t, a_t, z_next);
    Matrix *l_self_mat = self_model_loss(module, z_t, a_t, m_t, z_next);
    Matrix *l_cons_mat = consciousness_loss(module, z_t, a_t, z_next, m_t);

    if (!l_world_mat || !l_self_mat || !l_cons_mat) {
        matrix_free(l_world_mat);
        matrix_free(l_self_mat);
        matrix_free(l_cons_mat);
        return NULL;
    }

    double l_world = l_world_mat->data[0];
    double l_self = l_self_mat->data[0];
    double l_cons = l_cons_mat->data[0];

    // Self-model confidence (simplified)
    double confidence_score = 0.8;

    // Policy loss
    double reward_mean = matrix_sum(reward) / reward->size;
    double l_policy = -reward_mean + 0.1 * (1.0 - confidence_score);

    // Compute penalty
    double c_compute = num_params / 1000000.0;

    // Adaptive weights (simplified softmax)
    double weights[5] = {l_world, l_self, l_cons, l_policy, c_compute};
    double max_weight = weights[0];
    for (int i = 1; i < 5; i++) {
        if (weights[i] > max_weight) max_weight = weights[i];
    }

    double exp_weights[5];
    double sum_exp = 0.0;
    for (int i = 0; i < 5; i++) {
        exp_weights[i] = exp(weights[i] - max_weight);
        sum_exp += exp_weights[i];
    }

    double lambdas[5];
    for (int i = 0; i < 5; i++) {
        lambdas[i] = exp_weights[i] / sum_exp;
    }

    // Total loss
    double l_total = lambdas[0] * l_world + lambdas[1] * l_self +
                    lambdas[2] * l_cons + lambdas[3] * l_policy +
                    lambdas[4] * c_compute;

    // Consciousness score
    double consciousness_score = 1.0 / (1.0 + l_cons);

    // Return all losses in a matrix
    Matrix *result = matrix_create(7, 1);
    if (!result) {
        matrix_free(l_world_mat);
        matrix_free(l_self_mat);
        matrix_free(l_cons_mat);
        return NULL;
    }

    result->data[0] = l_world;
    result->data[1] = l_self;
    result->data[2] = l_cons;
    result->data[3] = l_policy;
    result->data[4] = c_compute;
    result->data[5] = l_total;
    result->data[6] = consciousness_score;

    // Update module statistics
    module->stats[0] = l_world;        // l_world_history
    module->stats[1] = l_self;         // l_self_history
    module->stats[2] = l_cons;         // l_cons_history
    module->stats[3] = l_total;        // l_total_history
    module->stats[4] = consciousness_score; // consciousness_score

    // Cleanup
    matrix_free(l_world_mat);
    matrix_free(l_self_mat);
    matrix_free(l_cons_mat);

    return result;
}
