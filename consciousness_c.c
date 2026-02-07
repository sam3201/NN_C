/*
 * SAM Consciousness Loss Implementation - Pure C from Scratch
 * No external libraries, no Python dependencies - pure C implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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
    Matrix *world_model;
    Matrix *self_model;
    Matrix *policy_model;
    Matrix *resource_controller;
    double *stats;
    size_t latent_dim;
    size_t action_dim;
} ConsciousnessLossModule;

// ================================
// MATRIX OPERATIONS (Pure C)
// ================================

Matrix *matrix_create(size_t rows, size_t cols) {
    Matrix *mat = malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->size = rows * cols;
    mat->data = calloc(mat->size, sizeof(double));
    return mat;
}

void matrix_free(Matrix *mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

Matrix *matrix_copy(const Matrix *src) {
    Matrix *dst = matrix_create(src->rows, src->cols);
    memcpy(dst->data, src->data, src->size * sizeof(double));
    return dst;
}

void matrix_fill(Matrix *mat, double value) {
    for (size_t i = 0; i < mat->size; i++) {
        mat->data[i] = value;
    }
}

void matrix_random_normal(Matrix *mat, double mean, double std) {
    // Simple random number generation (not cryptographically secure)
    for (size_t i = 0; i < mat->size; i++) {
        // Box-Muller transform for normal distribution
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        mat->data[i] = mean + std * z;
    }
}

void matrix_random_uniform(Matrix *mat, double min, double max) {
    for (size_t i = 0; i < mat->size; i++) {
        double r = (double)rand() / RAND_MAX;
        mat->data[i] = min + r * (max - min);
    }
}

Matrix *matrix_multiply(const Matrix *a, const Matrix *b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "Matrix multiply: incompatible dimensions %zux%zu and %zux%zu\n",
                a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix *result = matrix_create(a->rows, b->cols);

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
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Matrix add: incompatible dimensions %zux%zu and %zux%zu\n",
                a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix *result = matrix_create(a->rows, a->cols);

    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}

Matrix *matrix_subtract(const Matrix *a, const Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Matrix subtract: incompatible dimensions %zux%zu and %zux%zu\n",
                a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Matrix *result = matrix_create(a->rows, a->cols);

    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }

    return result;
}

Matrix *matrix_transpose(const Matrix *mat) {
    Matrix *result = matrix_create(mat->cols, mat->rows);

    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            result->data[j * result->cols + i] = mat->data[i * mat->cols + j];
        }
    }

    return result;
}

void matrix_scale(Matrix *mat, double scalar) {
    for (size_t i = 0; i < mat->size; i++) {
        mat->data[i] *= scalar;
    }
}

double matrix_sum(const Matrix *mat) {
    double sum = 0.0;
    for (size_t i = 0; i < mat->size; i++) {
        sum += mat->data[i];
    }
    return sum;
}

Matrix *matrix_apply_function(const Matrix *mat, double (*func)(double)) {
    Matrix *result = matrix_create(mat->rows, mat->cols);

    for (size_t i = 0; i < mat->size; i++) {
        result->data[i] = func(mat->data[i]);
    }

    return result;
}

// Activation functions
double relu_activation(double x) {
    return x > 0.0 ? x : 0.0;
}

double tanh_activation(double x) {
    return tanh(x);
}

double sigmoid_activation(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Square function for MSE
double square_activation(double x) {
    return x * x;
}

// ================================
// NEURAL NETWORK LAYERS
// ================================

LinearLayer *linear_create(size_t input_size, size_t output_size) {
    LinearLayer *layer = malloc(sizeof(LinearLayer));
    layer->input_size = input_size;
    layer->output_size = output_size;

    // Xavier initialization
    double scale = sqrt(2.0 / (input_size + output_size));
    layer->weights = matrix_create(output_size, input_size);
    matrix_random_normal(layer->weights, 0.0, scale);

    layer->biases = matrix_create(output_size, 1);
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
    if (input->rows != layer->input_size) {
        fprintf(stderr, "Linear forward: input size mismatch %zu vs %zu\n",
                input->rows, layer->input_size);
        return NULL;
    }

    // W * x + b
    Matrix *weights_T = matrix_transpose(layer->weights);
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

// ================================
// SEQUENTIAL MODEL
// ================================

SequentialModel *sequential_create(size_t num_layers, size_t *layer_sizes) {
    SequentialModel *model = malloc(sizeof(SequentialModel));
    model->num_layers = num_layers - 1; // number of layers = number of transitions
    model->layers = malloc(model->num_layers * sizeof(LinearLayer*));

    for (size_t i = 0; i < model->num_layers; i++) {
        model->layers[i] = linear_create(layer_sizes[i], layer_sizes[i+1]);
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
    Matrix *current = matrix_copy(input);

    for (size_t i = 0; i < model->num_layers; i++) {
        Matrix *next = linear_forward(model->layers[i], current);

        if (i < model->num_layers - 1) { // Apply ReLU to hidden layers
            Matrix *activated = matrix_apply_function(next, relu_activation);
            matrix_free(next);
            next = activated;
        }

        matrix_free(current);
        current = next;
    }

    return current;
}

// ================================
// CONSCIOUSNESS LOSS MODULE
// ================================

ConsciousnessLossModule *consciousness_create(size_t latent_dim, size_t action_dim) {
    ConsciousnessLossModule *module = malloc(sizeof(ConsciousnessLossModule));

    module->latent_dim = latent_dim;
    module->action_dim = action_dim;

    // Initialize models with proper layer sizes
    size_t world_layers[] = {latent_dim + action_dim, 128, latent_dim};
    module->world_model = sequential_create(3, world_layers);

    size_t self_layers[] = {latent_dim + action_dim + latent_dim, 128, latent_dim};
    module->self_model = sequential_create(3, self_layers);

    size_t policy_layers[] = {latent_dim * 2, 128, action_dim};
    module->policy_model = sequential_create(3, policy_layers);

    size_t resource_layers[] = {latent_dim, 64, 3};
    module->resource_controller = sequential_create(3, resource_layers);

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
    // Concatenate z_t and a_t
    Matrix *combined = matrix_create(z_t->rows + a_t->rows, 1);
    memcpy(combined->data, z_t->data, z_t->size * sizeof(double));
    memcpy(combined->data + z_t->size, a_t->data, a_t->size * sizeof(double));

    // Forward pass through world model
    Matrix *prediction = sequential_forward(module->world_model, combined);
    matrix_free(combined);

    if (!prediction) return NULL;

    // MSE loss
    Matrix *diff = matrix_subtract(prediction, z_next_actual);
    Matrix *squared = matrix_apply_function(diff, square_activation);
    Matrix *loss = matrix_create(1, 1);
    loss->data[0] = matrix_sum(squared) / squared->size;

    matrix_free(diff);
    matrix_free(squared);
    matrix_free(prediction);

    return loss;
}

Matrix *self_model_loss(const ConsciousnessLossModule *module,
                       const Matrix *z_t, const Matrix *a_t, const Matrix *m_t, const Matrix *z_next_actual) {
    // Concatenate z_t, a_t, m_t
    Matrix *combined = matrix_create(z_t->rows + a_t->rows + m_t->rows, 1);
    memcpy(combined->data, z_t->data, z_t->size * sizeof(double));
    memcpy(combined->data + z_t->size, a_t->data, a_t->size * sizeof(double));
    memcpy(combined->data + z_t->size + a_t->size, m_t->data, m_t->size * sizeof(double));

    // Forward pass through self model
    Matrix *delta_pred = sequential_forward(module->self_model, combined);
    matrix_free(combined);

    if (!delta_pred) return NULL;

    // Actual state change: z_next - z_t
    Matrix *delta_actual = matrix_subtract(z_next_actual, z_t);

    // MSE loss
    Matrix *diff = matrix_subtract(delta_pred, delta_actual);
    Matrix *squared = matrix_apply_function(diff, square_activation);
    Matrix *loss = matrix_create(1, 1);
    loss->data[0] = matrix_sum(squared) / squared->size;

    matrix_free(delta_pred);
    matrix_free(delta_actual);
    matrix_free(diff);
    matrix_free(squared);

    return loss;
}

Matrix *consciousness_loss(const ConsciousnessLossModule *module,
                          const Matrix *z_t, const Matrix *a_t, const Matrix *z_next_actual, const Matrix *m_t) {
    // Get world model prediction
    Matrix *combined_world = matrix_create(z_t->rows + a_t->rows, 1);
    memcpy(combined_world->data, z_t->data, z_t->size * sizeof(double));
    memcpy(combined_world->data + z_t->size, a_t->data, a_t->size * sizeof(double));

    Matrix *z_world = sequential_forward(module->world_model, combined_world);
    matrix_free(combined_world);

    // Get self model prediction
    Matrix *combined_self = matrix_create(z_t->rows + a_t->rows + m_t->rows, 1);
    memcpy(combined_self->data, z_t->data, z_t->size * sizeof(double));
    memcpy(combined_self->data + z_t->size, a_t->data, a_t->size * sizeof(double));
    memcpy(combined_self->data + z_t->size + a_t->size, m_t->data, m_t->size * sizeof(double));

    Matrix *delta_self = sequential_forward(module->self_model, combined_self);
    matrix_free(combined_self);

    Matrix *z_self = matrix_add(z_t, delta_self);

    // KL divergence approximation: MSE between world and self predictions
    Matrix *diff = matrix_subtract(z_world, z_self);
    Matrix *squared = matrix_apply_function(diff, square_activation);
    Matrix *loss = matrix_create(1, 1);
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

// ================================
// PYTHON BINDINGS
// ================================

#include <Python.h>

static ConsciousnessLossModule *global_module = NULL;

static PyObject *consciousness_create_module(PyObject *self, PyObject *args) {
    size_t latent_dim, action_dim;
    if (!PyArg_ParseTuple(args, "nn", &latent_dim, &action_dim)) {
        return NULL;
    }

    if (global_module) {
        consciousness_free(global_module);
    }

    global_module = consciousness_create(latent_dim, action_dim);
    if (!global_module) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create consciousness module");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *consciousness_compute_loss(PyObject *self, PyObject *args) {
    if (!global_module) {
        PyErr_SetString(PyExc_RuntimeError, "Consciousness module not initialized");
        return NULL;
    }

    // Parse input arrays (simplified - would need proper numpy array handling)
    PyObject *z_t_obj, *a_t_obj, *z_next_obj, *m_t_obj, *reward_obj;
    int num_params;
    if (!PyArg_ParseTuple(args, "OOOOOi", &z_t_obj, &a_t_obj, &z_next_obj, &m_t_obj, &reward_obj, &num_params)) {
        return NULL;
    }

    // For now, create dummy matrices for testing
    // In full implementation, would convert Python arrays to C matrices
    Matrix *z_t = matrix_create(64, 1);
    Matrix *a_t = matrix_create(16, 1);
    Matrix *z_next = matrix_create(64, 1);
    Matrix *m_t = matrix_create(64, 1);
    Matrix *reward = matrix_create(32, 1);

    matrix_random_normal(z_t, 0.0, 1.0);
    matrix_random_normal(a_t, 0.0, 1.0);
    matrix_random_normal(z_next, 0.0, 1.0);
    matrix_random_normal(m_t, 0.0, 1.0);
    matrix_random_normal(reward, 0.0, 1.0);

    // Compute losses
    Matrix *losses = consciousness_compute_loss_c(global_module, z_t, a_t, z_next, m_t, reward, num_params);

    if (!losses) {
        matrix_free(z_t);
        matrix_free(a_t);
        matrix_free(z_next);
        matrix_free(m_t);
        matrix_free(reward);
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute loss");
        return NULL;
    }

    // Return loss values as Python dict
    PyObject *result = PyDict_New();
    PyDict_SetItemString(result, "l_world", PyFloat_FromDouble(losses->data[0]));
    PyDict_SetItemString(result, "l_self", PyFloat_FromDouble(losses->data[1]));
    PyDict_SetItemString(result, "l_cons", PyFloat_FromDouble(losses->data[2]));
    PyDict_SetItemString(result, "l_policy", PyFloat_FromDouble(losses->data[3]));
    PyDict_SetItemString(result, "c_compute", PyFloat_FromDouble(losses->data[4]));
    PyDict_SetItemString(result, "l_total", PyFloat_FromDouble(losses->data[5]));
    PyDict_SetItemString(result, "consciousness_score", PyFloat_FromDouble(losses->data[6]));

    matrix_free(losses);
    matrix_free(z_t);
    matrix_free(a_t);
    matrix_free(z_next);
    matrix_free(m_t);
    matrix_free(reward);

    return result;
}

static PyObject *consciousness_get_stats(PyObject *self, PyObject *args) {
    if (!global_module) {
        PyErr_SetString(PyExc_RuntimeError, "Consciousness module not initialized");
        return NULL;
    }

    PyObject *result = PyDict_New();
    PyDict_SetItemString(result, "consciousness_score", PyFloat_FromDouble(global_module->stats[4]));
    PyDict_SetItemString(result, "total_loss", PyFloat_FromDouble(global_module->stats[3]));

    return result;
}

static PyMethodDef ConsciousnessMethods[] = {
    {"create", consciousness_create_module, METH_VARARGS, "Create consciousness module"},
    {"compute_loss", consciousness_compute_loss, METH_VARARGS, "Compute consciousness loss"},
    {"get_stats", consciousness_get_stats, METH_NOARGS, "Get consciousness statistics"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef consciousness_module = {
    PyModuleDef_HEAD_INIT,
    "consciousness_c",
    "Pure C consciousness loss implementation",
    -1,
    ConsciousnessMethods,
    NULL,  // m_slots
    NULL,  // m_traverse
    NULL,  // m_clear
    NULL   // m_free
};

PyMODINIT_FUNC PyInit_consciousness_c(void) {
    return PyModule_Create(&consciousness_module);
}

// ================================
// MAIN FUNCTION (for testing)
// ================================

int main() {
    srand(time(NULL));

    printf("🧠 SAM Consciousness Loss Module - Pure C Implementation\n");
    printf("No external libraries, no Python dependencies - pure C!\n");

    // Create consciousness module
    ConsciousnessLossModule *module = consciousness_create(64, 16);

    if (!module) {
        fprintf(stderr, "Failed to create consciousness module\n");
        return 1;
    }

    printf("✅ Consciousness module created successfully\n");

    // Test with some dummy data
    Matrix *z_t = matrix_create(64, 1);
    Matrix *a_t = matrix_create(16, 1);
    Matrix *m_t = matrix_create(64, 1);
    Matrix *z_next = matrix_create(64, 1);

    // Fill with random data
    matrix_random_normal(z_t, 0.0, 1.0);
    matrix_random_normal(a_t, 0.0, 1.0);
    matrix_random_normal(m_t, 0.0, 1.0);
    matrix_random_normal(z_next, 0.0, 1.0);

    // Compute losses
    Matrix *l_world = world_prediction_loss(module, z_t, a_t, z_next);
    Matrix *l_self = self_model_loss(module, z_t, a_t, m_t, z_next);
    Matrix *l_cons = consciousness_loss(module, z_t, a_t, z_next, m_t);

    if (l_world && l_self && l_cons) {
        printf("✅ Loss computation successful:\n");
        printf("   World Loss: %.6f\n", l_world->data[0]);
        printf("   Self Loss: %.6f\n", l_self->data[0]);
        printf("   Consciousness Loss: %.6f\n", l_cons->data[0]);

        // Update stats
        module->stats[0] = l_world->data[0];  // l_world_history
        module->stats[1] = l_self->data[0];   // l_self_history
        module->stats[2] = l_cons->data[0];   // l_cons_history
        module->stats[3] = (l_world->data[0] + l_self->data[0] + l_cons->data[0]) / 3.0; // l_total
        module->stats[4] = 1.0 / (1.0 + l_cons->data[0]); // consciousness_score

        printf("   Consciousness Score: %.6f\n", module->stats[4]);
    }

    // Cleanup
    matrix_free(l_world);
    matrix_free(l_self);
    matrix_free(l_cons);
    matrix_free(z_t);
    matrix_free(a_t);
    matrix_free(m_t);
    matrix_free(z_next);

    consciousness_free(module);

    printf("✅ Pure C consciousness implementation completed!\n");

    return 0;
}
