#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <math.h>

#define ACTIVATION_FUNCTIONS \
    X(SIGMOID, "sigmoid") \
    X(RELU, "relu") \
    X(TANH, "tanh") \
    X(ARGMAX, "argmax") \
    X(SOFTMAX, "softmax")

#define ACTIVATION_DERIVATIVES \
    X(SIGMOID_DERIVATIVE, "sigmoid_derivative") \
    X(RELU_DERIVATIVE, "relu_derivative") \
    X(TANH_DERIVATIVE, "tanh_derivative") \
    X(ARGMAX_DERIVATIVE, "argmax_derivative") \
    X(SOFTMAX_DERIVATIVE, "softmax_derivative")

#define LOSS_FUNCTIONS \
    X(CE, "cross-entropy") \
    X(MSE, "mean-squared-error")

#define LOSS_DERIVATIVES \
    X(CE_DERIVATIVE, "cross-entropy_derivative") \
    X(MSE_DERIVATIVE, "mean-squared-error_derivative")

typedef enum {
    #define X(name, str) name,
    ACTIVATION_FUNCTIONS
    #undef X
    ACTIVATION_FUNCTION_COUNT 
} ActivationFunction;

typedef enum {
    #define X(name, str) name,
    ACTIVATION_DERIVATIVES
    #undef X 
    ACTIVATION_DERIVATIVE_COUNT
} ActivationDerivative;

typedef enum {
    #define X(name, str) name,
    LOSS_FUNCTIONS
    #undef X
    LOSS_FUNCTION_COUNT
} LossFunction;

typedef enum {
    #define X(name, str) name,
    LOSS_DERIVATIVES
    #undef X
    LOSS_DERIVATIVE_COUNT
} LossDerivative;

typedef struct NN_t {
    size_t *layers;
    size_t numLayers;

    long double **weights;
    long double **biases;

    long double (**activationFunctions)(long double);
    long double (**activationDerivatives)(long double);

    long double (*lossFunction)(long double, long double);
    long double (*lossDerivative)(long double, long double);
} NN_t;

NN_t *NN_init(size_t *layers,
              ActivationFunction **activationFunctions, ActivationDerivative **activationDerivatives,
              LossFunction lossFunction, LossDerivative lossDerivative);

void NN_add_layer(NN_t *nn, size_t layerSize, ActivationFunction **activationFunctions, ActivationDerivative **activationDerivatives);

long double NN_matmul(long double *inputs, long double *weights, long double *biases); 
long double *NN_forward(NN_t *nn, long double *inputs);
void NN_backprop(NN_t *nn, long double *inputs, long double *outputs, long double *labels);

void NN_destroy(NN_t *nn);

long double sigmoid(long double x);
long double sigmoid_derivative(long double x);

long double relu(long double x);
long double relu_derivative(long double x);

long double tanh_activation(long double x);
long double tanh_derivative(long double x);

long double argmax(long double *x);
long double argmax_derivative(long double *x);

long double softmax(long double *x);
long double softmax_derivative(long double *x);

long double mse(long double y_true, long double y_pred);
long double mse_derivative(long double y_true, long double y_pred);

long double ce(long double y_true, long double y_pred);
long double ce_derivative(long double y_true, long double y_pred);

void NN_save(const char *filename, NN_t *nn);
void NN_load(const char *filename);

#endif // NN_H

