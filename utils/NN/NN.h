#ifndef NN_H
#define NN_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Forward declaration
typedef struct NN_t NN_t;

// Activation function types
typedef enum {
  SIGMOID = 0,
  TANH = 1,
  RELU = 2,
  LINEAR = 3,
  SOFTMAX = 4,
  ARGMAX = 5,
  ACTIVATION_TYPE_COUNT = 6
} ActivationFunctionType;

// Activation derivative types
typedef enum {
  SIGMOID_DERIVATIVE = 0,
  TANH_DERIVATIVE = 1,
  RELU_DERIVATIVE = 2,
  LINEAR_DERIVATIVE = 3,
  SOFTMAX_DERIVATIVE = 4,
  ARGMAX_DERIVATIVE = 5,
  ACTIVATION_DERIVATIVE_TYPE_COUNT = 6
} ActivationDerivativeType;

// Loss function types
typedef enum {
  MSE = 0,
  MAE = 1,
  HUBER = 2,
  LL = 3,
  CE = 4,
  LOSS_TYPE_COUNT = 5
} LossFunctionType;

// Loss derivative types
typedef enum {
  MSE_DERIVATIVE = 0,
  MAE_DERIVATIVE = 1,
  HUBER_DERIVATIVE = 2,
  LL_DERIVATIVE = 3,
  CE_DERIVATIVE = 4,
  LOSS_DERIVATIVE_TYPE_COUNT = 5
} LossDerivativeType;

// Optimizer types
typedef enum {
  SGD = 0,
  RMSPROP = 1,
  ADAGRAD = 2,
  ADAM = 3,
  NAG = 4,
  OPTIMIZER_TYPE_COUNT = 5
} OptimizerType;

// Regularization types
typedef enum {
  L1 = 0,
  L2 = 1,
  REGULARIZATION_TYPE_COUNT = 2
} RegularizationType;

// Function pointer types
typedef long double (*ActivationFunction)(long double);
typedef long double (*ActivationDerivative)(long double);
typedef long double (*LossFunction)(long double, long double);
typedef long double (*LossDerivative)(long double, long double);
typedef void (*OptimizerFunction)(NN_t *);
typedef long double (*RegularizationFunction)(long double);

// Neural Network structure
typedef struct NN_t {
  size_t numLayers; // Number of layers
  size_t *layers;   // Array of layer sizes

  long double **weights;   // Weight matrices
  long double **biases;    // Bias vectors
  long double **weights_v; // Velocity / m for optimizers
  long double **biases_v;  // Velocity / v for optimizers

  long double learningRate; // Learning rate
  size_t t;                 // Time step for Adam

  ActivationFunction *activationFunctions;
  ActivationDerivative *activationDerivatives;
  LossFunction loss;
  LossDerivative lossDerivative;
  OptimizerFunction optimizer;
  RegularizationFunction regularization;
} NN_t;

// Initialization
NN_t *NN_init(size_t *layers, ActivationFunctionType *actFuncs,
              ActivationDerivativeType *actDerivs, LossFunctionType lossFunc,
              LossDerivativeType lossDeriv, RegularizationType reg,
              OptimizerType opt, long double learningRate);

NN_t *NN_init_random(size_t num_inputs, size_t num_outputs);
void NN_destroy(NN_t *nn);

// Forward propagation
long double **NN_forward(NN_t *nn, long double inputs[]);
long double *NN_forward_softmax(NN_t *nn, long double inputs[]);

// Backpropagation
void NN_backprop_custom_delta(NN_t *nn, long double inputs[],
                              long double *output_delta);
void NN_backprop_softmax(NN_t *nn, long double inputs[], long double y_true[],
                         long double y_pred[]);
void NN_backprop_argmax(NN_t *nn, long double inputs[], long double y_true[],
                        long double y_pred[],
                        long double (*lossDerivative)(long double,
                                                      long double));

// Matrix multiplication
long double *NN_matmul(long double inputs[], long double weights[],
                       long double biases[], size_t input_size,
                       size_t output_size);

// Loss functions
long double NN_loss(NN_t *nn, long double y_true, long double y_predicted);

long double mse(long double predicted, long double target);
long double mae(long double predicted, long double target);
long double huber(long double predicted, long double target);
long double ll(long double predicted, long double target);
long double ce(long double predicted, long double target);

// Loss derivatives
long double mse_derivative(long double predicted, long double target);
long double mae_derivative(long double predicted, long double target);
long double huber_derivative(long double predicted, long double target);
long double ll_derivative(long double predicted, long double target);
long double ce_derivative(long double predicted, long double target);

// Activation functions
long double sigmoid(long double x);
long double tanh_activation(long double x);
long double relu(long double x);
long double linear(long double x);

// Activation derivatives
long double sigmoid_derivative(long double x);
long double tanh_derivative(long double x);
long double relu_derivative(long double x);
long double linear_derivative(long double x);

// Vector-based activations
void softmax(long double *vec, size_t size);
void argmax(long double *vec, size_t size);
void softmax_derivative(long double *predicted, long double *one_hot,
                        long double *gradients, size_t size);
long double *create_one_hot(size_t index, size_t size);
void argmax(long double *vec, size_t size);
void argmax_derivative(long double *predicted, long double *one_hot,
                       long double *gradients, size_t size);

// Optimizers
void sgd(NN_t *nn);
void rmsprop(NN_t *nn);
void adagrad(NN_t *nn);
void adam(NN_t *nn);
void nag(NN_t *nn);

// Regularization
long double l1(long double weight);
long double l2(long double weight);

// Helpers
ActivationFunction get_activation_function(ActivationFunctionType type);
ActivationDerivative get_activation_derivative(ActivationDerivativeType type);
LossFunction get_loss_function(LossFunctionType type);
LossDerivative get_loss_derivative(LossDerivativeType type);
RegularizationFunction get_regularization_function(RegularizationType type);
OptimizerFunction get_optimizer_function(OptimizerType type);

ActivationDerivativeType
map_activation_to_derivative(ActivationFunctionType actFunc);
LossDerivativeType map_loss_to_derivative(LossFunctionType lossFunc);

const char *activation_to_string(ActivationFunctionType type);
const char *loss_to_string(LossFunctionType type);
const char *optimizer_to_string(OptimizerType type);
const char *regularization_to_string(RegularizationType type);

ActivationFunctionType get_activation_function_type(const char *str);
LossFunctionType get_loss_function_type(const char *str);
OptimizerType get_optimizer_type(const char *str);
RegularizationType get_regularization_type(const char *str);

LossFunctionType get_loss_function_from_func(LossFunction func);
OptimizerType get_optimizer_from_func(OptimizerFunction func);
RegularizationType get_regularization_from_func(RegularizationFunction func);

// Save / Load
int NN_save(NN_t *nn, const char *filename);
NN_t *NN_load(const char *filename);

#endif // NN_H
