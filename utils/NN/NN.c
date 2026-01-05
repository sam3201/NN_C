#include "NN.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Lookup tables for functions
static const ActivationFunction ACTIVATION_FUNCTIONS[] = {
    sigmoid, tanh_activation, relu, linear};

static const ActivationDerivative ACTIVATION_DERIVATIVES[] = {
    sigmoid_derivative, tanh_derivative, relu_derivative, linear_derivative};

static const LossFunction LOSS_FUNCTIONS[] = {mse, mae, huber, ll, ce};

static const LossDerivative LOSS_DERIVATIVES[] = {
    mse_derivative, mae_derivative, huber_derivative, ll_derivative,
    ce_derivative};

static const OptimizerFunction OPTIMIZER_FUNCTIONS[] = {sgd, rmsprop, adagrad,
                                                        adam, nag};

static const RegularizationFunction REGULARIZATION_FUNCTIONS[] = {l1, l2};

// Helper functions
long double *create_one_hot(size_t index, size_t size) {
  if (index >= size)
    return NULL;
  long double *vec = calloc(size, sizeof(long double));
  if (!vec)
    return NULL;
  vec[index] = 1.0L;
  return vec;
}

NN_t *NN_init(size_t *layers, ActivationFunctionType *actFuncs,
              ActivationDerivativeType *actDerivs, LossFunctionType lossFunc,
              LossDerivativeType lossDeriv, RegularizationType reg,
              OptimizerType opt, long double learningRate) {

  if (!layers || !actFuncs || !actDerivs) {
    fprintf(stderr, "NN_init: NULL input parameters\n");
    return NULL;
  }

  NN_t *nn = (NN_t *)malloc(sizeof(NN_t));
  if (!nn)
    return NULL;

  nn->numLayers = 0;
  while (layers[nn->numLayers] != 0) {
    nn->numLayers++;
  }

  nn->layers = (size_t *)malloc(nn->numLayers * sizeof(size_t));
  if (!nn->layers) {
    free(nn);
    return NULL;
  }

  memcpy(nn->layers, layers, nn->numLayers * sizeof(size_t));

  // Allocate memory for network components
  nn->weights =
      (long double **)malloc((nn->numLayers - 1) * sizeof(long double *));
  nn->biases =
      (long double **)malloc((nn->numLayers - 1) * sizeof(long double *));

  nn->weights_grad = malloc((nn->numLayers - 1) * sizeof(long double *));
  nn->biases_grad = malloc((nn->numLayers - 1) * sizeof(long double *));

  nn->opt_m_w = malloc((nn->numLayers - 1) * sizeof(long double *));
  nn->opt_v_w = malloc((nn->numLayers - 1) * sizeof(long double *));
  nn->opt_m_b = malloc((nn->numLayers - 1) * sizeof(long double *));
  nn->opt_v_b = malloc((nn->numLayers - 1) * sizeof(long double *));

  if (!nn->weights || !nn->biases || !nn->weights_grad || !nn->biases_grad) {
    NN_destroy(nn);
    return NULL;
  }

  // Allocate and initialize activation functions
  nn->activationFunctions = (ActivationFunction *)malloc(
      (nn->numLayers - 1) * sizeof(ActivationFunction));
  if (!nn->activationFunctions) {
    fprintf(stderr, "Failed to allocate memory for activation functions\n");
    NN_destroy(nn);
    return NULL;
  }
  for (size_t i = 0; i < nn->numLayers - 1; i++) {
    nn->activationFunctions[i] = get_activation_function(actFuncs[i]);
    if (!nn->activationFunctions[i]) {
      fprintf(stderr, "Invalid activation function for layer %zu\n", i);
      NN_destroy(nn);
      return NULL;
    }
  }

  // Allocate and initialize activation derivatives
  nn->activationDerivatives = (ActivationDerivative *)malloc(
      (nn->numLayers - 1) * sizeof(ActivationDerivative));
  if (!nn->activationDerivatives) {
    fprintf(stderr, "Failed to allocate memory for activation derivatives\n");
    NN_destroy(nn);
    return NULL;
  }
  for (size_t i = 0; i < nn->numLayers - 1; i++) {
    nn->activationDerivatives[i] = get_activation_derivative(actDerivs[i]);
    if (!nn->activationDerivatives[i]) {
      fprintf(stderr, "Invalid activation derivative for layer %zu\n", i);
      NN_destroy(nn);
      return NULL;
    }
  }

  // Initialize weights and biases for each layer
  for (size_t i = 0; i < nn->numLayers - 1; i++) {
    size_t current_size = nn->layers[i] * nn->layers[i + 1];

    nn->weights[i] = (long double *)malloc(current_size * sizeof(long double));
    nn->biases[i] =
        (long double *)malloc(nn->layers[i + 1] * sizeof(long double));

    size_t wcount = nn->layers[i] * nn->layers[i + 1];
    size_t bcount = nn->layers[i + 1];

    nn->weights_grad[i] = calloc(wcount, sizeof(long double));
    nn->biases_grad[i] = calloc(bcount, sizeof(long double));

    nn->opt_m_w[i] = calloc(wcount, sizeof(long double));
    nn->opt_v_w[i] = calloc(wcount, sizeof(long double));
    nn->opt_m_b[i] = calloc(bcount, sizeof(long double));
    nn->opt_v_b[i] = calloc(bcount, sizeof(long double));

    if (!nn->weights[i] || !nn->biases[i] || !nn->weights_grad[i] ||
        !nn->biases_grad[i] || !nn->opt_m_w[i] || !nn->opt_v_w[i] ||
        !nn->opt_m_b[i]) {
      fprintf(stderr,
              "Failed to allocate memory for weights or biases at layer %zu\n",
              i);
      NN_destroy(nn);
      return NULL;
    }
  }

  // Initialize parameters
  nn->learningRate = learningRate;
  nn->t = 1;

  // Set functions
  nn->loss = get_loss_function(lossFunc);
  nn->lossDerivative = get_loss_derivative(lossDeriv);
  nn->regularization = get_regularization_function(reg);
  nn->optimizer = get_optimizer_function(opt);

  // Initialize weights with Xavier/Glorot initialization
  for (size_t i = 0; i < nn->numLayers - 1; i++) {
    size_t current_size = nn->layers[i] * nn->layers[i + 1];
    long double scale = sqrtl(2.0L / (nn->layers[i] + nn->layers[i + 1]));

    for (size_t j = 0; j < current_size; j++) {
      nn->weights[i][j] =
          ((long double)rand() / RAND_MAX * 2.0L - 1.0L) * scale;
    }

    for (size_t j = 0; j < nn->layers[i + 1]; j++) {
      nn->biases[i][j] = 0.0L; // Initialize biases to zero
    }
  }

  return nn;
}

NN_t *NN_init_random(size_t num_inputs, size_t num_outputs) {
  if (num_inputs == 0 || num_outputs == 0)
    return NULL;

  // Create random network architecture (1-2 hidden layers)
  size_t num_hidden_layers = 1 + (rand() % 2); // Random number between 1-2
  size_t *layers =
      malloc((num_hidden_layers + 3) *
             sizeof(size_t)); // +2 for input/output, +1 for terminator
  if (!layers)
    return NULL;

  // Set input and output layers
  layers[0] = num_inputs;
  layers[num_hidden_layers + 1] = num_outputs;
  layers[num_hidden_layers + 2] = 0; // Terminator

  // Set hidden layers (random size between input and output size)
  for (size_t i = 1; i <= num_hidden_layers; i++) {
    size_t min_size = num_outputs;
    size_t max_size = num_inputs;
    if (min_size > max_size) {
      size_t temp = min_size;
      min_size = max_size;
      max_size = temp;
    }
    layers[i] = min_size + (rand() % (max_size - min_size + 1));
  }

  // Random activation functions for each layer
  size_t total_layers = num_hidden_layers + 2;
  ActivationFunctionType *act_funcs =
      malloc((total_layers - 1) * sizeof(ActivationFunctionType));
  ActivationDerivativeType *act_derivs =
      malloc((total_layers - 1) * sizeof(ActivationDerivativeType));
  if (!act_funcs || !act_derivs) {
    free(layers);
    free(act_funcs);
    free(act_derivs);
    return NULL;
  }

  // Set random activation functions
  for (size_t i = 0; i < total_layers; i++) {
    act_funcs[i] = rand() % ACTIVATION_TYPE_COUNT;
    act_derivs[i] = map_activation_to_derivative(act_funcs[i]);
  }

  // Random hyperparameters
  LossFunctionType loss_func = rand() % LOSS_TYPE_COUNT;
  LossDerivativeType loss_deriv = map_loss_to_derivative(loss_func);
  RegularizationType reg_type = rand() % REGULARIZATION_TYPE_COUNT;
  OptimizerType opt_type = rand() % OPTIMIZER_TYPE_COUNT;

  // Initialize network with random parameters
  NN_t *nn =
      NN_init(layers, act_funcs, act_derivs, loss_func, loss_deriv, reg_type,
              opt_type, 0.001L + ((long double)rand() / RAND_MAX) * 0.099L);

  // Clean up
  free(layers);
  free(act_funcs);
  free(act_derivs);

  return nn;
}

// Memory cleanup function
void NN_destroy(NN_t *nn) {
  if (!nn)
    return;

  for (size_t i = 0; i < nn->numLayers - 1; i++) {
    free(nn->weights[i]);
    free(nn->biases[i]);
    free(nn->weights_grad[i]);
    free(nn->biases_grad[i]);
    free(nn->opt_m_w[i]);
    free(nn->opt_v_w[i]);
    free(nn->opt_m_b[i]);
    free(nn->opt_v_b[i]);
  }

  free(nn->weights);
  free(nn->biases);
  free(nn->weights_grad);
  free(nn->biases_grad);
  free(nn->opt_m_w);
  free(nn->opt_v_w);
  free(nn->opt_m_b);
  free(nn->opt_v_b);

  free(nn->layers);
  free(nn->activationFunctions);
  free(nn->activationDerivatives);
  free(nn);
}

// Optimizer Functions
void sgd(NN_t *nn) {
  for (size_t l = 0; l < nn->numLayers - 1; l++) {
    size_t wcount = nn->layers[l] * nn->layers[l + 1];
    size_t bcount = nn->layers[l + 1];

    for (size_t i = 0; i < wcount; i++)
      nn->weights[l][i] -= nn->learningRate * nn->weights_grad[l][i];

    for (size_t j = 0; j < bcount; j++)
      nn->biases[l][j] -= nn->learningRate * nn->biases_grad[l][j];
  }
}

void rmsprop(NN_t *nn) {
  const long double decay = 0.9L;
  const long double eps = 1e-8L;

  for (size_t l = 0; l < nn->numLayers - 1; l++) {
    size_t wcount = nn->layers[l] * nn->layers[l + 1];
    size_t bcount = nn->layers[l + 1];

    for (size_t i = 0; i < wcount; i++) {
      nn->opt_v_w[l][i] =
          decay * nn->opt_v_w[l][i] +
          (1 - decay) * nn->weights_grad[l][i] * nn->weights_grad[l][i];

      nn->weights[l][i] -= nn->learningRate * nn->weights_grad[l][i] /
                           (sqrtl(nn->opt_v_w[l][i]) + eps);
    }

    for (size_t j = 0; j < bcount; j++) {
      nn->opt_v_b[l][j] =
          decay * nn->opt_v_b[l][j] +
          (1 - decay) * nn->biases_grad[l][j] * nn->biases_grad[l][j];

      nn->biases[l][j] -= nn->learningRate * nn->biases_grad[l][j] /
                          (sqrtl(nn->opt_v_b[l][j]) + eps);
    }
  }
}

void adagrad(NN_t *nn) {
  const long double eps = 1e-8L;

  for (size_t l = 0; l < nn->numLayers - 1; l++) {
    size_t wcount = nn->layers[l] * nn->layers[l + 1];
    size_t bcount = nn->layers[l + 1];

    for (size_t i = 0; i < wcount; i++) {
      nn->opt_v_w[l][i] += nn->weights_grad[l][i] * nn->weights_grad[l][i];
      nn->weights[l][i] -= nn->learningRate * nn->weights_grad[l][i] /
                           (sqrtl(nn->opt_v_w[l][i]) + eps);
    }

    for (size_t j = 0; j < bcount; j++) {
      nn->opt_v_b[l][j] += nn->biases_grad[l][j] * nn->biases_grad[l][j];
      nn->biases[l][j] -= nn->learningRate * nn->biases_grad[l][j] /
                          (sqrtl(nn->opt_v_b[l][j]) + eps);
    }
  }
}

void adam(NN_t *nn) {
  const long double beta1 = 0.9L;
  const long double beta2 = 0.999L;
  const long double eps = 1e-8L;

  for (size_t l = 0; l < nn->numLayers - 1; l++) {
    size_t wcount = nn->layers[l] * nn->layers[l + 1];
    size_t bcount = nn->layers[l + 1];

    for (size_t i = 0; i < wcount; i++) {
      nn->opt_m_w[l][i] =
          beta1 * nn->opt_m_w[l][i] + (1 - beta1) * nn->weights_grad[l][i];

      nn->opt_v_w[l][i] =
          beta2 * nn->opt_v_w[l][i] +
          (1 - beta2) * nn->weights_grad[l][i] * nn->weights_grad[l][i];

      long double m_hat = nn->opt_m_w[l][i] / (1 - powl(beta1, nn->t));
      long double v_hat = nn->opt_v_w[l][i] / (1 - powl(beta2, nn->t));

      nn->weights[l][i] -= nn->learningRate * m_hat / (sqrtl(v_hat) + eps);
    }

    for (size_t j = 0; j < bcount; j++) {
      nn->opt_m_b[l][j] =
          beta1 * nn->opt_m_b[l][j] + (1 - beta1) * nn->biases_grad[l][j];
      nn->opt_v_b[l][j] =
          beta2 * nn->opt_v_b[l][j] +
          (1 - beta2) * nn->biases_grad[l][j] * nn->biases_grad[l][j];

      long double m_hat = nn->opt_m_b[l][j] / (1 - powl(beta1, nn->t));
      long double v_hat = nn->opt_v_b[l][j] / (1 - powl(beta2, nn->t));

      nn->biases[l][j] -= nn->learningRate * m_hat / (sqrtl(v_hat) + eps);
    }
  }

  nn->t++;
}

void nag(NN_t *nn) {
  const long double mu = 0.9L;

  for (size_t l = 0; l < nn->numLayers - 1; l++) {
    size_t wcount = nn->layers[l] * nn->layers[l + 1];
    size_t bcount = nn->layers[l + 1];

    for (size_t i = 0; i < wcount; i++) {
      long double v_prev = nn->opt_m_w[l][i];
      nn->opt_m_w[l][i] =
          mu * nn->opt_m_w[l][i] - nn->learningRate * nn->weights_grad[l][i];
      nn->weights[l][i] += -mu * v_prev + (1 + mu) * nn->opt_m_w[l][i];
    }

    for (size_t j = 0; j < bcount; j++) {
      long double v_prev = nn->opt_m_b[l][j];
      nn->opt_m_b[l][j] =
          mu * nn->opt_m_b[l][j] - nn->learningRate * nn->biases_grad[l][j];
      nn->biases[l][j] += -mu * v_prev + (1 + mu) * nn->opt_m_b[l][j];
    }
  }
}

// Matrix Multiplication Function
long double *NN_matmul(long double inputs[], long double weights[],
                       long double biases[], size_t input_size,
                       size_t output_size) {
  long double *output =
      (long double *)malloc(output_size * sizeof(long double));
  if (!output) {
    fprintf(stderr, "NN_matmul: Failed to allocate memory for output\n");
    return NULL;
  }

  for (size_t i = 0; i < output_size; i++) {
    output[i] = biases[i];

    for (size_t j = 0; j < input_size; j++) {
      long double weight = weights[i * input_size + j];
      long double contribution = weight * inputs[j];
      output[i] += contribution;
    }
  }

  return output;
}

// Loss calculation function
long double NN_loss(NN_t *nn, long double y_true, long double y_predicted) {
  if (!nn || !nn->loss)
    return INFINITY;

  // Calculate base loss
  long double loss = nn->loss(y_true, y_predicted);

  // Add regularization if configured
  if (nn->regularization) {
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
      size_t weights_size = nn->layers[i] * nn->layers[i + 1];
      for (size_t j = 0; j < weights_size; j++) {
        loss += nn->regularization(nn->weights[i][j]);
      }
    }
  }

  return loss;
}

// Forward Propagation Functions //Caller owns buffer, must free() after use
long double *NN_forward(NN_t *nn, long double inputs[]) {
  long double **acts = NN_forward_full(nn, inputs);
  long double *out = acts[nn->numLayers - 1];

  for (size_t i = 0; i < nn->numLayers - 1; i++)
    free(acts[i]);
  free(acts);

  return out;
}

long double **NN_forward_full(NN_t *nn, long double inputs[]) {
  if (!nn || !inputs)
    return NULL;

  long double **activations =
      (long double **)malloc(nn->numLayers * sizeof(long double *));
  if (!activations)
    return NULL;

  // Input layer
  activations[0] = (long double *)malloc(nn->layers[0] * sizeof(long double));
  memcpy(activations[0], inputs, nn->layers[0] * sizeof(long double));

  for (size_t l = 1; l < nn->numLayers; l++) {
    size_t curr_size = nn->layers[l];
    size_t prev_size = nn->layers[l - 1];

    activations[l] = (long double *)malloc(curr_size * sizeof(long double));
    for (size_t j = 0; j < curr_size; j++) {
      long double sum = nn->biases[l - 1][j];
      for (size_t k = 0; k < prev_size; k++)
        sum += nn->weights[l - 1][k * curr_size + j] * activations[l - 1][k];
      // Apply activation function (except for input layer)
      if (l < nn->numLayers - 1)
        activations[l][j] = nn->activationFunctions[l - 1](sum);
      else
        activations[l][j] = sum; // output raw for loss
    }
  }

  return activations;
}

// Compute element-wise error at output
// dL/dz = dL/dy * dy/dz
// Compute backpropagation using a precomputed delta for the output layer
void NN_backprop_custom_delta(NN_t *nn, long double inputs[],
                              long double *output_delta) {
  if (!nn || !inputs || !output_delta)
    return;

  size_t L = nn->numLayers;

  // Forward pass to store pre-activation sums (z) for all layers
  long double **z_values =
      (long double **)malloc((L - 1) * sizeof(long double *));
  long double **activations = (long double **)malloc(L * sizeof(long double *));
  if (!z_values || !activations)
    return;

  // Input layer
  activations[0] = (long double *)malloc(nn->layers[0] * sizeof(long double));
  memcpy(activations[0], inputs, nn->layers[0] * sizeof(long double));

  // Forward pass to compute z and activations
  for (size_t l = 1; l < L; l++) {
    size_t in_size = nn->layers[l - 1];
    size_t out_size = nn->layers[l];

    z_values[l - 1] = (long double *)malloc(out_size * sizeof(long double));
    activations[l] = (long double *)malloc(out_size * sizeof(long double));

    for (size_t j = 0; j < out_size; j++) {
      long double sum = nn->biases[l - 1][j];
      for (size_t i = 0; i < in_size; i++)
        sum += nn->weights[l - 1][i * out_size + j] * activations[l - 1][i];
      z_values[l - 1][j] = sum;

      if (l < L - 1)
        activations[l][j] = nn->activationFunctions[l - 1](sum);
      else
        activations[l][j] = sum; // raw output
    }
  }

  // Allocate memory for deltas
  long double **deltas =
      (long double **)malloc((L - 1) * sizeof(long double *));
  for (size_t l = 0; l < L - 1; l++)
    deltas[l] = (long double *)calloc(nn->layers[l + 1], sizeof(long double));

  // Set output layer delta
  size_t last_idx = L - 2;
  memcpy(deltas[last_idx], output_delta,
         nn->layers[L - 1] * sizeof(long double));

  // Backpropagate through hidden layers
  for (size_t l = L - 2; l > 0; l--) {
    size_t curr_size = nn->layers[l];
    size_t next_size = nn->layers[l + 1];

    for (size_t i = 0; i < curr_size; i++) {
      long double sum = 0.0L;
      for (size_t j = 0; j < next_size; j++) {
        sum += nn->weights[l][i * next_size + j] * deltas[l][j];
      }
      deltas[l - 1][i] =
          sum * nn->activationDerivatives[l - 1](z_values[l - 1][i]);
    }
  }

  // Compute gradients for weights and biases
  for (size_t l = 0; l < L - 1; l++) {
    size_t in_size = nn->layers[l];
    size_t out_size = nn->layers[l + 1];

    for (size_t j = 0; j < out_size; j++) {
      nn->biases_grad[l][j] = deltas[l][j];

      for (size_t i = 0; i < in_size; i++) {
        nn->weights_grad[l][i * out_size + j] =
            activations[l][i] * deltas[l][j];
      }
    }
  }

  // Apply optimizer update
  if (nn->optimizer)
    nn->optimizer(nn);

  // Free memory
  for (size_t l = 0; l < L - 1; l++) {
    free(z_values[l]);
    free(activations[l + 1]);
    free(deltas[l]);
  }
  free(z_values);
  free(activations[0]);
  free(activations);
  free(deltas);
}

void NN_backprop(NN_t *nn, long double inputs[], long double y_true[]) {
  size_t out_size = nn->layers[nn->numLayers - 1];

  long double *y_pred = NN_forward_softmax(nn, inputs);
  long double *grad = calloc(out_size, sizeof(long double));

  for (size_t i = 0; i < out_size; i++)
    grad[i] = nn->lossDerivative(y_pred[i], y_true[i]);

  NN_backprop_custom_delta(nn, inputs, grad);

  free(grad);
  free(y_pred);
}

long double *NN_forward_softmax(NN_t *nn, long double inputs[]) {
  size_t last_layer_idx = nn->numLayers - 2;
  size_t out_size = nn->layers[nn->numLayers - 1];

  long double *output =
      NN_matmul(inputs, nn->weights[last_layer_idx], nn->biases[last_layer_idx],
                nn->layers[last_layer_idx], out_size);

  softmax(output, out_size);
  return output;
}

void NN_backprop_softmax(NN_t *nn, long double inputs[], long double y_true[],
                         long double y_pred[]) {
  size_t output_size = nn->layers[nn->numLayers - 1];
  long double *grad = calloc(output_size, sizeof(long double));
  softmax_derivative(y_pred, y_true, grad, output_size);

  // feed grad backward through previous layers normally
  NN_backprop_custom_delta(nn, inputs, grad);
  free(grad);
}

void NN_backprop_argmax(NN_t *nn, long double inputs[], long double y_true[],
                        long double y_pred[]) {
  size_t output_size = nn->layers[nn->numLayers - 1];
  long double *grad = calloc(output_size, sizeof(long double));
  argmax_derivative(y_pred, y_true, grad, output_size);

  // feed grad backward through previous layers normally
  NN_backprop_custom_delta(nn, inputs, grad);
  free(grad);
}

// Function Getters
ActivationFunction get_activation_function(ActivationFunctionType type) {
  if (type < 0 || type >= ACTIVATION_TYPE_COUNT)
    return NULL;
  return ACTIVATION_FUNCTIONS[type];
}

ActivationDerivative get_activation_derivative(ActivationDerivativeType type) {
  if (type < 0 || type >= ACTIVATION_DERIVATIVE_TYPE_COUNT)
    return NULL;
  return ACTIVATION_DERIVATIVES[type];
}

LossFunction get_loss_function(LossFunctionType type) {
  if (type < 0 || type >= LOSS_TYPE_COUNT)
    return NULL;
  return LOSS_FUNCTIONS[type];
}

LossDerivative get_loss_derivative(LossDerivativeType type) {
  if (type < 0 || type >= LOSS_DERIVATIVE_TYPE_COUNT)
    return NULL;
  return LOSS_DERIVATIVES[type];
}

RegularizationFunction get_regularization_function(RegularizationType type) {
  if (type < 0 || type >= REGULARIZATION_TYPE_COUNT)
    return NULL;
  return REGULARIZATION_FUNCTIONS[type];
}

OptimizerFunction get_optimizer_function(OptimizerType type) {
  if (type < 0 || type >= OPTIMIZER_TYPE_COUNT)
    return NULL;
  return OPTIMIZER_FUNCTIONS[type];
}

// Function to map activation function to its derivative
ActivationDerivativeType
map_activation_to_derivative(ActivationFunctionType actFunc) {
  switch (actFunc) {
  case SIGMOID:
    return SIGMOID_DERIVATIVE;
  case TANH:
    return TANH_DERIVATIVE;
  case RELU:
    return RELU_DERIVATIVE;
  case LINEAR:
    return LINEAR_DERIVATIVE;
  default:
    fprintf(stderr, "Unhandled activation function type: %d\n", actFunc);
    return ACTIVATION_DERIVATIVE_TYPE_COUNT; // Invalid value
  }
}

// Function to map loss function to its derivative
LossDerivativeType map_loss_to_derivative(LossFunctionType lossFunc) {
  switch (lossFunc) {
  case MSE:
    return MSE_DERIVATIVE;
  case MAE:
    return MAE_DERIVATIVE;
  case HUBER:
    return HUBER_DERIVATIVE;
  case LL:
    return LL_DERIVATIVE;
  case CE:
    return CE_DERIVATIVE;
  default:
    fprintf(stderr, "Unhandled loss function type: %d\n", lossFunc);
    return LOSS_DERIVATIVE_TYPE_COUNT; // Invalid value
  }
}

// String Conversion Functions
const char *activation_to_string(ActivationFunctionType type) {
  switch (type) {
  case SIGMOID:
    return "sigmoid";
  case TANH:
    return "tanh";
  case RELU:
    return "relu";
  case LINEAR:
    return "linear";
  default:
    return "unknown";
  }
}

const char *loss_to_string(LossFunctionType type) {
  switch (type) {
  case MSE:
    return "mse";
  case MAE:
    return "mae";
  case HUBER:
    return "huber";
  case LL:
    return "log_loss";
  case CE:
    return "cross_entropy";
  default:
    return "unknown";
  }
}

const char *optimizer_to_string(OptimizerType type) {
  switch (type) {
  case SGD:
    return "sgd";
  case RMSPROP:
    return "rmsprop";
  case ADAGRAD:
    return "adagrad";
  case ADAM:
    return "adam";
  case NAG:
    return "nag";
  default:
    return "unknown";
  }
}

const char *regularization_to_string(RegularizationType type) {
  switch (type) {
  case L1:
    return "l1";
  case L2:
    return "l2";
  default:
    return "unknown";
  }
}

ActivationFunctionType get_activation_function_type(const char *str) {
  if (!str)
    return -1;
  if (strcmp(str, "sigmoid") == 0)
    return SIGMOID;
  if (strcmp(str, "tanh") == 0)
    return TANH;
  if (strcmp(str, "relu") == 0)
    return RELU;
  if (strcmp(str, "linear") == 0)
    return LINEAR;
  return -1;
}

LossFunctionType get_loss_function_type(const char *str) {
  if (!str)
    return -1;
  if (strcmp(str, "mse") == 0)
    return MSE;
  if (strcmp(str, "mae") == 0)
    return MAE;
  if (strcmp(str, "huber") == 0)
    return HUBER;
  if (strcmp(str, "log_loss") == 0)
    return LL;
  if (strcmp(str, "cross_entropy") == 0)
    return CE;
  return -1;
}

OptimizerType get_optimizer_type(const char *str) {
  if (!str)
    return -1;
  if (strcmp(str, "sgd") == 0)
    return SGD;
  if (strcmp(str, "rmsprop") == 0)
    return RMSPROP;
  if (strcmp(str, "adagrad") == 0)
    return ADAGRAD;
  if (strcmp(str, "adam") == 0)
    return ADAM;
  if (strcmp(str, "nag") == 0)
    return NAG;
  return -1;
}

RegularizationType get_regularization_type(const char *str) {
  if (!str)
    return -1;
  if (strcmp(str, "l1") == 0)
    return L1;
  if (strcmp(str, "l2") == 0)
    return L2;
  return -1;
}

// Type Getters from Function Pointers
ActivationFunctionType
get_activation_function_from_func(ActivationFunction func) {
  for (int i = 0; i < ACTIVATION_TYPE_COUNT; i++) {
    if (ACTIVATION_FUNCTIONS[i] == func)
      return (ActivationFunctionType)i;
  }
  return ACTIVATION_TYPE_COUNT;
}

ActivationDerivativeType
get_activation_derivative_from_func(ActivationDerivative func) {
  for (int i = 0; i < ACTIVATION_DERIVATIVE_TYPE_COUNT; i++) {
    if (ACTIVATION_DERIVATIVES[i] == func)
      return (ActivationDerivativeType)i;
  }
  return ACTIVATION_DERIVATIVE_TYPE_COUNT; // invalid
}

LossFunctionType get_loss_function_from_func(LossFunction func) {
  for (int i = 0; i < LOSS_TYPE_COUNT; i++) {
    if (LOSS_FUNCTIONS[i] == func)
      return i;
  }
  return -1;
}

OptimizerType get_optimizer_from_func(OptimizerFunction func) {
  for (int i = 0; i < OPTIMIZER_TYPE_COUNT; i++) {
    if (OPTIMIZER_FUNCTIONS[i] == func)
      return i;
  }
  return -1;
}

RegularizationType get_regularization_from_func(RegularizationFunction func) {
  for (int i = 0; i < REGULARIZATION_TYPE_COUNT; i++) {
    if (REGULARIZATION_FUNCTIONS[i] == func)
      return i;
  }
  return -1;
}

ActivationDerivativeType
get_derivative_from_activation(ActivationFunctionType type) {
  switch (type) {
  case SIGMOID:
    return SIGMOID_DERIVATIVE;
  case TANH:
    return TANH_DERIVATIVE;
  case RELU:
    return RELU_DERIVATIVE;
  case LINEAR:
    return LINEAR_DERIVATIVE;
  default:
    return LINEAR_DERIVATIVE;
  }
}

int NN_save(NN_t *nn, const char *filename) {
  if (!nn || !filename)
    return -1;

  FILE *f = fopen(filename, "wb");
  if (!f)
    return -1;

  uint32_t magic = 0x4E4E3031; // "NN01"
  fwrite(&magic, sizeof(uint32_t), 1, f);

  fwrite(&nn->numLayers, sizeof(size_t), 1, f);
  fwrite(nn->layers, sizeof(size_t), nn->numLayers, f);

  fwrite(&nn->learningRate, sizeof(long double), 1, f);
  fwrite(&nn->t, sizeof(size_t), 1, f);

  // Activation enums
  for (size_t i = 0; i < nn->numLayers - 1; i++) {
    ActivationFunctionType a =
        get_activation_function_from_func(nn->activationFunctions[i]);
    ActivationDerivativeType d =
        get_activation_derivative_from_func(nn->activationDerivatives[i]);

    fwrite(&a, sizeof(a), 1, f);
    fwrite(&d, sizeof(d), 1, f);
  }

  LossFunctionType loss = get_loss_function_from_func(nn->loss);
  OptimizerType opt = get_optimizer_from_func(nn->optimizer);
  RegularizationType reg = get_regularization_from_func(nn->regularization);

  fwrite(&loss, sizeof(loss), 1, f);
  fwrite(&opt, sizeof(opt), 1, f);
  fwrite(&reg, sizeof(reg), 1, f);

  // Layer data
  for (size_t l = 0; l < nn->numLayers - 1; l++) {
    size_t wcount = nn->layers[l] * nn->layers[l + 1];
    size_t bcount = nn->layers[l + 1];

    fwrite(nn->weights[l], sizeof(long double), wcount, f);
    fwrite(nn->biases[l], sizeof(long double), bcount, f);

    fwrite(nn->opt_m_w[l], sizeof(long double), wcount, f);
    fwrite(nn->opt_v_w[l], sizeof(long double), wcount, f);
    fwrite(nn->opt_m_b[l], sizeof(long double), bcount, f);
    fwrite(nn->opt_v_b[l], sizeof(long double), bcount, f);
  }

  fclose(f);
  return 0;
}

NN_t *NN_load(const char *filename) {
  if (!filename)
    return NULL;

  FILE *f = fopen(filename, "rb");
  if (!f)
    return NULL;

  uint32_t magic;
  fread(&magic, sizeof(uint32_t), 1, f);
  if (magic != 0x4E4E3031) {
    fclose(f);
    return NULL;
  }

  NN_t *nn = calloc(1, sizeof(NN_t));
  if (!nn) {
    fclose(f);
    return NULL;
  }

  fread(&nn->numLayers, sizeof(size_t), 1, f);

  nn->layers = malloc(nn->numLayers * sizeof(size_t));
  fread(nn->layers, sizeof(size_t), nn->numLayers, f);

  fread(&nn->learningRate, sizeof(long double), 1, f);
  fread(&nn->t, sizeof(size_t), 1, f);

  // Activations
  nn->activationFunctions =
      malloc((nn->numLayers - 1) * sizeof(ActivationFunction));
  nn->activationDerivatives =
      malloc((nn->numLayers - 1) * sizeof(ActivationDerivative));

  for (size_t i = 0; i < nn->numLayers - 1; i++) {
    ActivationFunctionType a;
    ActivationDerivativeType d;
    fread(&a, sizeof(a), 1, f);
    fread(&d, sizeof(d), 1, f);

    nn->activationFunctions[i] = get_activation_function(a);
    nn->activationDerivatives[i] = get_activation_derivative(d);
  }

  LossFunctionType loss;
  OptimizerType opt;
  RegularizationType reg;

  fread(&loss, sizeof(loss), 1, f);
  fread(&opt, sizeof(opt), 1, f);
  fread(&reg, sizeof(reg), 1, f);

  nn->loss = get_loss_function(loss);
  nn->lossDerivative = get_loss_derivative(map_loss_to_derivative(loss));
  nn->optimizer = get_optimizer_function(opt);
  nn->regularization = get_regularization_function(reg);

  // Allocate arrays
  size_t L = nn->numLayers - 1;
  nn->weights = malloc(L * sizeof(long double *));
  nn->biases = malloc(L * sizeof(long double *));
  nn->weights_grad = malloc(L * sizeof(long double *));
  nn->biases_grad = malloc(L * sizeof(long double *));
  nn->opt_m_w = malloc(L * sizeof(long double *));
  nn->opt_v_w = malloc(L * sizeof(long double *));
  nn->opt_m_b = malloc(L * sizeof(long double *));
  nn->opt_v_b = malloc(L * sizeof(long double *));

  for (size_t l = 0; l < L; l++) {
    size_t wcount = nn->layers[l] * nn->layers[l + 1];
    size_t bcount = nn->layers[l + 1];

    nn->weights[l] = malloc(wcount * sizeof(long double));
    nn->biases[l] = malloc(bcount * sizeof(long double));
    nn->weights_grad[l] = calloc(wcount, sizeof(long double));
    nn->biases_grad[l] = calloc(bcount, sizeof(long double));

    nn->opt_m_w[l] = malloc(wcount * sizeof(long double));
    nn->opt_v_w[l] = malloc(wcount * sizeof(long double));
    nn->opt_m_b[l] = malloc(bcount * sizeof(long double));
    nn->opt_v_b[l] = malloc(bcount * sizeof(long double));

    fread(nn->weights[l], sizeof(long double), wcount, f);
    fread(nn->biases[l], sizeof(long double), bcount, f);
    fread(nn->opt_m_w[l], sizeof(long double), wcount, f);
    fread(nn->opt_v_w[l], sizeof(long double), wcount, f);
    fread(nn->opt_m_b[l], sizeof(long double), bcount, f);
    fread(nn->opt_v_b[l], sizeof(long double), bcount, f);
  }

  fclose(f);
  return nn;
}

// Activation Functions
long double sigmoid(long double x) { return 1.0L / (1.0L + expl(-x)); }

long double tanh_activation(long double x) { return tanhl(x); }

long double relu(long double x) { return x > 0 ? x : 0; }

long double linear(long double x) { return x; }

void softmax(long double *vec, size_t size) {
  long double max = vec[0];
  for (size_t i = 1; i < size; i++)
    if (vec[i] > max)
      max = vec[i];

  long double sum = 0.0L;
  for (size_t i = 0; i < size; i++) {
    vec[i] = expl(vec[i] - max); // stability
    sum += vec[i];
  }
  for (size_t i = 0; i < size; i++)
    vec[i] /= sum;
}

void softmax_derivative(long double *predicted, long double *one_hot,
                        long double *gradients, size_t size) {
  for (size_t i = 0; i < size; i++)
    gradients[i] = predicted[i] - one_hot[i];
}

size_t argmax_index(long double *vec, size_t size) {
  size_t idx = 0;
  for (size_t i = 1; i < size; i++)
    if (vec[i] > vec[idx])
      idx = i;
  return idx;
}

void argmax(long double *vec, size_t size) {
  size_t idx = argmax_index(vec, size);
  for (size_t i = 0; i < size; i++)
    vec[i] = (i == idx) ? 1.0L : 0.0L;
}

void argmax_derivative(long double *predicted, long double *one_hot,
                       long double *gradients, size_t size) {
  // identical to softmax for cross-entropy/one-hot
  for (size_t i = 0; i < size; i++)
    gradients[i] = predicted[i] - one_hot[i];
}

// Activation Derivatives
long double sigmoid_derivative(long double x) {
  long double s = sigmoid(x);
  return s * (1.0L - s);
}

long double tanh_derivative(long double x) {
  long double t = tanhl(x);
  return 1.0L - t * t;
}

long double relu_derivative(long double x) { return x > 0 ? 1.0L : 0.0L; }

long double linear_derivative(long double x) { return 1.0L; }

// Loss Functions
long double mse(long double y_true, long double y_pred) {
  long double diff = y_true - y_pred;
  return 0.5L * diff * diff;
}

long double mae(long double y_true, long double y_pred) {
  return fabsl(y_true - y_pred);
}

long double huber(long double y_true, long double y_pred) { // Huber Loss
  const long double delta = 1.0L;
  long double diff = fabsl(y_true - y_pred);
  if (diff <= delta) {
    return 0.5L * diff * diff;
  }
  return delta * diff - 0.5L * delta * delta;
}

long double ll(long double y_true, long double y_pred) { // Log Loss
  const long double epsilon = 1e-15L;
  y_pred = fmaxl(fminl(y_pred, 1.0L - epsilon), epsilon);
  return -(y_true * logl(y_pred) + (1.0L - y_true) * logl(1.0L - y_pred));
}

long double ce(long double y_true, long double y_pred) { // Cross Entropy
  const long double epsilon = 1e-15L;
  y_pred = fmaxl(fminl(y_pred, 1.0L - epsilon), epsilon);
  return -y_true * logl(y_pred);
}

// Loss Derivatives
long double mse_derivative(long double y_true, long double y_pred) {
  return y_pred - y_true;
}

long double mae_derivative(long double y_true, long double y_pred) {
  return y_pred > y_true ? 1.0L : -1.0L;
}

long double huber_derivative(long double y_true,
                             long double y_pred) { // Huber Loss Derivative
  const long double delta = 1.0L;
  long double diff = y_pred - y_true;
  if (fabsl(diff) <= delta) {
    return diff;
  }
  return delta * (diff > 0 ? 1.0L : -1.0L);
}

long double ll_derivative(long double y_true,
                          long double y_pred) { // Log Loss Derivative
  const long double epsilon = 1e-15L;
  y_pred = fmaxl(fminl(y_pred, 1.0L - epsilon), epsilon);
  return (y_pred - y_true) / (y_pred * (1.0L - y_pred));
}

long double ce_derivative(long double y_true,
                          long double y_pred) { // Cross Entropy Derivative
  const long double epsilon = 1e-15L;
  y_pred = fmaxl(fminl(y_pred, 1.0L - epsilon), epsilon);
  return -y_true / y_pred;
}

// Regularization Functions
long double l1(long double weight) { return fabsl(weight); }

long double l2(long double weight) { return 0.5L * weight * weight; }
