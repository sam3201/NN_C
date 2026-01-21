#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
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
  ACTIVATION_TYPE_COUNT = 4
} ActivationFunctionType;

// Activation derivative types
typedef enum {
  SIGMOID_DERIVATIVE = 0,
  TANH_DERIVATIVE = 1,
  RELU_DERIVATIVE = 2,
  LINEAR_DERIVATIVE = 3,
  ACTIVATION_DERIVATIVE_TYPE_COUNT = 4
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
  ADADELTA = 5,         // Adaptive learning rate
  RMSPROP_NESTEROV = 6, // RMSProp with Nesterov
  ADAMAX = 7,           // Adam infinity norm
  NADAM = 8,            // Nesterov-accelerated Adam
  OPTIMIZER_TYPE_COUNT = 9
} OptimizerType;

// Regularization types
typedef enum {
  L1 = 0, // Lasso
  L2 = 1, // Ridge
  REGULARIZATION_TYPE_COUNT = 2
} RegularizationType;

// Weight initialization types
typedef enum {
  ZERO = 0,           // Zero/Constant initialization (bad due to symmetry)
  RANDOM_UNIFORM = 1, // Basic uniform distribution
  RANDOM_NORMAL = 2,  // Basic normal distribution (risks gradient issues)
  XAVIER = 3,         // Xavier/Glorot (for sigmoid/tanh)
  HE = 4,             // He initialization (for ReLU)
  LECUN = 5,          // LeCun initialization (for deeper models)
  ORTHOGONAL = 6,     // Orthogonal initialization (for complex models)
  WEIGHT_INIT_TYPE_COUNT = 7
} WeightInitType;

// Learning rate scheduler types
typedef enum {
  CONSTANT = 0,
  STEP = 1,        // Step decay
  EXPONENTIAL = 2, // Exponential decay
  COSINE = 3,      // Cosine annealing
  ONE_CYCLE = 4,   // One cycle policy
  WARMUP = 5,      // Learning rate warmup
  SCHEDULER_TYPE_COUNT = 6
} LRSchedulerType;

// Advanced network types
typedef enum {
  NETWORK_MLP = 0,  // Multi-Layer Perceptron
  NETWORK_RNN = 1,  // Recurrent Neural Network
  NETWORK_LSTM = 2, // Long Short-Term Memory
  NETWORK_GNN = 3,  // Graph Neural Network
  NETWORK_SNN = 4,  // Spiking Neural Network
  NETWORK_KAN = 5,  // Kolmogorov-Arnold Network
  NETWORK_GAN = 6,  // Generative Adversarial Network
  NETWORK_TYPE_COUNT = 7
} NetworkType;

// Function pointer types
typedef long double (*ActivationFunction)(long double);
typedef long double (*ActivationDerivative)(long double);
typedef long double (*LossFunction)(long double, long double);
typedef long double (*LossDerivative)(long double, long double);
typedef void (*OptimizerFunction)(NN_t *);
typedef long double (*RegularizationFunction)(long double);
typedef void (*NNGradHook)(NN_t *, void *);

// Neural Network structure
typedef struct NN_t {
  size_t numLayers; // Number of layers
  size_t *layers;   // Array of layer sizes

  long double **weights; // Weight matrices
  long double **biases;  // Bias vectors

  long double **weights_grad; // dL/dW
  long double **biases_grad;  // dL/db

  long double **opt_m_w; // Adam / momentum m
  long double **opt_v_w; // Adam / RMSProp v
  long double **opt_m_b;
  long double **opt_v_b;

  long double learningRate;     // Learning rate
  long double baseLearningRate; // Base learning rate before scheduling
  long double lr_sched_start;   // LR schedule multiplier start
  long double lr_sched_end;     // LR schedule multiplier end
  size_t lr_sched_steps;        // Number of schedule steps
  size_t lr_sched_step;         // Current schedule step
  long double global_grad_clip; // Global grad clip (L2 norm)
  size_t t;                     // Time step for Adam

  NNGradHook grad_hook;
  void *grad_hook_ctx;

  ActivationFunction *activationFunctions;
  ActivationDerivative *activationDerivatives;
  LossFunction loss;
  LossDerivative lossDerivative;
  OptimizerFunction optimizer;
  RegularizationFunction regularization;
  WeightInitType weightInit; // Weight initialization type

  // Enhanced optimizer parameters
  float beta1;        // Adam beta1
  float beta2;        // Adam beta2
  float epsilon;      // Adam epsilon
  float weight_decay; // Weight decay
  float dropout_rate; // Dropout rate

  // Learning rate scheduling
  LRSchedulerType lr_scheduler;
  float initial_lr;
  float final_lr;
  int total_steps;
  int current_step;
  float warmup_steps;

  // Gradient management
  float gradient_clip_value;
  int gradient_clip_type; // 0=none, 1=value, 2=norm
  float gradient_noise_std;
  int accumulation_steps;

  // Training monitoring
  float *loss_history;
  float *accuracy_history;
  int history_capacity;
  int history_size;
  int epoch_count;
} NN_t;

// Initialization
NN_t *NN_init(size_t *layers, ActivationFunctionType *actFuncs,
              ActivationDerivativeType *actDerivs, LossFunctionType lossFunc,
              LossDerivativeType lossDeriv, RegularizationType reg,
              OptimizerType opt, long double learningRate);
NN_t *NN_init_with_weight_init(size_t *layers, ActivationFunctionType *actFuncs,
                               ActivationDerivativeType *actDerivs,
                               LossFunctionType lossFunc,
                               LossDerivativeType lossDeriv,
                               RegularizationType reg, OptimizerType opt,
                               long double learningRate,
                               WeightInitType weightInit);

NN_t *NN_init_random(size_t num_inputs, size_t num_outputs);
void NN_destroy(NN_t *nn);
void NN_set_base_lr(NN_t *nn, long double base_lr);
void NN_set_lr_schedule(NN_t *nn, long double mult_start, long double mult_end,
                        size_t steps);
void NN_set_global_grad_clip(NN_t *nn, long double clip);
void NN_set_grad_hook(NN_t *nn, NNGradHook hook, void *ctx);

// Enhanced optimizer functions
void NN_set_optimizer_params(NN_t *nn, float beta1, float beta2, float epsilon);
void NN_set_weight_decay(NN_t *nn, float decay_rate);
void NN_set_dropout(NN_t *nn, float dropout_rate);

// Learning rate scheduling
void NN_set_lr_scheduler(NN_t *nn, LRSchedulerType type, float initial_lr,
                         float final_lr, int total_steps);
void NN_step_lr(NN_t *nn);
float NN_get_current_lr(NN_t *nn);

// Gradient management
void NN_set_gradient_clipping(NN_t *nn, int clip_type, float threshold);
void NN_enable_gradient_noise(NN_t *nn, float noise_std);
void NN_set_gradient_accumulation(NN_t *nn, int accumulation_steps);

// Training monitoring
void NN_enable_monitoring(NN_t *nn, int history_capacity);
void NN_log_metrics(NN_t *nn, float loss, float accuracy);
void NN_print_training_summary(NN_t *nn);

// Model management
void NN_save_model(NN_t *nn, const char *filename);
NN_t *NN_load_model(const char *filename);
void NN_print_model_summary(NN_t *nn);

// Forward propagation
long double **NN_forward_full(NN_t *nn, long double inputs[]);
long double *NN_forward(NN_t *nn, long double inputs[]);
long double *NN_forward_softmax(NN_t *nn, long double inputs[]);

// Backpropagation
void NN_backprop_custom_delta(NN_t *nn, long double inputs[],
                              long double *output_delta);
// Backprop using precomputed output delta, and ALSO return dL/dInputs.
// Caller owns returned buffer and must free().
long double *NN_backprop_custom_delta_inputgrad(NN_t *nn, long double inputs[],
                                                long double *output_delta);
void NN_backprop(NN_t *nn, long double inputs[], long double y_true[]);
void NN_backprop_softmax(NN_t *nn, long double inputs[], long double y_true[],
                         long double y_pred[]);
void NN_backprop_argmax(NN_t *nn, long double inputs[], long double y_true[],
                        long double y_pred[]);
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
void adadelta(NN_t *nn);
void rmsprop_nesterov(NN_t *nn);
void adamax(NN_t *nn);
void nadam(NN_t *nn);

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

ActivationFunctionType
get_activation_function_from_func(ActivationFunction func);

ActivationDerivativeType
get_activation_derivative_from_func(ActivationDerivative func);
LossFunctionType get_loss_function_from_func(LossFunction func);
OptimizerType get_optimizer_from_func(OptimizerFunction func);
RegularizationType get_regularization_from_func(RegularizationFunction func);

ActivationDerivativeType
get_derivative_from_activation(ActivationFunctionType type);

// Save / Load
int NN_save(NN_t *nn, const char *filename);
NN_t *NN_load(const char *filename);

int NN_save_fp(NN_t *nn, FILE *f);
NN_t *NN_load_fp(FILE *f);

// Enhanced random number generators
static double uniform_random(double min_val, double max_val) {
  return min_val + (max_val - min_val) * ((double)rand() / RAND_MAX);
}

static double normal_random(double mean, double stddev) {
  static double spare = 0.0;
  static bool has_spare = false;

  if (has_spare) {
    has_spare = false;
    return mean + stddev * spare;
  } else {
    has_spare = true;
    double u, v, s;
    do {
      u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
      v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
      s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + stddev * u * s;
  }
}

// Orthogonal initialization using Gram-Schmidt process
static void orthogonal_init(long double *matrix, size_t rows, size_t cols) {
  // Initialize with normal distribution
  for (size_t i = 0; i < rows * cols; i++) {
    matrix[i] = normal_random(0.0, 1.0);
  }

  // Gram-Schmidt orthogonalization
  for (size_t col = 0; col < cols; col++) {
    // Normalize column
    long double norm = 0.0L;
    for (size_t row = 0; row < rows; row++) {
      norm += matrix[row * cols + col] * matrix[row * cols + col];
    }
    norm = sqrtl(norm);

    if (norm > 1e-10L) {
      for (size_t row = 0; row < rows; row++) {
        matrix[row * cols + col] /= norm;
      }
    }

    // Orthogonalize against previous columns
    for (size_t prev_col = 0; prev_col < col; prev_col++) {
      long double dot = 0.0L;
      for (size_t row = 0; row < rows; row++) {
        dot += matrix[row * cols + col] * matrix[row * cols + prev_col];
      }

      for (size_t row = 0; row < rows; row++) {
        matrix[row * cols + col] -= dot * matrix[row * cols + prev_col];
      }
    }

    // Renormalize
    norm = 0.0L;
    for (size_t row = 0; row < rows; row++) {
      norm += matrix[row * cols + col] * matrix[row * cols + col];
    }
    norm = sqrtl(norm);

    if (norm > 1e-10L) {
      for (size_t row = 0; row < rows; row++) {
        matrix[row * cols + col] /= norm;
      }
    }
  }
}

// Lookup tables for functions
static const ActivationFunction ACTIVATION_FUNCTIONS[] = {
    sigmoid, tanh_activation, relu, linear};

static const ActivationDerivative ACTIVATION_DERIVATIVES[] = {
    sigmoid_derivative, tanh_derivative, relu_derivative, linear_derivative};

static const LossFunction LOSS_FUNCTIONS[] = {mse, mae, huber, ll, ce};

static const LossDerivative LOSS_DERIVATIVES[] = {
    mse_derivative, mae_derivative, huber_derivative, ll_derivative,
    ce_derivative};

static const OptimizerFunction OPTIMIZER_FUNCTIONS[] = {
    sgd,      rmsprop,          adagrad, adam, nag,
    adadelta, rmsprop_nesterov, adamax,  nadam};

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
  nn->baseLearningRate = learningRate;
  nn->lr_sched_start = 1.0L;
  nn->lr_sched_end = 1.0L;
  nn->lr_sched_steps = 0;
  nn->lr_sched_step = 0;
  nn->global_grad_clip = 0.0L;
  nn->grad_hook = NULL;
  nn->grad_hook_ctx = NULL;
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

NN_t *NN_init_with_weight_init(size_t *layers, ActivationFunctionType *actFuncs,
                               ActivationDerivativeType *actDerivs,
                               LossFunctionType lossFunc,
                               LossDerivativeType lossDeriv,
                               RegularizationType reg, OptimizerType opt,
                               long double learningRate,
                               WeightInitType weightInit) {
  if (!layers || !actFuncs || !actDerivs) {
    fprintf(stderr, "NN_init_with_weight_init: NULL input parameters\n");
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

  for (size_t i = 0; i < nn->numLayers; i++) {
    nn->layers[i] = layers[i];
  }

  // Allocate memory for weights and biases
  nn->weights =
      (long double **)malloc((nn->numLayers - 1) * sizeof(long double *));
  nn->biases =
      (long double **)malloc((nn->numLayers - 1) * sizeof(long double *));
  nn->weights_grad =
      (long double **)malloc((nn->numLayers - 1) * sizeof(long double *));
  nn->biases_grad =
      (long double **)malloc((nn->numLayers - 1) * sizeof(long double *));

  // Allocate optimizer memory
  nn->opt_m_w =
      (long double **)malloc((nn->numLayers - 1) * sizeof(long double *));
  nn->opt_v_w =
      (long double **)malloc((nn->numLayers - 1) * sizeof(long double *));
  nn->opt_m_b =
      (long double **)malloc((nn->numLayers - 1) * sizeof(long double *));
  nn->opt_v_b =
      (long double **)malloc((nn->numLayers - 1) * sizeof(long double *));

  // Allocate activation functions
  nn->activationFunctions =
      (ActivationFunction *)malloc(nn->numLayers * sizeof(ActivationFunction));
  nn->activationDerivatives = (ActivationDerivative *)malloc(
      nn->numLayers * sizeof(ActivationDerivative));

  if (!nn->weights || !nn->biases || !nn->weights_grad || !nn->biases_grad ||
      !nn->opt_m_w || !nn->opt_v_w || !nn->opt_m_b || !nn->opt_v_b ||
      !nn->activationFunctions || !nn->activationDerivatives) {
    fprintf(stderr, "Failed to allocate memory for network components\n");
    NN_destroy(nn);
    return NULL;
  }

  // Initialize weights and biases for each layer
  for (size_t i = 0; i < nn->numLayers - 1; i++) {
    size_t current_size = nn->layers[i] * nn->layers[i + 1];
    size_t bcount = nn->layers[i + 1];

    nn->weights[i] = (long double *)malloc(current_size * sizeof(long double));
    nn->biases[i] = (long double *)malloc(bcount * sizeof(long double));
    nn->weights_grad[i] =
        (long double *)malloc(current_size * sizeof(long double));
    nn->biases_grad[i] = (long double *)malloc(bcount * sizeof(long double));

    nn->opt_m_w[i] = (long double *)malloc(current_size * sizeof(long double));
    nn->opt_v_w[i] = (long double *)malloc(current_size * sizeof(long double));
    nn->opt_m_b[i] = (long double *)malloc(bcount * sizeof(long double));
    nn->opt_v_b[i] = (long double *)malloc(bcount * sizeof(long double));

    if (!nn->weights[i] || !nn->biases[i] || !nn->weights_grad[i] ||
        !nn->biases_grad[i] || !nn->opt_m_w[i] || !nn->opt_v_w[i] ||
        !nn->opt_m_b[i] || !nn->opt_v_b[i]) {
      fprintf(stderr,
              "Failed to allocate memory for weights or biases at layer %zu\n",
              i);
      NN_destroy(nn);
      return NULL;
    }
  }

  // Initialize parameters
  nn->learningRate = learningRate;
  nn->baseLearningRate = learningRate;
  nn->lr_sched_start = 1.0L;
  nn->lr_sched_end = 1.0L;
  nn->lr_sched_steps = 0;
  nn->lr_sched_step = 0;
  nn->global_grad_clip = 0.0L;
  nn->grad_hook = NULL;
  nn->grad_hook_ctx = NULL;
  nn->t = 1;
  nn->weightInit = weightInit;

  // Set functions
  nn->loss = get_loss_function(lossFunc);
  nn->lossDerivative = get_loss_derivative(lossDeriv);
  nn->regularization = get_regularization_function(reg);
  nn->optimizer = get_optimizer_function(opt);

  // Initialize weights with specified initialization method
  for (size_t i = 0; i < nn->numLayers - 1; i++) {
    size_t current_size = nn->layers[i] * nn->layers[i + 1];
    size_t bcount = nn->layers[i + 1];
    long double scale = 0.0L;
    size_t fan_in = nn->layers[i];
    size_t fan_out = nn->layers[i + 1];

    switch (weightInit) {
    case ZERO:
      // Zero/Constant initialization (bad due to symmetry)
      scale = 0.0L;
      break;
    case RANDOM_UNIFORM:
      // Basic uniform distribution [-1, 1]
      scale = 1.0L;
      break;
    case RANDOM_NORMAL:
      // Basic normal distribution (risks gradient issues)
      scale = 1.0L;
      break;
    case XAVIER:
      // Xavier/Glorot initialization (for sigmoid/tanh)
      // Variance = 2.0 / (fan_in + fan_out)
      scale = sqrtl(2.0L / (fan_in + fan_out));
      break;
    case HE:
      // He initialization (for ReLU)
      // Variance = 2.0 / fan_in
      scale = sqrtl(2.0L / fan_in);
      break;
    case LECUN:
      // LeCun initialization (for deeper models)
      // Variance = 1.0 / fan_in
      scale = sqrtl(1.0L / fan_in);
      break;
    case ORTHOGONAL:
      // Orthogonal initialization (for complex models)
      scale = 1.0L;
      break;
    default:
      // Default to Xavier
      scale = sqrtl(2.0L / (fan_in + fan_out));
      break;
    }

    // Initialize weights
    for (size_t j = 0; j < current_size; j++) {
      switch (weightInit) {
      case ZERO:
        nn->weights[i][j] = 0.0L;
        break;
      case RANDOM_UNIFORM:
        nn->weights[i][j] = uniform_random(-1.0L, 1.0L) * scale;
        break;
      case RANDOM_NORMAL:
        nn->weights[i][j] = normal_random(0.0L, scale);
        break;
      case XAVIER:
      case HE:
      case LECUN:
        nn->weights[i][j] = uniform_random(-1.0L, 1.0L) * scale;
        break;
      case ORTHOGONAL:
        // For orthogonal, we need to handle the entire weight matrix at once
        if (j == 0) {
          orthogonal_init(nn->weights[i], fan_in, fan_out);
          j = current_size - 1; // Skip rest since we handled all at once
        }
        break;
      default:
        nn->weights[i][j] = uniform_random(-1.0L, 1.0L) * scale;
        break;
      }
    }

    // Initialize biases (typically small random values or zeros)
    for (size_t j = 0; j < bcount; j++) {
      switch (weightInit) {
      case ZERO:
        nn->biases[i][j] = 0.0L;
        break;
      case RANDOM_UNIFORM:
      case RANDOM_NORMAL:
      case XAVIER:
      case HE:
      case LECUN:
      case ORTHOGONAL:
        nn->biases[i][j] = uniform_random(-0.1L, 0.1L);
        break;
      default:
        nn->biases[i][j] = 0.0L;
        break;
      }
    }

    // Initialize optimizer states
    for (size_t j = 0; j < current_size; j++) {
      nn->weights_grad[i][j] = 0.0L;
      nn->opt_m_w[i][j] = 0.0L;
      nn->opt_v_w[i][j] = 0.0L;
    }

    for (size_t j = 0; j < bcount; j++) {
      nn->biases_grad[i][j] = 0.0L;
      nn->opt_m_b[i][j] = 0.0L;
      nn->opt_v_b[i][j] = 0.0L;
    }
  }

  // Set activation functions
  for (size_t i = 0; i < nn->numLayers; i++) {
    nn->activationFunctions[i] = ACTIVATION_FUNCTIONS[actFuncs[i]];
    nn->activationDerivatives[i] = ACTIVATION_DERIVATIVES[actDerivs[i]];
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
  for (size_t i = 0; i < total_layers - 1; i++) {
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

  // Only free individual arrays if they were allocated
  if (nn->weights) {
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
      free(nn->weights[i]);
    }
  }
  if (nn->biases) {
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
      free(nn->biases[i]);
    }
  }
  if (nn->weights_grad) {
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
      free(nn->weights_grad[i]);
    }
  }
  if (nn->biases_grad) {
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
      free(nn->biases_grad[i]);
    }
  }
  if (nn->opt_m_w) {
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
      free(nn->opt_m_w[i]);
    }
  }
  if (nn->opt_v_w) {
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
      free(nn->opt_v_w[i]);
    }
  }
  if (nn->opt_m_b) {
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
      free(nn->opt_m_b[i]);
    }
  }
  if (nn->opt_v_b) {
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
      free(nn->opt_v_b[i]);
    }
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

void NN_set_base_lr(NN_t *nn, long double base_lr) {
  if (!nn)
    return;
  nn->baseLearningRate = base_lr;
  if (nn->lr_sched_steps == 0)
    nn->learningRate = base_lr;
}

void NN_set_lr_schedule(NN_t *nn, long double mult_start, long double mult_end,
                        size_t steps) {
  if (!nn)
    return;
  nn->lr_sched_start = mult_start;
  nn->lr_sched_end = mult_end;
  nn->lr_sched_steps = steps;
  nn->lr_sched_step = 0;
  if (steps == 0)
    nn->learningRate = nn->baseLearningRate * mult_start;
}

void NN_set_global_grad_clip(NN_t *nn, long double clip) {
  if (!nn)
    return;
  nn->global_grad_clip = clip;
}

void NN_set_grad_hook(NN_t *nn, NNGradHook hook, void *ctx) {
  if (!nn)
    return;
  nn->grad_hook = hook;
  nn->grad_hook_ctx = ctx;
}

static void nn_apply_lr_schedule(NN_t *nn) {
  if (!nn)
    return;
  if (nn->lr_sched_steps == 0) {
    nn->learningRate = nn->baseLearningRate;
    return;
  }

  size_t steps = nn->lr_sched_steps;
  size_t step = nn->lr_sched_step;
  if (steps <= 1)
    step = 0;
  else if (step >= steps)
    step = steps - 1;

  long double t =
      (steps <= 1) ? 1.0L : ((long double)step / (long double)(steps - 1));
  long double mult =
      nn->lr_sched_start + (nn->lr_sched_end - nn->lr_sched_start) * t;
  nn->learningRate = nn->baseLearningRate * mult;

  if (nn->lr_sched_step < steps)
    nn->lr_sched_step++;
}

static void nn_apply_global_grad_clip(NN_t *nn) {
  if (!nn || !(nn->global_grad_clip > 0.0L))
    return;

  long double sumsq = 0.0L;
  for (size_t l = 0; l < nn->numLayers - 1; l++) {
    size_t wcount = nn->layers[l] * nn->layers[l + 1];
    size_t bcount = nn->layers[l + 1];
    for (size_t i = 0; i < wcount; i++) {
      long double g = nn->weights_grad[l][i];
      sumsq += g * g;
    }
    for (size_t j = 0; j < bcount; j++) {
      long double g = nn->biases_grad[l][j];
      sumsq += g * g;
    }
  }
  if (sumsq <= 0.0L)
    return;
  long double norm = sqrtl(sumsq);
  if (norm <= nn->global_grad_clip)
    return;
  long double scale = nn->global_grad_clip / norm;
  for (size_t l = 0; l < nn->numLayers - 1; l++) {
    size_t wcount = nn->layers[l] * nn->layers[l + 1];
    size_t bcount = nn->layers[l + 1];
    for (size_t i = 0; i < wcount; i++)
      nn->weights_grad[l][i] *= scale;
    for (size_t j = 0; j < bcount; j++)
      nn->biases_grad[l][j] *= scale;
  }
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

// Enhanced optimizers
void adadelta(NN_t *nn) {
  const long double rho = 0.9L;
  const long double eps = 1e-6L;

  for (size_t l = 0; l < nn->numLayers - 1; l++) {
    size_t wcount = nn->layers[l] * nn->layers[l + 1];
    size_t bcount = nn->layers[l + 1];

    for (size_t i = 0; i < wcount; i++) {
      long double accu_grad = nn->weights_grad[l][i] * nn->weights_grad[l][i];
      nn->opt_v_w[l][i] = rho * nn->opt_v_w[l][i] + (1 - rho) * accu_grad;

      long double update =
          sqrtl((nn->opt_m_w[l][i] + eps) / (nn->opt_v_w[l][i] + eps)) *
          nn->weights_grad[l][i];
      nn->weights[l][i] -= update;
      nn->opt_m_w[l][i] = rho * nn->opt_m_w[l][i] + (1 - rho) * update * update;
    }

    for (size_t j = 0; j < bcount; j++) {
      long double accu_grad = nn->biases_grad[l][j] * nn->biases_grad[l][j];
      nn->opt_v_b[l][j] = rho * nn->opt_v_b[l][j] + (1 - rho) * accu_grad;

      long double update =
          sqrtl((nn->opt_m_b[l][j] + eps) / (nn->opt_v_b[l][j] + eps)) *
          nn->biases_grad[l][j];
      nn->biases[l][j] -= update;
      nn->opt_m_b[l][j] = rho * nn->opt_m_b[l][j] + (1 - rho) * update * update;
    }
  }
}

void rmsprop_nesterov(NN_t *nn) {
  const long double decay = 0.9L;
  const long double eps = 1e-8L;
  const long double momentum = 0.9L;

  for (size_t l = 0; l < nn->numLayers - 1; l++) {
    size_t wcount = nn->layers[l] * nn->layers[l + 1];
    size_t bcount = nn->layers[l + 1];

    for (size_t i = 0; i < wcount; i++) {
      nn->opt_v_w[l][i] =
          decay * nn->opt_v_w[l][i] +
          (1 - decay) * nn->weights_grad[l][i] * nn->weights_grad[l][i];

      long double nesterov_grad =
          nn->weights_grad[l][i] + momentum * nn->opt_m_w[l][i];
      nn->weights[l][i] -=
          nn->learningRate * nesterov_grad / (sqrtl(nn->opt_v_w[l][i]) + eps);
      nn->opt_m_w[l][i] =
          momentum * nn->opt_m_w[l][i] -
          nn->learningRate * nesterov_grad / (sqrtl(nn->opt_v_w[l][i]) + eps);
    }

    for (size_t j = 0; j < bcount; j++) {
      nn->opt_v_b[l][j] =
          decay * nn->opt_v_b[l][j] +
          (1 - decay) * nn->biases_grad[l][j] * nn->biases_grad[l][j];

      long double nesterov_grad =
          nn->biases_grad[l][j] + momentum * nn->opt_m_b[l][j];
      nn->biases[l][j] -=
          nn->learningRate * nesterov_grad / (sqrtl(nn->opt_v_b[l][j]) + eps);
      nn->opt_m_b[l][j] =
          momentum * nn->opt_m_b[l][j] -
          nn->learningRate * nesterov_grad / (sqrtl(nn->opt_v_b[l][j]) + eps);
    }
  }
}

void adamax(NN_t *nn) {
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
          fmaxl(beta2 * nn->opt_v_w[l][i], fabsl(nn->weights_grad[l][i]));

      long double m_hat = nn->opt_m_w[l][i] / (1 - powl(beta1, nn->t));
      nn->weights[l][i] -= nn->learningRate * m_hat / (nn->opt_v_w[l][i] + eps);
    }

    for (size_t j = 0; j < bcount; j++) {
      nn->opt_m_b[l][j] =
          beta1 * nn->opt_m_b[l][j] + (1 - beta1) * nn->biases_grad[l][j];
      nn->opt_v_b[l][j] =
          fmaxl(beta2 * nn->opt_v_b[l][j], fabsl(nn->biases_grad[l][j]));

      long double m_hat = nn->opt_m_b[l][j] / (1 - powl(beta1, nn->t));
      nn->biases[l][j] -= nn->learningRate * m_hat / (nn->opt_v_b[l][j] + eps);
    }
  }
}

void nadam(NN_t *nn) {
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

      // Nesterov momentum
      long double momentum =
          beta1 * m_hat + (1 - beta1) * nn->weights_grad[l][i];
      nn->weights[l][i] -= nn->learningRate * momentum / (sqrtl(v_hat) + eps);
    }

    for (size_t j = 0; j < bcount; j++) {
      nn->opt_m_b[l][j] =
          beta1 * nn->opt_m_b[l][j] + (1 - beta1) * nn->biases_grad[l][j];
      nn->opt_v_b[l][j] =
          beta2 * nn->opt_v_b[l][j] +
          (1 - beta2) * nn->biases_grad[l][j] * nn->biases_grad[l][j];

      long double m_hat = nn->opt_m_b[l][j] / (1 - powl(beta1, nn->t));
      long double v_hat = nn->opt_v_b[l][j] / (1 - powl(beta2, nn->t));

      long double momentum =
          beta1 * m_hat + (1 - beta1) * nn->biases_grad[l][j];
      nn->biases[l][j] -= nn->learningRate * momentum / (sqrtl(v_hat) + eps);
    }
  }
}

// Enhanced utility functions
void NN_set_optimizer_params(NN_t *nn, float beta1, float beta2,
                             float epsilon) {
  if (!nn)
    return;
  nn->beta1 = beta1;
  nn->beta2 = beta2;
  nn->epsilon = epsilon;
}

void NN_set_weight_decay(NN_t *nn, float decay_rate) {
  if (!nn)
    return;
  nn->weight_decay = decay_rate;
}

void NN_set_dropout(NN_t *nn, float dropout_rate) {
  if (!nn)
    return;
  nn->dropout_rate = dropout_rate;
}

void NN_set_lr_scheduler(NN_t *nn, LRSchedulerType type, float initial_lr,
                         float final_lr, int total_steps) {
  if (!nn)
    return;
  nn->lr_scheduler = type;
  nn->initial_lr = initial_lr;
  nn->final_lr = final_lr;
  nn->total_steps = total_steps;
  nn->current_step = 0;
  nn->learningRate = initial_lr;
}

void NN_step_lr(NN_t *nn) {
  if (!nn || nn->current_step >= nn->total_steps)
    return;

  nn->current_step++;
  float progress = (float)nn->current_step / nn->total_steps;

  switch (nn->lr_scheduler) {
  case CONSTANT:
    // No change
    break;
  case STEP:
    // Simple linear decay
    nn->learningRate =
        nn->initial_lr + (nn->final_lr - nn->initial_lr) * progress;
    break;
  case EXPONENTIAL:
    // Exponential decay
    nn->learningRate =
        nn->initial_lr * powl(nn->final_lr / nn->initial_lr, progress);
    break;
  case COSINE:
    // Cosine annealing
    nn->learningRate =
        nn->final_lr + (nn->initial_lr - nn->final_lr) *
                           (0.5f * (1.0f + cosf(3.14159f * progress)));
    break;
  case ONE_CYCLE:
    // One cycle policy
    nn->learningRate =
        nn->final_lr +
        (nn->initial_lr - nn->final_lr) *
            (0.5f *
             (1.0f + cosf(3.14159f * (progress - 1.0f / nn->total_steps))));
    break;
  case WARMUP:
    // Linear warmup then decay
    if (progress < 0.1f) {
      nn->learningRate = nn->initial_lr * (progress / 0.1f);
    } else {
      nn->learningRate = nn->initial_lr * powl(0.1f / progress, 0.5f);
    }
    break;
  }
}

float NN_get_current_lr(NN_t *nn) { return nn ? nn->learningRate : 0.0f; }

void NN_set_gradient_clipping(NN_t *nn, int clip_type, float threshold) {
  if (!nn)
    return;
  nn->gradient_clip_type = clip_type;
  nn->gradient_clip_value = threshold;
}

void NN_enable_monitoring(NN_t *nn, int history_capacity) {
  if (!nn)
    return;

  nn->loss_history = malloc(history_capacity * sizeof(float));
  nn->accuracy_history = malloc(history_capacity * sizeof(float));
  nn->history_capacity = history_capacity;
  nn->history_size = 0;
  nn->epoch_count = 0;
}

void NN_log_metrics(NN_t *nn, float loss, float accuracy) {
  if (!nn || !nn->loss_history)
    return;

  if (nn->history_size < nn->history_capacity) {
    nn->loss_history[nn->history_size] = loss;
    nn->accuracy_history[nn->history_size] = accuracy;
    nn->history_size++;
  }
}

void NN_print_model_summary(NN_t *nn) {
  if (!nn)
    return;

  printf("=== Neural Network Summary ===\n");
  printf("Architecture: ");
  for (size_t i = 0; i < nn->numLayers; i++) {
    printf("%zu", nn->layers[i]);
    if (i < nn->numLayers - 1)
      printf(" -> ");
  }
  printf("\n");

  printf("Total parameters: ");
  size_t total_params = 0;
  for (size_t i = 0; i < nn->numLayers - 1; i++) {
    size_t params = nn->layers[i] * nn->layers[i + 1] + nn->layers[i + 1];
    total_params += params;
    printf("Layer %zu: %zu, ", i, params);
  }
  printf("\nTotal: %zu\n", total_params);

  printf("Learning rate: %.6f\n", nn->learningRate);
  printf("Optimizer: %d\n", nn->optimizer);
  printf("Weight init: %d\n", nn->weightInit);
  printf("============================\n");
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

  // weights layout is: weights[i * output_size + j]
  // i = input index, j = output index
  for (size_t j = 0; j < output_size; j++) {
    long double sum = biases[j];
    for (size_t i = 0; i < input_size; i++) {
      sum += weights[i * output_size + j] * inputs[i];
    }
    output[j] = sum;
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
  if (!z_values || !activations) {
    free(z_values);
    free(activations);
    return;
  }

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

  if (nn->grad_hook)
    nn->grad_hook(nn, nn->grad_hook_ctx);

  // Apply optimizer update
  nn_apply_lr_schedule(nn);
  nn_apply_global_grad_clip(nn);
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

long double *NN_backprop_custom_delta_inputgrad(NN_t *nn, long double inputs[],
                                                long double *output_delta) {
  if (!nn || !inputs || !output_delta)
    return NULL;

  size_t L = nn->numLayers;

  long double **z_values =
      (long double **)malloc((L - 1) * sizeof(long double *));
  long double **activations = (long double **)malloc(L * sizeof(long double *));
  if (!z_values || !activations) {
    free(z_values);
    free(activations);
    return NULL;
  }

  // Input layer activations
  activations[0] = (long double *)malloc(nn->layers[0] * sizeof(long double));
  if (!activations[0]) {
    free(z_values);
    free(activations);
    return NULL;
  }
  memcpy(activations[0], inputs, nn->layers[0] * sizeof(long double));

  // Forward: compute z + activations
  for (size_t l = 1; l < L; l++) {
    size_t in_size = nn->layers[l - 1];
    size_t out_size = nn->layers[l];

    z_values[l - 1] = (long double *)malloc(out_size * sizeof(long double));
    activations[l] = (long double *)malloc(out_size * sizeof(long double));
    if (!z_values[l - 1] || !activations[l]) {
      // cleanup partial
      for (size_t k = 0; k < l; k++) {
        free(activations[k]);
        if (k < l - 1)
          free(z_values[k]);
      }
      free(z_values);
      free(activations);
      return NULL;
    }

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

  // Deltas for each layer output (excluding input)
  long double **deltas =
      (long double **)malloc((L - 1) * sizeof(long double *));
  if (!deltas) {
    for (size_t l = 0; l < L; l++)
      free(activations[l]);
    for (size_t l = 0; l < L - 1; l++)
      free(z_values[l]);
    free(z_values);
    free(activations);
    return NULL;
  }

  for (size_t l = 0; l < L - 1; l++) {
    deltas[l] = (long double *)calloc(nn->layers[l + 1], sizeof(long double));
    if (!deltas[l]) {
      for (size_t k = 0; k < l; k++)
        free(deltas[k]);
      free(deltas);
      for (size_t k = 0; k < L; k++)
        free(activations[k]);
      for (size_t k = 0; k < L - 1; k++)
        free(z_values[k]);
      free(z_values);
      free(activations);
      return NULL;
    }
  }

  // Output delta
  size_t last_idx = L - 2;
  memcpy(deltas[last_idx], output_delta,
         nn->layers[L - 1] * sizeof(long double));

  // Backprop hidden layers
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

  // --- compute dL/dInputs BEFORE optimizer step ---
  size_t in0 = nn->layers[0];
  size_t out0 = nn->layers[1];
  long double *grad_input = (long double *)calloc(in0, sizeof(long double));
  if (!grad_input) {
    // still update weights then cleanup
  } else {
    // dX[i] = sum_j W0[i,j] * delta0[j]
    for (size_t i = 0; i < in0; i++) {
      long double s = 0.0L;
      for (size_t j = 0; j < out0; j++) {
        s += nn->weights[0][i * out0 + j] * deltas[0][j];
      }
      grad_input[i] = s;
    }
  }

  // Gradients for weights/biases
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

  nn_apply_lr_schedule(nn);
  nn_apply_global_grad_clip(nn);
  if (nn->grad_hook)
    nn->grad_hook(nn, nn->grad_hook_ctx);

  if (nn->optimizer)
    nn->optimizer(nn);

  // Cleanup
  for (size_t l = 0; l < L - 1; l++) {
    free(z_values[l]);
    free(activations[l + 1]);
    free(deltas[l]);
  }
  free(z_values);
  free(activations[0]);
  free(activations);
  free(deltas);

  return grad_input; // caller frees
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
  if (type >= ACTIVATION_TYPE_COUNT)
    return NULL;
  return ACTIVATION_FUNCTIONS[type];
}

ActivationDerivative get_activation_derivative(ActivationDerivativeType type) {
  if (type >= ACTIVATION_DERIVATIVE_TYPE_COUNT)
    return NULL;
  return ACTIVATION_DERIVATIVES[type];
}

LossFunction get_loss_function(LossFunctionType type) {
  if (type >= LOSS_TYPE_COUNT)
    return NULL;
  return LOSS_FUNCTIONS[type];
}

LossDerivative get_loss_derivative(LossDerivativeType type) {
  if (type >= LOSS_DERIVATIVE_TYPE_COUNT)
    return NULL;
  return LOSS_DERIVATIVES[type];
}

RegularizationFunction get_regularization_function(RegularizationType type) {
  if (type >= REGULARIZATION_TYPE_COUNT)
    return NULL;
  return REGULARIZATION_FUNCTIONS[type];
}

OptimizerFunction get_optimizer_function(OptimizerType type) {
  if (type >= OPTIMIZER_TYPE_COUNT)
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

  fwrite(&nn->baseLearningRate, sizeof(long double), 1, f);
  fwrite(&nn->lr_sched_start, sizeof(long double), 1, f);
  fwrite(&nn->lr_sched_end, sizeof(long double), 1, f);
  fwrite(&nn->lr_sched_steps, sizeof(size_t), 1, f);
  fwrite(&nn->lr_sched_step, sizeof(size_t), 1, f);
  fwrite(&nn->global_grad_clip, sizeof(long double), 1, f);

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

  nn->baseLearningRate = nn->learningRate;
  nn->lr_sched_start = 1.0L;
  nn->lr_sched_end = 1.0L;
  nn->lr_sched_steps = 0;
  nn->lr_sched_step = 0;
  nn->global_grad_clip = 0.0L;
  {
    long double base_lr = 0.0L;
    long double sched_start = 0.0L;
    long double sched_end = 0.0L;
    size_t sched_steps = 0;
    size_t sched_step = 0;
    long double gclip = 0.0L;
    if (fread(&base_lr, sizeof(long double), 1, f) == 1 &&
        fread(&sched_start, sizeof(long double), 1, f) == 1 &&
        fread(&sched_end, sizeof(long double), 1, f) == 1 &&
        fread(&sched_steps, sizeof(size_t), 1, f) == 1 &&
        fread(&sched_step, sizeof(size_t), 1, f) == 1 &&
        fread(&gclip, sizeof(long double), 1, f) == 1) {
      nn->baseLearningRate = base_lr;
      nn->lr_sched_start = sched_start;
      nn->lr_sched_end = sched_end;
      nn->lr_sched_steps = sched_steps;
      nn->lr_sched_step = sched_step;
      nn->global_grad_clip = gclip;
    }
  }

  fclose(f);
  return nn;
}

int NN_save_fp(NN_t *nn, FILE *f) {
  if (!nn || !f)
    return -1;

  uint32_t magic = 0x4E4E3031; // "NN01"
  if (fwrite(&magic, sizeof(uint32_t), 1, f) != 1)
    return -1;

  if (fwrite(&nn->numLayers, sizeof(size_t), 1, f) != 1)
    return -1;
  if (fwrite(nn->layers, sizeof(size_t), nn->numLayers, f) != nn->numLayers)
    return -1;

  if (fwrite(&nn->learningRate, sizeof(long double), 1, f) != 1)
    return -1;
  if (fwrite(&nn->t, sizeof(size_t), 1, f) != 1)
    return -1;

  // Activation enums
  for (size_t i = 0; i < nn->numLayers - 1; i++) {
    ActivationFunctionType a =
        get_activation_function_from_func(nn->activationFunctions[i]);
    ActivationDerivativeType d =
        get_activation_derivative_from_func(nn->activationDerivatives[i]);
    if (fwrite(&a, sizeof(a), 1, f) != 1)
      return -1;
    if (fwrite(&d, sizeof(d), 1, f) != 1)
      return -1;
  }

  LossFunctionType loss = get_loss_function_from_func(nn->loss);
  OptimizerType opt = get_optimizer_from_func(nn->optimizer);
  RegularizationType reg = get_regularization_from_func(nn->regularization);

  if (fwrite(&loss, sizeof(loss), 1, f) != 1)
    return -1;
  if (fwrite(&opt, sizeof(opt), 1, f) != 1)
    return -1;
  if (fwrite(&reg, sizeof(reg), 1, f) != 1)
    return -1;

  // Layer data
  for (size_t l = 0; l < nn->numLayers - 1; l++) {
    size_t wcount = nn->layers[l] * nn->layers[l + 1];
    size_t bcount = nn->layers[l + 1];

    if (fwrite(nn->weights[l], sizeof(long double), wcount, f) != wcount)
      return -1;
    if (fwrite(nn->biases[l], sizeof(long double), bcount, f) != bcount)
      return -1;

    if (fwrite(nn->opt_m_w[l], sizeof(long double), wcount, f) != wcount)
      return -1;
    if (fwrite(nn->opt_v_w[l], sizeof(long double), wcount, f) != wcount)
      return -1;
    if (fwrite(nn->opt_m_b[l], sizeof(long double), bcount, f) != bcount)
      return -1;
    if (fwrite(nn->opt_v_b[l], sizeof(long double), bcount, f) != bcount)
      return -1;
  }

  if (fwrite(&nn->baseLearningRate, sizeof(long double), 1, f) != 1)
    return -1;
  if (fwrite(&nn->lr_sched_start, sizeof(long double), 1, f) != 1)
    return -1;
  if (fwrite(&nn->lr_sched_end, sizeof(long double), 1, f) != 1)
    return -1;
  if (fwrite(&nn->lr_sched_steps, sizeof(size_t), 1, f) != 1)
    return -1;
  if (fwrite(&nn->lr_sched_step, sizeof(size_t), 1, f) != 1)
    return -1;
  if (fwrite(&nn->global_grad_clip, sizeof(long double), 1, f) != 1)
    return -1;

  return 0;
}

NN_t *NN_load_fp(FILE *f) {
  if (!f)
    return NULL;

  uint32_t magic;
  if (fread(&magic, sizeof(uint32_t), 1, f) != 1)
    return NULL;
  if (magic != 0x4E4E3031)
    return NULL;

  NN_t *nn = calloc(1, sizeof(NN_t));
  if (!nn)
    return NULL;

  if (fread(&nn->numLayers, sizeof(size_t), 1, f) != 1) {
    free(nn);
    return NULL;
  }
  if (nn->numLayers == 0 || nn->numLayers > 256) {
    free(nn);
    return NULL;
  }

  nn->layers = malloc(nn->numLayers * sizeof(size_t));
  if (!nn->layers) {
    free(nn);
    return NULL;
  }

  if (fread(nn->layers, sizeof(size_t), nn->numLayers, f) != nn->numLayers) {
    free(nn->layers);
    free(nn);
    return NULL;
  }

  if (fread(&nn->learningRate, sizeof(long double), 1, f) != 1) {
    NN_destroy(nn);
    return NULL;
  }
  if (fread(&nn->t, sizeof(size_t), 1, f) != 1) {
    NN_destroy(nn);
    return NULL;
  }

  nn->activationFunctions =
      malloc((nn->numLayers - 1) * sizeof(ActivationFunction));
  nn->activationDerivatives =
      malloc((nn->numLayers - 1) * sizeof(ActivationDerivative));
  if (!nn->activationFunctions || !nn->activationDerivatives) {
    NN_destroy(nn);
    return NULL;
  }

  for (size_t i = 0; i < nn->numLayers - 1; i++) {
    ActivationFunctionType a;
    ActivationDerivativeType d;
    if (fread(&a, sizeof(a), 1, f) != 1) {
      NN_destroy(nn);
      return NULL;
    }
    if (fread(&d, sizeof(d), 1, f) != 1) {
      NN_destroy(nn);
      return NULL;
    }
    nn->activationFunctions[i] = get_activation_function(a);
    nn->activationDerivatives[i] = get_activation_derivative(d);
    if (!nn->activationFunctions[i] || !nn->activationDerivatives[i]) {
      NN_destroy(nn);
      return NULL;
    }
  }

  LossFunctionType loss;
  OptimizerType opt;
  RegularizationType reg;

  if (fread(&loss, sizeof(loss), 1, f) != 1) {
    NN_destroy(nn);
    return NULL;
  }
  if (fread(&opt, sizeof(opt), 1, f) != 1) {
    NN_destroy(nn);
    return NULL;
  }
  if (fread(&reg, sizeof(reg), 1, f) != 1) {
    NN_destroy(nn);
    return NULL;
  }

  nn->loss = get_loss_function(loss);
  nn->lossDerivative = get_loss_derivative(map_loss_to_derivative(loss));
  nn->optimizer = get_optimizer_function(opt);
  nn->regularization = get_regularization_function(reg);

  size_t L = nn->numLayers - 1;
  nn->weights = malloc(L * sizeof(long double *));
  nn->biases = malloc(L * sizeof(long double *));
  nn->weights_grad = malloc(L * sizeof(long double *));
  nn->biases_grad = malloc(L * sizeof(long double *));
  nn->opt_m_w = malloc(L * sizeof(long double *));
  nn->opt_v_w = malloc(L * sizeof(long double *));
  nn->opt_m_b = malloc(L * sizeof(long double *));
  nn->opt_v_b = malloc(L * sizeof(long double *));
  if (!nn->weights || !nn->biases || !nn->weights_grad || !nn->biases_grad ||
      !nn->opt_m_w || !nn->opt_v_w || !nn->opt_m_b || !nn->opt_v_b) {
    NN_destroy(nn);
    return NULL;
  }

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

    if (!nn->weights[l] || !nn->biases[l] || !nn->weights_grad[l] ||
        !nn->biases_grad[l] || !nn->opt_m_w[l] || !nn->opt_v_w[l] ||
        !nn->opt_m_b[l] || !nn->opt_v_b[l]) {
      NN_destroy(nn);
      return NULL;
    }

    if (fread(nn->weights[l], sizeof(long double), wcount, f) != wcount) {
      NN_destroy(nn);
      return NULL;
    }
    if (fread(nn->biases[l], sizeof(long double), bcount, f) != bcount) {
      NN_destroy(nn);
      return NULL;
    }
    if (fread(nn->opt_m_w[l], sizeof(long double), wcount, f) != wcount) {
      NN_destroy(nn);
      return NULL;
    }
    if (fread(nn->opt_v_w[l], sizeof(long double), wcount, f) != wcount) {
      NN_destroy(nn);
      return NULL;
    }
    if (fread(nn->opt_m_b[l], sizeof(long double), bcount, f) != bcount) {
      NN_destroy(nn);
      return NULL;
    }
    if (fread(nn->opt_v_b[l], sizeof(long double), bcount, f) != bcount) {
      NN_destroy(nn);
      return NULL;
    }
  }

  nn->baseLearningRate = nn->learningRate;
  nn->lr_sched_start = 1.0L;
  nn->lr_sched_end = 1.0L;
  nn->lr_sched_steps = 0;
  nn->lr_sched_step = 0;
  nn->global_grad_clip = 0.0L;
  {
    long double base_lr = 0.0L;
    long double sched_start = 0.0L;
    long double sched_end = 0.0L;
    size_t sched_steps = 0;
    size_t sched_step = 0;
    long double gclip = 0.0L;
    if (fread(&base_lr, sizeof(long double), 1, f) == 1 &&
        fread(&sched_start, sizeof(long double), 1, f) == 1 &&
        fread(&sched_end, sizeof(long double), 1, f) == 1 &&
        fread(&sched_steps, sizeof(size_t), 1, f) == 1 &&
        fread(&sched_step, sizeof(size_t), 1, f) == 1 &&
        fread(&gclip, sizeof(long double), 1, f) == 1) {
      nn->baseLearningRate = base_lr;
      nn->lr_sched_start = sched_start;
      nn->lr_sched_end = sched_end;
      nn->lr_sched_steps = sched_steps;
      nn->lr_sched_step = sched_step;
      nn->global_grad_clip = gclip;
    }
  }

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
