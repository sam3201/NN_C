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
