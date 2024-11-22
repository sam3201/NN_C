#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>

// Forward declaration
typedef struct NN_t NN_t;

// Activation function types
typedef enum {
    SIGMOID,
    TANH,
    RELU,
    LINEAR,
    ACTIVATION_TYPE_COUNT
} ActivationFunctionType;

// Activation Derivative Types
typedef enum {
    SIGMOID_DERIVATIVE,
    TANH_DERIVATIVE,
    RELU_DERIVATIVE,
    LINEAR_DERIVATIVE,
    ACTIVATION_DERIVATIVE_TYPE_COUNT
} ActivationDerivativeType;

// Loss function types
typedef enum {
    MSE,
    MAE,
    HUBER,
    LL,
    CE,
    LOSS_TYPE_COUNT
} LossFunctionType;

// Loss Derivative Types
typedef enum {
    MSE_DERIVATIVE,
    MAE_DERIVATIVE,
    HUBER_DERIVATIVE,
    LL_DERIVATIVE,
    CE_DERIVATIVE,
    LOSS_DERIVATIVE_TYPE_COUNT
} LossDerivativeType;

// Optimizer types
typedef enum {
    SGD,
    RMSPROP,
    ADAGRAD,
    ADAM,
    NAG,
    OPTIMIZER_TYPE_COUNT
} OptimizerType;

// Regularization types
typedef enum {
    L1,
    L2,
    REGULARIZATION_TYPE_COUNT
} RegularizationType;

// Function pointer types
typedef long double (*ActivationFunction)(long double);
typedef long double (*ActivationDerivative)(long double);
typedef long double (*LossFunction)(long double, long double);
typedef long double (*LossDerivative)(long double, long double);
typedef void (*OptimizerFunction)(struct NN_t*);
typedef long double (*RegularizationFunction)(long double);

// Neural Network structure
typedef struct NN_t {
    size_t numLayers;            // Number of layers in the network
    size_t* layers;             // Array storing the size of each layer
    long double** weights;       // Network weights
    long double** biases;        // Network biases
    long double** weights_gradients;     // Weight gradients
    long double** biases_gradients;     // Bias gradients
    long double** weights_m;            // First moment estimates for weights
    long double** weights_v;            // Second moment estimates for weights
    long double** biases_m;             // First moment estimates for biases
    long double** biases_v;             // Second moment estimates for biases
    unsigned int t;                   // Time step counter
    long double learningRate;          // Learning rate
    
    // Function pointers for activation and loss
    ActivationFunction* activationFunctions;
    ActivationDerivative* activationDerivatives;
    LossFunction loss;
    LossDerivative lossDerivative;
    
    // Regularization and optimization
    RegularizationFunction regularization;
    OptimizerFunction optimizer;
    RegularizationType regType;
    OptimizerType optType;
} NN_t;

NN_t* NN_init(size_t* layers,
              ActivationFunctionType* actFuncs,
              ActivationDerivativeType* actDerivs,
              LossFunctionType lossFunc,
              LossDerivativeType lossDeriv,
              long double learningRate,
              RegularizationType reg,
              OptimizerType opt);

NN_t* NN_init_random(size_t num_inputs, size_t num_outputs);

void NN_destroy(NN_t* nn);
long double* NN_forward(NN_t* nn, long double inputs[]);
void NN_backprop(NN_t* nn, long double inputs[], long double y_true, long double y_predicted);
long double NN_loss(NN_t* nn, long double y_true, long double y_predicted);
long double* NN_matmul(long double inputs[], long double weights[], long double biases[], 
                      size_t input_size, size_t output_size);

// Helper Functions for Type Conversion
ActivationFunction get_activation_function(ActivationFunctionType type);
ActivationDerivative get_activation_derivative(ActivationFunctionType type);
const char* activation_to_string(ActivationFunctionType type);
ActivationFunctionType string_to_activation(const char* str);
ActivationDerivativeType string_to_activation_derivative(const char* str);

LossFunction get_loss_function(LossFunctionType type);
LossDerivative get_loss_derivative(LossFunctionType type);
const char* loss_to_string(LossFunctionType type);
LossFunctionType string_to_loss(const char* str);
LossDerivativeType string_to_loss_derivative(const char* str);

OptimizerFunction get_optimizer_function(OptimizerType type);
const char* optimizer_to_string(OptimizerType type);
OptimizerType string_to_optimizer(const char* str);

RegularizationFunction get_regularization_function(RegularizationType type);
const char* regularization_to_string(RegularizationType type);
RegularizationType string_to_regularization(const char* str);

// Save and Load Functions
int NN_save(NN_t* nn, const char* filename);
NN_t* NN_load(const char* filename);

// Genetic Algorithm / NEAT Operations
typedef struct {
    size_t innovation_number;
    size_t from_node;
    size_t to_node;
    long double weight;
    bool enabled;
} ConnectionGene;

typedef struct {
    size_t node_id;
    size_t layer;
    ActivationFunctionType activation;
} NodeGene;

typedef struct {
    size_t species_id;
    long double fitness;
    long double adjusted_fitness;
    size_t num_nodes;
    size_t num_connections;
    NodeGene* nodes;
    ConnectionGene* connections;
} Genome;

// NEAT Operations
void NN_mutate(NN_t* nn, long double mutation_rate);
void NN_crossover(NN_t* parent1, NN_t* parent2, NN_t* child1, NN_t* child2);
void NN_add_node(NN_t* nn);
void NN_add_connection(NN_t* nn);
void NN_toggle_connection(NN_t* nn, size_t connection_id);
long double NN_get_compatibility_distance(NN_t* nn1, NN_t* nn2);
void NN_speciate(NN_t** population, size_t pop_size);

// Transformer-specific Operations
typedef struct {
    size_t num_heads;
    size_t head_dim;
    long double** key_weights;
    long double** query_weights;
    long double** value_weights;
    long double** output_weights;
} MultiHeadAttention;

typedef struct {
    MultiHeadAttention* self_attention;
    MultiHeadAttention* cross_attention;
    NN_t* feedforward;
    long double* layer_norm1;
    long double* layer_norm2;
    long double* layer_norm3;
} TransformerLayer;

// Transformer Operations
void NN_self_attention(NN_t* nn, long double* inputs, size_t seq_length);
void NN_cross_attention(NN_t* nn, long double* encoder_outputs, long double* decoder_inputs);
void NN_position_encoding(NN_t* nn, long double* inputs, size_t seq_length);
void NN_layer_norm(long double* inputs, long double* gamma, long double* beta, size_t dim);

// LLM-specific Operations
typedef struct {
    size_t vocab_size;
    size_t max_seq_length;
    size_t embedding_dim;
    TransformerLayer** encoder_layers;
    TransformerLayer** decoder_layers;
    long double** token_embeddings;
    long double** position_embeddings;
} LLM_Config;

// LLM Operations
void NN_tokenize(NN_t* nn, const char* input, size_t* tokens, size_t* length);
void NN_detokenize(NN_t* nn, size_t* tokens, size_t length, char* output);
void NN_beam_search(NN_t* nn, size_t* input_tokens, size_t beam_width);
void NN_cache_attention(NN_t* nn, size_t layer_idx, long double* key_cache, long double* value_cache);

// Training and Inference Utilities
typedef struct {
    size_t batch_size;
    size_t seq_length;
    size_t num_epochs;
    long double warmup_steps;
    bool use_mixed_precision;
    bool use_gradient_checkpointing;
} TrainingConfig;

// Utility Functions
void NN_save_checkpoint(NN_t* nn, const char* filename);
NN_t* NN_load_checkpoint(const char* filename);
void NN_quantize(NN_t* nn, size_t bits);
void NN_prune(NN_t* nn, long double threshold);
void NN_profile_performance(NN_t* nn, const char* input);

// Memory Management
void NN_free_attention(MultiHeadAttention* attention);
void NN_free_transformer_layer(TransformerLayer* layer);
void NN_free_llm_config(LLM_Config* config);

// Activation Functions
long double sigmoid(long double x);
long double tanh_activation(long double x);
long double relu(long double x);
long double linear(long double x);

// Activation Derivatives
long double sigmoid_derivative(long double x);
long double tanh_derivative(long double x);
long double relu_derivative(long double x);
long double linear_derivative(long double x);

// Loss Functions
long double mse(long double y_true, long double y_pred);
long double mae(long double y_true, long double y_pred);
long double h(long double y_true, long double y_pred);
long double ll(long double y_true, long double y_pred);
long double ce(long double y_true, long double y_pred);

// Loss Derivatives
long double mse_derivative(long double y_true, long double y_pred);
long double mae_derivative(long double y_true, long double y_pred);
long double h_derivative(long double y_true, long double y_pred);
long double ll_derivative(long double y_true, long double y_pred);
long double ce_derivative(long double y_true, long double y_pred);

// Regularization Functions
long double l1(long double weight);
long double l2(long double weight);

// Optimizer Functions
void sgd(NN_t* nn);
void rmsprop(NN_t* nn);
void adagrad(NN_t* nn);
void adam(NN_t* nn);
void nag(NN_t* nn);

// Function Getters
ActivationFunction get_activation_function(ActivationFunctionType type);
ActivationDerivative get_activation_derivative(ActivationFunctionType type);
LossFunction get_loss_function(LossFunctionType type);
LossDerivative get_loss_derivative(LossFunctionType type);
RegularizationFunction get_regularization_function(RegularizationType type);
OptimizerFunction get_optimizer_function(OptimizerType type);

LossFunctionType get_loss_function_type(LossFunction func);
LossDerivativeType get_loss_derivative_type(LossDerivative func);

#endif // NN_H
