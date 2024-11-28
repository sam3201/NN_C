#include "NN.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Lookup tables for functions
static const ActivationFunction ACTIVATION_FUNCTIONS[] = {
    sigmoid,
    tanh_activation,
    relu,
    linear
};

static const ActivationDerivative ACTIVATION_DERIVATIVES[] = {
    sigmoid_derivative,
    tanh_derivative,
    relu_derivative,
    linear_derivative
};

static const LossFunction LOSS_FUNCTIONS[] = {
    mse,
    mae,
    huber,
    ll,
    ce
};

static const LossDerivative LOSS_DERIVATIVES[] = {
    mse_derivative,
    mae_derivative,
    huber_derivative,
    ll_derivative,
    ce_derivative
};

static const OptimizerFunction OPTIMIZER_FUNCTIONS[] = {
    sgd,
    rmsprop,
    adagrad,
    adam,
    nag
};

static const RegularizationFunction REGULARIZATION_FUNCTIONS[] = {
    l1,
    l2
};

NN_t* NN_init(size_t* layers,
              ActivationFunctionType* actFuncs,
              ActivationDerivativeType* actDerivs,
              LossFunctionType lossFunc,
              LossDerivativeType lossDeriv,
              RegularizationType reg,
              OptimizerType opt,
              long double learningRate) {
    
    if (!layers || !actFuncs || !actDerivs) {
        fprintf(stderr, "NN_init: NULL input parameters\n");
        return NULL;
    }
    
    NN_t* nn = (NN_t*)malloc(sizeof(NN_t));
    if (!nn) return NULL;

    nn->numLayers = 0;
    while (layers[nn->numLayers] != 0) {
        nn->numLayers++;
    }
    
    nn->layers = (size_t*)malloc(nn->numLayers * sizeof(size_t));
    if (!nn->layers) {
        free(nn);
        return NULL;
    }
    
    memcpy(nn->layers, layers, nn->numLayers * sizeof(size_t));
    
    // Allocate memory for network components
    nn->weights = (long double**)malloc((nn->numLayers - 1) * sizeof(long double*));
    nn->biases = (long double**)malloc((nn->numLayers - 1) * sizeof(long double*));
    nn->weights_v = (long double**)malloc((nn->numLayers - 1) * sizeof(long double*));
    nn->biases_v = (long double**)malloc((nn->numLayers - 1) * sizeof(long double*));
    
    if (!nn->weights || !nn->biases || !nn->weights_v || !nn->biases_v) {
        NN_destroy(nn);
        return NULL;
    }
    
    // Allocate and initialize activation functions
    nn->activationFunctions = (ActivationFunction*)malloc((nn->numLayers - 1) * sizeof(ActivationFunction));
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
    nn->activationDerivatives = (ActivationDerivative*)malloc((nn->numLayers - 1) * sizeof(ActivationDerivative));
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
        
        nn->weights[i] = (long double*)malloc(current_size * sizeof(long double));
        nn->biases[i] = (long double*)malloc(nn->layers[i + 1] * sizeof(long double));
        nn->weights_v[i] = (long double*)calloc(current_size, sizeof(long double));
        nn->biases_v[i] = (long double*)calloc(nn->layers[i + 1], sizeof(long double));
        
        if (!nn->weights[i] || !nn->biases[i] || !nn->weights_v[i] || !nn->biases_v[i]) {
            fprintf(stderr, "Failed to allocate memory for weights or biases at layer %zu\n", i);
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
            nn->weights[i][j] = ((long double)rand() / RAND_MAX * 2.0L - 1.0L) * scale;
        }
        
        for (size_t j = 0; j < nn->layers[i + 1]; j++) {
            nn->biases[i][j] = 0.0L;  // Initialize biases to zero
        }
    }
    
    return nn;
}

NN_t* NN_init_random(size_t num_inputs, size_t num_outputs) {
    if (num_inputs == 0 || num_outputs == 0) return NULL;
    
    // Create random network architecture (1-2 hidden layers)
    size_t num_hidden_layers = 1 + (rand() % 2); // Random number between 1-2
    size_t* layers = malloc((num_hidden_layers + 3) * sizeof(size_t));  // +2 for input/output, +1 for terminator
    if (!layers) return NULL;
    
    // Set input and output layers
    layers[0] = num_inputs;
    layers[num_hidden_layers + 1] = num_outputs;
    layers[num_hidden_layers + 2] = 0;  // Terminator
    
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
    ActivationFunctionType* act_funcs = malloc((total_layers - 1) * sizeof(ActivationFunctionType));
    ActivationDerivativeType* act_derivs = malloc((total_layers - 1) * sizeof(ActivationDerivativeType));
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
    NN_t* nn = NN_init(layers,
                       act_funcs,
                       act_derivs,
                       loss_func,
                       loss_deriv,
                       reg_type,
                       opt_type,
                       0.001L + ((long double)rand() / RAND_MAX) * 0.099L);
    
    // Clean up
    free(layers);
    free(act_funcs);
    free(act_derivs);
    
    return nn;
}

// Memory cleanup function
void NN_destroy(NN_t* nn) {
    if (!nn) return;
    // Free layers
    free(nn->layers);
    
    // Free weights and biases
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        free(nn->weights[i]);
        free(nn->biases[i]);
        free(nn->weights_v[i]);
        free(nn->biases_v[i]);
    }
    
    // Free weights and biases arrays
    free(nn->weights);
    free(nn->biases);
    free(nn->weights_v);
    free(nn->biases_v);
    
    // Free activation functions
    free(nn->activationFunctions);
    free(nn->activationDerivatives);

    // Free network
    free(nn);
}

// Optimizer Functions
void sgd(NN_t* nn) {
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        size_t current_layer = nn->layers[i];
        size_t next_layer = nn->layers[i + 1];
        
        for (size_t j = 0; j < current_layer * next_layer; j++) {
            nn->weights[i][j] -= nn->learningRate * nn->weights_v[i][j];
        }
        
        for (size_t j = 0; j < next_layer; j++) {
            nn->biases[i][j] -= nn->learningRate * nn->biases_v[i][j];
        }
    }
}

void rmsprop(NN_t* nn) {
    const long double decay_rate = 0.9L;
    const long double epsilon = 1e-8L;
    
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        size_t current_layer = nn->layers[i];
        size_t next_layer = nn->layers[i + 1];
        
        for (size_t j = 0; j < current_layer * next_layer; j++) {
            nn->weights_v[i][j] = decay_rate * nn->weights_v[i][j] + 
                                 (1 - decay_rate) * nn->weights_v[i][j] * nn->weights_v[i][j];
            nn->weights[i][j] -= nn->learningRate * nn->weights_v[i][j] / 
                                (sqrtl(nn->weights_v[i][j]) + epsilon);
        }
        
        for (size_t j = 0; j < next_layer; j++) {
            nn->biases_v[i][j] = decay_rate * nn->biases_v[i][j] + 
                                (1 - decay_rate) * nn->biases_v[i][j] * nn->biases_v[i][j];
            nn->biases[i][j] -= nn->learningRate * nn->biases_v[i][j] / 
                               (sqrtl(nn->biases_v[i][j]) + epsilon);
        }
    }
}

void adagrad(NN_t* nn) {
    const long double epsilon = 1e-8L;
    
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        size_t current_layer = nn->layers[i];
        size_t next_layer = nn->layers[i + 1];
        
        for (size_t j = 0; j < current_layer * next_layer; j++) {
            nn->weights_v[i][j] += nn->weights_v[i][j] * nn->weights_v[i][j];
            nn->weights[i][j] -= nn->learningRate * nn->weights_v[i][j] / 
                                (sqrtl(nn->weights_v[i][j]) + epsilon);
        }
        
        for (size_t j = 0; j < next_layer; j++) {
            nn->biases_v[i][j] += nn->biases_v[i][j] * nn->biases_v[i][j];
            nn->biases[i][j] -= nn->learningRate * nn->biases_v[i][j] / 
                               (sqrtl(nn->biases_v[i][j]) + epsilon);
        }
    }
}

void adam(NN_t* nn) {
    const long double beta1 = 0.9L;
    const long double beta2 = 0.999L;
    const long double epsilon = 1e-8L;
    
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        size_t current_layer = nn->layers[i];
        size_t next_layer = nn->layers[i + 1];
        
        for (size_t j = 0; j < current_layer * next_layer; j++) {
            nn->weights_v[i][j] = beta2 * nn->weights_v[i][j] + 
                                 (1 - beta2) * nn->weights_v[i][j] * nn->weights_v[i][j];
            
            long double m_hat = nn->weights_v[i][j] / (1 - powl(beta1, nn->t));
            long double v_hat = nn->weights_v[i][j] / (1 - powl(beta2, nn->t));
            
            nn->weights[i][j] -= nn->learningRate * m_hat / (sqrtl(v_hat) + epsilon);
        }
        
        for (size_t j = 0; j < next_layer; j++) {
            nn->biases_v[i][j] = beta2 * nn->biases_v[i][j] + 
                                (1 - beta2) * nn->biases_v[i][j] * nn->biases_v[i][j];
            
            long double m_hat = nn->biases_v[i][j] / (1 - powl(beta1, nn->t));
            long double v_hat = nn->biases_v[i][j] / (1 - powl(beta2, nn->t));
            
            nn->biases[i][j] -= nn->learningRate * m_hat / (sqrtl(v_hat) + epsilon);
        }
    }
    nn->t++;
}

void nag(NN_t* nn) {
    const long double momentum = 0.9L;
    
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        size_t current_layer = nn->layers[i];
        size_t next_layer = nn->layers[i + 1];
        
        for (size_t j = 0; j < current_layer * next_layer; j++) {
            long double prev_velocity = nn->weights_v[i][j];
            nn->weights_v[i][j] = momentum * nn->weights_v[i][j] - nn->learningRate * nn->weights_v[i][j];
            nn->weights[i][j] += -momentum * prev_velocity + (1 + momentum) * nn->weights_v[i][j];
        }
        
        for (size_t j = 0; j < next_layer; j++) {
            long double prev_velocity = nn->biases_v[i][j];
            nn->biases_v[i][j] = momentum * nn->biases_v[i][j] - nn->learningRate * nn->biases_v[i][j];
            nn->biases[i][j] += -momentum * prev_velocity + (1 + momentum) * nn->biases_v[i][j];
        }
    }
}

// Matrix Multiplication Function
long double* NN_matmul(long double inputs[], long double weights[], long double biases[], 
                      size_t input_size, size_t output_size) {
    long double* output = (long double*)malloc(output_size * sizeof(long double));
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

// Forward Propagation Functions
long double* NN_forward(NN_t* nn, long double inputs[]) {
    if (!nn || !inputs) {
        fprintf(stderr, "NN_forward: NULL input parameters\n");
        return NULL;
    }
    
    long double* current = inputs;
    long double* next = NULL;
    
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        next = NN_matmul(current, nn->weights[i], nn->biases[i], 
                        nn->layers[i], nn->layers[i + 1]);
        
        // Apply activation function
        for (size_t j = 0; j < nn->layers[i + 1]; j++) {
            next[j] = nn->activationFunctions[i](next[j]);
        }
        
        if (i > 0) free(current);
        current = next;
    }
    
    return current;
}

// Loss calculation function
long double NN_loss(NN_t* nn, long double y_true, long double y_predicted) {
    if (!nn || !nn->loss) return INFINITY;
    
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

// Backward Propagation Functions
void NN_backprop(NN_t* nn, long double inputs[], long double y_true, long double y_predicted) {
    if (!nn || !inputs) {
        fprintf(stderr, "NN_backprop: NULL input parameters\n");
        return;
    }

    // Calculate output layer error
    long double error = NN_loss(nn, y_true, y_predicted);

    // Backpropagate through layers
    for (size_t i = nn->numLayers - 1; i > 0; i--) {
        size_t current_layer = nn->layers[i];
        size_t prev_layer = nn->layers[i - 1];
        
        // Calculate gradients
        for (size_t j = 0; j < current_layer; j++) {
            if (!nn->activationDerivatives[i - 1]) {
                fprintf(stderr, "Activation derivative is NULL for layer %zu\n", i - 1);
                return;
            }
            long double delta = error * nn->activationDerivatives[i - 1](y_predicted);
            
            // Update bias gradients
            nn->biases_v[i - 1][j] = delta;

            // Update weight gradients
            for (size_t k = 0; k < prev_layer; k++) {
                nn->weights_v[i - 1][k * current_layer + j] = delta * inputs[k];
            }
        }
    }

    // Apply optimizer
    if (!nn->optimizer) {
        fprintf(stderr, "Optimizer function is NULL\n");
        return;
    }
    nn->optimizer(nn);
}

// Function Getters
ActivationFunction get_activation_function(ActivationFunctionType type) {
    if (type < 0 || type >= ACTIVATION_TYPE_COUNT) return NULL;
    return ACTIVATION_FUNCTIONS[type];
}

ActivationDerivative get_activation_derivative(ActivationDerivativeType type) {
    if (type < 0 || type >= ACTIVATION_DERIVATIVE_TYPE_COUNT) return NULL;
    return ACTIVATION_DERIVATIVES[type];
}

LossFunction get_loss_function(LossFunctionType type) {
    if (type < 0 || type >= LOSS_TYPE_COUNT) return NULL;
    return LOSS_FUNCTIONS[type];
}

LossDerivative get_loss_derivative(LossDerivativeType type) {
    if (type < 0 || type >= LOSS_DERIVATIVE_TYPE_COUNT) return NULL;
    return LOSS_DERIVATIVES[type];
}

RegularizationFunction get_regularization_function(RegularizationType type) {
    if (type < 0 || type >= REGULARIZATION_TYPE_COUNT) return NULL;
    return REGULARIZATION_FUNCTIONS[type];
}

OptimizerFunction get_optimizer_function(OptimizerType type) {
    if (type < 0 || type >= OPTIMIZER_TYPE_COUNT) return NULL;
    return OPTIMIZER_FUNCTIONS[type];
}

// Function to map activation function to its derivative
ActivationDerivativeType map_activation_to_derivative(ActivationFunctionType actFunc) {
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
const char* activation_to_string(ActivationFunctionType type) {
    switch(type) {
        case SIGMOID: return "sigmoid";
        case TANH: return "tanh";
        case RELU: return "relu";
        case LINEAR: return "linear";
        default: return "unknown";
    }
}

const char* loss_to_string(LossFunctionType type) {
    switch(type) {
        case MSE: return "mse";
        case MAE: return "mae";
        case HUBER: return "huber";
        case LL: return "log_loss";
        case CE: return "cross_entropy";
        default: return "unknown";
    }
}

const char* optimizer_to_string(OptimizerType type) {
    switch(type) {
        case SGD: return "sgd";
        case RMSPROP: return "rmsprop";
        case ADAGRAD: return "adagrad";
        case ADAM: return "adam";
        case NAG: return "nag";
        default: return "unknown";
    }
}

const char* regularization_to_string(RegularizationType type) {
    switch(type) {
        case L1: return "l1";
        case L2: return "l2";
        default: return "unknown";
    }
}

ActivationFunctionType get_activation_function_type(const char* str) {
    if (!str) return -1;
    if (strcmp(str, "sigmoid") == 0) return SIGMOID;
    if (strcmp(str, "tanh") == 0) return TANH;
    if (strcmp(str, "relu") == 0) return RELU;
    if (strcmp(str, "linear") == 0) return LINEAR;
    return -1;
}

LossFunctionType get_loss_function_type(const char* str) {
    if (!str) return -1;
    if (strcmp(str, "mse") == 0) return MSE;
    if (strcmp(str, "mae") == 0) return MAE;
    if (strcmp(str, "huber") == 0) return HUBER;
    if (strcmp(str, "log_loss") == 0) return LL;
    if (strcmp(str, "cross_entropy") == 0) return CE;
    return -1;
}

OptimizerType get_optimizer_type(const char* str) {
    if (!str) return -1;
    if (strcmp(str, "sgd") == 0) return SGD;
    if (strcmp(str, "rmsprop") == 0) return RMSPROP;
    if (strcmp(str, "adagrad") == 0) return ADAGRAD;
    if (strcmp(str, "adam") == 0) return ADAM;
    if (strcmp(str, "nag") == 0) return NAG;
    return -1;
}

RegularizationType get_regularization_type(const char* str) {
    if (!str) return -1;
    if (strcmp(str, "l1") == 0) return L1;
    if (strcmp(str, "l2") == 0) return L2;
    return -1;
}

// Type Getters from Function Pointers
LossFunctionType get_loss_function_from_func(LossFunction func) {
    for (int i = 0; i < LOSS_TYPE_COUNT; i++) {
        if (LOSS_FUNCTIONS[i] == func) return i;
    }
    return -1;
}

OptimizerType get_optimizer_from_func(OptimizerFunction func) {
    for (int i = 0; i < OPTIMIZER_TYPE_COUNT; i++) {
        if (OPTIMIZER_FUNCTIONS[i] == func) return i;
    }
    return -1;
}

RegularizationType get_regularization_from_func(RegularizationFunction func) {
    for (int i = 0; i < REGULARIZATION_TYPE_COUNT; i++) {
        if (REGULARIZATION_FUNCTIONS[i] == func) return i;
    }
    return -1;
}

int NN_save(NN_t* nn, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) return -1;
    
    // Write network structure
    fwrite(&nn->numLayers, sizeof(size_t), 1, file);
    fwrite(nn->layers, sizeof(size_t), nn->numLayers, file);
    
    // Write weights, biases, gradients, and optimizer states
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        size_t weights_size = nn->layers[i] * nn->layers[i + 1];
        
        // Weights and biases
        fwrite(nn->weights[i], sizeof(long double), weights_size, file);
        fwrite(nn->biases[i], sizeof(long double), nn->layers[i + 1], file);
        
        // Gradients
        fwrite(nn->weights_v[i], sizeof(long double), weights_size, file);
        fwrite(nn->biases_v[i], sizeof(long double), nn->layers[i + 1], file);
    }
    
    // Write optimizer state
    fwrite(&nn->t, sizeof(unsigned int), 1, file);
    fwrite(&nn->learningRate, sizeof(long double), 1, file);
    
    // Convert function pointers to types and save as strings
    RegularizationType reg_type = get_regularization_from_func(nn->regularization);
    OptimizerType opt_type = get_optimizer_from_func(nn->optimizer);
    
    const char* reg_str = regularization_to_string(reg_type);
    const char* opt_str = optimizer_to_string(opt_type);
    
    size_t reg_len = strlen(reg_str) + 1;
    size_t opt_len = strlen(opt_str) + 1;
    
    fwrite(&reg_len, sizeof(size_t), 1, file);
    fwrite(&opt_len, sizeof(size_t), 1, file);
    fwrite(reg_str, sizeof(char), reg_len, file);
    fwrite(opt_str, sizeof(char), opt_len, file);
    
    fclose(file);
    return 0;
}

NN_t* NN_load(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    
    NN_t* nn = (NN_t*)malloc(sizeof(NN_t));
    if (!nn) {
        fclose(file);
        return NULL;
    }
    
    // Read network structure
    if (fread(&nn->numLayers, sizeof(size_t), 1, file) != 1) {
        free(nn);
        fclose(file);
        return NULL;
    }
    
    // Allocate and read layers array
    nn->layers = (size_t*)malloc(nn->numLayers * sizeof(size_t));
    if (!nn->layers || fread(nn->layers, sizeof(size_t), nn->numLayers, file) != nn->numLayers) {
        free(nn->layers);
        free(nn);
        fclose(file);
        return NULL;
    }
    
    // Allocate memory for all network components
    nn->weights = (long double**)malloc((nn->numLayers - 1) * sizeof(long double*));
    nn->biases = (long double**)malloc((nn->numLayers - 1) * sizeof(long double*));
    nn->weights_v = (long double**)malloc((nn->numLayers - 1) * sizeof(long double*));
    nn->biases_v = (long double**)malloc((nn->numLayers - 1) * sizeof(long double*));
    
    if (!nn->weights || !nn->biases || !nn->weights_v || !nn->biases_v) {
        NN_destroy(nn);
        fclose(file);
        return NULL;
    }
    
    // Allocate and initialize activation functions
    nn->activationFunctions = (ActivationFunction*)malloc((nn->numLayers - 1) * sizeof(ActivationFunction));
    if (!nn->activationFunctions) {
        NN_destroy(nn);
        fclose(file);
        return NULL;
    }
    
    // Allocate and initialize activation derivatives
    nn->activationDerivatives = (ActivationDerivative*)malloc((nn->numLayers - 1) * sizeof(ActivationDerivative));
    if (!nn->activationDerivatives) {
        NN_destroy(nn);
        fclose(file);
        return NULL;
    }
    
    // Read weights, biases, gradients and optimizer states
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        size_t weights_size = nn->layers[i] * nn->layers[i + 1];
        
        // Allocate memory for each layer
        nn->weights[i] = (long double*)malloc(weights_size * sizeof(long double));
        nn->biases[i] = (long double*)malloc(nn->layers[i + 1] * sizeof(long double));
        nn->weights_v[i] = (long double*)malloc(weights_size * sizeof(long double));
        nn->biases_v[i] = (long double*)malloc(nn->layers[i + 1] * sizeof(long double));
        
        if (!nn->weights[i] || !nn->biases[i] || !nn->weights_v[i] || !nn->biases_v[i]) {
            NN_destroy(nn);
            fclose(file);
            return NULL;
        }
        
        // Read the data
        if (fread(nn->weights[i], sizeof(long double), weights_size, file) != weights_size ||
            fread(nn->biases[i], sizeof(long double), nn->layers[i + 1], file) != nn->layers[i + 1] ||
            fread(nn->weights_v[i], sizeof(long double), weights_size, file) != weights_size ||
            fread(nn->biases_v[i], sizeof(long double), nn->layers[i + 1], file) != nn->layers[i + 1]) {
            NN_destroy(nn);
            fclose(file);
            return NULL;
        }
    }
    
    // Read optimizer state
    if (fread(&nn->t, sizeof(unsigned int), 1, file) != 1 ||
        fread(&nn->learningRate, sizeof(long double), 1, file) != 1) {
        NN_destroy(nn);
        fclose(file);
        return NULL;
    }
    
    // Read string lengths
    size_t reg_len, opt_len;
    if (fread(&reg_len, sizeof(size_t), 1, file) != 1 ||
        fread(&opt_len, sizeof(size_t), 1, file) != 1) {
        NN_destroy(nn);
        fclose(file);
        return NULL;
    }
    
    // Read and convert strings back to function pointers
    char* reg_str = (char*)malloc(reg_len);
    char* opt_str = (char*)malloc(opt_len);
    
    if (!reg_str || !opt_str) {
        free(reg_str);
        free(opt_str);
        NN_destroy(nn);
        fclose(file);
        return NULL;
    }
    
    if (fread(reg_str, sizeof(char), reg_len, file) != reg_len ||
        fread(opt_str, sizeof(char), opt_len, file) != opt_len) {
        free(reg_str);
        free(opt_str);
        NN_destroy(nn);
        fclose(file);
        return NULL;
    }
    
    RegularizationType reg_type = get_regularization_type(reg_str);
    OptimizerType opt_type = get_optimizer_type(opt_str);
    
    nn->regularization = REGULARIZATION_FUNCTIONS[reg_type];
    nn->optimizer = OPTIMIZER_FUNCTIONS[opt_type];
    
    free(reg_str);
    free(opt_str);
    
    fclose(file);
    return nn;
}

// Activation Functions
long double sigmoid(long double x) {
    return 1.0L / (1.0L + expl(-x));
}

long double tanh_activation(long double x) {
    return tanhl(x);
}

long double relu(long double x) {
    return x > 0 ? x : 0;
}

long double linear(long double x) {
    return x;
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

long double relu_derivative(long double x) {
    return x > 0 ? 1.0L : 0.0L;
}

long double linear_derivative(long double x) {
    return 1.0L;
}

// Loss Functions
long double mse(long double y_true, long double y_pred) {
    long double diff = y_true - y_pred;
    return 0.5L * diff * diff;
}

long double mae(long double y_true, long double y_pred) {
    return fabsl(y_true - y_pred);
}

long double huber(long double y_true, long double y_pred) {  // Huber Loss
    const long double delta = 1.0L;
    long double diff = fabsl(y_true - y_pred);
    if (diff <= delta) {
        return 0.5L * diff * diff;
    }
    return delta * diff - 0.5L * delta * delta;
}

long double ll(long double y_true, long double y_pred) {  // Log Loss
    const long double epsilon = 1e-15L;
    y_pred = fmaxl(fminl(y_pred, 1.0L - epsilon), epsilon);
    return -(y_true * logl(y_pred) + (1.0L - y_true) * logl(1.0L - y_pred));
}

long double ce(long double y_true, long double y_pred) {  // Cross Entropy
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

long double huber_derivative(long double y_true, long double y_pred) {  // Huber Loss Derivative
    const long double delta = 1.0L;
    long double diff = y_pred - y_true;
    if (fabsl(diff) <= delta) {
        return diff;
    }
    return delta * (diff > 0 ? 1.0L : -1.0L);
}

long double ll_derivative(long double y_true, long double y_pred) {  // Log Loss Derivative
    const long double epsilon = 1e-15L;
    y_pred = fmaxl(fminl(y_pred, 1.0L - epsilon), epsilon);
    return (y_pred - y_true) / (y_pred * (1.0L - y_pred));
}

long double ce_derivative(long double y_true, long double y_pred) {  // Cross Entropy Derivative
    const long double epsilon = 1e-15L;
    y_pred = fmaxl(fminl(y_pred, 1.0L - epsilon), epsilon);
    return -y_true / y_pred;
}

// Regularization Functions
long double l1(long double weight) {
    return fabsl(weight);
}

long double l2(long double weight) {
    return 0.5L * weight * weight;
}
