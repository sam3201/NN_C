#include "NN.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

ActivationFunction activation_functions[ACTIVATION_FUNCTION_COUNT] = {
    sigmoid,
    relu,
    tanh_activation,
    //argmax,  // argmax is not a standard activation function
    softmax
};

ActivationDerivative activation_derivatives[ACTIVATION_DERIVATIVE_COUNT] = {
    sigmoid_derivative,
    relu_derivative,
    tanh_derivative,
    //argmax_derivative,  // argmax_derivative is not a standard activation derivative
    softmax_derivative
};

LossFunction loss_functions[LOSS_FUNCTION_COUNT] = {
    mse,
    ce,
    mae,
};

LossDerivative loss_derivatives[LOSS_DERIVATIVE_COUNT] = {
    mse_derivative,
    ce_derivative,
    mae_derivative
};

char *activation_function_names[ACTIVATION_FUNCTION_COUNT] = {
    "SIGMOID",
    "RELU",
    "TANH",
    //"ARGMAX",
    "SOFTMAX"
};

char *activation_derivative_names[ACTIVATION_DERIVATIVE_COUNT] = {
  "SIGMOID_DERIVATIVE",
    "RELU_DERIVATIVE",
    "TANH_DERIVATIVE",
    //"ARGMAX_DERIVATIVE",
    "SOFTMAX_DERIVATIVE"
};

char *loss_function_names[LOSS_FUNCTION_COUNT] = {
    "MSE",
    "CE",  
    "MAE"
};

char *loss_derivative_names[LOSS_DERIVATIVE_COUNT] = {
    "MSE_DERIVATIVE",
    "CE_DERIVATIVE",
    "MAE_DERIVATIVE"
};

char* activation_function_type_to_string(ActivationFunctionType code) {
  char *activationFunction = (char *)malloc(64 * sizeof(char)); 

  switch (code) {
    case SIGMOID: return "SIGMOID";
    case RELU: return "RELU";
    case TANH: return "TANH"; 
    //case ARGMAX: return "ARGMAX";
    case SOFTMAX: return "SOFTMAX"; 
    default: return "UNKNOWN";
  }

  return activationFunction;
}

char *activation_derivative_type_to_string(ActivationDerivativeType deriv) {
  char *activationDerivative = (char *)malloc(64 * sizeof(char));

  switch (deriv) {
    case SIGMOID: return "SIGMOID_DERIVATIVE";
    case RELU: return "RELU_DERIVATIVE";
    case TANH: return "TANH_DERIVATIVE";
    //case ARGMAX: return "ARGMAX_DERIVATIVE";
    case SOFTMAX: return "SOFTMAX_DERIVATIVE";

    default: return "unknown";
  }
}

char* loss_function_type_to_string(LossFunctionType func) {
  char *lossFunction = (char *)malloc(64 * sizeof(char));
  switch (func) {
    case MSE: return "MSE";
    case CE: return "CE";
    case MAE: return "MAE";
    default: return "unknown";
  }
}

char* loss_derivative_type_to_string(LossDerivativeType deriv) {
  char *lossDerivative = (char *)malloc(64 * sizeof(char));
  switch (deriv) {
    case MSE: return "MSE_DERIVATIVE";
    case CE: return "CE_DERIVATIVE";
    case MAE: return "MAE_DERIVATIVE";
    default: return "unknown";
  }
}

ActivationFunctionType string_to_activation_function_type(char *func) {
  if (strcmp(func, "SIGMOID") == 0) {
    return SIGMOID;
  } else if (strcmp(func, "RELU") == 0) {
    return RELU;
  } else if (strcmp(func, "TANH") == 0) {
    return TANH;
  } else if (strcmp(func, "SOFTMAX") == 0) {
    return SOFTMAX;
  } else {
    printf("Invalid activation function. Supported functions are SIGMOID, RELU, TANH, and SOFTMAX.\n");
    exit(1);
  }
}

ActivationDerivativeType string_to_activation_derivative_type(char *deriv) {
  if (strcmp(deriv, "SIGMOID") == 0) {
    return SIGMOID_DERIVATIVE;
  } else if (strcmp(deriv, "RELU") == 0) {
    return RELU_DERIVATIVE;
  } else if (strcmp(deriv, "TANH") == 0) {
    return TANH_DERIVATIVE;
  } else if (strcmp(deriv, "SOFTMAX") == 0) {
    return SOFTMAX_DERIVATIVE;
  } else {
    printf("Invalid activation function. Supported functions are SIGMOID, RELU, TANH, and SOFTMAX.\n");
    exit(1);
  }
}

LossFunctionType string_to_loss_function_type(char *func) {
  if (strcmp(func, "MSE") == 0) {
    return MSE;
  } else if (strcmp(func, "CE") == 0) {
    return CE;
  } else if (strcmp(func, "MAE") == 0) {
    return MAE;
  } else {  
    printf("Invalid loss function: %s. Supported functions are MSE and CE.\n", func);
    exit(1);
  }
}

LossDerivativeType string_to_loss_derivative_type(char *deriv) {
  if (strcmp(deriv, "MSE") == 0) {
    return MSE_DERIVATIVE;
  } else if (strcmp(deriv, "CE") == 0) {
    return CE_DERIVATIVE;
  } else if (strcmp(deriv, "MAE") == 0) {
    return MAE_DERIVATIVE;
  } else {
    printf("Invalid loss derivative. Supported functions are MSE and CE.\n");
    exit(1);
  }
}

ActivationFunctionType activation_function_to_enum(ActivationFunction func) {
    if (func == sigmoid) return SIGMOID;
    if (func == relu) return RELU;
    if (func == tanh_activation) return TANH;
    if (func == softmax) return SOFTMAX;
    return SIGMOID; // Default to sigmoid for safety
}

ActivationDerivativeType activation_derivative_to_enum(ActivationDerivative func) {
    if (func == sigmoid_derivative) return SIGMOID_DERIVATIVE;
    if (func == relu_derivative) return RELU_DERIVATIVE;
    if (func == tanh_derivative) return TANH_DERIVATIVE;
    if (func == softmax_derivative) return SOFTMAX_DERIVATIVE;
    return SIGMOID_DERIVATIVE; // Default to sigmoid derivative for safety
}

ActivationFunctionType activation_function_to_enum2(ActivationFunction activation_function) {
    if (activation_function == sigmoid) {
        return SIGMOID;
    } else if (activation_function == relu) {
        return RELU;
    } else if (activation_function == tanh_activation) {
        return TANH;
    }
  
    return ACTIVATION_FUNCTION_COUNT; 
}

ActivationDerivativeType activation_derivative_to_enum2(ActivationDerivative activation_derivative) {
    if (activation_derivative == sigmoid_derivative) {
        return SIGMOID_DERIVATIVE;
    } else if (activation_derivative == relu_derivative) {
        return RELU_DERIVATIVE;
    } else if (activation_derivative == tanh_derivative) {
        return TANH_DERIVATIVE;
    }
    
    return ACTIVATION_DERIVATIVE_COUNT;
}

LossFunctionType loss_function_to_enum(LossFunction loss_function) {
    if (loss_function == mse) {
        return MSE;
    } else if (loss_function == ce) {
        return CE;
    } else if (loss_function == mae) {
        return MAE;
    }
    
    return LOSS_FUNCTION_COUNT; 
}

LossDerivativeType loss_derivative_to_enum(LossDerivative loss_derivative) {
    if (loss_derivative == mse_derivative) {
        return MSE_DERIVATIVE;
    } else if (loss_derivative == ce_derivative) {
        return CE_DERIVATIVE;
    } else if (loss_derivative == mae_derivative) {
        return MAE_DERIVATIVE;
    }
    
    return LOSS_DERIVATIVE_COUNT; // Default to MSE derivative on error
}

ActivationFunction get_activation_function(ActivationFunctionType type) {
    if (type >= 0 && type < ACTIVATION_FUNCTION_COUNT) {
        return activation_functions[type];
  }
    return NULL; // Error: invalid type
}

ActivationDerivative get_activation_derivative(ActivationDerivativeType type) {
    if (type >= 0 && type < ACTIVATION_DERIVATIVE_COUNT) {
      return activation_derivatives[type];
  }
    return NULL; // Error: invalid type
}

LossFunction get_loss_function(LossFunctionType type) {
    if (type >= 0 && type < LOSS_FUNCTION_COUNT) {
    return loss_functions[type];
  }
    return NULL; // Error: invalid type
}

LossDerivative get_loss_derivative(LossDerivativeType type) {
    if (type >= 0 && type < LOSS_DERIVATIVE_COUNT) {
    return loss_derivatives[type];
  }
    return NULL; // Error: invalid type
}

char* activation_function_to_string(ActivationFunctionType type) {
    if (type >= 0 && type < ACTIVATION_FUNCTION_COUNT) {
    return activation_function_names[type];
  }
    return "UNKNOWN";
}
  

char* activation_derivative_to_string(ActivationDerivativeType type) {
  if (type >= 0 && type < ACTIVATION_DERIVATIVE_COUNT) {
    return activation_derivative_names[type];
  }
    return "UNKNOWN";
}

char* loss_function_to_string(LossFunctionType type) {
    if (type >= 0 && type < LOSS_FUNCTION_COUNT) {
    return loss_function_names[type];
  }
    return "UNKNOWN";
}

NN_t *NN_init_random(unsigned int num_inputs, unsigned int num_outputs) {
    if (num_inputs == 0 || num_outputs == 0) {
        fprintf(stderr, "Invalid input/output dimensions: inputs=%u, outputs=%u\n", num_inputs, num_outputs);
        return NULL;
    }

    printf("Initializing neural network with %u inputs and %u outputs...\n", num_inputs, num_outputs);

    unsigned int num_layers = 2 + (rand() % 9); // Random number of layers between 2 and 10 (including input/output)
    printf("Total number of layers: %u\n", num_layers);

    // Allocate space for layers array including zero terminator
    size_t *layers = calloc(num_layers + 1, sizeof(size_t));
    if (!layers) {
        fprintf(stderr, "Failed to allocate memory for layers array\n");
        return NULL;
    }

    // Set input and output layers
    layers[0] = num_inputs;
    layers[num_layers - 1] = num_outputs;
    layers[num_layers] = 0;  // Zero terminator

    // Set hidden layer sizes
    for (unsigned int i = 1; i < num_layers - 1; i++) {
        layers[i] = 1 + (rand() % 10); // Random layer sizes between 1 and 10
        printf("Layer %u size: %zu\n", i, layers[i]);
    }

    // Allocate activation function arrays
    ActivationFunctionType *activationFunctions = malloc(num_layers * sizeof(ActivationFunctionType));
    ActivationDerivativeType *activationDerivatives = malloc(num_layers * sizeof(ActivationDerivativeType));
    if (!activationFunctions || !activationDerivatives) {
        fprintf(stderr, "Failed to allocate memory for activation functions\n");
        free(layers);
        if (activationFunctions) free(activationFunctions);
        if (activationDerivatives) free(activationDerivatives);
        return NULL;
    }

    // Set activation functions for all layers
    for (unsigned int i = 0; i < num_layers; i++) {
        if (i == num_layers - 1) {
            // Use sigmoid for output layer
            activationFunctions[i] = SIGMOID;
            activationDerivatives[i] = SIGMOID_DERIVATIVE;
        } else {
            // Random activation for hidden layers
            activationFunctions[i] = (ActivationFunctionType)(rand() % (ACTIVATION_FUNCTION_COUNT - 1)); // Exclude ARGMAX
            activationDerivatives[i] = (ActivationDerivativeType)(rand() % (ACTIVATION_DERIVATIVE_COUNT - 1)); // Exclude ARGMAX
        }
        printf("Layer %u activation function: %s, derivative: %s\n", 
               i, 
               activation_function_type_to_string(activationFunctions[i]),
               activation_derivative_type_to_string(activationDerivatives[i]));
    }

    LossFunctionType lossFunction = (LossFunctionType)rand() % LOSS_FUNCTION_COUNT;
    LossDerivativeType lossDerivative = (LossDerivativeType)rand() % LOSS_FUNCTION_COUNT;
    printf("Loss function: %s\n", loss_function_type_to_string(lossFunction));

    // Random learning rate between 0. and 1
    long double learningRate = 0. + ((long double)rand() / RAND_MAX) * 1;
    printf("Learning rate: %Lf\n", learningRate);

    // Initialize the neural network
    NN_t *nn = NN_init(layers, activationFunctions, activationDerivatives, lossFunction, lossDerivative, learningRate);

    // Clean up
    free(layers);
    free(activationFunctions);
    free(activationDerivatives);

    if (!nn) {
        fprintf(stderr, "Failed to initialize neural network\n");
        return NULL;
    }

    printf("Neural network initialized successfully.\n");
    return nn;
}

NN_t *NN_init(size_t layers[],
              ActivationFunctionType activationFunctions[], ActivationDerivativeType activationDerivatives[],
              LossFunctionType lossFunction, LossDerivativeType lossDerivative, long double learningRate) {

    // Validate input parameters
    if (!layers || !activationFunctions || !activationDerivatives) {
        fprintf(stderr, "Invalid input parameters in NN_init\n");
        return NULL;
    }

    NN_t *nn = (NN_t*)calloc(1, sizeof(NN_t));  // Use calloc to zero-initialize
    if (!nn) {
        fprintf(stderr, "Failed to allocate memory for neural network\n");
        return NULL;
    }

    // Count layers until we hit the zero terminator
    nn->numLayers = 0;
    while (layers[nn->numLayers] != 0) {
        nn->numLayers++;
    }
    
    if (nn->numLayers < 2) {
        fprintf(stderr, "Neural network must have at least 2 layers\n");
        free(nn);
        return NULL;
    }
    
    printf("Initializing neural network with %zu layers\n", nn->numLayers);
    
    // Allocate and copy layer sizes
    nn->layers = (size_t*)malloc(nn->numLayers * sizeof(size_t));
    if (!nn->layers) {
        fprintf(stderr, "Failed to allocate memory for layers\n");
        free(nn);
        return NULL;
    }
    memcpy(nn->layers, layers, nn->numLayers * sizeof(size_t));

    // Validate layer sizes
    for (size_t i = 0; i < nn->numLayers; i++) {
        if (nn->layers[i] == 0) {
            fprintf(stderr, "Invalid layer size 0 at position %zu\n", i);
            free(nn->layers);
            free(nn);
            return NULL;
        }
        printf("Layer %zu size: %zu\n", i, nn->layers[i]);
    }

    // Allocate and set activation functions
    nn->activationFunctions = malloc(nn->numLayers * sizeof(ActivationFunction));
    nn->activationDerivatives = malloc(nn->numLayers * sizeof(ActivationDerivative));
    if (!nn->activationFunctions || !nn->activationDerivatives) {
        fprintf(stderr, "Failed to allocate memory for activation functions\n");
        free(nn->layers);
        if (nn->activationFunctions) free(nn->activationFunctions);
        if (nn->activationDerivatives) free(nn->activationDerivatives);
        free(nn);
        return NULL;
    }

    // Set activation functions for all layers
    for (size_t i = 0; i < nn->numLayers; i++) {
        if (activationFunctions[i] >= ACTIVATION_FUNCTION_COUNT || activationFunctions[i] < 0) {
            fprintf(stderr, "Invalid activation function type %d at layer %zu\n", activationFunctions[i], i);
            free(nn->activationFunctions);
            free(nn->activationDerivatives);
            free(nn->layers);
            free(nn);
            return NULL;
        }
        
        if (activationDerivatives[i] >= ACTIVATION_DERIVATIVE_COUNT || activationDerivatives[i] < 0) {
            fprintf(stderr, "Invalid activation derivative type %d at layer %zu\n", activationDerivatives[i], i);
            free(nn->activationFunctions);
            free(nn->activationDerivatives);
            free(nn->layers);
            free(nn);
            return NULL;
        }
        
        nn->activationFunctions[i] = get_activation_function(activationFunctions[i]);
        nn->activationDerivatives[i] = get_activation_derivative(activationDerivatives[i]);
        printf("Layer %zu activation: %s\n", i, activation_function_type_to_string(activationFunctions[i]));
    }

    // Set loss functions
    if (lossFunction >= LOSS_FUNCTION_COUNT || lossFunction < 0) {
        fprintf(stderr, "Invalid loss function type %d\n", lossFunction);
        free(nn->activationFunctions);
        free(nn->activationDerivatives);
        free(nn->layers);
        free(nn);
        return NULL;
    }
    
    if (lossDerivative >= LOSS_DERIVATIVE_COUNT || lossDerivative < 0) {
        fprintf(stderr, "Invalid loss derivative type %d\n", lossDerivative);
        free(nn->activationFunctions);
        free(nn->activationDerivatives);
        free(nn->layers);
        free(nn);
        return NULL;
    }
    
    nn->lossFunction = get_loss_function(lossFunction);
    nn->lossDerivative = get_loss_derivative(lossDerivative);

    // Allocate and initialize weights and biases
    nn->weights = (long double**)calloc(nn->numLayers - 1, sizeof(long double*));
    nn->biases = (long double**)calloc(nn->numLayers - 1, sizeof(long double*));
    if (!nn->weights || !nn->biases) {
        fprintf(stderr, "Failed to allocate memory for weights/biases\n");
        free(nn->activationFunctions);
        free(nn->activationDerivatives);
        free(nn->layers);
        if (nn->weights) free(nn->weights);
        if (nn->biases) free(nn->biases);
        free(nn);
        return NULL;
    }

    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        size_t num_weights = nn->layers[i + 1] * nn->layers[i];
        nn->weights[i] = (long double*)malloc(num_weights * sizeof(long double));
        nn->biases[i] = (long double*)malloc(nn->layers[i + 1] * sizeof(long double));
        
        if (!nn->weights[i] || !nn->biases[i]) {
            fprintf(stderr, "Failed to allocate memory for layer %zu weights/biases\n", i);
            for (size_t j = 0; j < i; j++) {
                free(nn->weights[j]);
                free(nn->biases[j]);
            }
            if (nn->weights[i]) free(nn->weights[i]);
            if (nn->biases[i]) free(nn->biases[i]);
            free(nn->weights);
            free(nn->biases);
            free(nn->activationFunctions);
            free(nn->activationDerivatives);
            free(nn->layers);
            free(nn);
            return NULL;
        }

        // Initialize weights using Xavier initialization
        long double scale = sqrt(2.0 / (nn->layers[i] + nn->layers[i + 1]));
        for (size_t j = 0; j < num_weights; j++) {
            nn->weights[i][j] = ((long double)rand() / RAND_MAX * 2 - 1) * scale;
        }

        // Initialize biases to small random values
        for (size_t j = 0; j < nn->layers[i + 1]; j++) {
            nn->biases[i][j] = ((long double)rand() / RAND_MAX * 0.2 - 0.1);
        }
        
        printf("Layer %zu: Initialized %zu weights and %zu biases\n", 
               i, num_weights, nn->layers[i + 1]);
    }

    // Set learning rate
    nn->learningRate = learningRate;
    printf("Learning rate set to %Lf\n", learningRate);
    
    return nn;
}

void NN_destroy(NN_t *nn) {
    if (nn == NULL) return;

    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        free(nn->weights[i]);
        free(nn->biases[i]);
    }

    free(nn->weights);
    free(nn->biases);
    
    free(nn->activationFunctions);
    free(nn->activationDerivatives);

    free(nn);
}

void NN_add_layer(NN_t *nn, size_t layerSize, ActivationFunctionType activationFunctions[], ActivationDerivativeType activationDerivatives[]) {
  size_t prevNumLayers = nn->numLayers - layerSize;
  nn->layers = (size_t *)realloc(nn->layers, (nn->numLayers + 1) * sizeof(size_t));
  nn->layers[nn->numLayers] = layerSize;
  nn->numLayers+=layerSize;

  nn->activationFunctions = (ActivationFunction *)realloc(nn->activationFunctions, (nn->numLayers + layerSize) * sizeof(ActivationFunction));
  nn->activationDerivatives = (ActivationDerivative *)realloc(nn->activationDerivatives, (nn->numLayers + layerSize) * sizeof(ActivationDerivative));

  for (size_t neuron = prevNumLayers; neuron < nn->numLayers; neuron++) {
    switch(activationFunctions[neuron]) {
      case SIGMOID:
        nn->activationFunctions[neuron] = sigmoid;
        nn->activationDerivatives[neuron] = sigmoid_derivative;
        break;
      case RELU:
        nn->activationFunctions[neuron] = relu;
        nn->activationDerivatives[neuron] = relu_derivative;
        break;
      case TANH:
        nn->activationFunctions[neuron] = tanh_activation;
        nn->activationDerivatives[neuron] = tanh_derivative;
        break;
      case ARGMAX:
        printf("Argmax not implemented yet!\n");
        exit(1);
      //nn->activationFunctions[neuron] = argmax;
      //nn->activationDerivatives[neuron] = argmax_derivative;
      case SOFTMAX:
        printf("Softmax not implemented yet!\n");
        exit(1);
      default:
        printf("Invalid activation function!\n");
        exit(1);
    }
  }
  nn->weights = (long double **)realloc(nn->weights, (nn->numLayers) * sizeof(long double *));
  nn->biases = (long double **)realloc(nn->biases, (nn->numLayers) * sizeof(long double *));
  for (size_t layer = prevNumLayers; layer < nn->numLayers; layer++) {
    nn->weights[layer] = (long double *)malloc(nn->layers[layer] * nn->layers[layer + 1] * sizeof(long double));
    nn->biases[layer] = (long double *)malloc(nn->layers[layer + 1] * sizeof(long double));

    for (size_t neuron = 0; neuron < nn->numLayers; neuron++) {
      nn->weights[layer][neuron] = (long double)rand() / RAND_MAX;
      nn->biases[layer][neuron] = (long double)rand() / RAND_MAX;    
    }
  }
}

long double *NN_matmul(long double inputs[], long double weights[], long double biases[], size_t input_size, size_t output_size) {
    long double *results = (long double *)malloc(output_size * sizeof(long double));

    for (size_t j = 0; j < output_size; j++) {
        results[j] = biases[j];  // Start with the bias
        for (size_t i = 0; i < input_size; i++) {
            results[j] += inputs[i] * weights[i + j * input_size];
        }
    }

    return results;
}

long double *NN_forward(NN_t *nn, long double inputs[]) {
    long double *outputs = (long double *)malloc(sizeof(long double) * nn->layers[nn->numLayers - 1]);
    if (!outputs) {
        fprintf(stderr, "Failed to allocate memory for outputs\n");
        return NULL;
    }

    long double *current_inputs = inputs;
    long double *layer_outputs = NULL;

    for (size_t layer = 0; layer < nn->numLayers - 1; layer++) {
        // Compute matrix multiplication for this layer
        layer_outputs = NN_matmul(current_inputs, 
                               nn->weights[layer], 
                               nn->biases[layer],
                               nn->layers[layer],
                               nn->layers[layer + 1]);

        // Apply activation function if it exists
        if (nn->activationFunctions[layer + 1]) {
            for (size_t j = 0; j < nn->layers[layer + 1]; j++) {
                layer_outputs[j] = nn->activationFunctions[layer + 1](layer_outputs[j]);
            }
        }

        if (layer > 0) {
            free(current_inputs);
        }
        current_inputs = layer_outputs;
    }

    memcpy(outputs, layer_outputs, sizeof(long double) * nn->layers[nn->numLayers - 1]);
    free(layer_outputs);

    return outputs;
}

long double NN_loss(NN_t *nn, long double y_true, long double y_predicted) {
  return nn->lossFunction(y_true, y_predicted);
}

long double NN_loss_derivative(NN_t *nn, long double y_true, long double y_predicted) {
  return nn->lossDerivative(y_true, y_predicted);
}

void NN_backprop(NN_t *nn, long double inputs[], long double y_true, long double y_pred) {
    // Calculate the error using the loss function
    long double error = nn->lossFunction(y_true, y_pred); 

    // Backpropagation through the network
    for (int layer = nn->numLayers - 1; layer >= 0; layer--) {
        for (int neuron = 0; neuron < nn->layers[layer]; neuron++) {
            // Calculate the derivative of the loss function
            long double derivative = nn->activationDerivatives[layer](nn->biases[layer][neuron]);

            // Calculate the gradient
            long double gradient = error * derivative;

             // Update weights and biases
           for (int prev_neuron = 0; prev_neuron < nn->layers[layer - 1]; prev_neuron++) {
             nn->weights[layer - 1][neuron] += nn->learningRate * gradient * inputs[prev_neuron]; // Update weights
           }
            
            nn->biases[layer][neuron] += nn->learningRate * gradient; // Apply learning rate to bias update
        }
    }
}

NN_t *NN_crossover(NN_t *parent1, NN_t *parent2) {
    if (!parent1 || !parent2 || parent1->numLayers != parent2->numLayers) {
        fprintf(stderr, "Parent networks are incompatible for crossover.\n");
        return NULL;
    }

    // Create arrays for activation functions and derivatives
    ActivationFunctionType *activationFuncs = malloc((parent1->numLayers - 1) * sizeof(ActivationFunctionType));
    ActivationDerivativeType *activationDerivs = malloc((parent1->numLayers - 1) * sizeof(ActivationDerivativeType));
    
    if (!activationFuncs || !activationDerivs) {
        fprintf(stderr, "Failed to allocate memory for activation functions\n");
        free(activationFuncs);
        free(activationDerivs);
        return NULL;
    }

    // Get function types from parent1's function pointers
    for (size_t i = 0; i < parent1->numLayers - 1; i++) {
        // Convert function pointers to their corresponding types
      activationFuncs[i] = activation_function_to_enum(parent1->activationFunctions[i]); 
      activationDerivs[i] = activation_derivative_to_enum(parent1->activationDerivatives[i]); 
    }

    // Create Loss functino and derivatives
     LossFunctionType lossType = loss_function_to_enum(parent1->lossFunction);
     LossDerivativeType lossDeriv = loss_derivative_to_enum(parent1->lossDerivative);
    // Add other loss functions here if needed

    // Create child network with parent1's configuration
    NN_t *child = NN_init(parent1->layers, 
                         activationFuncs,
                         activationDerivs,
                         lossType,
                         lossDeriv,
                         parent1->learningRate);

    // Free temporary arrays
    free(activationFuncs);
    free(activationDerivs);

    if (!child) {
        fprintf(stderr, "Failed to create child network\n");
        return NULL;
    }

    // Perform crossover of weights and biases
    for (size_t layer = 0; layer < child->numLayers - 1; layer++) {
        for (size_t i = 0; i < child->layers[layer + 1]; i++) {
            // Combine weights
            for (size_t j = 0; j < child->layers[layer]; j++) {
                size_t idx = i * child->layers[layer] + j;
                // Average the weights from both parents
                child->weights[layer][idx] = (parent1->weights[layer][idx] + parent2->weights[layer][idx]) / 2.0L;
            }
            // Average the biases from both parents
            child->biases[layer][i] = (parent1->biases[layer][i] + parent2->biases[layer][i]) / 2.0L;
        }
    }

    return child;
}

NN_t *NN_copy(NN_t *nn) {
  if (!nn) {
    fprintf(stderr, "Cannot copy a NULL network.\n");
    return NULL;
  }

  ActivationFunctionType *activationFuncs = malloc((nn->numLayers - 1) * sizeof(ActivationFunctionType));
  ActivationDerivativeType *activationDerivs = malloc((nn->numLayers - 1) * sizeof(ActivationDerivativeType));
  
  if (!activationFuncs || !activationDerivs) {
    fprintf(stderr, "Failed to allocate memory for activation functions\n");
    free(activationFuncs);
    free(activationDerivs);
    return NULL;
  }

  for (size_t i = 0; i < nn->numLayers - 1; i++) {
    activationFuncs[i] = activation_function_to_enum(nn->activationFunctions[i]); 
    activationDerivs[i] = activation_derivative_to_enum(nn->activationDerivatives[i]); 
  }
  LossFunctionType lossType = loss_function_to_enum(nn->lossFunction);
  LossDerivativeType lossDeriv = loss_derivative_to_enum(nn->lossDerivative);

  NN_t *child = NN_init(nn->layers, 
                         activationFuncs,
                         activationDerivs,
                         lossType,
                         lossDeriv,
                         nn->learningRate);

  free(activationFuncs);
  free(activationDerivs);

  if (!child) {
    fprintf(stderr, "Failed to create child network\n");
    return NULL;
  }

  for (size_t layer = 0; layer < child->numLayers - 1; layer++) {
        for (size_t i = 0; i < child->layers[layer + 1]; i++) {
            for (size_t j = 0; j < child->layers[layer]; j++) {
                size_t idx = i * child->layers[layer] + j;
                child->weights[layer][idx] = nn->weights[layer][idx];
            }
            child->biases[layer][i] = nn->biases[layer][i];
        }
    }

  return child;
}

void NN_mutate(NN_t *nn, long double mutationRate, long double mutationStrength) {
    if (!nn) {
        fprintf(stderr, "Cannot mutate a NULL network.\n");
        return;
    }

    for (size_t layer = 0; layer < nn->numLayers - 1; layer++) {
        for (size_t i = 0; i < nn->layers[layer + 1]; i++) {
            // Mutate weights
            for (size_t j = 0; j < nn->layers[layer]; j++) {
                if ((long double)rand() / RAND_MAX < mutationRate) {
                    nn->weights[layer][i * nn->layers[layer] + j] += ((long double)rand() / RAND_MAX * 2 - 1) * mutationStrength;
                }
            }

            // Mutate biases
            if ((long double)rand() / RAND_MAX < mutationRate) {
                nn->biases[layer][i] += ((long double)rand() / RAND_MAX * 2 - 1) * mutationStrength;
            }
        }
    }
}

void NN_rl_backprop(NN_t *nn, long double *inputs, long double *y_true, 
                    long double *y_predicted, long double *rewards, 
                    long double gamma) {

  // Calculate gradients for reinforcement learning
    size_t outputSize = nn->layers[nn->numLayers - 1];
    long double *outputGradients = (long double *)calloc(outputSize, sizeof(long double));

    // Calculate discounted reward
    long double *discountedRewards = (long double *)malloc(outputSize * sizeof(long double));

    for (size_t i = 0; i < outputSize; i++) {
        discountedRewards[i] = rewards[i];
        for (size_t j = i + 1; j < outputSize; j++) {
            discountedRewards[i] += gamma * rewards[j];
        }
    }

    // Gradient computation for reinforcement learning
    for (size_t i = 0; i < outputSize; i++) {
        outputGradients[i] = (long double )(*nn->lossFunction)(y_true[i], y_predicted[i]) * discountedRewards[i];
    }

    // Backpropagate through layers
    for (int layer = nn->numLayers - 2; layer >= 0; --layer) {
        long double *newGradients = (long double *)calloc(nn->layers[layer], sizeof(long double));
        for (size_t i = 0; i < nn->layers[layer]; i++) {
            for (size_t j = 0; j < nn->layers[layer + 1]; j++) {
                long double weightGradient = outputGradients[j] * inputs[i];
                nn->weights[layer][i * nn->layers[layer + 1] + j] -= weightGradient; // Update weights
                newGradients[i] += outputGradients[j] * nn->weights[layer][i * nn->layers[layer + 1] + j];
            }
        }
        
        if (layer < nn->numLayers - 2) {
            free(outputGradients);
        }
        outputGradients = newGradients;
    }

    free(outputGradients);
    free(discountedRewards);
}

long double relu(long double x) {
  return x > 0 ? x : 0;
}

long double relu_derivative(long double x) {
  return x > 0 ? 1.0L : 0.0L;
}

long double sigmoid(long double x) {
  return 1.0L / (1.0L + expl(-x));
}

long double sigmoid_derivative(long double x) {
  long double s = sigmoid(x);
  return s * (1.0L - s);
}

long double tanh_activation(long double x) {
  return tanhl(x);
}

long double tanh_derivative(long double x) {
  long double t = tanhl(x);
  return 1.0L - t * t;
}

long double argmax(long double x[]) {
  long double max = x[0];
  size_t maxIndex = 0;
  // Note: size must be passed in or determined from context
  // For now, we'll just use a fixed size since this is only used in specific contexts
  size_t size = 10;  // This should be replaced with actual size
  for (size_t i = 1; i < size; i++) {
    if (x[i] > max) {
      max = x[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

long double argmax_derivative(long double x[]) {
  return 0; // Derivative of argmax is not well-defined
}

long double softmax(long double x) {
  return exp(x); 
}

long double softmax_derivative(long double x) {
  return softmax(x) * (1 - softmax(x)); 
}

long double ce(long double y_true, long double y_pred) {
    return -y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred);
}

long double ce_derivative(long double y_true, long double y_pred) {
    return (y_pred - y_true) / (y_pred * (1 - y_pred));
}

long double mse(long double x, long double y) {
  return 0.5 * pow(x - y, 2);
}

long double mse_derivative(long double x, long double y) {
  return 2 * (x - y);
}

long double mae(long double x, long double y) {
  return fabs(x - y);
}

long double mae_derivative(long double x, long double y) {
  return x < y ? 1 : -1;
}

int NN_save(NN_t *nn, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return 0;
    }

    // Save number of layers and layer sizes
    fprintf(fp, "%zu\n", nn->numLayers);
    for (size_t i = 0; i < nn->numLayers; i++) {
        fprintf(fp, "%zu ", nn->layers[i]);
    }
    fprintf(fp, "\n");

    // Save activation functions and derivatives
    for (size_t i = 0; i < nn->numLayers; i++) {
        fprintf(fp, "%d ", activation_function_to_enum(nn->activationFunctions[i]));
    }
    fprintf(fp, "\n");
    for (size_t i = 0; i < nn->numLayers; i++) {
        fprintf(fp, "%d ", activation_derivative_to_enum(nn->activationDerivatives[i]));
    }
    fprintf(fp, "\n");

    // Save loss function and derivative
    fprintf(fp, "%d\n", loss_function_to_enum(nn->lossFunction));
    fprintf(fp, "%d\n", loss_derivative_to_enum(nn->lossDerivative));

    // Save learning rate
    fprintf(fp, "%Lf\n", nn->learningRate);

    // Save weights
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        for (size_t j = 0; j < nn->layers[i + 1]; j++) {
            for (size_t k = 0; k < nn->layers[i]; k++) {
                fprintf(fp, "%Lf ", nn->weights[i][j * nn->layers[i] + k]);
            }
            fprintf(fp, "\n");
        }
    }

    // Save biases
    for (size_t i = 0; i < nn->numLayers; i++) {
        for (size_t j = 0; j < nn->layers[i]; j++) {
            fprintf(fp, "%Lf ", nn->biases[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 1;
}

NN_t *NN_load(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file %s for reading\n", filename);
        return NULL;
    }

    // Read number of layers
    size_t numLayers;
    if (fscanf(fp, "%zu", &numLayers) != 1) {
        fprintf(stderr, "Error: Could not read number of layers\n");
        fclose(fp);
        return NULL;
    }

    // Read layer sizes
    size_t *layers = (size_t *)malloc(numLayers * sizeof(size_t));
    for (size_t i = 0; i < numLayers; i++) {
        if (fscanf(fp, "%zu", &layers[i]) != 1) {
            fprintf(stderr, "Error: Could not read layer size\n");
            free(layers);
            fclose(fp);
            return NULL;
        }
    }

    // Read activation functions and derivatives
    int *actFuncs = (int *)malloc(numLayers * sizeof(int));
    int *actDerivs = (int *)malloc(numLayers * sizeof(int));
    for (size_t i = 0; i < numLayers; i++) {
        if (fscanf(fp, "%d", &actFuncs[i]) != 1) {
            fprintf(stderr, "Error: Could not read activation function\n");
            free(layers);
            free(actFuncs);
            free(actDerivs);
            fclose(fp);
            return NULL;
        }
    }
    for (size_t i = 0; i < numLayers; i++) {
        if (fscanf(fp, "%d", &actDerivs[i]) != 1) {
            fprintf(stderr, "Error: Could not read activation derivative\n");
            free(layers);
            free(actFuncs);
            free(actDerivs);
            fclose(fp);
            return NULL;
        }
    }

    // Read loss function and derivative
    int lossFunc, lossDeriv;
    if (fscanf(fp, "%d", &lossFunc) != 1 || fscanf(fp, "%d", &lossDeriv) != 1) {
        fprintf(stderr, "Error: Could not read loss function/derivative\n");
        free(layers);
        free(actFuncs);
        free(actDerivs);
        fclose(fp);
        return NULL;
    }

    // Read learning rate
    long double learningRate;
    if (fscanf(fp, "%Lf", &learningRate) != 1) {
        fprintf(stderr, "Error: Could not read learning rate\n");
        free(layers);
        free(actFuncs);
        free(actDerivs);
        fclose(fp);
        return NULL;
    }

    // Initialize the network
    NN_t *nn = NN_init(layers, (ActivationFunctionType *)actFuncs, 
                       (ActivationDerivativeType *)actDerivs,
                       (LossFunctionType)lossFunc, 
                       (LossDerivativeType)lossDeriv, 
                       learningRate);

    if (!nn) {
        fprintf(stderr, "Error: Could not initialize network\n");
        free(layers);
        free(actFuncs);
        free(actDerivs);
        fclose(fp);
        return NULL;
    }

    // Read weights
    for (size_t i = 0; i < numLayers - 1; i++) {
        for (size_t j = 0; j < layers[i + 1]; j++) {
            for (size_t k = 0; k < layers[i]; k++) {
                if (fscanf(fp, "%Lf", &nn->weights[i][j * layers[i] + k]) != 1) {
                    fprintf(stderr, "Error: Could not read weight\n");
                    NN_destroy(nn);
                    free(layers);
                    free(actFuncs);
                    free(actDerivs);
                    fclose(fp);
                    return NULL;
                }
            }
        }
    }

    // Read biases
    for (size_t i = 0; i < numLayers; i++) {
        for (size_t j = 0; j < layers[i]; j++) {
            if (fscanf(fp, "%Lf", &nn->biases[i][j]) != 1) {
                fprintf(stderr, "Error: Could not read bias\n");
                NN_destroy(nn);
                free(layers);
                free(actFuncs);
                free(actDerivs);
                fclose(fp);
                return NULL;
            }
        }
    }

    // Clean up
    free(layers);
    free(actFuncs);
    free(actDerivs);
    fclose(fp);

    return nn;
}
