#include "NN.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

char* activation_function_to_string(ActivationFunction code) {
    char *activationFunction = (char *)malloc(64 * sizeof(char)); 

    switch (code) {
        case SIGMOID: return "SIGMOID";
        case RELU: return "RELU";
        case TANH: return "TANH"; 
        case ARGMAX: return "ARGMAX";
        case SOFTMAX: return "SOFTMAX"; 
       default: return "UNKNOWN";
  }

  return activationFunction;
}

char* activation_derivative_to_string(ActivationDerivative deriv) {
   char *activationDerivative = (char *)malloc(64 * sizeof(char));

    switch (deriv) {
      case SIGMOID: return "SIGMOID_DERIVATIVE";
      case RELU: return "RELU_DERIVATIVE";
      case TANH: return "TANH_DERIVATIVE";
      case ARGMAX: return "ARGMAX_DERIVATIVE";
      case SOFTMAX: return "SOFTMAX_DERIVATIVE";

        default: return "unknown";
    }
}

char* loss_function_to_string(LossFunction func) {
   char *lossFunction = (char *)malloc(64 * sizeof(char));
    switch (func) {
    case MSE: return "MSE";
    case CE: return "CROSS_ENTROPY";
        default: return "unknown";
    }
}

char* loss_derivative_to_string(LossDerivative deriv) {
   char *lossDerivative = (char *)malloc(64 * sizeof(char));
    switch (deriv) {
    case MSE: return "MSE_DERIVATIVE";
    case CE: return "CROSS_ENTROPY_DERIVATIVE";
        default: return "unknown";
    }
}

NN_t *NN_init(size_t layers[],
              ActivationFunction activationFunctions[], ActivationDerivative activationDerivatives[],
              LossFunction lossFunction, LossDerivative lossDerivative) {

 
    NN_t *nn = (NN_t*)malloc(sizeof(NN_t));
  
    nn->numLayers = 0;
    while (layers[nn->numLayers] != 0) {
      nn->numLayers++;
    }
    
    nn->layers = layers;

    nn->activationFunctions = malloc(nn->numLayers * sizeof(long double (*)(long double)));
    nn->activationDerivatives = malloc(nn->numLayers * sizeof(long double (*)(long double)));
    for (unsigned int a = 0; a < nn->numLayers; a++) { 
      char *activationFunction = activation_function_to_string(activationFunctions[a]);
      if (strcmp(activationFunction, "SIGMOID")) {
          nn->activationFunctions[a] = sigmoid;
          nn->activationDerivatives[a] = sigmoid_derivative;
      } else if (strcmp(activationFunction,"RELU")) {
          nn->activationFunctions[a] = relu;
          nn->activationDerivatives[a] = relu_derivative;
      } else if (strcmp(activationFunction,"TANH")) {
          nn->activationFunctions[a] = tanh_activation;
          nn->activationDerivatives[a] = tanh_derivative;
      } else if (strcmp(activationFunction,"ARGMAX")) {
          printf("Argmax not implemented yet!\n");
          exit(1);
      } else if (strcmp(activationFunction,"SOFTMAX")) {
          printf("Softmax not implemented yet!\n");
          exit(1);
      } 
     }
  
    nn->lossFunction = malloc(sizeof(long double (*)(long double)));; 
    nn->lossDerivative = malloc(sizeof(long double (*)(long double)));
    switch (lossFunction) {
      case MSE:
        nn->lossFunction = mse;
        nn->lossDerivative = mse_derivative;
        break;
      case CE:
        printf("Cross entropy not implemented yet!\n");
        exit(1); 
    }

    nn->weights = (long double**)malloc((nn->numLayers - 1) * sizeof(long double*));
    nn->biases = (long double**)malloc((nn->numLayers - 1) * sizeof(long double*));
    for (size_t i = 1; i < nn->numLayers; i++) {
        nn->weights[i - 1] = (long double*)malloc(nn->layers[i] * nn->layers[i - 1] * sizeof(long double));
        nn->biases[i - 1] = (long double*)malloc(nn->layers[i] * sizeof(long double));

        for (size_t j = 0; j < nn->layers[i] * nn->layers[i - 1]; j++) {
            nn->weights[i - 1][j] = (long double)rand() / RAND_MAX * 2 - 1;
        }
    }

    return nn;
}

void NN_destroy(NN_t *nn) {
    for (size_t i = 0; i < sizeof(nn->layers) / sizeof(size_t); i++) {
        free(nn->weights[i]);
        free(nn->biases[i]);
    }
    free(nn->weights);
    free(nn->biases);
    free(nn->layers);
    free(nn->activationFunctions);
    free(nn->activationDerivatives);
    free(nn->lossFunction);
    free(nn->lossDerivative);
    free(nn);
}


void NN_add_layer(NN_t *nn, size_t layerSize, ActivationFunction activationFunctions[], ActivationDerivative activationDerivatives[]) {
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

long double NN_matmul(long double *inputs, long double *weights, long double *biases) { 
    long double sum = 0.;
    for (size_t i = 0; i < sizeof(inputs) / sizeof(long double); i++) {
        for (size_t j = 0; j < sizeof(weights) / sizeof(long double); j++) {
            sum += (inputs[j] * weights[j]) + biases[j];
        }
    }
    return sum;
}

long double *NN_forward(NN_t *nn, long double *inputs) {
    long double *outputs = (long double *)malloc(sizeof(long double) * nn->layers[0]);
    
    for (size_t layer = 0; layer < nn->numLayers - 1; layer++) {
        outputs = (long double *)realloc(outputs, sizeof(long double) * nn->layers[layer + 1]);
        outputs[layer] = NN_matmul(inputs, nn->weights[layer], nn->biases[layer]); 
        for (size_t i = 0; i < nn->layers[layer + 1]; i++) {
            outputs[i] = nn->activationFunctions[layer + 1](outputs[i]);
        }
        
        inputs = outputs;
    }
    
    return outputs;
} 

void NN_backprop(NN_t *nn, long double *inputs, long double *y_true, long double *y_predicted) {
    printf("Backprop\n");
    
    long double **deltas = (long double **)malloc(sizeof(long double *) * (nn->numLayers - 1));

    // Calculate the loss (optional, depending on whether you need this)
    long double loss = 0.0;
    for (size_t i = 0; i < nn->layers[nn->numLayers - 1]; i++) {
        loss += nn->lossFunction(y_true[i], y_predicted[i]);
    }
    printf("Loss: %Lf\n", loss);

    // Calculate the delta for the output layer
    deltas[nn->numLayers - 2] = (long double *)malloc(sizeof(long double) * nn->layers[nn->numLayers - 1]);
    for (unsigned int i = 0; i < nn->layers[nn->numLayers - 1]; i++) {
        deltas[nn->numLayers - 2][i] = nn->lossDerivative(y_true[i], y_predicted[i]) * nn->activationDerivatives[nn->numLayers - 2](y_predicted[i]);
    }
    printf("Deltas\n");

    // Backpropagate through the hidden layers
    for (unsigned int layer = nn->numLayers - 2; layer > 0; layer--) {
        deltas[layer - 1] = (long double *)malloc(sizeof(long double) * nn->layers[layer]);
        for (unsigned int neuron = 0; neuron < nn->layers[layer]; neuron++) {
            long double sum = 0.0;
            for (unsigned int nextNeuron = 0; nextNeuron < nn->layers[layer + 1]; nextNeuron++) {
                sum += deltas[layer][nextNeuron] * nn->weights[layer][neuron * nn->layers[layer + 1] + nextNeuron];
            }
            deltas[layer - 1][neuron] = sum * nn->activationDerivatives[layer - 1](inputs[neuron]);
        }
    }
    printf("Deltas2\n");

    // Update weights and biases
    for (unsigned int layer = 0; layer < nn->numLayers - 1; layer++) {
        for (unsigned int neuron = 0; neuron < nn->layers[layer]; neuron++) {
            for (unsigned int nextNeuron = 0; nextNeuron < nn->layers[layer + 1]; nextNeuron++) {
                nn->weights[layer][neuron * nn->layers[layer + 1] + nextNeuron] -= deltas[layer][nextNeuron] * inputs[neuron];
            }
        }
        for (unsigned int neuron = 0; neuron < nn->layers[layer + 1]; neuron++) {
            nn->biases[layer][neuron] -= deltas[layer][neuron];
        }
        *inputs = NN_matmul(inputs, nn->weights[layer], nn->biases[layer]);
        for (size_t i = 0; i < nn->layers[layer + 1]; i++) {
            inputs[i] = nn->activationFunctions[layer](inputs[i]);
        }
    }
    printf("Weight Update\n");
    printf("Biases Update\n");

    // Clean up
    for (unsigned int layer = 0; layer < nn->numLayers - 1; layer++) {
        free(deltas[layer]);
    }
    free(deltas);
    printf("Done\n");
}

long double relu(long double x) {
  return x > 0 ? x : 0;
}

long double relu_derivative(long double x) {
  return x > 0 ? 1 : 0;
}

long double sigmoid(long double x) {
  return 1 / (1 + exp(-x));
}

long double sigmoid_derivative(long double x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

long double tanh_activation(long double x) {
  return tanh(x);
}

long double tanh_derivative(long double x) {
  return 1 - tanh_activation(x) * tanh_activation(x);
}

long double argmax(long double *x) {
  long double max = x[0];
  size_t maxIndex = 0;
  for (size_t i = 1; i < sizeof(x) / sizeof(x[0]); i++) {
    if (x[i] > max) {
      max = x[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

long double argmax_derivative(long double *x) {
  return 0; 
}

long double softmax(long double *x) {
  long double sum = 0.0;
  for (size_t i = 0; i < sizeof(x) / sizeof(x[0]); i++) {
    sum += exp(x[i]);
  }
  for (size_t i = 0; i < sizeof(x) / sizeof(x[0]); i++) {
    x[i] = exp(x[i]) / sum;
  }
  return x[0];
}

long double softmax_derivative(long double *x) {
  return softmax(x) * (1 - softmax(x)); 
}

long double cross_entropy(long double x, long double y) {
  return -log(x) * y;
}

long double cross_entropy_derivative(long double x, long double y) {
  return x - y;
}

long double mse(long double x, long double y) {
  return 0.5 * pow(x - y, 2);
}

long double mse_derivative(long double x, long double y) {
  return 2 * (x - y);
}


/*
void NN_save(const char* filename, NN_t *nn) {
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        printf("Error opening file: %s\n", filename);
        return;
    }

    // Save network structure
    for (size_t i = 0; i < nn->numLayers; i++) {
        fwrite(&nn->layers[i], sizeof(size_t), 1, fp);
    }
    
    for (size_t i = 0; i < sizeof(nn->weights) / sizeof(nn->weights[0]); i++) {
        fwrite(nn->weights[i], sizeof(long double), nn->layers[i] * nn->layers[i + 1], fp);
        fwrite(nn->biases[i], sizeof(long double), nn->layers[i + 1], fp);
    }
    
    // Save activation functions
    for (size_t i = 0; i < nn->numLayers; i++) {
        size_t func_id = activation_function_to_id(nn->activationFunctions[i]);
        fwrite(&func_id, sizeof(size_t), 1, fp);

        size_t deriv_id = activation_derivative_to_id(nn->activationDerivatives[i]);
        fwrite(&deriv_id, sizeof(size_t), 1, fp);
    }
    
    // Save loss functions
    size_t loss_func_id = loss_function_to_id(nn->lossFunction);
    fwrite(&loss_func_id, sizeof(size_t), 1, fp);

    const char *loss_deriv_id = loss_derivative_to_string(nn->lossDerivative);
    fwrite(&loss_deriv_id, sizeof(size_t), 1, fp);
    
    fclose(fp);
}

NN_t *NN_load(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Error opening file: %s\n", filename);
    return NULL;
  }

  NN_t *nn = (NN_t *)malloc(sizeof(NN_t));

  size_t numLayers;
  fread(&numLayers, sizeof(size_t), 1, fp);
  nn->numLayers = numLayers;
  nn->layers = (size_t *)malloc(numLayers * sizeof(size_t));
  for (size_t i = 0; i < numLayers; i++) {
    fread(&nn->layers[i], sizeof(size_t), 1, fp);
  }

  nn->weights = (long double **)malloc(numLayers * sizeof(long double *));
  nn->biases = (long double **)malloc(numLayers * sizeof(long double *));
  for ( size_t i = 0; i < numLayers; i++) {
    nn->weights[i] = (long double *)malloc(nn->layers[i] * nn->layers[i + 1] * sizeof(long double));
    nn->biases[i] = (long double *)malloc(nn->layers[i + 1] * sizeof(long double));
    fread(nn->weights[i], sizeof(long double), nn->layers[i] * nn->layers[i + 1], fp);
    fread(nn->biases[i], sizeof(long double), nn->layers[i + 1], fp);
  }
  
  char activationFunctionNames[2][10];
  fread(activationFunctionNames[0], sizeof(char), 10, fp);
  fread(activationFunctionNames[1], sizeof(char), 10, fp);
  nn->activationFunctions[0] = activation_function_from_string(activationFunctionNames[0]);
  nn->activationFunctions[1] = activation_function_from_string(activationFunctionNames[1]);
  
  char lossFunctionName[10];
  fread(lossFunctionName, sizeof(char), 10, fp);
  nn->lossFunction = loss_function_from_string(lossFunctionName);
  
  fclose(fp);
}

*/
