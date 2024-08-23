#include "NN.h"
#include <stdlib.h>
#include <stdio.h>

NN_t *NN_init(size_t *layers, size_t numLayers,
              long double (**activationFunctions)(long double), long double (**activationDerivatives)(long double),
              long double (*lossFunction)(long double, long double), long double (*lossDerivative)(long double, long double)) {

    NN_t *nn = (NN_t *)malloc(sizeof(NN_t));

    nn->layers = (size_t *)malloc(numLayers * sizeof(size_t));
    for (size_t i = 0; i < numLayers; i++) {
        nn->layers[i] = layers[i];
    }
    nn->numLayers = numLayers;

    nn->activationFunctions = activationFunctions;
    nn->activationDerivatives = activationDerivatives;
    nn->lossFunction = lossFunction;
    nn->lossDerivative = lossDerivative;

    nn->weights = (long double **)malloc((numLayers - 1) * sizeof(long double *));
    nn->biases = (long double **)malloc((numLayers - 1) * sizeof(long double *));

    for (size_t layer = 0; layer < numLayers - 1; layer++) {
        nn->weights[layer] = (long double *)malloc(nn->layers[layer] * nn->layers[layer + 1] * sizeof(long double));
        nn->biases[layer] = (long double *)malloc(nn->layers[layer + 1] * sizeof(long double));
        for (size_t neuron = 0; neuron < nn->layers[layer] * nn->layers[layer + 1]; neuron++) {
            nn->weights[layer][neuron] = (long double)rand() / RAND_MAX;
        }
        for (size_t neuron = 0; neuron < nn->layers[layer + 1]; neuron++) {
            nn->biases[layer][neuron] = (long double)rand() / RAND_MAX;
        }
    }

    return nn;
}

void NN_destroy(NN_t *nn) {
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        free(nn->weights[i]);
        free(nn->biases[i]);
    }
    free(nn->weights);
    free(nn->biases);
    free(nn->layers);
    free(nn);
}


void NN_add_layer(NN_t *nn, size_t layerSize, long double (**activationFunctions)(long double), long double (**activationDerivatives)(long double)) {
   size_t prevNumLayers = nn->numLayers;

   nn->layers = (size_t *)realloc(nn->layers, (nn->numLayers + 1) * sizeof(size_t));
   nn->layers[nn->numLayers] = layerSize;
   nn->numLayers+=layerSize;

   nn->activationFunctions = (long double (**)(long double))realloc(nn->activationFunctions, (nn->numLayers + layerSize) * sizeof(long double (*)(long double)));
   nn->activationDerivatives = (long double (**)(long double))realloc(nn->activationDerivatives, (nn->numLayers + layerSize) * sizeof(long double (*)(long double)));
   for (size_t neuron = prevNumLayers; neuron < nn->numLayers; neuron++) {
     nn->activationFunctions[neuron] = activationFunctions[neuron];
     nn->activationDerivatives[neuron] = activationDerivatives[neuron];
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
  for (unsigned int i = 0; i < sizeof(inputs) / sizeof(long double); i++) {
      for (unsigned int j = 0; j < sizeof(weights) / sizeof(long double); j++) {
        sum += (inputs[i] * weights[j]) + biases[j];; 
    }
  }

  return sum;
}

long double *NN_forward(NN_t *nn, long double *inputs) {
    printf("forward\n");

    long double *currentOutput = (long double *)malloc(sizeof(long double) * nn->layers[0]);
    for (size_t i = 0; i < nn->layers[0]; i++) {
        currentOutput[i] = inputs[i];
    }

    for (size_t layer = 0; layer < nn->numLayers; layer++) {
        printf("layer %zu\n", layer);
        size_t numNeurons = nn->layers[layer];
        long double *nextOutput = (long double *)malloc(sizeof(long double) * numNeurons);

        for (size_t neuron = 0; neuron < numNeurons; neuron++) {
            nextOutput[neuron] = NN_matmul(currentOutput, nn->weights[layer] + neuron * nn->layers[layer], nn->biases[layer] + neuron);
            printf("Matmul: %Lf\n", nextOutput[neuron]);
            nextOutput[neuron] = nn->activationFunctions[neuron](nextOutput[neuron]);
            printf("Activation: %Lf\n", nextOutput[neuron]);

        }

        free(currentOutput);
        currentOutput = nextOutput;
    }

    return currentOutput;
}

void NN_backprop(NN_t *nn, long double *inputs, long double *outputs, long double *labels) {
    long double **deltas = (long double **)malloc(sizeof(long double *) * nn->numLayers);

    deltas[nn->numLayers - 1] = (long double *)malloc(sizeof(long double) * nn->layers[nn->numLayers - 1]);
    for (unsigned int i = 0; i < nn->layers[nn->numLayers - 1]; i++) {
        deltas[nn->numLayers - 1][i] = (outputs[i] - labels[i]) * nn->lossDerivative(outputs[i], labels[i]);
    }

    for (unsigned int layer = nn->numLayers - 2; layer != (unsigned int)-1; layer--) {
        deltas[layer] = (long double *)malloc(sizeof(long double) * nn->layers[layer]);
        for (unsigned int neuron = 0; neuron < nn->layers[layer]; neuron++) {
            long double sum = 0.;
            for (unsigned int nextNeuron = 0; nextNeuron < nn->layers[layer + 1]; nextNeuron++) {
                sum += deltas[layer + 1][nextNeuron] * nn->weights[layer][neuron * nn->layers[layer + 1] + nextNeuron];
            }
            deltas[layer][neuron] = sum * nn->activationDerivatives[layer](outputs[neuron]);
        }
    }

    for (unsigned int layer = 0; layer < nn->numLayers - 1; layer++) {
        for (unsigned int neuron = 0; neuron < nn->layers[layer]; neuron++) {
            for (unsigned int nextNeuron = 0; nextNeuron < nn->layers[layer + 1]; nextNeuron++) {
                nn->weights[layer][neuron * nn->layers[layer + 1] + nextNeuron] -= deltas[layer + 1][nextNeuron] * inputs[neuron];
            }
            nn->biases[layer + 1][neuron] -= deltas[layer + 1][neuron];
        }
    }

    for (unsigned int layer = 0; layer < nn->numLayers; layer++) {
        free(deltas[layer]);
    }
    free(deltas);
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

