#include "NN.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    size_t layers[] = {3, 5, 2, 0};  
    ActivationFunction activationFunctions[] = {RELU, RELU, SIGMOID};  
    ActivationDerivative activationDerivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, SIGMOID_DERIVATIVE};
    
    LossFunction lossFunction = MSE;
    LossDerivative lossDerivative = MSE_DERIVATIVE;

    NN_t *nn = NN_init(layers, activationFunctions, activationDerivatives, lossFunction, lossDerivative);

    long double inputs[] = {0.5, 0.1, 0.3};

    long double *outputs = NN_forward(nn, inputs);

    printf("Predicted Outputs: ");
    for (size_t i = 0; i < layers[2]; i++) {
        printf("%Lf ", outputs[i]);
    }
    printf("\n");

    long double y_true[] = {0.0, 1.0};

    printf("Running Backpropagation...\n");
    NN_backprop(nn, inputs, y_true, outputs[0]);

    free(outputs);
    NN_destroy(nn);

    return 0;
}

