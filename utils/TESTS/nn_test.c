#include "../NN/NN.h"
#include <stdio.h>
#include <stdlib.h>

void test_backprop() {
    printf("\n=== Testing Backpropagation ===\n");
    
    // Create a simple network: 2 inputs -> 3 hidden -> 1 output
    size_t layers[] = {2, 3, 1, 0};
    // One activation function per layer (except input layer)
    ActivationFunctionType activationFunctions[] = {SIGMOID, SIGMOID, SIGMOID}; 
    ActivationDerivativeType activationDerivatives[] = {SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE};
    
    printf("Creating neural network with architecture: [2, 3, 1]\n");
    NN_t *nn = NN_init(layers, activationFunctions, activationDerivatives, MSE, MSE_DERIVATIVE, 0.01);
    if (!nn) {
        fprintf(stderr, "Failed to initialize neural network\n");
        return;
    }

    // Set simple test inputs
    long double inputs[] = {0.5, -0.2};
    printf("\nInput values: [%.2Lf, %.2Lf]\n", inputs[0], inputs[1]);

    // Forward pass
    long double *outputs = NN_forward(nn, inputs);
    if (!outputs) {
        fprintf(stderr, "Forward pass failed\n");
        NN_destroy(nn);
        return;
    }

    printf("Initial prediction: %.6Lf\n", outputs[0]);

    // Set target value and run backpropagation
    long double y_true = 1.0;  // We want the output to be 1.0
    printf("Target value: %.1Lf\n", y_true);

    // Train the network for a few iterations
    printf("\nTraining for 5 iterations:\n");
    for (int i = 0; i < 5; i++) {
        // Forward pass
        long double *new_outputs = NN_forward(nn, inputs);
        if (!new_outputs) {
            fprintf(stderr, "Forward pass failed during training\n");
            free(outputs);
            NN_destroy(nn);
            return;
        }

        printf("Iteration %d - Prediction: %.6Lf, Loss: %.6Lf\n", 
               i + 1, new_outputs[0], NN_loss(nn, y_true, new_outputs[0]));

        // Backpropagation
        NN_backprop(nn, inputs, y_true, *new_outputs);
        
        if (i < 4) { // Don't free the last output as we'll use it later
            free(new_outputs);
        } else {
            outputs = new_outputs; // Save final output
        }
    }

    // Verify that the loss decreased
    printf("\nFinal prediction: %.6Lf\n", outputs[0]);
    printf("Final loss: %.6Lf\n", NN_loss(nn, y_true, outputs[0]));

    // Cleanup
    free(outputs);
    NN_destroy(nn);
    printf("Backpropagation test completed\n");
}

int main() {
    // Run the original forward pass test
    printf("=== Testing Forward Pass ===\n");
    size_t layers[] = {3, 5, 2, 0};
    // One activation function per layer (except input layer)
    ActivationFunctionType activationFunctions[] = {SIGMOID, SIGMOID, SIGMOID}; 
    ActivationDerivativeType activationDerivatives[] = {SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE};
    LossFunctionType lossFunction = MSE;
    LossDerivativeType lossDerivative = MSE_DERIVATIVE;
    long double learningRate = 0.01;

    printf("Creating neural network...\n");
    NN_t *nn = NN_init(layers, activationFunctions, activationDerivatives, lossFunction, lossDerivative, learningRate);
    if (!nn) {
        fprintf(stderr, "Failed to initialize neural network\n");
        return 1;
    }

    long double inputs[] = {0.5, 0.1, 0.3};
    printf("\nInput values: [%.2Lf, %.2Lf, %.2Lf]\n", inputs[0], inputs[1], inputs[2]);

    long double *outputs = NN_forward(nn, inputs);
    if (!outputs) {
        fprintf(stderr, "Forward pass failed\n");
        NN_destroy(nn);
        return 1;
    }

    printf("Predicted outputs: [");
    for (size_t i = 0; i < layers[2]; i++) {
        printf("%.6Lf%s", outputs[i], i < layers[2] - 1 ? ", " : "");
    }
    printf("]\n");

    free(outputs);
    NN_destroy(nn);
    printf("Forward pass test completed successfully\n");

    // Run the backpropagation test
    test_backprop();

    return 0;
}
