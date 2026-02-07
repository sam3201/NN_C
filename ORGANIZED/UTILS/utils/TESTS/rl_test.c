#include "../NN/NN.h"
#include <stdio.h>
#include <stdlib.h>

// Simple Reinforcement Learning Test with Neural Network
int main() {
    size_t layers[] = {3, 5, 2, 0}; // Input, hidden, and output layers (MUST END IN ZERO)
    ActivationFunctionType activationFunctions[] = {RELU, RELU, SIGMOID};  // Activation functions for each layer
    ActivationDerivativeType activationDerivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, SIGMOID_DERIVATIVE};  // Derivatives for each layer

    // Loss function and its derivative (mean squared error)
    LossFunctionType lossFunction = CE;
    LossDerivativeType lossDerivative = CE_DERIVATIVE;

    // Initialize the neural network
    NN_t *nn = NN_init(layers, activationFunctions, activationDerivatives, lossFunction, lossDerivative, 0.01);

    // Sample input for testing (3 features)
    long double inputs[] = {0.5, 0.1, 0.3};

    // Run a forward pass through the network to get predicted outputs
    long double *outputs = NN_forward(nn, inputs);

    // Print predicted outputs (for reinforcement learning, these would represent Q-values or state-action values)
    printf("Predicted Outputs (Q-values): ");
    for (size_t i = 0; i < layers[2]; i++) {
        printf("%Lf ", outputs[i]);
    }
    printf("\n");

    // Sample true outputs (for RL, these would come from environment feedback)
    long double y_true[] = {0.0, 1.0};  // Example: agent expected to prefer the second action

    // Print the true values (expected outcomes)
    printf("True Outputs (expected Q-values): ");
    for (size_t i = 0; i < layers[2]; i++) {
        printf("%Lf ", y_true[i]);
    }
    printf("\n");

    // Sample rewards (for RL, these would come from the environment after an action)
    long double rewards[] = {1.0, 0.5};  // Example: rewards corresponding to each action

    // Discount factor (gamma), typically between 0 and 1
    long double gamma = 0.9;

    // Perform reinforcement learning backpropagation to adjust the neural network weights
    printf("Running Reinforcement Learning Backpropagation...\n");
    NN_rl_backprop(nn, inputs, y_true, outputs, rewards, gamma);  // Pass all required arguments

    // Free the memory used by the network's outputs
    free(outputs);

    // Destroy the neural network to free up allocated memory
    NN_destroy(nn);

    return 0;
}
