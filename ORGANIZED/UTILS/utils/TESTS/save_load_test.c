#include "../NN/NN.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper function to print network predictions
void print_predictions(NN_t *nn, long double inputs[][2], long double targets[], const char *prefix) {
    printf("\n%s predictions:\n", prefix);
    for (int i = 0; i < 4; i++) {
        long double *output = NN_forward(nn, inputs[i]);
        if (!output) {
            fprintf(stderr, "Forward pass failed\n");
            return;
        }
        printf("Input: [%.0Lf, %.0Lf] Target: %.0Lf Prediction: %.6Lf\n",
               inputs[i][0], inputs[i][1], targets[i], *output);
        free(output);
    }
}

int main() {
    // Create a simple neural network
    size_t layers[] = {2, 3, 1, 0};  // 2 inputs, 3 hidden, 1 output, zero terminator
    ActivationFunctionType actFuncs[] = {SIGMOID, SIGMOID, SIGMOID};
    ActivationDerivativeType actDerivs[] = {SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE};
    
    printf("Creating neural network...\n");
    NN_t *nn = NN_init(layers, actFuncs, actDerivs, MSE, MSE_DERIVATIVE, 0.1);
    if (!nn) {
        fprintf(stderr, "Failed to create neural network\n");
        return 1;
    }

    // Training data for XOR function
    long double inputs[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    long double targets[] = {0, 1, 1, 0};

    printf("\nTraining network on XOR function...\n");
    // Train for a few epochs
    for (int epoch = 0; epoch < 1000; epoch++) {
        double total_error = 0;
        for (int i = 0; i < 4; i++) {
            long double *output = NN_forward(nn, inputs[i]);
            if (!output) {
                fprintf(stderr, "Forward pass failed\n");
                NN_destroy(nn);
                return 1;
            }
            total_error += NN_loss(nn, targets[i], *output);
            NN_backprop(nn, inputs[i], targets[i], *output);
            free(output);
        }
        
        // Print progress every 100 epochs
        if ((epoch + 1) % 100 == 0) {
            printf("Epoch %d, Average Error: %.6f\n", epoch + 1, total_error / 4);
        }
    }

    // Test original network
    print_predictions(nn, inputs, targets, "Original network");

    // Save the network
    printf("\nSaving network to 'network.txt'...\n");
    if (!NN_save(nn, "network.txt")) {
        fprintf(stderr, "Failed to save network\n");
        NN_destroy(nn);
        return 1;
    }
    printf("Network saved successfully!\n");

    // Destroy original network
    NN_destroy(nn);
    printf("Original network destroyed\n");

    // Load the network
    printf("\nLoading network from 'network.txt'...\n");
    NN_t *loaded_nn = NN_load("network.txt");
    if (!loaded_nn) {
        fprintf(stderr, "Failed to load network\n");
        return 1;
    }
    printf("Network loaded successfully!\n");

    // Test loaded network
    print_predictions(loaded_nn, inputs, targets, "Loaded network");

    // Clean up
    NN_destroy(loaded_nn);
    printf("\nTest completed successfully!\n");
    return 0;
}
