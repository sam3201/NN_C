#include "../Raylib/raylib.h"
#include "../NN/NN.h"
#include "../VISUALIZER/NN_visualizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Test neural network configurations
#define TEST_INPUT_NODES 4
#define TEST_HIDDEN_NODES 5
#define TEST_OUTPUT_NODES 3
#define UPDATE_INTERVAL_MS 500

// Test scenarios
void test_random_weights(NN_t* nn) {
    size_t layer = rand() % (nn->numLayers - 1);
    size_t from = rand() % nn->layers[layer];
    size_t to = rand() % nn->layers[layer + 1];
    nn->weights[layer][from * nn->layers[layer + 1] + to] = 
        ((long double)rand() / RAND_MAX) * 2.0L - 1.0L;
}

void test_all_positive_weights(NN_t* nn) {
    for (size_t l = 0; l < nn->numLayers - 1; l++) {
        for (size_t i = 0; i < nn->layers[l]; i++) {
            for (size_t j = 0; j < nn->layers[l + 1]; j++) {
                nn->weights[l][i * nn->layers[l + 1] + j] = 
                    (long double)rand() / RAND_MAX;  // 0 to 1
            }
        }
    }
}

void test_all_negative_weights(NN_t* nn) {
    for (size_t l = 0; l < nn->numLayers - 1; l++) {
        for (size_t i = 0; i < nn->layers[l]; i++) {
            for (size_t j = 0; j < nn->layers[l + 1]; j++) {
                nn->weights[l][i * nn->layers[l + 1] + j] = 
                    -((long double)rand() / RAND_MAX);  // -1 to 0
            }
        }
    }
}

void test_extreme_weights(NN_t* nn) {
    for (size_t l = 0; l < nn->numLayers - 1; l++) {
        for (size_t i = 0; i < nn->layers[l]; i++) {
            for (size_t j = 0; j < nn->layers[l + 1]; j++) {
                nn->weights[l][i * nn->layers[l + 1] + j] = 
                    (rand() % 2) ? 1.0L : -1.0L;  // Only -1 or 1
            }
        }
    }
}

int main(void) {
    // Seed random number generator
    srand(time(NULL));

    // Create layer configuration
    size_t layers[] = {TEST_INPUT_NODES, TEST_HIDDEN_NODES, TEST_OUTPUT_NODES, 0};
    
    // Create activation functions array
    ActivationFunctionType activations[] = {SIGMOID, SIGMOID, SIGMOID};
    ActivationDerivativeType derivatives[] = {SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE};
    
    printf("Creating neural network for visualization test...\n");
    printf("Configuration:\n");
    printf("- Input nodes: %d\n", TEST_INPUT_NODES);
    printf("- Hidden nodes: %d\n", TEST_HIDDEN_NODES);
    printf("- Output nodes: %d\n", TEST_OUTPUT_NODES);
    
    NN_t* test_nn = NN_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, 0.1);
    
    if (!test_nn) {
        printf("Failed to create neural network!\n");
        return 1;
    }
    
    printf("Neural network created successfully!\n");
    
    // Initialize visualizer
    InitializeVisualizer();
    
    printf("\nNeural Network Visualization Test\n");
    printf("=================================\n");
    printf("Testing different weight scenarios:\n");
    printf("1. Random weights (0-15s)\n");
    printf("2. All positive weights (15-30s)\n");
    printf("3. All negative weights (30-45s)\n");
    printf("4. Extreme weights (-1 or 1) (45-60s)\n");
    printf("\nPress ESC in the window to exit early.\n");
    
    time_t start_time = time(NULL);
    int current_test = 0;
    
    // Main test loop
    while (!WindowShouldClose()) {
        // Update test scenarios
        time_t elapsed = time(NULL) - start_time;
        
        // Switch between test scenarios every 15 seconds
        int new_test = (int)(elapsed / 15);
        if (new_test != current_test && new_test < 4) {
            current_test = new_test;
            printf("\nSwitching to test scenario %d\n", current_test + 1);
            
            // Apply the new test scenario immediately
            switch(current_test) {
                case 0: test_random_weights(test_nn); break;
                case 1: test_all_positive_weights(test_nn); break;
                case 2: test_all_negative_weights(test_nn); break;
                case 3: test_extreme_weights(test_nn); break;
            }
        }
        
        // Update weights periodically within the current scenario
        if (elapsed % 1 == 0) {  // Update every second
            switch(current_test) {
                case 0: test_random_weights(test_nn); break;
                case 1: test_all_positive_weights(test_nn); break;
                case 2: test_all_negative_weights(test_nn); break;
                case 3: test_extreme_weights(test_nn); break;
            }
        }
        
        // Draw the network
        DrawNeuralNetwork(test_nn);
        
        // Add small delay to prevent excessive updates
        struct timespec ts = {0, 16666667}; // ~60 FPS
        nanosleep(&ts, NULL);
    }

    // Cleanup
    printf("\nCleaning up...\n");
    CloseVisualizer();
    NN_destroy(test_nn);

    printf("Test completed successfully!\n");
    return 0;
}
