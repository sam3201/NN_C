#include "../NN/NN.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// Helper function to check if two long doubles are approximately equal
static bool approximately_equal(long double a, long double b, long double epsilon) {
    return fabsl(a - b) < epsilon;
}

// Test initialization
void test_nn_init() {
    printf("Testing NN initialization...\n");
    
    size_t layers[] = {2, 3, 1};
    ActivationFunctionType actFuncs[] = {SIGMOID, SIGMOID, SIGMOID};
    ActivationDerivativeType actDerivatives[] = {SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE};
    LossFunctionType lossFunc = MSE;
    LossDerivativeType lossDerivative = MSE_DERIVATIVE;
    RegularizationType reg = L1;
    OptimizerType optimizer = SGD;
    
    NN_t* nn = NN_init(layers, actFuncs, actDerivatives, lossFunc, lossDerivative, 0.01L, reg, optimizer);
    assert(nn != NULL);
    assert(nn->numLayers == 3);
    assert(nn->layers[0] == 2);
    assert(nn->layers[1] == 3);
    assert(nn->layers[2] == 1);
    
    NN_destroy(nn);
    printf("NN initialization test passed!\n");
}

// Test forward propagation
void test_forward_prop() {
    printf("Testing forward propagation...\n");
    
    size_t layers[] = {2, 2, 1};
    ActivationFunctionType actFuncs[] = {LINEAR, LINEAR, LINEAR};
    ActivationDerivativeType actDerivatives[] = {LINEAR_DERIVATIVE, LINEAR_DERIVATIVE, LINEAR_DERIVATIVE};
    LossFunctionType lossFunc = MSE;
    LossDerivativeType lossDerivative = MSE_DERIVATIVE;
    RegularizationType reg = L1;
    OptimizerType optimizer = SGD;
    NN_t* nn = NN_init(layers, actFuncs, actDerivatives, lossFunc, lossDerivative, 0.01L, reg, optimizer);
    
    // Set weights and biases manually for predictable output
    nn->weights[0][0] = 0.5L;
    nn->weights[0][1] = 0.5L;
    nn->weights[0][2] = 0.5L;
    nn->weights[0][3] = 0.5L;
    nn->weights[1][0] = 1.0L;
    nn->weights[1][1] = 1.0L;
    
    nn->biases[0][0] = 0.0L;
    nn->biases[0][1] = 0.0L;
    nn->biases[1][0] = 0.0L;
    
    long double inputs[] = {1.0L, 1.0L};
    long double* output = NN_forward(nn, inputs);
    
    assert(output != NULL);
    // With these weights and linear activation, output should be 2.0
    assert(approximately_equal(*output, 2.0L, 1e-6L));
    
    free(output);
    NN_destroy(nn);
    printf("Forward propagation test passed!\n");
}

// Test backpropagation
void test_backprop() {
    printf("Testing backpropagation...\n");
    
    size_t layers[] = {2, 2, 1};
    ActivationFunctionType actFuncs[] = {LINEAR, LINEAR, LINEAR};
    ActivationDerivativeType actDerivatives[] = {LINEAR_DERIVATIVE, LINEAR_DERIVATIVE, LINEAR_DERIVATIVE};
    LossFunctionType lossFunc = MSE;
    LossDerivativeType lossDerivative = MSE_DERIVATIVE;
    RegularizationType reg = L1;
    OptimizerType optimizer = SGD;
    
    NN_t* nn = NN_init( 
        layers, 
        actFuncs, 
        actDerivatives, 
        lossFunc, 
        lossDerivative, 
        0.01L, 
        reg, 
        optimizer
    ); 
    
    // Initial weights and biases
    nn->weights[0][0] = 0.5L;
    nn->weights[0][1] = 0.5L;
    nn->weights[0][2] = 0.5L;
    nn->weights[0][3] = 0.5L;
    nn->weights[1][0] = 1.0L;
    nn->weights[1][1] = 1.0L;
    
    long double inputs[] = {1.0L, 1.0L};
    long double* output = NN_forward(nn, inputs);
    NN_backprop(nn, inputs, 1.0L, *output);
    
    // Weights should have changed
    assert(!approximately_equal(nn->weights[0][0], 0.5L, 1e-10L));
    
    free(output);
    NN_destroy(nn);
    printf("Backpropagation test passed!\n");
}

int main() {
    printf("Starting Neural Network tests...\n\n");
    
    test_nn_init();
    test_forward_prop();
    test_backprop();
    
    printf("\nAll tests passed successfully!\n");
    return 0;
}
