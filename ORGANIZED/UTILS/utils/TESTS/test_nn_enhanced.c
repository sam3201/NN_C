#include "../NN/NN/NN.h"
#include "../NN/NN/NN.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

// Test helper functions
static int test_passed = 0;
static int test_failed = 0;

#define TEST_ASSERT(condition, message) \
    do { \
        if (condition) { \
            printf("✓ PASS: %s\n", message); \
            test_passed++; \
        } else { \
            printf("✗ FAIL: %s\n", message); \
            test_failed++; \
        } \
    } while(0)

#define TEST_ASSERT_CLOSE(a, b, tolerance, message) \
    TEST_ASSERT(fabs(a - b) < tolerance, message)

// Test data
static long double simple_inputs[] = {0.5L, -0.3L, 0.8L, 0.1L};
static long double simple_targets[] = {0.1L, 0.7L, -0.2L, 0.9L};

// Test weight initialization methods
void test_weight_initialization() {
    printf("\n=== Testing Weight Initialization Methods ===\n");
    
    size_t layers[] = {4, 8, 8, 4, 0};
    ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
    ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    
    // Test each initialization method
    WeightInitType methods[] = {ZERO, RANDOM_UNIFORM, RANDOM_NORMAL, XAVIER, HE, LECUN, ORTHOGONAL};
    const char* method_names[] = {"Zero", "Random Uniform", "Random Normal", "Xavier", "He", "LeCun", "Orthogonal"};
    
    for (int i = 0; i < 7; i++) {
        printf("Testing %s initialization...\n", method_names[i]);
        
        NN_t *nn = NN_init_with_weight_init(layers, activations, derivatives, 
                                            MSE, MSE_DERIVATIVE, L2, ADAM, 0.001f, methods[i]);
        TEST_ASSERT(nn != NULL, "Network creation failed");
        
        // Test that weights are properly initialized
        if (nn && nn->weights && nn->weights[0]) {
            // Check that weights are not all zero (except for ZERO initialization)
            if (methods[i] != ZERO) {
                int has_nonzero = 0;
                for (int j = 0; j < layers[0] * layers[1]; j++) {
                    if (fabs(nn->weights[0][j]) > 1e-6f) {
                        has_nonzero = 1;
                        break;
                    }
                }
                TEST_ASSERT(has_nonzero, "Weights should not be zero");
            } else {
                // Zero initialization should have all zero weights
                int all_zero = 1;
                for (int j = 0; j < layers[0] * layers[1]; j++) {
                    if (fabs(nn->weights[0][j]) > 1e-6f) {
                        all_zero = 0;
                        break;
                    }
                }
                TEST_ASSERT(all_zero, "Zero initialization should have all zero weights");
            }
        }
        
        if (nn) NN_destroy(nn);
    }
}

// Test advanced optimizers
void test_advanced_optimizers() {
    printf("\n=== Testing Advanced Optimizers ===\n");
    
    size_t layers[] = {4, 8, 8, 4, 0};
    ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
    ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    
    OptimizerType optimizers[] = {SGD, RMSPROP, ADAGRAD, ADAM, NAG, ADADELTA, RMSPROP_NESTEROV, ADAMAX, NADAM};
    const char* optimizer_names[] = {"SGD", "RMSProp", "AdaGrad", "Adam", "NAG", "AdaDelta", "RMSProp Nesterov", "Adamax", "Nadam"};
    
    for (int i = 0; i < 9; i++) {
        printf("Testing %s optimizer...\n", optimizer_names[i]);
        
        NN_t *nn = NN_init_with_weight_init(layers, activations, derivatives, 
                                            MSE, MSE_DERIVATIVE, L2, optimizers[i], 0.001f, HE);
        TEST_ASSERT(nn != NULL, "Network creation failed");
        
        // Test forward pass
        const long double *output = NN_forward(nn, simple_inputs);
        TEST_ASSERT(output != NULL, "Forward pass failed");
        
        // Test backward pass
        long double *grad_input = NN_backprop_custom_delta_inputgrad(nn, simple_inputs, simple_targets);
        TEST_ASSERT(grad_input != NULL, "Backward pass failed");
        
        // Test optimizer step
        nn->t = 1; // Set time step for Adam
        nn->optimizer(nn);
        
        if (output) free((void*)output);
        if (grad_input) free(grad_input);
        if (nn) NN_destroy(nn);
    }
}

// Test learning rate scheduling
void test_learning_rate_scheduling() {
    printf("\n=== Testing Learning Rate Scheduling ===\n");
    
    size_t layers[] = {4, 8, 8, 4, 0};
    ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
    ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    
    LRSchedulerType schedulers[] = {CONSTANT, STEP, EXPONENTIAL, COSINE, ONE_CYCLE, WARMUP};
    const char* scheduler_names[] = {"Constant", "Step", "Exponential", "Cosine", "One Cycle", "Warmup"};
    
    for (int i = 0; i < 6; i++) {
        printf("Testing %s scheduler...\n", scheduler_names[i]);
        
        NN_t *nn = NN_init_with_weight_init(layers, activations, derivatives, 
                                            MSE, MSE_DERIVATIVE, L2, ADAM, 0.001f, HE);
        TEST_ASSERT(nn != NULL, "Network creation failed");
        
        // Set up scheduler
        NN_set_lr_scheduler(nn, schedulers[i], 0.001f, 0.0001f, 100);
        
        // Test initial learning rate
        float initial_lr = NN_get_current_lr(nn);
        TEST_ASSERT_CLOSE(initial_lr, 0.001f, 1e-6f, "Initial learning rate incorrect");
        
        // Test learning rate step
        for (int step = 0; step < 10; step++) {
            NN_step_lr(nn);
            float current_lr = NN_get_current_lr(nn);
            printf("  Step %d: LR = %.6f\n", step + 1, current_lr);
        }
        
        if (nn) NN_destroy(nn);
    }
}

// Test enhanced utility functions
void test_enhanced_utilities() {
    printf("\n=== Testing Enhanced Utilities ===\n");
    
    size_t layers[] = {4, 8, 8, 4, 0};
    ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
    ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    
    NN_t *nn = NN_init_with_weight_init(layers, activations, derivatives, 
                                        MSE, MSE_DERIVATIVE, L2, ADAM, 0.001f, HE);
    TEST_ASSERT(nn != NULL, "Network creation failed");
    
    // Test optimizer parameters
    printf("Testing optimizer parameters...\n");
    NN_set_optimizer_params(nn, 0.9f, 0.999f, 1e-8f);
    TEST_ASSERT_CLOSE(nn->beta1, 0.9f, 1e-6f, "Beta1 parameter incorrect");
    TEST_ASSERT_CLOSE(nn->beta2, 0.999f, 1e-6f, "Beta2 parameter incorrect");
    TEST_ASSERT_CLOSE(nn->epsilon, 1e-8f, 1e-10f, "Epsilon parameter incorrect");
    
    // Test weight decay
    printf("Testing weight decay...\n");
    NN_set_weight_decay(nn, 1e-4f);
    TEST_ASSERT_CLOSE(nn->weight_decay, 1e-4f, 1e-6f, "Weight decay parameter incorrect");
    
    // Test dropout
    printf("Testing dropout...\n");
    NN_set_dropout(nn, 0.5f);
    TEST_ASSERT_CLOSE(nn->dropout_rate, 0.5f, 1e-6f, "Dropout rate incorrect");
    
    // Test gradient clipping
    printf("Testing gradient clipping...\n");
    NN_set_gradient_clipping(nn, 2, 1.0f);
    TEST_ASSERT(nn->gradient_clip_type == 2, "Gradient clip type incorrect");
    TEST_ASSERT_CLOSE(nn->gradient_clip_value, 1.0f, 1e-6f, "Gradient clip value incorrect");
    
    // Test monitoring
    printf("Testing monitoring...\n");
    NN_enable_monitoring(nn, 100);
    TEST_ASSERT(nn->loss_history != NULL, "Loss history not allocated");
    TEST_ASSERT(nn->accuracy_history != NULL, "Accuracy history not allocated");
    TEST_ASSERT(nn->history_capacity == 100, "History capacity incorrect");
    
    // Test metrics logging
    NN_log_metrics(nn, 0.5f, 0.8f);
    TEST_ASSERT(nn->history_size == 1, "History size not incremented");
    TEST_ASSERT_CLOSE(nn->loss_history[0], 0.5f, 1e-6f, "Loss logging failed");
    TEST_ASSERT_CLOSE(nn->accuracy_history[0], 0.8f, 1e-6f, "Accuracy logging failed");
    
    // Test model summary
    printf("Testing model summary...\n");
    NN_print_model_summary(nn);
    
    if (nn) NN_destroy(nn);
}

// Test convergence with different methods
void test_convergence_comparison() {
    printf("\n=== Testing Convergence Comparison ===\n");
    
    size_t layers[] = {4, 8, 8, 4, 0};
    ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
    ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    
    WeightInitType methods[] = {ZERO, RANDOM_UNIFORM, HE, ORTHOGONAL};
    const char* method_names[] = {"Zero", "Random Uniform", "He", "Orthogonal"};
    
    for (int i = 0; i < 4; i++) {
        printf("Testing convergence with %s initialization...\n", method_names[i]);
        
        NN_t *nn = NN_init_with_weight_init(layers, activations, derivatives, 
                                            MSE, MSE_DERIVATIVE, L2, ADAM, 0.001f, methods[i]);
        TEST_ASSERT(nn != NULL, "Network creation failed");
        
        // Set up enhanced features
        NN_set_lr_scheduler(nn, COSINE, 0.001f, 0.0001f, 50);
        NN_enable_monitoring(nn, 50);
        
        // Simulate training
        float initial_loss = 10.0f;
        for (int epoch = 0; epoch < 20; epoch++) {
            // Forward pass
            const long double *output = NN_forward(nn, simple_inputs);
            
            // Simple loss calculation
            float loss = 0.0f;
            for (int j = 0; j < 4; j++) {
                long double diff = output[j] - simple_targets[j];
                loss += (float)(diff * diff);
            }
            loss /= 4.0f;
            
            // Backward pass
            long double *grad_input = NN_backprop_custom_delta_inputgrad(nn, simple_inputs, simple_targets);
            
            // Optimizer step
            nn->t = epoch + 1;
            nn->optimizer(nn);
            NN_step_lr(nn);
            
            // Log metrics
            NN_log_metrics(nn, loss, 1.0f - loss / 10.0f);
            
            if (output) free((void*)output);
            if (grad_input) free(grad_input);
            
            printf("  Epoch %d: Loss = %.6f, LR = %.6f, Accuracy = %.6f\n", 
                   epoch + 1, loss, NN_get_current_lr(nn), 1.0f - loss / 10.0f);
            
            // Stop if converged
            if (loss < 0.01f) break;
        }
        
        printf("Final loss: %.6f\n", initial_loss);
        
        if (nn) NN_destroy(nn);
    }
}

// Test memory management
void test_memory_management() {
    printf("\n=== Testing Memory Management ===\n");
    
    size_t layers[] = {4, 8, 8, 4, 0};
    ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
    ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    
    // Test multiple network creation and destruction
    for (int i = 0; i < 10; i++) {
        NN_t *nn = NN_init_with_weight_init(layers, activations, derivatives, 
                                            MSE, MSE_DERIVATIVE, L2, ADAM, 0.001f, HE);
        TEST_ASSERT(nn != NULL, "Network creation failed");
        
        // Test forward/backward passes
        const long double *output = NN_forward(nn, simple_inputs);
        long double *grad_input = NN_backprop_custom_delta_inputgrad(nn, simple_inputs, simple_targets);
        
        TEST_ASSERT(output != NULL, "Forward pass failed");
        TEST_ASSERT(grad_input != NULL, "Backward pass failed");
        
        if (output) free((void*)output);
        if (grad_input) free(grad_input);
        
        NN_destroy(nn);
    }
    
    printf("Successfully created and destroyed 10 networks\n");
}

// Test edge cases
void test_edge_cases() {
    printf("\n=== Testing Edge Cases ===\n");
    
    // Test with single layer
    size_t single_layer[] = {4, 4, 0};
    NN_t *single_nn = NN_init_with_weight_init(single_layer, 
                                             (ActivationFunctionType[]){LINEAR}, 
                                             (ActivationDerivativeType[]){LINEAR_DERIVATIVE}, 
                                             MSE, MSE_DERIVATIVE, L2, ADAM, 0.001f, HE);
    TEST_ASSERT(single_nn != NULL, "Single layer network creation failed");
    
    const long double *output = NN_forward(single_nn, simple_inputs);
    TEST_ASSERT(output != NULL, "Single layer forward pass failed");
    
    long double *grad_input = NN_backprop_custom_delta_inputgrad(single_nn, simple_inputs, simple_targets);
    TEST_ASSERT(grad_input != NULL, "Single layer backward pass failed");
    
    if (output) free((void*)output);
    if (grad_input) free(grad_input);
    NN_destroy(single_nn);
    
    // Test with very small network
    size_t tiny_layers[] = {2, 2, 0};
    NN_t *tiny_nn = NN_init_with_weight_init(tiny_layers, 
                                           (ActivationFunctionType[]){LINEAR}, 
                                           (ActivationDerivativeType[]){LINEAR_DERIVATIVE}, 
                                           MSE, MSE_DERIVATIVE, L2, ADAM, 0.001f, HE);
    TEST_ASSERT(tiny_nn != NULL, "Tiny network creation failed");
    
    const long double *tiny_output = NN_forward(tiny_nn, simple_inputs);
    TEST_ASSERT(tiny_output != NULL, "Tiny network forward pass failed");
    
    long double *tiny_grad_input = NN_backprop_custom_delta_inputgrad(tiny_nn, simple_inputs, simple_targets);
    TEST_ASSERT(tiny_grad_input != NULL, "Tiny network backward pass failed");
    
    if (tiny_output) free((void*)tiny_output);
    if (tiny_grad_input) free(tiny_grad_input);
    NN_destroy(tiny_nn);
    
    printf("Edge cases passed\n");
}

// Main test runner
int main(void) {
    printf("=== Enhanced Neural Network Framework Tests ===\n");
    
    test_weight_initialization();
    test_advanced_optimizers();
    test_learning_rate_scheduling();
    test_enhanced_utilities();
    test_convergence_comparison();
    test_memory_management();
    test_edge_cases();
    
    printf("\n=== Test Results ===\n");
    printf("✓ Passed: %d\n", test_passed);
    printf("✗ Failed: %d\n", test_failed);
    printf("Total: %d\n", test_passed + test_failed);
    
    if (test_failed == 0) {
        printf("\n🎉 All tests passed! Enhanced NN framework is working correctly.\n");
        return 0;
    } else {
        printf("\n⚠️  Some tests failed. Please check the implementation.\n");
        return 1;
    }
}
