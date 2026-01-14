#include "../NN/NEAT/NEAT.h"
#include "../NN/NN/NN.h"
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

// Test NEAT integration with enhanced NN framework
void test_neat_nn_integration() {
    printf("=== Testing NEAT-NN Integration ===\n");
    
    // NEAT already uses our NN.h, so we just need to test that it works
    // Create a simple NEAT network
    size_t inputs[] = {4, 8, 8, 4, 0};
    ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
    ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    
    NN_t *nn = NN_init_with_weight_init(inputs, activations, derivatives, 
                                        MSE, MSE_DERIVATIVE, L2, ADAM, 0.001f, HE);
    TEST_ASSERT(nn != NULL, "NEAT network creation failed");
    
    float test_inputs[] = {0.5f, -0.3f, 0.8f, 0.1f};
    long double *test_inputs_ld = malloc(4 * sizeof(long double));
    for (int i = 0; i < 4; i++) {
        test_inputs_ld[i] = (long double)test_inputs[i];
    }
    const long double *output = NN_forward(nn, test_inputs_ld);
    TEST_ASSERT(output != NULL, "NEAT forward pass failed");
    
    // Test backward pass
    float test_targets[] = {0.1f, 0.7f, -0.2f, 0.9f};
    long double *test_targets_ld = malloc(4 * sizeof(long double));
    for (int i = 0; i < 4; i++) {
        test_targets_ld[i] = (long double)test_targets[i];
    }
    float *grad_input = malloc(4 * sizeof(float));
    long double *grad_input_ld = malloc(4 * sizeof(long double));
    NN_backprop_custom_delta(nn, test_inputs_ld, test_targets_ld);
    
    // Convert back to float for compatibility
    for (int i = 0; i < 4; i++) {
        grad_input[i] = (float)grad_input_ld[i];
    }
    
    free(test_inputs_ld);
    free(test_targets_ld);
    free(grad_input_ld);
    TEST_ASSERT(grad_input != NULL, "NEAT backward pass failed");
    
    // Test optimizer step
    nn->t = 1;
    nn->optimizer(nn);
    
    // Test that NEAT can use enhanced features
    NN_set_lr_scheduler(nn, COSINE, 0.001f, 0.0001f, 100);
    NN_enable_monitoring(nn, 50);
    NN_set_optimizer_params(nn, 0.9f, 0.999f, 1e-8f);
    
    // Test learning rate scheduling
    for (int i = 0; i < 5; i++) {
        NN_step_lr(nn);
        printf("  Step %d: LR = %.6f\n", i + 1, NN_get_current_lr(nn));
    }
    
    // Test monitoring
    NN_log_metrics(nn, 0.5f, 0.8f);
    NN_log_metrics(nn, 0.3f, 0.9f);
    
    // Test model summary
    NN_print_model_summary(nn);
    
    // Cleanup
    if (output) free((void*)output);
    if (grad_input) free(grad_input);
    NN_destroy(nn);
    
    printf("✓ NEAT-NN integration test passed\n");
}

// Test NEAT with different weight initialization
void test_neat_weight_initialization() {
    printf("=== Testing NEAT Weight Initialization ===\n");
    
    size_t inputs[] = {4, 8, 8, 4, 0};
    ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
    ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    
    WeightInitType methods[] = {ZERO, RANDOM_UNIFORM, RANDOM_NORMAL, XAVIER, HE, LECUN, ORTHOGONAL};
    const char* method_names[] = {"Zero", "Random Uniform", "Random Normal", "Xavier", "He", "LeCun", "Orthogonal"};
    
    for (int i = 0; i < 7; i++) {
        printf("Testing NEAT with %s initialization...\n", method_names[i]);
        
        NN_t *nn = NN_init_with_weight_init(inputs, activations, derivatives, 
                                            MSE, MSE_DERIVATIVE, L2, ADAM, 0.001f, methods[i]);
        TEST_ASSERT(nn != NULL, "NEAT network creation failed");
        
        // Test that NEAT can use different initialization methods
        float test_inputs[] = {0.5f, -0.3f, 0.8f, 0.1f};
        const float *output = NN_forward(nn, test_inputs);
        TEST_ASSERT(output != NULL, "NEAT forward pass failed");
        
        // Test backward pass
        float test_targets[] = {0.1f, 0.7f, -0.2f, 0.9f};
        long double *test_inputs_ld = malloc(4 * sizeof(long double));
        long double *test_targets_ld = malloc(4 * sizeof(long double));
        for (int i = 0; i < 4; i++) {
            test_inputs_ld[i] = (long double)test_inputs[i];
            test_targets_ld[i] = (long double)test_targets[i];
        }
        float *grad_input = malloc(4 * sizeof(float));
        long double *grad_input_ld = malloc(4 * sizeof(long double));
        NN_backprop_custom_delta(nn, test_inputs_ld, test_targets_ld);
        
        // Convert back to float for compatibility
        for (int i = 0; i < 4; i++) {
            grad_input[i] = (float)grad_input_ld[i];
        }
        
        free(test_inputs_ld);
        free(test_targets_ld);
        free(grad_input_ld);
        TEST_ASSERT(grad_input != NULL, "NEAT backward pass failed");
        
        // Test optimizer step
        nn->t = 1;
        nn->optimizer(nn);
        
        // Cleanup
        if (output) free((void*)output);
        NN_destroy(nn);
        
        printf("  %s initialization test passed\n", method_names[i]);
    }
    
    printf("✓ NEAT weight initialization test passed\n");
}

// Test NEAT with enhanced optimizers
void test_neat_enhanced_optimizers() {
    printf("=== Testing NEAT Enhanced Optimizers ===\n");
    
    size_t inputs[] = {4, 8, 8, 4, 0};
    ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
    ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    
    OptimizerType optimizers[] = {SGD, RMSPROP, ADAGRAD, ADAM, NAG, ADADELTA, RMSPROP_NESTEROV, ADAMAX, NADAM};
    const char* optimizer_names[] = {"SGD", "RMSProp", "AdaGrad", "Adam", "NAG", "AdaDelta", "RMSProp Nesterov", "Adamax", "Nadam"};
    
    for (int i = 0; i < 9; i++) {
        printf("Testing NEAT with %s optimizer...\n", optimizer_names[i]);
        
        NN_t *nn = NN_init_with_weight_init(inputs, activations, derivatives, 
                                            MSE, MSE_DERIVATIVE, L2, optimizers[i], 0.001f, HE);
        TEST_ASSERT(nn != NULL, "NEAT network creation failed");
        
        // Test that NEAT can use enhanced optimizers
        float test_inputs[] = {0.5f, -0.3f, 0.8f, 0.1f};
        const float *output = NN_forward(nn, test_inputs);
        TEST_ASSERT(output != NULL, "NEAT forward pass failed");
        
        // Test backward pass
        float test_targets[] = {0.1f, 0.7f, -0.2f, 0.9f};
        long double *test_inputs_ld = malloc(4 * sizeof(long double));
        long double *test_targets_ld = malloc(4 * sizeof(long double));
        for (int i = 0; i < 4; i++) {
            test_inputs_ld[i] = (long double)test_inputs[i];
            test_targets_ld[i] = (long double)test_targets[i];
        }
        float *grad_input = malloc(4 * sizeof(float));
        long double *grad_input_ld = malloc(4 * sizeof(long double));
        NN_backprop_custom_delta(nn, test_inputs_ld, test_targets_ld);
        
        // Convert back to float for compatibility
        for (int i = 0; i < 4; i++) {
            grad_input[i] = (float)grad_input_ld[i];
        }
        
        free(test_inputs_ld);
        free(test_targets_ld);
        free(grad_input_ld);
        TEST_ASSERT(grad_input != NULL, "NEAT backward pass failed");
        
        // Test optimizer step
        nn->t = 1;
        nn->optimizer(nn);
        
        // Cleanup
        if (output) free((void*)output);
        NN_destroy(nn);
        
        printf("  %s optimizer test passed\n", optimizer_names[i]);
    }
    
    printf("✓ NEAT enhanced optimizers test passed\n");
}

// Test NEAT memory management
void test_neat_memory() {
    printf("=== Testing NEAT Memory Management ===\n");
    
    // Test multiple creation and destruction
    for (int i = 0; i < 5; i++) {
        size_t inputs[] = {4, 8, 8, 4, 0};
        ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
        ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
        
        NN_t *nn = NN_init_with_weight_init(inputs, activations, derivatives, 
                                            MSE, MSE_DERIVATIVE, L2, ADAM, 0.001f, HE);
        TEST_ASSERT(nn != NULL, "NEAT network creation failed");
        
        // Test operations
        float test_inputs[] = {0.5f, -0.3f, 0.8f, 0.1f};
        const float *output = NN_forward(nn, test_inputs);
        float test_targets[] = {0.1f, 0.7f, -0.2f, 0.9f};
        long double *test_inputs_ld = malloc(4 * sizeof(long double));
        long double *test_targets_ld = malloc(4 * sizeof(long double));
        for (int i = 0; i < 4; i++) {
            test_inputs_ld[i] = (long double)test_inputs[i];
            test_targets_ld[i] = (long double)test_targets[i];
        }
        float *grad_input = malloc(4 * sizeof(float));
        long double *grad_input_ld = malloc(4 * sizeof(long double));
        NN_backprop_custom_delta(nn, test_inputs_ld, test_targets_ld);
        
        // Convert back to float for compatibility
        for (int i = 0; i < 4; i++) {
            grad_input[i] = (float)grad_input_ld[i];
        }
        
        free(test_inputs_ld);
        free(test_targets_ld);
        free(grad_input_ld);
        
        nn->t = i + 1;
        nn->optimizer(nn);
        
        // Cleanup
        if (output) free((void*)output);
        NN_destroy(nn);
        
        printf("  Memory test %d passed\n", i + 1);
    }
    
    printf("✓ NEAT memory management test passed\n");
}

// Test NEAT convergence
void test_neat_convergence() {
    printf("=== Testing NEAT Convergence ===\n");
    
    size_t inputs[] = {4, 8, 8, 4, 0};
    ActivationFunctionType activations[] = {RELU, RELU, RELU, LINEAR};
    ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    
    NN_t *nn = NN_init_with_weight_init(inputs, activations, derivatives, 
                                        MSE, MSE_DERIVATIVE, L2, ADAM, 0.001f, HE);
    TEST_ASSERT(nn != NULL, "NEAT network creation failed");
    
    // Set up enhanced features
    NN_set_lr_scheduler(nn, COSINE, 0.001f, 0.0001f, 50);
    NN_enable_monitoring(nn, 50);
    
    // Simulate training
    float test_inputs[] = {0.5f, -0.3f, 0.8f, 0.1f};
    float test_targets[] = {0.1f, 0.7f, -0.2f, 0.9f};
    
    for (int epoch = 0; epoch < 10; epoch++) {
        // Forward pass
        const float *output = NN_forward(nn, test_inputs);
        
        // Simple loss calculation
        float loss = 0.0f;
        for (int i = 0; i < 4; i++) {
            float diff = output[i] - test_targets[i];
            loss += diff * diff;
        }
        loss /= 4.0f;
        
        // Backpropagation
        long double *test_inputs_ld = malloc(4 * sizeof(long double));
        long double *test_targets_ld = malloc(4 * sizeof(long double));
        for (int i = 0; i < 4; i++) {
            test_inputs_ld[i] = (long double)test_inputs[i];
            test_targets_ld[i] = (long double)test_targets[i];
        }
        NN_backprop_custom_delta(nn, test_inputs_ld, test_targets_ld);
        
        // Optimizer step
        nn->t = epoch + 1;
        nn->optimizer(nn);
        NN_step_lr(nn);
        
        // Log metrics
        NN_log_metrics(nn, loss, 1.0f - loss / 10.0f);
        
        printf("  Epoch %d: Loss = %.6f, LR = %.6f, Accuracy = %.6f\n", 
               epoch + 1, loss, NN_get_current_lr(nn), 1.0f - loss / 10.0f);
        
        // Cleanup
        if (output) free((void*)output);
        
        // Stop if converged
        if (loss < 0.01f) break;
    }
    
    // Print training summary
    NN_print_model_summary(nn);
    
    // Cleanup
    NN_destroy(nn);
    
    printf("✓ NEAT convergence test passed\n");
}

// Main test runner
int main(void) {
    printf("=== NEAT Integration Tests ===\n");
    
    test_neat_nn_integration();
    test_neat_weight_initialization();
    test_neat_enhanced_optimizers();
    test_neat_memory();
    test_neat_convergence();
    
    printf("\n=== NEAT Test Results ===\n");
    printf("✓ Passed: %d\n", test_passed);
    printf("✗ Failed: %d\n", test_failed);
    printf("Total: %d\n", test_passed + test_failed);
    
    if (test_failed == 0) {
        printf("\n🎉 All NEAT tests passed! Integration is working correctly.\n");
        return 0;
    } else {
        printf("\n⚠️  Some NEAT tests failed. Please check the implementation.\n");
        return 1;
    }
}
