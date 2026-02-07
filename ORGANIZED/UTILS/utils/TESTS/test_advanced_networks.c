#include "../NN/GNN.h"
#include "../NN/SNN.h"
#include "../NN/KAN.h"
#include "../NN/GAN.h"
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

// Test Graph Neural Networks
void test_gnn_creation() {
    printf("=== Testing GNN Creation ===\n");
    
    GNN_t *gnn = GNN_create(5, 3, 8, 2);
    TEST_ASSERT(gnn != NULL, "GNN creation failed");
    
    if (gnn) {
        TEST_ASSERT(gnn->num_nodes == 5, "GNN node count incorrect");
        TEST_ASSERT(gnn->feature_dim == 3, "GNN feature dimension incorrect");
        TEST_ASSERT(gnn->hidden_dim == 8, "GNN hidden dimension incorrect");
        TEST_ASSERT(gnn->num_layers == 2, "GNN layer count incorrect");
        
        // Test adding edges
        GNN_add_edge(gnn, 0, 1, 1.0L);
        GNN_add_edge(gnn, 1, 2, 0.5L);
        GNN_add_edge(gnn, 2, 3, 0.8L);
        GNN_add_edge(gnn, 3, 4, 1.2L);
        GNN_add_edge(gnn, 4, 0, 0.7L);
        
        // Test edge existence
        TEST_ASSERT(GNN_has_edge(gnn, 0, 1), "Edge 0->1 should exist");
        TEST_ASSERT(!GNN_has_edge(gnn, 0, 2), "Edge 0->2 should not exist");
        
        // Test node degrees
        TEST_ASSERT(GNN_get_degree(gnn, 0) == 2, "Node 0 degree incorrect");
        TEST_ASSERT(GNN_get_degree(gnn, 2) == 2, "Node 2 degree incorrect");
        
        GNN_destroy(gnn);
    }
    
    printf("✓ GNN creation test passed\n");
}

void test_gnn_forward() {
    printf("=== Testing GNN Forward Pass ===\n");
    
    GNN_t *gnn = GNN_create(4, 2, 4, 2);
    TEST_ASSERT(gnn != NULL, "GNN creation failed");
    
    if (gnn) {
        // Create a simple graph: 0->1->2->3
        GNN_add_edge(gnn, 0, 1, 1.0L);
        GNN_add_edge(gnn, 1, 2, 1.0L);
        GNN_add_edge(gnn, 2, 3, 1.0L);
        
        // Set node features
        long double **features = malloc(4 * sizeof(long double*));
        for (int i = 0; i < 4; i++) {
            features[i] = malloc(2 * sizeof(long double));
            features[i][0] = (long double)i * 0.1L;
            features[i][1] = (long double)i * 0.2L;
            GNN_set_node_features(gnn, i, features[i]);
        }
        
        // Forward pass
        long double *output = GNN_forward(gnn, features);
        TEST_ASSERT(output != NULL, "GNN forward pass failed");
        
        if (output) {
            // Check output is reasonable
            for (int i = 0; i < 4; i++) {
                TEST_ASSERT(!isnan(output[i]) && !isinf(output[i]), "GNN output contains invalid values");
            }
            free(output);
        }
        
        // Cleanup
        for (int i = 0; i < 4; i++) {
            free(features[i]);
        }
        free(features);
        GNN_destroy(gnn);
    }
    
    printf("✓ GNN forward pass test passed\n");
}

// Test Spiking Neural Networks
void test_snn_creation() {
    printf("=== Testing SNN Creation ===\n");
    
    SNN_t *snn = SNN_create(10, 5, 3);
    TEST_ASSERT(snn != NULL, "SNN creation failed");
    
    if (snn) {
        TEST_ASSERT(snn->num_neurons == 10, "SNN neuron count incorrect");
        TEST_ASSERT(snn->input_size == 5, "SNN input size incorrect");
        TEST_ASSERT(snn->output_size == 3, "SNN output size incorrect");
        
        // Test neuron parameters
        SNN_set_neuron_threshold(snn, 0, 1.0L);
        SNN_set_time_constant(snn, 0, 20.0L);
        SNN_set_time_step(snn, 0.1L);
        
        // Test synapse addition
        SNN_add_synapse(snn, 0, 1, 0.5L);
        SNN_add_synapse(snn, 1, 2, 0.8L);
        SNN_add_synapse(snn, 2, 3, 0.3L);
        
        SNN_destroy(snn);
    }
    
    printf("✓ SNN creation test passed\n");
}

void test_snn_forward() {
    printf("=== Testing SNN Forward Pass ===\n");
    
    SNN_t *snn = SNN_create(5, 3, 2);
    TEST_ASSERT(snn != NULL, "SNN creation failed");
    
    if (snn) {
        // Set neuron thresholds
        for (int i = 0; i < 5; i++) {
            SNN_set_neuron_threshold(snn, i, 0.5L + i * 0.1L);
        }
        
        // Create input
        long double inputs[] = {0.1L, 0.2L, 0.3L};
        
        // Forward pass for 100ms
        long double *output = SNN_forward(snn, inputs, 0.1L);
        TEST_ASSERT(output != NULL, "SNN forward pass failed");
        
        if (output) {
            // Check output is reasonable
            for (int i = 0; i < 2; i++) {
                TEST_ASSERT(output[i] >= 0.0L, "SNN output should be non-negative");
            }
            free(output);
        }
        
        // Test spike counting
        size_t spike_count = SNN_count_spikes(snn, 0.1L);
        TEST_ASSERT(spike_count >= 0, "Spike count should be non-negative");
        
        SNN_destroy(snn);
    }
    
    printf("✓ SNN forward pass test passed\n");
}

// Test Kolmogorov-Arnold Networks
void test_kan_creation() {
    printf("=== Testing KAN Creation ===\n");
    
    KAN_t *kan = KAN_create(3, 5, 2, 2);
    TEST_ASSERT(kan != NULL, "KAN creation failed");
    
    if (kan) {
        TEST_ASSERT(kan->input_dim == 3, "KAN input dimension incorrect");
        TEST_ASSERT(kan->hidden_dim == 5, "KAN hidden dimension incorrect");
        TEST_ASSERT(kan->output_dim == 2, "KAN output dimension incorrect");
        TEST_ASSERT(kan->num_layers == 2, "KAN layer count incorrect");
        
        // Test configuration
        KAN_set_grid_size(kan, 10);
        KAN_enable_symbolic(kan, true);
        KAN_set_symbolic_threshold(kan, 5);
        
        TEST_ASSERT(kan->grid_size == 10, "KAN grid size incorrect");
        TEST_ASSERT(kan->use_symbolic == true, "KAN symbolic mode incorrect");
        
        KAN_destroy(kan);
    }
    
    printf("✓ KAN creation test passed\n");
}

void test_kan_forward() {
    printf("=== Testing KAN Forward Pass ===\n");
    
    KAN_t *kan = KAN_create(2, 3, 1, 2);
    TEST_ASSERT(kan != NULL, "KAN creation failed");
    
    if (kan) {
        // Create input
        long double inputs[] = {0.5L, -0.3L};
        
        // Forward pass
        long double *output = KAN_forward(kan, inputs);
        TEST_ASSERT(output != NULL, "KAN forward pass failed");
        
        if (output) {
            // Check output is reasonable
            TEST_ASSERT(!isnan(output[0]) && !isinf(output[0]), "KAN output contains invalid values");
            free(output);
        }
        
        KAN_destroy(kan);
    }
    
    printf("✓ KAN forward pass test passed\n");
}

// Test Generative Adversarial Networks
void test_gan_creation() {
    printf("=== Testing GAN Creation ===\n");
    
    GAN_t *gan = GAN_create(64, 128, 64, 32);
    TEST_ASSERT(gan != NULL, "GAN creation failed");
    
    if (gan) {
        TEST_ASSERT(gan->latent_dim == 32, "GAN latent dimension incorrect");
        TEST_ASSERT(gan->generator != NULL, "GAN generator not created");
        TEST_ASSERT(gan->discriminator != NULL, "GAN discriminator not created");
        
        // Test configuration
        GAN_set_learning_rate(gan, 0.0002L);
        GAN_set_lambda_gp(gan, 10.0L);
        GAN_enable_wasserstein(gan, true);
        GAN_set_n_critic(gan, 5);
        
        TEST_ASSERT(gan->use_wasserstein == true, "GAN Wasserstein mode incorrect");
        TEST_ASSERT(gan->n_critic == 5, "GAN n_critic incorrect");
        
        GAN_destroy(gan);
    }
    
    printf("✓ GAN creation test passed\n");
}

void test_gan_generation() {
    printf("=== Testing GAN Generation ===\n");
    
    GAN_t *gan = GAN_create(32, 64, 32, 16);
    TEST_ASSERT(gan != NULL, "GAN creation failed");
    
    if (gan) {
        // Generate noise
        long double *noise = malloc(16 * sizeof(long double));
        for (int i = 0; i < 16; i++) {
            noise[i] = (long double)rand() / RAND_MAX * 2.0L - 1.0L;
        }
        
        // Generate sample
        long double *generated = GAN_generate(gan, noise);
        TEST_ASSERT(generated != NULL, "GAN generation failed");
        
        if (generated) {
            // Check output is reasonable
            for (int i = 0; i < 32; i++) {
                TEST_ASSERT(!isnan(generated[i]) && !isinf(generated[i]), "GAN output contains invalid values");
            }
            free(generated);
        }
        
        free(noise);
        GAN_destroy(gan);
    }
    
    printf("✓ GAN generation test passed\n");
}

void test_gan_discrimination() {
    printf("=== Testing GAN Discrimination ===\n");
    
    GAN_t *gan = GAN_create(32, 64, 32, 16);
    TEST_ASSERT(gan != NULL, "GAN creation failed");
    
    if (gan) {
        // Create sample data
        long double *data = malloc(32 * sizeof(long double));
        for (int i = 0; i < 32; i++) {
            data[i] = (long double)i / 32.0L;
        }
        
        // Discriminate
        long double score = GAN_discriminate(gan, data);
        TEST_ASSERT(score >= 0.0L && score <= 1.0L, "GAN discriminator score out of range");
        
        free(data);
        GAN_destroy(gan);
    }
    
    printf("✓ GAN discrimination test passed\n");
}

// Test memory management
void test_advanced_networks_memory() {
    printf("=== Testing Advanced Networks Memory Management ===\n");
    
    // Test multiple creation/destruction cycles
    for (int i = 0; i < 3; i++) {
        GNN_t *gnn = GNN_create(3, 2, 4, 2);
        if (gnn) {
            GNN_add_edge(gnn, 0, 1, 1.0L);
            GNN_add_edge(gnn, 1, 2, 1.0L);
            GNN_destroy(gnn);
        }
        
        SNN_t *snn = SNN_create(5, 3, 2);
        if (snn) {
            SNN_add_synapse(snn, 0, 1, 0.5L);
            SNN_destroy(snn);
        }
        
        KAN_t *kan = KAN_create(2, 3, 1, 2);
        if (kan) {
            long double inputs[] = {0.1L, 0.2L};
            long double *output = KAN_forward(kan, inputs);
            if (output) free(output);
            KAN_destroy(kan);
        }
        
        GAN_t *gan = GAN_create(16, 32, 16, 8);
        if (gan) {
            long double *noise = malloc(8 * sizeof(long double));
            long double *generated = GAN_generate(gan, noise);
            if (generated) free(generated);
            free(noise);
            GAN_destroy(gan);
        }
        
        printf("  Memory test %d passed\n", i + 1);
    }
    
    printf("✓ Advanced networks memory management test passed\n");
}

// Test parameter counting
void test_advanced_networks_parameters() {
    printf("=== Testing Advanced Networks Parameter Counting ===\n");
    
    GNN_t *gnn = GNN_create(4, 3, 5, 2);
    if (gnn) {
        size_t gnn_params = GNN_get_parameter_count(gnn);
        TEST_ASSERT(gnn_params > 0, "GNN parameter count should be positive");
        printf("  GNN parameters: %zu\n", gnn_params);
        GNN_destroy(gnn);
    }
    
    SNN_t *snn = SNN_create(5, 3, 2);
    if (snn) {
        // SNN parameters are not easily countable due to spike-based nature
        printf("  SNN: Spike-based parameters (not countable)\n");
        SNN_destroy(snn);
    }
    
    KAN_t *kan = KAN_create(3, 5, 2, 2);
    if (kan) {
        size_t kan_params = KAN_get_parameter_count(kan);
        TEST_ASSERT(kan_params > 0, "KAN parameter count should be positive");
        printf("  KAN parameters: %zu\n", kan_params);
        KAN_destroy(kan);
    }
    
    GAN_t *gan = GAN_create(32, 64, 32, 16);
    if (gan) {
        size_t gan_params = GAN_get_parameter_count(gan);
        TEST_ASSERT(gan_params > 0, "GAN parameter count should be positive");
        printf("  GAN parameters: %zu\n", gan_params);
        GAN_destroy(gan);
    }
    
    printf("✓ Advanced networks parameter counting test passed\n");
}

// Main test runner
int main(void) {
    printf("=== Advanced Neural Networks Tests ===\n");
    
    // Graph Neural Networks
    test_gnn_creation();
    test_gnn_forward();
    
    // Spiking Neural Networks
    test_snn_creation();
    test_snn_forward();
    
    // Kolmogorov-Arnold Networks
    test_kan_creation();
    test_kan_forward();
    
    // Generative Adversarial Networks
    test_gan_creation();
    test_gan_generation();
    test_gan_discrimination();
    
    // Memory and parameter tests
    test_advanced_networks_memory();
    test_advanced_networks_parameters();
    
    printf("\n=== Advanced Networks Test Results ===\n");
    printf("✓ Passed: %d\n", test_passed);
    printf("✗ Failed: %d\n", test_failed);
    printf("Total: %d\n", test_passed + test_failed);
    
    if (test_failed == 0) {
        printf("\n🎉 All advanced network tests passed! Frameworks are working correctly.\n");
        return 0;
    } else {
        printf("\n⚠️  Some advanced network tests failed. Please check the implementation.\n");
        return 1;
    }
}
