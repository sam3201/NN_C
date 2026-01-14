#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Test runner for all NN framework tests
int main(int argc, char *argv[]) {
    printf("=== Comprehensive NN Framework Test Suite ===\n\n");
    
    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;
    
    // Test configurations
    struct {
        const char* name;
        const char* executable;
        const char* description;
    } tests[] = {
        {"Enhanced NN Framework", "./test_nn_enhanced", "Test enhanced NN features"},
        {"CONVOLUTION Integration", "./test_convolution_integration", "Test CONVOLUTION integration"},
        {"TRANSFORMER Integration", "./test_transformer_integration", "Test TRANSFORMER integration"},
        {"RNN/LSTM Support", "./test_rnn_lstm", "Test RNN/LSTM functionality"},
        {"NEAT Framework", "./test_neat_integration", "Test NEAT integration"},
        {"MUZE Framework", "./test_muze_integration", "Test MUZE integration"},
        {"Memory Framework", "./test_memory_integration", "Test MEMORY integration"},
        {"Tokenizer Framework", "./test_tokenizer_integration", "Test TOKENIZER integration"},
    };
    
    int num_tests = sizeof(tests) / sizeof(tests[0]);
    
    // Build all tests
    printf("Building all tests...\n");
    for (int i = 0; i < num_tests; i++) {
        printf("Building %s...\n", tests[i].name);
        
        char build_cmd[256];
        snprintf(build_cmd, sizeof(build_cmd), 
                "gcc -o %s %s.c ../NN/NN.c ../NN/CONVOLUTION.c ../NN/TRANSFORMER.c ../NN/NEAT.c ../NN/TOKENIZER.c -I../NN -lm -std=c99", 
                tests[i].executable, tests[i].executable);
        
        int result = system(build_cmd);
        if (result != 0) {
            printf("✗ Failed to build %s\n", tests[i].name);
            failed_tests++;
        } else {
            printf("✓ Built %s\n", tests[i].name);
        }
    }
    
    printf("\nRunning all tests...\n");
    
    // Run all tests
    for (int i = 0; i < num_tests; i++) {
        printf("\n=== Running %s ===\n", tests[i].name);
        printf("Description: %s\n", tests[i].description);
        
        // Check if executable exists
        if (access(tests[i].executable, X_OK) != 0) {
            printf("✗ Test executable not found: %s\n", tests[i].executable);
            failed_tests++;
            total_tests++;
            continue;
        }
        
        // Run the test
        int result = system(tests[i].executable);
        
        total_tests++;
        if (result == 0) {
            printf("✓ %s PASSED\n", tests[i].name);
            passed_tests++;
        } else {
            printf("✗ %s FAILED\n", tests[i].name);
            failed_tests++;
        }
    }
    
    // Print summary
    printf("\n=== Test Suite Summary ===\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", failed_tests);
    printf("Success rate: %.1f%%\n", total_tests > 0 ? (float)passed_tests / total_tests * 100.0f : 0.0f);
    
    if (failed_tests == 0) {
        printf("\n🎉 All tests passed! The NN framework is working correctly.\n");
        return 0;
    } else {
        printf("\n⚠️  Some tests failed. Please check the implementation.\n");
        return 1;
    }
}
