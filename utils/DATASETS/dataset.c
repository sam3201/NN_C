#include "../DATASETS/dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

Dataset* create_empty_dataset(void) {
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) return NULL;
    
    dataset->num_samples = 0;
    dataset->inputs = NULL;
    dataset->targets = NULL;
    
    return dataset;
}

void add_to_dataset(Dataset* dataset, const long double* input, const long double* target) {
    if (!dataset) return;
    
    // Reallocate memory for new sample
    dataset->inputs = (long double**)realloc(dataset->inputs, (dataset->num_samples + 1) * sizeof(long double*));
    dataset->targets = (long double**)realloc(dataset->targets, (dataset->num_samples + 1) * sizeof(long double*));
    
    if (!dataset->inputs || !dataset->targets) {
        fprintf(stderr, "Failed to allocate memory for dataset\n");
        return;
    }
    
    // Allocate memory for input and target data
    dataset->inputs[dataset->num_samples] = (long double*)malloc(sizeof(long double));
    dataset->targets[dataset->num_samples] = (long double*)malloc(sizeof(long double));
    
    if (!dataset->inputs[dataset->num_samples] || !dataset->targets[dataset->num_samples]) {
        fprintf(stderr, "Failed to allocate memory for sample data\n");
        return;
    }
    
    // Copy input and target data
    memcpy(dataset->inputs[dataset->num_samples], input, sizeof(long double));
    memcpy(dataset->targets[dataset->num_samples], target, sizeof(long double));
    
    dataset->num_samples++;
}

void destroy_dataset(Dataset* dataset) {
    if (!dataset) return;
    
    for (size_t i = 0; i < dataset->num_samples; i++) {
        free(dataset->inputs[i]);
        free(dataset->targets[i]);
    }
    free(dataset->inputs);
    free(dataset->targets);
    free(dataset);
}

long double* load_input(Dataset* dataset, size_t index) {
    if (!dataset || index >= dataset->num_samples) return NULL;
    
    long double* input = malloc(sizeof(long double));
    if (!input) return NULL;
    
    memcpy(input, dataset->inputs[index], sizeof(long double));
    return input;
}

long double* load_target(Dataset* dataset, size_t index) {
    if (!dataset || index >= dataset->num_samples) return NULL;
    
    long double* target = malloc(sizeof(long double));
    if (!target) return NULL;
    
    memcpy(target, dataset->targets[index], sizeof(long double));
    return target;
}

long double** load_input_batch(Dataset* dataset, size_t batch_index, size_t batch_size) {
    if (!dataset) return NULL;
    
    size_t start_idx = batch_index * batch_size;
    if (start_idx + batch_size > dataset->num_samples) return NULL;
    
    long double** input_batch = malloc(batch_size * sizeof(long double*));
    if (!input_batch) return NULL;
    
    for (size_t i = 0; i < batch_size; i++) {
        input_batch[i] = load_input(dataset, start_idx + i);
        if (!input_batch[i]) {
            // Cleanup on failure
            for (size_t j = 0; j < i; j++) {
                free(input_batch[j]);
            }
            free(input_batch);
            return NULL;
        }
    }
    
    return input_batch;
}

long double** load_target_batch(Dataset* dataset, size_t batch_index, size_t batch_size) {
    if (!dataset) return NULL;
    
    size_t start_idx = batch_index * batch_size;
    if (start_idx + batch_size > dataset->num_samples) return NULL;
    
    long double** target_batch = malloc(batch_size * sizeof(long double*));
    if (!target_batch) return NULL;
    
    for (size_t i = 0; i < batch_size; i++) {
        target_batch[i] = load_target(dataset, start_idx + i);
        if (!target_batch[i]) {
            // Cleanup on failure
            for (size_t j = 0; j < i; j++) {
                free(target_batch[j]);
            }
            free(target_batch);
            return NULL;
        }
    }
    
    return target_batch;
}
