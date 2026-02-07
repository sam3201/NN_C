#ifndef DATASET_H
#define DATASET_H

#include <stdlib.h>

typedef struct {
    size_t num_samples;
    long double** inputs;
    long double** targets;
} Dataset;

// Dataset management functions
Dataset* create_empty_dataset(void);
void add_to_dataset(Dataset* dataset, const long double* input, const long double* target);
void destroy_dataset(Dataset* dataset);

// Data loading functions
long double* load_input(Dataset* dataset, size_t index);
long double* load_target(Dataset* dataset, size_t index);
long double** load_input_batch(Dataset* dataset, size_t batch_index, size_t batch_size);
long double** load_target_batch(Dataset* dataset, size_t batch_index, size_t batch_size);

#endif // DATASET_H
