#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../DATASETS/dataset.h"
#include "../NN/TOKENIZER.h"

int main() {
    // Initialize random seed
    srand(time(NULL));

    // Create tokenizer and build vocabulary
    printf("Creating tokenizer...\n");
    Tokenizer* tokenizer = create_tokenizer("datasets/sample.txt");
    if (!tokenizer) {
        fprintf(stderr, "Failed to create tokenizer\n");
        return 1;
    }
    printf("Vocabulary size: %zu\n", tokenizer->vocab_size);

    // Create dataset
    printf("Loading dataset...\n");
    Dataset* dataset = create_masked_word_dataset("datasets/sample.txt", tokenizer);
    if (!dataset) {
        fprintf(stderr, "Failed to load dataset\n");
        free_tokenizer(tokenizer);
        return 1;
    }
    printf("Number of samples: %zu\n", dataset->num_samples);

    // Print some examples
    printf("\nExample samples:\n");
    for (size_t i = 0; i < 5 && i < dataset->num_samples; i++) {
        printf("\nSample %zu:\n", i + 1);
        printf("Input  : %s\n", tokenizer_decode(tokenizer, dataset->inputs[i]));
        printf("Target : %s\n", tokenizer_decode(tokenizer, dataset->targets[i]));
    }

    // Clean up
    free_dataset(dataset);
    free_tokenizer(tokenizer);

    return 0;
}
