#define POP_SIZE 50
#define INPUT_DIM 5
#define OUTPUT_DIM 2
#define NUM_SAMPLES 10
#define GENERATIONS 100

int main() {
  NEAT_t *neat = NEAT_init(INPUT_DIM, OUTPUT_DIM, POP_SIZE);

  // Example input/target arrays (replace with your simulation data)
  long double *inputs[NUM_SAMPLES];
  long double *targets[NUM_SAMPLES];
  for (size_t i = 0; i < NUM_SAMPLES; i++) {
    inputs[i] = malloc(INPUT_DIM * sizeof(long double));
    targets[i] = malloc(OUTPUT_DIM * sizeof(long double));

    // Fill with random example data
    for (size_t j = 0; j < INPUT_DIM; j++)
      inputs[i][j] = ((rand() % 2000) / 1000.0L - 1.0L);
    for (size_t j = 0; j < OUTPUT_DIM; j++)
      targets[i][j] = ((rand() % 2000) / 1000.0L - 1.0L);
  }

  // Training loop
  for (size_t gen = 0; gen < GENERATIONS; gen++) {
    NEAT_train(neat, inputs, targets, NUM_SAMPLES);

    // Print best fitness
    long double bestFitness = neat->pop->genomes[0]->fitness;
    printf("Generation %zu: Best Fitness = %.6Lf\n", gen, bestFitness);
  }

  // Clean up
  for (size_t i = 0; i < NUM_SAMPLES; i++) {
    free(inputs[i]);
    free(targets[i]);
  }

  NEAT_destroy(neat);
  return 0;
}
