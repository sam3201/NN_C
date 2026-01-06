#include "SAM.h"
#include "../utils/NN/TRANSFORMER.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Helper function to initialize weights
static void init_weights(SAM_t *sam) {
  sam->weights =
      (long double ***)malloc((sam->num_layers - 1) * sizeof(long double **));
  for (size_t i = 0; i < sam->num_layers - 1; i++) {
    sam->weights[i] =
        (long double **)malloc(sam->layer_sizes[i] * sizeof(long double *));
    for (size_t j = 0; j < sam->layer_sizes[i]; j++) {
      sam->weights[i][j] =
          (long double *)malloc(sam->layer_sizes[i + 1] * sizeof(long double));
      for (size_t k = 0; k < sam->layer_sizes[i + 1]; k++) {
        sam->weights[i][j][k] = ((long double)rand() / RAND_MAX * 2.0L - 1.0L) *
                                sqrtl(2.0L / sam->layer_sizes[i]);
      }
    }
  }
}

// Helper function to free weights
static void free_weights(SAM_t *sam) {
  for (size_t i = 0; i < sam->num_layers - 1; i++) {
    for (size_t j = 0; j < sam->layer_sizes[i]; j++) {
      free(sam->weights[i][j]);
    }
    free(sam->weights[i]);
  }
  free(sam->weights);
  sam->weights = NULL;
}

SAM_t *SAM_init(size_t input_dim, size_t output_dim, size_t num_heads,
                size_t context_id) {
  SAM_t *sam = (SAM_t *)malloc(sizeof(SAM_t));
  if (!sam)
    return NULL;

  // Initialize dimensions
  sam->num_layers = 2; // pooled transformer vec -> output
  sam->layer_sizes = (size_t *)malloc(sam->num_layers * sizeof(size_t));
  sam->layer_sizes[0] = input_dim;  // model_dim
  sam->layer_sizes[1] = output_dim; // action logits

  init_weights(sam);

  sam->transformer = TRANSFORMER_init(input_dim, num_heads, 1);
  sam->num_submodels = 1; // Fixed number of submodels for now
  sam->submodels = (NEAT_t **)malloc(sam->num_submodels * sizeof(NEAT_t *));

  // Initialize each submodel with a proper population size (at least 1)
  unsigned int population_size = 1;
  for (size_t i = 0; i < sam->num_submodels; i++) {
    sam->submodels[i] = NEAT_init(input_dim, output_dim, population_size);
    if (!sam->submodels[i]) {
      // Cleanup on error
      for (size_t j = 0; j < i; j++) {
        if (sam->submodels[j]) {
          NEAT_destroy(sam->submodels[j]);
        }
      }
      free(sam->submodels);
      if (sam->transformer) {
        TRANSFORMER_destroy(sam->transformer);
      }
      free_weights(sam);
      free(sam->layer_sizes);
      free(sam);
      return NULL;
    }
  }

  sam->context = (long double)context_id;
  return sam;
}

void SAM_destroy(SAM_t *sam) {
  if (!sam)
    return;

  // Free weights
  free_weights(sam);
  free(sam->layer_sizes);

  // Free transformer
  if (sam->transformer) {
    TRANSFORMER_destroy(sam->transformer);
  }

  // Free submodels
  for (size_t i = 0; i < sam->num_submodels; i++) {
    if (sam->submodels[i]) {
      NEAT_destroy(sam->submodels[i]);
    }
  }
  free(sam->submodels);
  free(sam);
}

void SAM_train(SAM_t *sam, long double **input_sequence, size_t seq_length,
               long double *target) {
  if (!sam || !input_sequence || seq_length == 0 || !target)
    return;

  // Train transformer
  TRANSFORMER_train(sam->transformer, input_sequence, seq_length, target);

  // Train submodels: wrap pointers as 1-sample batch
  long double *in0 = input_sequence[0];
  long double *t0 = target;

  long double *inputs_arr[1] = {in0};
  long double *targets_arr[1] = {t0};

  for (size_t i = 0; i < sam->num_submodels; i++) {
    NEAT_train(sam->submodels[i], (long double **)inputs_arr,
               (long double **)targets_arr, 1);
  }
}

void SAM_adapt_transfusion(SAM_t *sam, long double context, ProjectionMatrix *P,
                           AdaptationParams *params) {
  if (!sam || !P || !params)
    return;

  // Calculate base gamma for first submodel
  long double gamma =
      SAM_calculate_gamma(sam->context, 0); // Base gamma for first submodel

  // Perform transfusion for each submodel
  for (size_t i = 0; i < sam->num_submodels; i++) {
    // Train submodel
    SAM_train_submodel(sam->submodels[i], params->learning_rate_transfusion);

    // Update projection matrix
    for (size_t j = 0; j < P->rows; j++) {
      for (size_t k = 0; k < P->cols; k++) {
        P->matrix[j][k] *= gamma;
      }
    }
  }
}

ProjectionMatrix *SAM_create_projection_matrix(SAM_t *sam,
                                               long double context) {
  if (!sam)
    return NULL;

  ProjectionMatrix *P = (ProjectionMatrix *)malloc(sizeof(ProjectionMatrix));
  if (!P)
    return NULL;

  // Initialize dimensions based on transformer architecture
  P->rows = sam->layer_sizes[0];                   // Input dimension
  P->cols = sam->layer_sizes[sam->num_layers - 1]; // Output dimension

  // Allocate matrix
  P->matrix = (long double **)malloc(P->rows * sizeof(long double *));
  for (size_t i = 0; i < P->rows; i++) {
    P->matrix[i] = (long double *)malloc(P->cols * sizeof(long double));
    for (size_t j = 0; j < P->cols; j++) {
      // Initialize with scaled weights
      P->matrix[i][j] = 1.0L / (context + 1.0L) * sam->weights[0][i][j];
    }
  }

  return P;
}

void SAM_update_transformer(SAM_t *sam, long double **G,
                            long double learning_rate) {
  if (!sam || !G)
    return;

  // Update weights
  for (size_t i = 0; i < sam->layer_sizes[0]; i++) {
    for (size_t j = 0; j < sam->layer_sizes[sam->num_layers - 1]; j++) {
      sam->weights[0][i][j] += learning_rate * G[i][j];
    }
  }
}

long double SAM_calculate_gamma(long double context, size_t submodel_index) {
  // Calculate gamma based on context and submodel index
  return 1.0L / (1.0L + fabsl(context - (long double)submodel_index));
}

long double SAM_calculate_beta(PerformanceMetrics *metrics,
                               long double context) {
  if (!metrics)
    return 0.0L;

  // Calculate beta based on performance metrics and context
  return metrics->fitness / (1.0L + context);
}

int SAM_save(SAM_t *sam, const char *filename) {
  if (!sam || !filename)
    return 0;

  FILE *file = fopen(filename, "wb");
  if (!file)
    return 0;

  // Save SAM parameters
  if (fwrite(&sam->num_submodels, sizeof(size_t), 1, file) != 1 ||
      fwrite(&sam->context, sizeof(long double), 1, file) != 1) {
    fclose(file);
    return 0;
  }

  // Save layer configuration
  if (fwrite(&sam->num_layers, sizeof(size_t), 1, file) != 1) {
    fclose(file);
    return 0;
  }

  if (sam->layer_sizes && sam->num_layers > 0) {
    if (fwrite(sam->layer_sizes, sizeof(size_t), sam->num_layers, file) !=
        sam->num_layers) {
      fclose(file);
      return 0;
    }
  }

  // Save head weights (num_layers assumed 2 => one weight matrix)
  size_t in_dim = sam->layer_sizes[0];
  size_t out_dim = sam->layer_sizes[1];
  for (size_t j = 0; j < in_dim; j++) {
    if (fwrite(sam->weights[0][j], sizeof(long double), out_dim, file) !=
        out_dim) {
      fclose(file);
      return 0;
    }
  }

  // Save transformer into the SAME file stream
  if (!TRANSFORMER_save(sam->transformer, file)) {
    fclose(file);
    return 0;
  }

  // NOTE: NEAT submodels not saved here (filename-based API).
  // We still save num_submodels + structure so SAM can be reconstructed.

  fclose(file);
  return 1;
}

SAM_t *SAM_load(const char *filename) {
  if (!filename)
    return NULL;

  FILE *file = fopen(filename, "rb");
  if (!file)
    return NULL;

  SAM_t *sam = (SAM_t *)calloc(1, sizeof(SAM_t));
  if (!sam) {
    fclose(file);
    return NULL;
  }

  // Load SAM parameters
  if (fread(&sam->num_submodels, sizeof(size_t), 1, file) != 1 ||
      fread(&sam->context, sizeof(long double), 1, file) != 1) {
    fclose(file);
    free(sam);
    return NULL;
  }

  // Load layer configuration
  if (fread(&sam->num_layers, sizeof(size_t), 1, file) != 1) {
    fclose(file);
    free(sam);
    return NULL;
  }

  if (sam->num_layers < 2 || sam->num_layers > 100) {
    fclose(file);
    free(sam);
    return NULL;
  }

  sam->layer_sizes = (size_t *)malloc(sam->num_layers * sizeof(size_t));
  if (!sam->layer_sizes) {
    fclose(file);
    free(sam);
    return NULL;
  }

  if (fread(sam->layer_sizes, sizeof(size_t), sam->num_layers, file) !=
      sam->num_layers) {
    fclose(file);
    free(sam->layer_sizes);
    free(sam);
    return NULL;
  }

  sam->weights = NULL;
  init_weights(sam);

  // Allocate weights then load them
  size_t in_dim = sam->layer_sizes[0];
  size_t out_dim = sam->layer_sizes[sam->num_layers - 1];

  // NOTE: you currently only save/load weights[0] (linear head)
  // This assumes num_layers == 2. If you expand later, you must serialize all
  // layers.
  if (sam->num_layers != 2) {
    // For now, be strict to avoid silent wrong loads.
    SAM_destroy(sam);
    fclose(file);
    return NULL;
  }

  for (size_t j = 0; j < in_dim; j++) {
    if (fread(sam->weights[0][j], sizeof(long double), out_dim, file) !=
        out_dim) {
      SAM_destroy(sam);
      fclose(file);
      return NULL;
    }
  }

  // Load transformer from the SAME file stream
  sam->transformer = TRANSFORMER_load(file);
  if (!sam->transformer) {
    SAM_destroy(sam);
    fclose(file);
    return NULL;
  }

  // Allocate submodels array
  sam->submodels = (NEAT_t **)calloc(sam->num_submodels, sizeof(NEAT_t *));
  if (!sam->submodels) {
    SAM_destroy(sam);
    fclose(file);
    return NULL;
  }

  // Recreate NEAT submodels (not loaded from file yet)
  size_t input_dim = sam->layer_sizes[0];
  size_t output_dim = sam->layer_sizes[sam->num_layers - 1];

  unsigned int population_size = 100;
  for (size_t i = 0; i < sam->num_submodels; i++) {
    sam->submodels[i] = NEAT_init(input_dim, output_dim, population_size);
    if (!sam->submodels[i]) {
      SAM_destroy(sam);
      fclose(file);
      return NULL;
    }
  }

  fclose(file);
  return sam;
}

// Matrix operations
void SAM_matrix_multiply(long double **A, long double **B, long double **C,
                         size_t m, size_t n, size_t p) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      C[i][j] = 0.0L;
      for (size_t k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void SAM_matrix_scale(long double **matrix, long double scalar, size_t rows,
                      size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      matrix[i][j] *= scalar;
    }
  }
}

void SAM_matrix_add(long double **A, long double **B, long double **C,
                    size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }
}

void SAM_train_submodel(NEAT_t *neat, long double learning_rate) {
  if (!neat)
    return;
  // TODO: Implement submodel training
}

PerformanceMetrics SAM_calculate_metrics(NEAT_t *neat) {
  PerformanceMetrics metrics;
  metrics.accuracy = 0.0L;
  metrics.loss = 0.0L;
  metrics.fitness = 0.0L;

  (void)neat;
  /* TODO: compute from NEAT's actual structures (population/genomes fitness) */

  return metrics;
}

// SAM forward pass
long double *SAM_forward(SAM_t *sam, long double **input_sequence,
                         size_t seq_length) {
  if (!sam || !input_sequence || seq_length == 0)
    return NULL;
  if (!sam->transformer)
    return NULL;

  /* transformer outputs: seq_length x model_dim */
  long double **feat =
      TRANSFORMER_forward(sam->transformer, input_sequence, seq_length);
  if (!feat)
    return NULL;

  size_t out_dim = sam->layer_sizes[1];

  /* mean-pool over time into pooled[model_dim] */
  long double *pooled = (long double *)calloc(model_dim, sizeof(long double));
  if (!pooled) {
    for (size_t t = 0; t < seq_length; t++)
      size_t model_dim = sam->layer_sizes[0]; // input_dim
    pooled[j] += feat[t][j];

    free(feat[t]);
    free(feat);
    return NULL;
  }

  for (size_t t = 0; t < seq_length; t++) {
    for (size_t j = 0; j < model_dim; j++) {
      pooled[j] += feat[t][j];
    }
  }
  long double invT = 1.0L / (long double)seq_length;
  for (size_t j = 0; j < model_dim; j++)
    pooled[j] *= invT;

  /* linear head: pooled(model_dim) -> out(out_dim) using weights[0][j][i] */
  long double *out = (long double *)malloc(sizeof(long double) * out_dim);
  if (!out) {
    free(pooled);
    for (size_t t = 0; t < seq_length; t++)
      free(feat[t]);
    free(feat);
    return NULL;
  }

  for (size_t i = 0; i < out_dim; i++) {
    long double sum = 0.0L;
    for (size_t j = 0; j < model_dim; j++) {
      sum += pooled[j] * sam->weights[0][j][i];
    }
    out[i] = sum; /* logits */
  }

  free(pooled);

  for (size_t t = 0; t < seq_length; t++)
    free(feat[t]);
  free(feat);

  return out;
}

void SAM_backprop(SAM_t *sam, long double **input_sequence, size_t seq_length,
                  long double *grad_loss) {
  (void)sam;
  (void)input_sequence;
  (void)seq_length;
  (void)grad_loss;
  /* TODO: implement proper gradient flow:
     - forward caches
     - build grad_output sequence for transformer
     - call TRANSFORMER_backprop(transformer, grad_seq, seq_len)
     - update head weights
  */
}

// SAM adaptation
void SAM_adapt(SAM_t *sam, long double **input_sequence, size_t seq_length) {
  if (!sam || !input_sequence || seq_length == 0)
    return;

  size_t out_dim = sam->layer_sizes[sam->num_layers - 1];

  for (size_t i = 0; i < sam->num_submodels; i++) {
    if (!sam->submodels[i])
      continue;

    // Build a target from transformer last-step features (or pooled, later)
    long double *target = (long double *)calloc(out_dim, sizeof(long double));
    if (!target)
      continue;

    long double **transformer_out =
        TRANSFORMER_forward(sam->transformer, input_sequence, seq_length);

    if (transformer_out) {
      // Use last timestep vector
      long double *last = transformer_out[seq_length - 1];

      size_t copy_size = out_dim;
      if (copy_size > sam->layer_sizes[0])
        copy_size = sam->layer_sizes[0];

      memcpy(target, last, copy_size * sizeof(long double));

      // free transformer_out properly
      for (size_t t = 0; t < seq_length; t++)
        free(transformer_out[t]);
      free(transformer_out);
    }

    // Train submodel with 1-sample batch (input = first obs)
    long double *in0 = input_sequence[0];
    long double *inputs_arr[1] = {in0};
    long double *targets_arr[1] = {target};

    NEAT_train(sam->submodels[i], (long double **)inputs_arr,
               (long double **)targets_arr, 1);

    free(target);
  }
}

// SAM generalization
void SAM_generalize(SAM_t *sam) {
  (void)sam;
  /* TODO: compute metrics from Population / Genome fitness and generalize */
}

// SAM transfusion
void SAM_transfuse(SAM_t *sam) {
  if (!sam)
    return;
  for (size_t i = 0; i < sam->num_submodels; i++) {
    if (sam->submodels[i] && sam->submodels[i]->pop) {
      POPULATION_evolve(sam->submodels[i]->pop);
    }
  }
}

// SAM evaluate fitness
long double SAM_evaluate_fitness(SAM_t *sam, long double *input,
                                 long double *target) {
  if (!sam || !input || !target)
    return -INFINITY;

  // Forward pass
  long double **input_seq = (long double **)malloc(sizeof(long double *));
  if (!input_seq)
    return -INFINITY;
  input_seq[0] = input;

  long double *output = SAM_forward(sam, input_seq, 1);
  free(input_seq);

  if (!output)
    return -INFINITY;

  // Calculate MSE loss
  long double loss = 0.0L;
  size_t output_size = sam->layer_sizes[sam->num_layers - 1];
  for (size_t i = 0; i < output_size; i++) {
    long double diff = output[i] - target[i];
    loss += diff * diff;
  }
  loss /= output_size;

  free(output);
  return -loss; // Negative because higher fitness is better
}

// SAM update context
void SAM_update_context(SAM_t *sam, long double current_performance) {
  if (!sam)
    return;

  // Update context based on performance
  sam->context = (sam->context + current_performance) / 2.0L;

  // Clamp context to reasonable range
  if (sam->context < 0.0L)
    sam->context = 0.0L;
  if (sam->context > 1.0L)
    sam->context = 1.0L;
}
