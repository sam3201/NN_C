#include "NEAT.h"
#include "../VISUALIZER/NN_visualizer.h"
#include "NN.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static size_t innovation_number = 0;
static size_t species_id_counter = 0;

Perceptron_t *Perceptron_init(NN_t *nn, long double fitness,
                              unsigned int num_connections,
                              Perceptron_t *connections, bool enabled) {
  if (!nn) {
    fprintf(stderr, "Neural network cannot be NULL\n");
    return NULL;
  }

  Perceptron_t *perceptron = malloc(sizeof(Perceptron_t));
  if (!perceptron) {
    fprintf(stderr, "Failed to allocate memory for perceptron\n");
    return NULL;
  }

  perceptron->nn = nn;
  perceptron->fitness = fitness;
  perceptron->num_connections = num_connections;
  perceptron->connections = connections;
  perceptron->enabled = enabled;

  return perceptron;
}

Perceptron_t *Perceptron_init_random(unsigned int num_inputs,
                                     unsigned int num_outputs) {
  printf(
      "Perceptron_init_random: Creating network with %u inputs, %u outputs\n",
      num_inputs, num_outputs);
  fflush(stdout);

  if (num_inputs == 0 || num_outputs == 0) {
    fprintf(stderr, "Invalid input/output dimensions\n");
    return NULL;
  }

  // Create layer sizes array with terminating 0
  size_t num_layers = 3;
  size_t layers[num_layers + 1];
  for (size_t i = 0; i < num_layers; i++) {
    // Random number between 1 and 2
    layers[i] = 1 + (rand() % 2);
  }
  layers[num_layers] = 0;

  // Create activation functions array
  ActivationFunctionType actFuncs[num_layers - 1];
  ActivationDerivativeType actDerivs[num_layers - 1];
  for (size_t i = 0; i < num_layers - 1; i++) {
    actFuncs[i] = rand() % ACTIVATION_TYPE_COUNT;
    actDerivs[i] = rand() % ACTIVATION_DERIVATIVE_TYPE_COUNT;
  }

  LossFunctionType lossFunc = rand() % LOSS_TYPE_COUNT;
  LossDerivativeType lossDeriv = map_loss_to_derivative(lossFunc);
  RegularizationType regFunc = rand() % REGULARIZATION_TYPE_COUNT;
  OptimizerType optFunc = rand() % OPTIMIZER_TYPE_COUNT;

  long double learningRate = ((long double)rand() / RAND_MAX) * 0.09L + 0.01L;

  printf("Perceptron_init_random: Initializing neural network\n");
  fflush(stdout);

  NN_t *nn = NN_init(layers, actFuncs, actDerivs, lossFunc, lossDeriv, regFunc,
                     optFunc, learningRate);
  if (!nn) {
    fprintf(stderr, "Failed to initialize neural network\n");
    return NULL;
  }

  printf("Perceptron_init_random: Neural network initialized successfully\n");
  fflush(stdout);

  Perceptron_t *perceptron = Perceptron_init(nn, 0.0, 0, NULL, true);
  if (!perceptron) {
    NN_destroy(nn);
    return NULL;
  }

  return perceptron;
}

void Perceptron_destroy(Perceptron_t *perceptron) {
  if (perceptron) {
    if (perceptron->nn) {
      NN_destroy(perceptron->nn);
    }
    free(perceptron);
  }
}

NEAT_t *NEAT_init(unsigned int num_inputs, unsigned int num_outputs,
                  unsigned int initial_population) {
  srand(time(NULL));
  innovation_number = 0;
  species_id_counter = 0;

  NEAT_t *neat = (NEAT_t *)malloc(sizeof(NEAT_t));
  if (!neat)
    return NULL;

  neat->population_size = initial_population;

  // Initialize population
  neat->population =
      (Perceptron_t **)malloc(initial_population * sizeof(Perceptron_t *));
  if (!neat->population) {
    free(neat);
    return NULL;
  }

  for (unsigned int i = 0; i < initial_population; i++) {
    neat->population[i] = Perceptron_init_random(num_inputs, num_outputs);
    if (!neat->population[i]) {
      // Cleanup previously allocated perceptrons
      for (unsigned int j = 0; j < i; j++) {
        Perceptron_destroy(neat->population[j]);
      }
      free(neat->population);
      free(neat);
      return NULL;
    }
    neat->population[i]->fitness = 0.0;  // Set default fitness value
    neat->population[i]->enabled = true; // Ensure perceptron is enabled
    printf("Initializing perceptron %u with fitness: %Lf\n", i,
           neat->population[i]->fitness);
  }

  neat->nodes =
      (Perceptron_t **)malloc(initial_population * sizeof(Perceptron_t *));
  if (!neat->nodes) {
    free(neat->population);
    free(neat);
    return NULL;
  }
  for (unsigned int i = 0; i < initial_population; i++) {
    neat->nodes[i] = neat->population[i];
  }

  neat->num_nodes = initial_population; // Set num_nodes to initial_population

  return neat;
}

void NEAT_add_random(NEAT_t *neat, unsigned int num_new_nodes) {
  if (!neat || num_new_nodes == 0) {
    fprintf(stderr, "Invalid NEAT structure or number of nodes\n");
    return;
  }

  // Get input/output dimensions from existing node
  if (!neat->nodes || !neat->nodes[0] || !neat->nodes[0]->nn) {
    fprintf(stderr, "Invalid NEAT structure: no valid initial node\n");
    return;
  }

  unsigned int num_inputs = neat->nodes[0]->nn->layers[0];
  unsigned int num_outputs =
      neat->nodes[0]->nn->layers[neat->nodes[0]->nn->numLayers - 1];

  // Allocate new array with increased size
  Perceptron_t **new_nodes = realloc(
      neat->nodes, (neat->num_nodes + num_new_nodes) * sizeof(Perceptron_t *));
  if (!new_nodes) {
    fprintf(stderr, "Failed to reallocate memory for new nodes\n");
    return;
  }
  neat->nodes = new_nodes;

  // Initialize new nodes
  for (unsigned int i = 0; i < num_new_nodes; i++) {
    neat->nodes[neat->num_nodes + i] =
        Perceptron_init_random(num_inputs, num_outputs);
    if (!neat->nodes[neat->num_nodes + i]) {
      fprintf(stderr, "Failed to initialize node %u\n", neat->num_nodes + i);
      continue;
    }

    // Randomly set species ID (for now, just increment)
    neat->species_id++;

    printf("Added new node %u with species ID %d\n", neat->num_nodes + i,
           neat->species_id);
  }

  neat->num_nodes += num_new_nodes;
  printf("NEAT network now has %u nodes\n", neat->num_nodes);
}

void NEAT_destroy(NEAT_t *neat) {
  printf("\nDestroying NEAT network...\n");
  if (!neat) {
    printf("NEAT is NULL, nothing to destroy\n");
    return;
  }

  printf("Destroying nodes...\n");
  if (neat->nodes) {
    for (int i = 0; i < neat->num_nodes; i++) {
      if (neat->nodes[i]) {
        printf("Destroying node %d...\n", i);
        if (neat->nodes[i]->nn) {
          printf("Destroying neural network for node %d...\n", i);
          NN_destroy(neat->nodes[i]->nn);
          neat->nodes[i]->nn = NULL;
        }
        NEAT_destroy_node(neat->nodes[i]);
        neat->nodes[i] = NULL;
      }
    }
    printf("Freeing nodes array...\n");
    free(neat->nodes);
    neat->nodes = NULL;
  }

  printf("Freeing NEAT structure...\n");
  free(neat);
  printf("NEAT network destroyed successfully\n");
}

void NEAT_destroy_node(Perceptron_t *node) {
  printf("Destroying node...\n");
  if (!node) {
    printf("Node is NULL, nothing to destroy\n");
    return;
  }

  if (node->nn) {
    printf("Destroying node's neural network...\n");
    NN_destroy(node->nn);
    node->nn = NULL;
  }

  printf("Freeing node structure...\n");
  free(node);
  printf("Node destroyed successfully\n");
}

void NEAT_add_neuron_random(NEAT_t *neat) {
  if (!neat || !neat->nodes || neat->num_nodes == 0) {
    fprintf(stderr, "Invalid NEAT structure for adding neuron\n");
    return;
  }

  // Expand nodes array
  neat->num_nodes++;
  Perceptron_t **new_nodes =
      realloc(neat->nodes, neat->num_nodes * sizeof(Perceptron_t *));
  if (!new_nodes) {
    fprintf(stderr, "Failed to reallocate memory for nodes\n");
    neat->num_nodes--; // Revert the increment
    return;
  }
  neat->nodes = new_nodes;

  // Get a random existing node as template
  unsigned int rand_idx = rand() % (neat->num_nodes - 1);
  Perceptron_t *template_node = neat->nodes[rand_idx];

  if (!template_node || !template_node->nn) {
    fprintf(stderr, "Invalid template node\n");
    neat->num_nodes--;
    return;
  }

  // Create a new node with random layer sizes between template's bounds
  unsigned int min_layer_size = 1;
  unsigned int max_layer_size = 10;
  unsigned int num_hidden_layers = (rand() % 3) + 1; // 1 to 3 hidden layers

  // Initialize layer sizes array
  size_t *layer_sizes =
      malloc((num_hidden_layers + 2) *
             sizeof(size_t)); // +2 for input and output layers
  if (!layer_sizes) {
    fprintf(stderr, "Failed to allocate layer sizes\n");
    neat->num_nodes--;
    return;
  }

  // Set input and output layer sizes
  layer_sizes[0] = template_node->nn->layers[0]; // Input layer
  layer_sizes[num_hidden_layers + 1] =
      template_node->nn
          ->layers[template_node->nn->numLayers - 1]; // Output layer

  // Generate random hidden layer sizes
  for (unsigned int i = 1; i <= num_hidden_layers; i++) {
    layer_sizes[i] =
        min_layer_size + (rand() % (max_layer_size - min_layer_size + 1));
  }

  // Initialize activation functions and derivatives
  ActivationFunctionType *activation_funcs =
      malloc((num_hidden_layers + 2) * sizeof(ActivationFunctionType));
  ActivationDerivativeType *activation_derivs =
      malloc((num_hidden_layers + 2) * sizeof(ActivationDerivativeType));

  if (!activation_funcs || !activation_derivs) {
    fprintf(stderr, "Failed to allocate activation functions\n");
    free(layer_sizes);
    if (activation_funcs)
      free(activation_funcs);
    if (activation_derivs)
      free(activation_derivs);
    neat->num_nodes--;
    return;
  }

  // Assign activation functions with preference for stable ones
  for (unsigned int i = 0; i < num_hidden_layers + 2; i++) {
    int r = rand() % 100;
    if (r < 40) { // 40% chance for ReLU
      activation_funcs[i] = RELU;
      activation_derivs[i] = RELU_DERIVATIVE;
    } else if (r < 70) { // 30% chance for tanh
      activation_funcs[i] = TANH;
      activation_derivs[i] = TANH_DERIVATIVE;
    } else { // 30% chance for sigmoid
      activation_funcs[i] = SIGMOID;
      activation_derivs[i] = SIGMOID_DERIVATIVE;
    }
    switch (activation_funcs[i]) {
    case SIGMOID:
      activation_derivs[i] = SIGMOID_DERIVATIVE;
      break;
    case TANH:
      activation_derivs[i] = TANH_DERIVATIVE;
      break;
    case RELU:
      activation_derivs[i] = RELU_DERIVATIVE;
      break;
    case LINEAR:
      activation_derivs[i] = LINEAR_DERIVATIVE;
      break;
    default:
      break;
    }
  }

  LossFunctionType lossFunc = rand() % LOSS_TYPE_COUNT;
  LossDerivativeType lossDeriv;
  switch (lossFunc) {
  case MSE:
    lossDeriv = MSE_DERIVATIVE;
    break;
  case MAE:
    lossDeriv = MAE_DERIVATIVE;
    break;
  case HUBER:
    lossDeriv = HUBER_DERIVATIVE;
    break;
  case LL:
    lossDeriv = LL_DERIVATIVE;
    break;
  case CE:
    lossDeriv = CE_DERIVATIVE;
    break;
  default:
    lossDeriv = MSE_DERIVATIVE;
    break;
  }

  RegularizationType regFunc = rand() % REGULARIZATION_TYPE_COUNT;
  OptimizerType optFunc = rand() % OPTIMIZER_TYPE_COUNT;
  long double learningRate = ((long double)rand() / RAND_MAX) * 0.09L + 0.01L;

  // Create new neural network with proper initialization
  NN_t *new_nn =
      NN_init(layer_sizes, activation_funcs, activation_derivs, lossFunc,
              lossDeriv, regFunc, optFunc, learningRate); // Learning rate

  if (!new_nn) {
    fprintf(stderr, "Failed to create new neural network\n");
    free(layer_sizes);
    free(activation_funcs);
    free(activation_derivs);
    neat->num_nodes--;
    return;
  }

  // Initialize the new perceptron
  neat->nodes[neat->num_nodes - 1] = malloc(sizeof(Perceptron_t));
  if (!neat->nodes[neat->num_nodes - 1]) {
    fprintf(stderr, "Failed to allocate new perceptron\n");
    NN_destroy(new_nn);
    free(layer_sizes);
    free(activation_funcs);
    free(activation_derivs);
    neat->num_nodes--;
    return;
  }

  neat->nodes[neat->num_nodes - 1]->nn = new_nn;
  neat->nodes[neat->num_nodes - 1]->enabled = true;
  neat->nodes[neat->num_nodes - 1]->fitness = 0.0f;

  // Initialize weights with Xavier initialization
  for (unsigned int i = 0; i < new_nn->numLayers - 1; i++) {
    double scale = sqrt(2.0 / (new_nn->layers[i] + new_nn->layers[i + 1]));
    for (unsigned int j = 0; j < new_nn->layers[i + 1] * new_nn->layers[i];
         j++) {
      // Generate random number between -1 and 1
      double rand_val = ((double)rand() / RAND_MAX) * 2 - 1;
      new_nn->weights[i][j] = rand_val * scale;
    }

    // Initialize biases to small random values
    for (unsigned int j = 0; j < new_nn->layers[i + 1]; j++) {
      new_nn->biases[i][j] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
    }
  }

  free(layer_sizes);
  free(activation_funcs);
  free(activation_derivs);
}

Perceptron_t *NEAT_crossover(Perceptron_t *parent1, Perceptron_t *parent2) {
  if (!parent1 || !parent1->nn)
    return NULL;

  // Create new network with same structure as parent1
  size_t *layers = malloc((parent1->nn->numLayers + 1) * sizeof(size_t));
  if (!layers)
    return NULL;

  // Copy layer structure
  memcpy(layers, parent1->nn->layers, parent1->nn->numLayers * sizeof(size_t));
  layers[parent1->nn->numLayers] = 0; // Terminating zero

  // Copy activation functions
  ActivationFunctionType *actFuncs =
      malloc((parent1->nn->numLayers - 1) * sizeof(ActivationFunctionType));
  ActivationDerivativeType *actDerivs =
      malloc((parent1->nn->numLayers - 1) * sizeof(ActivationDerivativeType));
  if (!actFuncs || !actDerivs) {
    free(layers);
    free(actFuncs);
    free(actDerivs);
    return NULL;
  }

  LossFunctionType lossFunc = rand() % LOSS_TYPE_COUNT;
  LossDerivativeType lossDeriv;
  switch (lossFunc) {
  case MSE:
    lossDeriv = MSE_DERIVATIVE;
    break;
  case MAE:
    lossDeriv = MAE_DERIVATIVE;
    break;
  case HUBER:
    lossDeriv = HUBER_DERIVATIVE;
    break;
  case LL:
    lossDeriv = LL_DERIVATIVE;
    break;
  case CE:
    lossDeriv = CE_DERIVATIVE;
    break;
  default:
    lossDeriv = MSE_DERIVATIVE;
    break;
  }

  RegularizationType regFunc = rand() % REGULARIZATION_TYPE_COUNT;
  OptimizerType optFunc = rand() % OPTIMIZER_TYPE_COUNT;
  long double learningRate = ((long double)rand() / RAND_MAX) * 0.09L + 0.01L;

  // Initialize new network with parent's structure
  NN_t *nn = NN_init(layers, actFuncs, actDerivs, lossFunc, lossDeriv, regFunc,
                     optFunc, learningRate);
  free(layers);
  free(actFuncs);
  free(actDerivs);

  if (!nn)
    return NULL;

  // Create child perceptron
  Perceptron_t *child = Perceptron_init(nn, 0.0, 0, NULL, true);
  if (!child) {
    NN_destroy(nn);
    return NULL;
  }

  // Perform crossover of weights and biases
  for (size_t i = 0; i < parent1->nn->numLayers - 1; i++) {
    size_t weights_size = parent1->nn->layers[i] * parent1->nn->layers[i + 1];

    for (size_t j = 0; j < weights_size; j++) {
      // Random mutation
      long double mutation = ((rand() % 200) - 100) / 1000.0L;
      child->nn->weights[i][j] = parent1->nn->weights[i][j] + mutation;
    }

    // Mutate biases
    size_t bias_size = parent1->nn->layers[i + 1];
    for (size_t j = 0; j < bias_size; j++) {
      long double mutation = ((rand() % 200) - 100) / 1000.0L;
      child->nn->biases[i][j] = parent1->nn->biases[i][j] + mutation;
    }
  }

  return child;
}

void NEAT_evolve(NEAT_t *neat) {
  // Find best performer
  Perceptron_t *best = NULL;
  long double best_fitness = -INFINITY;
  unsigned int best_idx = 0;

  for (unsigned int i = 0; i < neat->num_nodes; i++) {
    if (neat->nodes[i]->enabled && neat->nodes[i]->fitness > best_fitness) {
      best_fitness = neat->nodes[i]->fitness;
      best = neat->nodes[i];
      best_idx = i;
    }
  }

  if (!best)
    return;

  // Create new generation based on best performer
  for (unsigned int i = 0; i < neat->num_nodes; i++) {
    if (i == best_idx)
      continue; // Keep the best performer

    if (neat->nodes[i]) {
      Perceptron_destroy(neat->nodes[i]);
    }

    neat->nodes[i] = NEAT_crossover(best, best); // Self-crossover for mutation
    if (neat->nodes[i]) {
      neat->nodes[i]->enabled = true;
      neat->nodes[i]->species_id = -1;
      neat->nodes[i]->fitness = 0;
    }
  }
}

long double *NEAT_forward(NEAT_t *neat, long double inputs[]) {
  if (!neat || !inputs) {
    printf("NEAT or inputs are NULL\n");
    return NULL;
  }

  // Find best enabled perceptron
  Perceptron_t *best = NULL;
  long double best_fitness = -INFINITY;

  for (unsigned int i = 0; i < neat->num_nodes; i++) {
    if (neat->nodes[i] && neat->nodes[i]->enabled) {
      if (neat->nodes[i]->fitness > best_fitness) {
        best_fitness = neat->nodes[i]->fitness;
        best = neat->nodes[i];
      }
    }
  }

  // If no perceptron with fitness found, use first enabled one
  if (!best) {
    for (unsigned int i = 0; i < neat->num_nodes; i++) {
      if (neat->nodes[i] && neat->nodes[i]->enabled) {
        best = neat->nodes[i];
        break;
      }
    }
  }

  if (!best || !best->nn) {
    printf("No enabled perceptron found\n");
    return NULL;
  }

  long double *output = NN_forward(best->nn, inputs);
  if (!output) {
    printf("Failed to compute output\n");
  }
  return output;
}

void NEAT_backprop(NEAT_t *neat, long double inputs[], long double y_true,
                   long double y_pred) {
  if (!neat || !inputs) {
    return;
  }

  // Apply backpropagation to all enabled nodes
  for (unsigned int i = 0; i < neat->num_nodes; i++) {
    if (neat->nodes[i]->enabled) {
      size_t input_size = neat->nodes[i]->nn->layers[0];
      size_t output_size =
          neat->nodes[i]->nn->layers[neat->nodes[i]->nn->numLayers - 1];

      long double target[output_size];
      target[0] = y_true; // Assuming single output for now

      NN_backprop(neat->nodes[i]->nn, inputs, y_true, y_pred);
    }
  }
}

void NEAT_speciate(NEAT_t *neat) {
  if (!neat || !neat->nodes || neat->num_nodes < 2)
    return;

  // Reset species IDs
  for (unsigned int i = 0; i < neat->num_nodes; i++) {
    neat->nodes[i]->species_id = -1;
  }

  // Assign species based on compatibility distance
  int current_species = 0;
  for (unsigned int i = 0; i < neat->num_nodes; i++) {
    if (neat->nodes[i]->species_id == -1) {
      neat->nodes[i]->species_id = current_species;

      // Check other nodes for compatibility
      for (unsigned int j = i + 1; j < neat->num_nodes; j++) {
        if (neat->nodes[j]->species_id == -1) {
          long double distance =
              NEAT_compatibility_distance(neat->nodes[i], neat->nodes[j]);
          if (distance < 1.0L) { // Threshold for species membership
            neat->nodes[j]->species_id = current_species;
          }
        }
      }
      current_species++;
    }
  }
}

void NEAT_adjust_species_fitness(NEAT_t *neat) {
  if (!neat || !neat->nodes)
    return;

  // Count members per species
  int max_species_id = -1;
  for (unsigned int i = 0; i < neat->num_nodes; i++) {
    if (neat->nodes[i]->species_id > max_species_id) {
      max_species_id = neat->nodes[i]->species_id;
    }
  }

  int *species_count = calloc(max_species_id + 1, sizeof(int));
  if (!species_count)
    return;

  // Count members in each species
  for (unsigned int i = 0; i < neat->num_nodes; i++) {
    if (neat->nodes[i]->species_id >= 0) {
      species_count[neat->nodes[i]->species_id]++;
    }
  }

  // Adjust fitness based on species size
  for (unsigned int i = 0; i < neat->num_nodes; i++) {
    if (neat->nodes[i]->species_id >= 0) {
      neat->nodes[i]->fitness /= species_count[neat->nodes[i]->species_id];
    }
  }

  free(species_count);
}

void NEAT_remove_stagnant_species(NEAT_t *neat,
                                  unsigned int stagnation_threshold) {
  if (!neat || !neat->nodes)
    return;

  // Calculate average fitness per species
  int max_species_id = -1;
  for (unsigned int i = 0; i < neat->num_nodes; i++) {
    if (neat->nodes[i]->species_id > max_species_id) {
      max_species_id = neat->nodes[i]->species_id;
    }
  }

  long double *species_fitness =
      calloc(max_species_id + 1, sizeof(long double));
  int *species_count = calloc(max_species_id + 1, sizeof(int));
  if (!species_fitness || !species_count) {
    free(species_fitness);
    free(species_count);
    return;
  }

  // Calculate average fitness
  for (unsigned int i = 0; i < neat->num_nodes; i++) {
    if (neat->nodes[i]->species_id >= 0) {
      species_fitness[neat->nodes[i]->species_id] += neat->nodes[i]->fitness;
      species_count[neat->nodes[i]->species_id]++;
    }
  }

  // Remove stagnant species
  for (unsigned int i = 0; i < neat->num_nodes; i++) {
    if (neat->nodes[i]->species_id >= 0) {
      long double avg_fitness = species_fitness[neat->nodes[i]->species_id] /
                                species_count[neat->nodes[i]->species_id];
      if (avg_fitness < stagnation_threshold) {
        neat->nodes[i]->enabled = false;
      }
    }
  }

  free(species_fitness);
  free(species_count);
}

void NEAT_train(NEAT_t *neat, long double *input, long double *target) {
  if (!neat || !input || !target)
    return;

  // Forward pass
  long double *output = NEAT_forward(neat, input);
  if (!output)
    return;

  // Calculate prediction (use first output value)
  long double y_pred =
      (neat->num_nodes > 0 && neat->nodes[0] && neat->nodes[0]->nn) ? output[0]
                                                                    : 0.0L;
  long double y_true = target[0]; // Assuming single output for now

  // Backpropagate
  NEAT_backprop(neat, input, y_true, y_pred);

  free(output);
}

int NEAT_save(NEAT_t *neat, const char *filename) {
  if (!neat || !filename)
    return 0;

  FILE *file = fopen(filename, "wb");
  if (!file)
    return 0;

  // Save global state
  fwrite(&innovation_number, sizeof(size_t), 1, file);
  fwrite(&species_id_counter, sizeof(size_t), 1, file);

  fclose(file);
  return 1;
}

void NEAT_load(NEAT_t *neat, const char *filename) {
  if (!neat || !filename)
    return;

  FILE *file = fopen(filename, "rb");
  if (!file)
    return;

  // Load global state
  size_t loaded_innovation, loaded_species;
  if (fread(&loaded_innovation, sizeof(size_t), 1, file) == 1 &&
      fread(&loaded_species, sizeof(size_t), 1, file) == 1) {
    innovation_number = loaded_innovation;
    species_id_counter = loaded_species;
  }

  fclose(file);
}

void save_neat_state(const char *filename) {
  FILE *file = fopen(filename, "wb");
  if (!file)
    return;

  // Save global state
  fwrite(&innovation_number, sizeof(size_t), 1, file);
  fwrite(&species_id_counter, sizeof(size_t), 1, file);
}

long double NEAT_compatibility_distance(Perceptron_t *perceptron1,
                                        Perceptron_t *perceptron2) {
  if (!perceptron1 || !perceptron2 || !perceptron1->nn || !perceptron2->nn) {
    return INFINITY;
  }

  // Simple distance metric based on network weights
  long double distance = 0.0L;
  size_t weight_count = 0;

  // Compare weights layer by layer
  for (size_t i = 0; i < perceptron1->nn->numLayers - 1; i++) {
    size_t weights_size =
        perceptron1->nn->layers[i] * perceptron1->nn->layers[i + 1];

    for (size_t j = 0; j < weights_size; j++) {
      distance += fabsl(perceptron1->nn->weights[i][j] -
                        perceptron2->nn->weights[i][j]);
      weight_count++;
    }
  }

  // Return average weight difference
  return weight_count > 0 ? distance / weight_count : INFINITY;
}

long double *perceptron_forward(Perceptron_t *perceptron,
                                long double inputs[]) {
  if (!perceptron || !perceptron->nn || !inputs) {
    return NULL;
  }

  // Use the neural network's forward pass
  long double *output = NN_forward(perceptron->nn, inputs);
  return output;
}

void perceptron_backprop(Perceptron_t *perceptron, long double inputs[],
                         long double y_true, long double y_pred) {
  if (!perceptron || !perceptron->nn || !inputs) {
    return;
  }

  // Calculate loss
  long double loss = perceptron->nn->loss(y_true, y_pred);

  // Update fitness based on loss (lower loss = higher fitness)
  perceptron->fitness = 1.0L / (1.0L + loss);

  // Use the neural network's backpropagation
  size_t input_size = perceptron->nn->layers[0];
  size_t output_size = perceptron->nn->layers[perceptron->nn->numLayers - 1];

  long double target[output_size];
  target[0] = y_true; // Assuming single output for now

  NN_backprop(perceptron->nn, inputs, y_true, y_pred);

  // Update weights using the optimizer
  perceptron->nn->optimizer(perceptron->nn);
}
