#include "NEAT.h"
#include <string.h>

// ------------------- Activation -------------------
long double activate(long double x, ActivationFunctionType type) {
  switch (type) {
  case SIGMOID:
    return 1.0L / (1.0L + expl(-x));
  case TANH:
    return tanhl(x);
  case RELU:
    return x > 0 ? x : 0;
  case LINEAR:
    return x;
  default:
    return x;
  }
}

// ------------------- Genome_t ----------------------
Genome_t *GENOME_init(size_t numInputs, size_t numOutputs) {
  Genome_t *genome = (Genome_t *)malloc(sizeof(Genome_t));
  genome->numNodes = numInputs + numOutputs + 1; // bias
  genome->nodes = (Node **)malloc(genome->numNodes * sizeof(Node *));

  size_t idx = 0;
  for (size_t i = 0; i < numInputs; i++) {
    genome->nodes[idx] = (Node *)malloc(sizeof(Node));
    genome->nodes[idx]->id = idx;
    genome->nodes[idx]->type = INPUT_NODE;
    genome->nodes[idx]->actFunc = LINEAR;
    genome->nodes[idx]->value = 0.0L;
    idx++;
  }

  // bias node
  genome->nodes[idx] = (Node *)malloc(sizeof(Node));
  genome->nodes[idx]->id = idx;
  genome->nodes[idx]->type = BIAS_NODE;
  genome->nodes[idx]->actFunc = LINEAR;
  genome->nodes[idx]->value = 1.0L;
  idx++;

  for (size_t i = 0; i < numOutputs; i++) {
    genome->nodes[idx] = (Node *)malloc(sizeof(Node));
    genome->nodes[idx]->id = idx;
    genome->nodes[idx]->type = OUTPUT_NODE;
    genome->nodes[idx]->actFunc = SIGMOID;
    genome->nodes[idx]->value = 0.0L;
    idx++;
  }

  genome->numConnections = 0;
  genome->connections = NULL;
  genome->fitness = 0.0L;

  return genome;
}

void GENOME_destroy(Genome_t *genome) {
  if (!genome)
    return;
  for (size_t i = 0; i < genome->numNodes; i++)
    free(genome->nodes[i]);
  free(genome->nodes);

  for (size_t i = 0; i < genome->numConnections; i++)
    free(genome->connections[i]);
  free(genome->connections);
  free(genome);
}

// ------------------- Forward Propagation (Topological) -------------------
void GENOME_forward(Genome_t *genome, long double *input, long double *output) {
  for (size_t i = 0; i < genome->numNodes; i++) {
    if (genome->nodes[i]->type != BIAS_NODE)
      genome->nodes[i]->value = 0.0L;
  }

  size_t inputIdx = 0;
  for (size_t i = 0; i < genome->numNodes; i++) {
    if (genome->nodes[i]->type == INPUT_NODE)
      genome->nodes[i]->value = input[inputIdx++];
  }

  size_t topoSize;
  size_t *order = topological_sort(genome, &topoSize);

  for (size_t i = 0; i < topoSize; i++) {
    Node *n = genome->nodes[order[i]];
    for (size_t j = 0; j < genome->numConnections; j++) {
      Connection *c = genome->connections[j];
      if (c->enabled && c->from == n) {
        c->to->value += c->from->value * c->weight;
      }
    }
    if (n->type != INPUT_NODE && n->type != BIAS_NODE)
      n->value = activate(n->value, n->actFunc);
  }

  free(order);

  size_t outIdx = 0;
  for (size_t i = 0; i < genome->numNodes; i++) {
    if (genome->nodes[i]->type == OUTPUT_NODE)
      output[outIdx++] = genome->nodes[i]->value;
  }
}

// ------------------- Mutation -------------------
void GENOME_add_connection(Genome_t *genome, size_t fromNode, size_t toNode,
                           long double weight) {
  genome->connections = realloc(
      genome->connections, (genome->numConnections + 1) * sizeof(Connection *));
  Connection *c = (Connection *)malloc(sizeof(Connection));
  c->from = genome->nodes[fromNode];
  c->to = genome->nodes[toNode];
  c->weight = weight;
  c->enabled = true;
  c->innovation = get_innovation_number(fromNode, toNode);
  genome->connections[genome->numConnections++] = c;
}

void GENOME_add_node(Genome_t *genome, size_t connectionIndex) {
  if (connectionIndex >= genome->numConnections)
    return;
  Connection *c = genome->connections[connectionIndex];
  c->enabled = false;

  Node *n = (Node *)malloc(sizeof(Node));
  n->id = genome->numNodes;
  n->type = HIDDEN_NODE;
  n->actFunc = SIGMOID;
  n->value = 0.0L;

  genome->nodes =
      realloc(genome->nodes, (genome->numNodes + 1) * sizeof(Node *));
  genome->nodes[genome->numNodes++] = n;

  GENOME_add_connection(genome, c->from->id, n->id, 1.0L);
  GENOME_add_connection(genome, n->id, c->to->id, c->weight);
}

void GENOME_mutate_weights(Genome_t *genome, long double perturbRate,
                           long double perturbAmount) {
  for (size_t i = 0; i < genome->numConnections; i++) {
    if ((rand() % 10000) / 10000.0L < perturbRate) {
      genome->connections[i]->weight +=
          (((rand() % 20000) / 10000.0L) - 1.0L) * perturbAmount;
    } else {
      genome->connections[i]->weight =
          (((rand() % 20000) / 10000.0L) * 2.0L - 1.0L);
    }
  }
}

// Helper: toggle a random connection
static void toggle_random_connection(Genome_t *genome) {
  if (genome->numConnections == 0)
    return;
  Connection *c = genome->connections[rand() % genome->numConnections];
  c->enabled = !c->enabled;
}

// Extended mutation
// ------------------- Extended NEAT Mutation -------------------
void GENOME_mutate(Genome_t *genome, long double weightPerturbRate,
                   long double weightPerturbAmount, long double addNodeRate,
                   long double addConnectionRate,
                   long double toggleConnectionRate,
                   long double biasPerturbRate,
                   long double actFuncMutateRate) // new
{
  // 1️⃣ Mutate connection weights
  for (size_t i = 0; i < genome->numConnections; i++) {
    Connection *c = genome->connections[i];
    if ((rand() % 10000) / 10000.0L < weightPerturbRate) {
      c->weight += (((rand() % 20000) / 10000.0L) - 1.0L) * weightPerturbAmount;
    } else {
      c->weight = ((rand() % 20000) / 10000.0L) * 2.0L - 1.0L;
    }

    // 2️⃣ Toggle connection enabled/disabled
    if ((rand() % 10000) / 10000.0L < toggleConnectionRate) {
      c->enabled = !c->enabled;
    }
  }

  // 3️⃣ Mutate bias nodes
  for (size_t i = 0; i < genome->numNodes; i++) {
    Node *n = genome->nodes[i];
    if (n->type == BIAS_NODE &&
        ((rand() % 10000) / 10000.0L < biasPerturbRate)) {
      n->value += (((rand() % 20000) / 10000.0L) - 1.0L) * weightPerturbAmount;
    }
  }

  // 4️⃣ Mutate activation functions (hidden/output nodes)
  for (size_t i = 0; i < genome->numNodes; i++) {
    Node *n = genome->nodes[i];
    if ((n->type == HIDDEN_NODE || n->type == OUTPUT_NODE) &&
        ((rand() % 10000) / 10000.0L < actFuncMutateRate)) {
      int choice = rand() % 4;
      switch (choice) {
      case 0:
        n->actFunc = SIGMOID;
        break;
      case 1:
        n->actFunc = TANH;
        break;
      case 2:
        n->actFunc = RELU;
        break;
      case 3:
        n->actFunc = LINEAR;
        break;
      }
    }
  }

  // 5️⃣ Add new node
  if (genome->numConnections > 0 &&
      ((rand() % 10000) / 10000.0L < addNodeRate)) {
    GENOME_add_node(genome, rand() % genome->numConnections);
  }

  // 6️⃣ Add new connection
  if ((rand() % 10000) / 10000.0L < addConnectionRate) {
    size_t from = rand() % genome->numNodes;
    size_t to = rand() % genome->numNodes;
    if (from != to) {
      GENOME_add_connection(genome, from, to,
                            ((rand() % 2000) / 1000.0L - 1.0L));
    }
  }
}

// ------------------- Crossover -------------------
Genome_t *GENOME_crossover(Genome_t *p1, Genome_t *p2) {
  if (!p1 || !p2)
    return NULL;
  Genome_t *child = GENOME_init(0, 0);

  // Copy nodes (just take union of node IDs)
  for (size_t i = 0; i < p1->numNodes; i++) {
    Node *n = malloc(sizeof(Node));
    *n = *(p1->nodes[i]);
    child->nodes =
        realloc(child->nodes, (child->numNodes + 1) * sizeof(Node *));
    child->nodes[child->numNodes++] = n;
  }

  // Merge connections using innovations
  size_t i1 = 0, i2 = 0;
  while (i1 < p1->numConnections || i2 < p2->numConnections) {
    Connection *c = NULL;

    if (i1 >= p1->numConnections)
      c = p2->connections[i2++];
    else if (i2 >= p2->numConnections)
      c = p1->connections[i1++];
    else {
      Connection *conn1 = p1->connections[i1];
      Connection *conn2 = p2->connections[i2];
      if (conn1->innovation == conn2->innovation) {
        c = rand() % 2 ? conn1 : conn2;
        i1++;
        i2++;
      } else if (conn1->innovation < conn2->innovation) {
        c = conn1;
        i1++;
      } else {
        c = conn2;
        i2++;
      }
    }

    Connection *newC = malloc(sizeof(Connection));
    *newC = *c;
    newC->enabled = c->enabled; // keep disabled status
    child->connections = realloc(
        child->connections, (child->numConnections + 1) * sizeof(Connection *));
    child->connections[child->numConnections++] = newC;
  }

  return child;
}

// ------------------- Serialization -------------------
void GENOME_save(Genome_t *genome, const char *filename) {
  FILE *f = fopen(filename, "wb");
  fwrite(&genome->numNodes, sizeof(size_t), 1, f);
  for (size_t i = 0; i < genome->numNodes; i++) {
    fwrite(genome->nodes[i], sizeof(Node), 1, f);
  }
  fwrite(&genome->numConnections, sizeof(size_t), 1, f);
  for (size_t i = 0; i < genome->numConnections; i++) {
    fwrite(genome->connections[i], sizeof(Connection), 1, f);
  }
  fwrite(&genome->fitness, sizeof(long double), 1, f);
  fclose(f);
}

Genome_t *GENOME_load(const char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f)
    return NULL;
  Genome_t *genome = (Genome_t *)malloc(sizeof(Genome_t));
  fread(&genome->numNodes, sizeof(size_t), 1, f);
  genome->nodes = (Node **)malloc(genome->numNodes * sizeof(Node *));
  for (size_t i = 0; i < genome->numNodes; i++) {
    genome->nodes[i] = (Node *)malloc(sizeof(Node));
    fread(genome->nodes[i], sizeof(Node), 1, f);
  }
  fread(&genome->numConnections, sizeof(size_t), 1, f);
  genome->connections =
      (Connection **)malloc(genome->numConnections * sizeof(Connection *));
  for (size_t i = 0; i < genome->numConnections; i++) {
    genome->connections[i] = (Connection *)malloc(sizeof(Connection));
    fread(genome->connections[i], sizeof(Connection), 1, f);
  }
  fread(&genome->fitness, sizeof(long double), 1, f);
  fclose(f);
  return genome;
}

// Initialize population
Population *POPULATION_init(size_t popSize, size_t numInputs,
                            size_t numOutputs) {
  Population *pop = malloc(sizeof(Population));
  pop->size = popSize;
  pop->genomes = malloc(popSize * sizeof(Genome_t *));
  for (size_t i = 0; i < popSize; i++)
    pop->genomes[i] = GENOME_init(numInputs, numOutputs);
  return pop;
}

// Free population
void POPULATION_destroy(Population *pop) {
  for (size_t i = 0; i < pop->size; i++)
    GENOME_destroy(pop->genomes[i]);
  free(pop->genomes);
  free(pop);
}

// ------------------- Compatibility Distance -------------------
long double compatibility_distance(Genome_t *g1, Genome_t *g2, long double c1,
                                   long double c2, long double c3) {
  size_t i1 = 0, i2 = 0;
  size_t excess = 0, disjoint = 0;
  long double weightDiff = 0.0L;
  size_t matching = 0;

  while (i1 < g1->numConnections || i2 < g2->numConnections) {
    if (i1 >= g1->numConnections) {
      excess++;
      i2++;
    } else if (i2 >= g2->numConnections) {
      excess++;
      i1++;
    } else {
      Connection *conn1 = g1->connections[i1];
      Connection *conn2 = g2->connections[i2];
      if (conn1->innovation == conn2->innovation) {
        weightDiff += fabsl(conn1->weight - conn2->weight);
        matching++;
        i1++;
        i2++;
      } else if (conn1->innovation < conn2->innovation) {
        disjoint++;
        i1++;
      } else {
        disjoint++;
        i2++;
      }
    }
  }

  long double avgWeightDiff = matching > 0 ? weightDiff / matching : 0.0L;
  size_t N = g1->numConnections > g2->numConnections ? g1->numConnections
                                                     : g2->numConnections;
  if (N < 20)
    N = 1; // small genome adjustment

  return (c1 * excess / N) + (c2 * disjoint / N) + (c3 * avgWeightDiff);
}

// ------------------- Species Assignment -------------------
size_t assign_species(Population *pop) {
  // Simplest: cluster by compatibility threshold
  const long double threshold = 3.0L; // adjust as needed
  size_t numSpecies = 0;
  size_t *species = calloc(pop->size, sizeof(size_t));

  for (size_t i = 0; i < pop->size; i++) {
    int found = 0;
    for (size_t j = 0; j < numSpecies; j++) {
      if (compatibility_distance(pop->genomes[i], pop->genomes[j], 1.0L, 1.0L,
                                 0.4L) < threshold) {
        species[i] = j;
        found = 1;
        break;
      }
    }
    if (!found) {
      species[i] = numSpecies++;
    }
  }
  free(species);
  return numSpecies;
}

// ------------------- Next Generation -------------------
// ------------------- Next Generation -------------------
void POPULATION_evolve(Population *pop) {
  if (!pop)
    return;

  // 1. Assign species
  size_t numSpecies = assign_species(pop);

  // 2. Sort genomes by fitness descending
  for (size_t i = 0; i < pop->size - 1; i++) {
    for (size_t j = i + 1; j < pop->size; j++) {
      if (pop->genomes[j]->fitness > pop->genomes[i]->fitness) {
        Genome_t *tmp = pop->genomes[i];
        pop->genomes[i] = pop->genomes[j];
        pop->genomes[j] = tmp;
      }
    }
  }

  // 3. Keep top 20% as elites
  size_t eliteCount = pop->size / 5;

  Genome_t **nextGen = malloc(pop->size * sizeof(Genome_t *));
  for (size_t i = 0; i < eliteCount; i++) {
    nextGen[i] =
        GENOME_crossover(pop->genomes[i], pop->genomes[i]); // self-copy
  }

  // 4. Fill remaining with crossover + extended mutation
  for (size_t i = eliteCount; i < pop->size; i++) {
    size_t p1 = rand() % eliteCount;
    size_t p2 = rand() % eliteCount;
    Genome_t *child = GENOME_crossover(pop->genomes[p1], pop->genomes[p2]);

    // Extended mutation parameters:
    // weightPerturbRate, weightPerturbAmount,
    // addNodeRate, addConnectionRate,
    // toggleConnectionRate, biasPerturbRate
    GENOME_mutate(child, 0.8L, 0.2L, 0.3L, 0.5L, 0.1L, 0.05L, 0.05L);

    nextGen[i] = child;
  }

  // 5. Destroy old population genomes
  for (size_t i = 0; i < pop->size; i++) {
    GENOME_destroy(pop->genomes[i]);
  }

  free(pop->genomes);
  pop->genomes = nextGen;
}

// Return innovation number, create if new
size_t get_innovation_number(size_t from, size_t to) {
  for (size_t i = 0; i < innovationCount; i++) {
    if (innovationHistory[i].from == from && innovationHistory[i].to == to)
      return innovationHistory[i].innovation;
  }
  // New innovation
  innovationHistory = realloc(innovationHistory,
                              (innovationCount + 1) * sizeof(InnovationRecord));
  innovationHistory[innovationCount].from = from;
  innovationHistory[innovationCount].to = to;
  innovationHistory[innovationCount].innovation = innovationCount;
  return innovationCount++;
}

// Helper: returns an array of node indices in topological order
size_t *topological_sort(Genome_t *genome, size_t *outSize) {
  size_t *inDegree = calloc(genome->numNodes, sizeof(size_t));
  for (size_t i = 0; i < genome->numConnections; i++) {
    Connection *c = genome->connections[i];
    if (c->enabled)
      inDegree[c->to->id]++;
  }

  size_t *order = malloc(genome->numNodes * sizeof(size_t));
  size_t idx = 0;

  size_t *queue = malloc(genome->numNodes * sizeof(size_t));
  size_t qstart = 0, qend = 0;

  for (size_t i = 0; i < genome->numNodes; i++) {
    if (inDegree[i] == 0)
      queue[qend++] = i;
  }

  while (qstart < qend) {
    size_t n = queue[qstart++];
    order[idx++] = n;

    for (size_t i = 0; i < genome->numConnections; i++) {
      Connection *c = genome->connections[i];
      if (c->enabled && c->from->id == n) {
        inDegree[c->to->id]--;
        if (inDegree[c->to->id] == 0)
          queue[qend++] = c->to->id;
      }
    }
  }

  free(queue);
  free(inDegree);
  *outSize = idx;
  return order;
}

void NEAT_train(NEAT_t *neat, long double *input, long double *target) {
  // Evaluate each genome
  for (size_t i = 0; i < neat->pop->size; i++) {
    long double output[neat->output_dims];
    GENOME_forward(neat->pop->genomes[i], input, output);

    // Simple fitness example: negative MSE
    long double fitness = 0.0L;
    for (size_t j = 0; j < neat->output_dims; j++)
      fitness -= (output[j] - target[j]) * (output[j] - target[j]);
    neat->pop->genomes[i]->fitness = fitness;
  }
  POPULATION_evolve(neat->pop);
}
