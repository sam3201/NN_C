#include "NEAT.h"

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

// ------------------- Genome ----------------------
Genome *Genome_init(size_t numInputs, size_t numOutputs) {
  Genome *genome = (Genome *)malloc(sizeof(Genome));
  genome->numNodes = numInputs + numOutputs + 1; // +1 for bias
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
  // Bias node
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

void Genome_destroy(Genome *genome) {
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

// ------------------- Forward ----------------------
void Genome_forward(Genome *genome, long double *input, long double *output) {
  // Load inputs
  size_t inputIdx = 0;
  for (size_t i = 0; i < genome->numNodes; i++) {
    if (genome->nodes[i]->type == INPUT_NODE)
      genome->nodes[i]->value = input[inputIdx++];
  }

  // TODO: Topological sort & propagate connections
  // For now, assume fully connected input → output for prototype
  for (size_t i = 0; i < genome->numConnections; i++) {
    Connection *c = genome->connections[i];
    if (c->enabled) {
      c->to->value += c->from->value * c->weight;
    }
  }

  // Apply activation for non-input nodes
  for (size_t i = 0; i < genome->numNodes; i++) {
    Node *n = genome->nodes[i];
    if (n->type != INPUT_NODE && n->type != BIAS_NODE)
      n->value = activate(n->value, n->actFunc);
  }

  // Fill output array
  size_t outIdx = 0;
  for (size_t i = 0; i < genome->numNodes; i++) {
    if (genome->nodes[i]->type == OUTPUT_NODE)
      output[outIdx++] = genome->nodes[i]->value;
  }
}
