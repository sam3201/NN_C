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

// ------------------- Forward Propagation (Topological) -------------------
#include <string.h>
void Genome_forward(Genome *genome, long double *input, long double *output) {
  // reset all node values except bias
  for (size_t i = 0; i < genome->numNodes; i++) {
    if (genome->nodes[i]->type != BIAS_NODE)
      genome->nodes[i]->value = 0.0L;
  }

  size_t inputIdx = 0;
  for (size_t i = 0; i < genome->numNodes; i++) {
    if (genome->nodes[i]->type == INPUT_NODE)
      genome->nodes[i]->value = input[inputIdx++];
  }

  // Simple topological propagation: naive, repeat passes until stable
  for (size_t pass = 0; pass < genome->numNodes; pass++) {
    for (size_t i = 0; i < genome->numConnections; i++) {
      Connection *c = genome->connections[i];
      if (c->enabled)
        c->to->value += c->from->value * c->weight;
    }
  }

  for (size_t i = 0; i < genome->numNodes; i++) {
    Node *n = genome->nodes[i];
    if (n->type != INPUT_NODE && n->type != BIAS_NODE)
      n->value = activate(n->value, n->actFunc);
  }

  // output
  size_t outIdx = 0;
  for (size_t i = 0; i < genome->numNodes; i++) {
    if (genome->nodes[i]->type == OUTPUT_NODE)
      output[outIdx++] = genome->nodes[i]->value;
  }
}

// ------------------- Mutation -------------------
void Genome_add_connection(Genome *genome, size_t fromNode, size_t toNode,
                           long double weight) {
  genome->connections = realloc(
      genome->connections, (genome->numConnections + 1) * sizeof(Connection *));
  Connection *c = (Connection *)malloc(sizeof(Connection));
  c->from = genome->nodes[fromNode];
  c->to = genome->nodes[toNode];
  c->weight = weight;
  c->enabled = true;
  c->innovation = genome->numConnections; // simple innovation
  genome->connections[genome->numConnections++] = c;
}

void Genome_add_node(Genome *genome, size_t connectionIndex) {
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

  Genome_add_connection(genome, c->from->id, n->id, 1.0L);
  Genome_add_connection(genome, n->id, c->to->id, c->weight);
}

void Genome_mutate_weights(Genome *genome, long double perturbRate,
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

// ------------------- Crossover -------------------
Genome *Genome_crossover(Genome *p1, Genome *p2) {
  if (!p1 || !p2)
    return NULL;
  Genome *child = Genome_init(0, 0);
  // copy nodes
  for (size_t i = 0; i < p1->numNodes; i++) {
    Node *n = (Node *)malloc(sizeof(Node));
    *n = *(p1->nodes[i]);
    child->nodes =
        realloc(child->nodes, (child->numNodes + 1) * sizeof(Node *));
    child->nodes[child->numNodes++] = n;
  }
  // copy connections randomly
  for (size_t i = 0; i < p1->numConnections; i++) {
    Connection *c1 = p1->connections[i];
    Connection *c2 = i < p2->numConnections ? p2->connections[i] : NULL;
    Connection *c = (Connection *)malloc(sizeof(Connection));
    if (c2 && rand() % 2)
      *c = *c2;
    else
      *c = *c1;
    child->connections = realloc(
        child->connections, (child->numConnections + 1) * sizeof(Connection *));
    child->connections[child->numConnections++] = c;
  }
  return child;
}

// ------------------- Serialization -------------------
void Genome_save(Genome *genome, const char *filename) {
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

Genome *Genome_load(const char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f)
    return NULL;
  Genome *genome = (Genome *)malloc(sizeof(Genome));
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
