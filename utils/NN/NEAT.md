# NEAT C Implementation

This is a C implementation of the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. It supports:

- Genome construction with nodes and connections
- Forward propagation with topological sorting
- Mutation (weights, nodes, connections)
- Crossover between genomes
- Serialization (save/load)
- Population evolution with speciation and elitism

---

## Getting Started

### 1. Initialize a Population

Create a population with a given size and number of input/output nodes:

```c
Population *pop = POPULATION_init(popSize, numInputs, numOutputs);


2. Evaluate Fitness

Run your simulation/game for each genome and assign a fitness score:

pop->genomes[i]->fitness = calculate_fitness(pop->genomes[i]);

