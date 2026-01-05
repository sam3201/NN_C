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


3. Evolve Population

Once fitness is assigned, create the next generation:

POPULATION_evolve(pop);


Uses compatibility distance for speciation

Preserves top 20% elite genomes

Fills remaining population with crossover and mutation

4. Repeat Evaluation + Evolution

Loop evaluation and evolution for the desired number of generations:

for (int gen = 0; gen < numGenerations; gen++) {
    evaluate_population(pop);
    POPULATION_evolve(pop);
}

5. Forward Propagation

To run a genome on input data:

long double input[numInputs] = {...};
long double output[numOutputs];

GENOME_forward(genome, input, output);

6. Save and Load Genomes

Save a genome:

GENOME_save(pop->genomes[0], "best_genome.dat");


Load a genome:

Genome_t *loaded = GENOME_load("best_genome.dat");

Function Order Summary

POPULATION_init – initialize population

evaluate_population – assign fitness to each genome

POPULATION_evolve – produce next generation

Repeat evaluation + evolution for N generations

GENOME_forward – test genomes on inputs

GENOME_save / GENOME_load – serialize best genome


