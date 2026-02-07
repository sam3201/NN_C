#ifndef MEMORY_H
#define MEMORY_H

#include <stdlib.h>

typedef struct {
  long double *vision_inputs;
  int action_taken;
  float reward;
  float value_estimate;
} MemoryEntry;

typedef struct {
  MemoryEntry *buffer;
  int size;
  int capacity;
  int input_size;
  int index;
} Memory;

void init_memory(Memory *memory, int initial_capacity, int input_size);
void store_memory(Memory *m, const long double *x, int action, float reward,
                  float value_estimate);
void free_memory(Memory *memory);

#endif
