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
  MemoryEntry *buffer; // Dynamic array
  int size;            // Current number of entries
  int capacity;        // Maximum entries before resize
  int index;           // Current write head
} Memory;

void init_memory(Memory *memory, int initial_capacity, int input_size);
void store_memory(Memory *memory, long double *vision_inputs, int action,
                  float reward, float value_estimate, int input_size);
void free_memory(Memory *memory);

#endif
