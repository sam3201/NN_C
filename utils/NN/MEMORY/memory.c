#include "MEMORY.h"
#include <stdlib.h>
#include <string.h>

void init_memory(Memory *memory, int input_size) {
  memory->size = 0;
  memory->index = 0;
  for (int i = 0; i < MEMORY_CAPACITY; i++) {
    memory->buffer[i].vision_inputs = malloc(input_size * sizeof(long double));
  }
}

void store_memory(Memory *memory, long double *vision_inputs, int action,
                  float reward, float value_estimate, int input_size) {
  int idx = memory->index % MEMORY_CAPACITY;
  memcpy(memory->buffer[idx].vision_inputs, vision_inputs,
         input_size * sizeof(long double));
  memory->buffer[idx].action_taken = action;
  memory->buffer[idx].reward = reward;
  memory->buffer[idx].value_estimate = value_estimate;

  memory->index++;
  if (memory->size < MEMORY_CAPACITY)
    memory->size++;
}

MemoryEntry *sample_memory(Memory *memory, int idx) {
  if (idx >= memory->size)
    return NULL;
  return &memory->buffer[idx];
}
