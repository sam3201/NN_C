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
  // Check if we need to expand
  if (memory->size >= memory->capacity) {
    memory->capacity *= 2; // Double the "infinite" space
    memory->buffer =
        realloc(memory->buffer, sizeof(MemoryEntry) * memory->capacity);
  }

  int idx = memory->size;
  memory->buffer[idx].vision_inputs = malloc(input_size * sizeof(long double));
  memcpy(memory->buffer[idx].vision_inputs, vision_inputs,
         input_size * sizeof(long double));

  memory->buffer[idx].action_taken = action;
  memory->buffer[idx].reward = reward;
  memory->buffer[idx].value_estimate = value_estimate;

  memory->size++;
}
