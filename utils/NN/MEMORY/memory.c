#include "MEMORY.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void init_memory(Memory *memory, int initial_capacity, int input_size) {
  memory->size = 0;
  memory->capacity = initial_capacity;

  // Allocate the initial array of entries
  memory->buffer =
      (MemoryEntry *)malloc(initial_capacity * sizeof(MemoryEntry));

  if (memory->buffer == NULL) {
    fprintf(stderr, "Failed to allocate memory buffer\n");
  }
}

void store_memory(Memory *memory, long double *vision_inputs, int action,
                  float reward, float value_estimate) {

  // Check if we need more "infinite" space
  if (memory->size >= memory->capacity) {
    int new_capacity = memory->capacity * 2;
    MemoryEntry *new_buffer =
        realloc(memory->buffer, new_capacity * sizeof(MemoryEntry));

    if (new_buffer == NULL) {
      fprintf(stderr, "Critical: Could not expand memory capacity!\n");
      return;
    }

    memory->buffer = new_buffer;
    memory->capacity = new_capacity;
  }

  int idx = memory->size;

  // Allocate space for the vision data for this specific entry
  memory->buffer[idx].vision_inputs =
      malloc(memory->input_size * sizeof(long double));

  if (memory->buffer[idx].vision_inputs != NULL) {
    memcpy(memory->buffer[idx].vision_inputs, vision_inputs,
           memory->input_size * sizeof(long double));
  }

  memory->buffer[idx].action_taken = action;
  memory->buffer[idx].reward = reward;
  memory->buffer[idx].value_estimate = value_estimate;

  memory->size++;
}

void free_memory(Memory *memory) {
  if (memory->buffer == NULL)
    return;

  // We must free the vision_inputs for EVERY entry we created
  for (int i = 0; i < memory->size; i++) {
    free(memory->buffer[i].vision_inputs);
  }

  free(memory->buffer);
  memory->buffer = NULL;
  memory->size = 0;
  memory->capacity = 0;
}
