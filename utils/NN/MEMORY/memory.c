#include "MEMORY.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void init_memory(Memory *memory, int initial_capacity, int input_size) {
  memory->size = 0;
  memory->capacity = initial_capacity;
  memory->input_size = input_size;
  memory->index = 0;

  memory->buffer =
      (MemoryEntry *)calloc((size_t)initial_capacity, sizeof(MemoryEntry));

  if (memory->buffer == NULL) {
    fprintf(stderr, "Failed to allocate memory buffer\n");
  }
}

void store_memory(Memory *m, const long double *x, int action, float reward,
                  float value_estimate) {
  if (!m || !m->buffer)
    return;

  int idx = m->index % m->capacity;

  // allocate once per slot, reuse forever
  if (!m->buffer[idx].vision_inputs) {
    m->buffer[idx].vision_inputs =
        (long double *)malloc(sizeof(long double) * m->input_size);
    if (!m->buffer[idx].vision_inputs)
      return;
  }

  memcpy(m->buffer[idx].vision_inputs, x, sizeof(long double) * m->input_size);
  m->buffer[idx].action_taken = action;
  m->buffer[idx].reward = reward;
  m->buffer[idx].value_estimate = value_estimate;

  m->index++;
  if (m->size < m->capacity)
    m->size++;
}

void free_memory(Memory *memory) {
  if (memory->buffer == NULL)
    return;

  for (int i = 0; i < memory->size; i++) {
    if (memory->buffer[i].vision_inputs != NULL) {
      free(memory->buffer[i].vision_inputs);
    }
  }

  free(memory->buffer);
  memory->buffer = NULL;
  memory->size = 0;
  memory->capacity = 0;
}
