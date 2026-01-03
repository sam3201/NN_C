#ifndef MEMORY_H
#define MEMORY_H

#define MEMORY_CAPACITY 1000

typedef struct {
  long double *vision_inputs;
  int action_taken;
  float reward;
  float value_estimate;
} MemoryEntry;

typedef struct {
  MemoryEntry buffer[MEMORY_CAPACITY];
  int size;
  int index; // Circular buffer
} Memory;

// Memory interface
void init_memory(Memory *memory, int input_size);
void store_memory(Memory *memory, long double *vision_inputs, int action,
                  float reward, float value_estimate, int input_size);
MemoryEntry *sample_memory(Memory *memory, int idx);

#endif
