// agent.h

#include "../NN_C/memory/memory.h"

typedef struct {
  Vector2 position;
  Vector2 velocity;
  Rectangle rect;
  unsigned int size;
  int level;
  int total_xp;
  float time_alive;
  int agent_id;
  int parent_id;
  int num_offsprings;
  int num_eaten;
  bool is_breeding;
  float breeding_timer;
  Color color;

  // Neural Network
  NEAT_t *brain;

  // MEMORY SYSTEM
  Memory memory;     // Replay memory for MuZero / RL
  size_t input_size; // Total input size for vision + additional inputs
} Agent;
