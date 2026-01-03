#include "AGENT.h"

void init_agent(Agent *agent, int id) {
  agent->level = 0;
  agent->total_xp = 0;
  agent->size = INITIAL_AGENT_SIZE;
  agent->time_alive = 0;
  agent->agent_id = id;
  agent->parent_id = -1;
  agent->num_offsprings = 0;
  agent->num_eaten = 0;
  agent->is_breeding = false;
  agent->breeding_timer = 0;
  agent->color = WHITE;

  // Random starting position
  agent->rect.x = (float)(rand() % (SCREEN_WIDTH - 10));
  agent->rect.y = (float)(rand() % (SCREEN_HEIGHT - 10));
  agent->rect.width = agent->size;
  agent->rect.height = agent->size;

  // Neural network
  agent->input_size = get_total_input_size();
  agent->brain = NEAT_init(agent->input_size, ACTION_COUNT, 1);

  // Initialize memory
  init_memory(&agent->memory, agent->input_size);
}
