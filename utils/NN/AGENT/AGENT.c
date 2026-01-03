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

void update_agent(GameState *game, int agent_idx) {
  Agent *agent = &game->agents[agent_idx];
    agent->time_alive += GetFrameTime();

    // 1️⃣ Encode vision
    encode_vision(game, agent_idx, game->vision_inputs);

    // 2️⃣ Get NN output (action)
    long double *outputs = NEAT_forward(agent->brain, game->vision_inputs);
    Action action = ACTION_NONE;

    if (outputs) {
        action = get_action_from_output(outputs);
        execute_action(game, agent_idx, action);
        game->last_actions[agent_idx] = action;
    }

    // 3️⃣ Store in memory
    float reward = (float)agent->total_xp; // Simple reward: XP gained
    float value_estimate = 0.0f;           // Placeholder for MuZero value function
    store_memory(&agent->memory, game->vision_inputs, (int)action, reward, value_estimate, agent->input_size);

    // 4️⃣ Handle breeding
    if (agent->is_breeding) {
        agent->breeding_timer += GetFrameTime();
        if (agent->breeding_timer >= BREEDING_DURATION) {
            handle_breeding(game, agent_idx);
        }
    }

    // 5️⃣ Update agent stats
    update_agent_color(agent);
    update_agent_size(agent);
}

