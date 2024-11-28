#include "../utils/NN/NN.h"
#include "../utils/NN/NEAT.h"
#include "../utils/Raylib/raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define POPULATION_SIZE 20 
#define MAX_FOOD 100
#define MAX_GROUNDSKEEPERS POPULATION_SIZE / 2
#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define FRAME_RATE 60
#define XP_PER_LEVEL 100
#define XP_FROM_FOOD 1  // Added XP gain values
#define XP_FROM_AGENT 25
#define XP_FROM_OFFSPRING 50
#define BREEDING_DURATION 2.0f
#define INITIAL_AGENT_SIZE 1
#define MOVEMENT_SPEED 2.0f
#define FOOD_SIZE 5
#define FOOD_SPAWN_CHANCE 0.1f  // 10% chance per frame to spawn new food

// Add these constants for entity types
#define VISION_EMPTY 0
#define VISION_SELF 1
#define VISION_FOOD 2
#define VISION_OFFSPRING 3
#define VISION_SAME_SIZE 5
#define VISION_OTHER 6

// Add this constant for label size
#define LABEL_SIZE 10

typedef enum {
    ACTION_NONE = 0,
    ACTION_MOVE_LEFT = 1,
    ACTION_MOVE_RIGHT = 2,
    ACTION_MOVE_UP = 3,
    ACTION_MOVE_DOWN = 4,
    ACTION_COUNT = 5
} Action;

// Add these struct definitions before GameState
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
    NEAT_t* brain;
} Agent;

typedef struct {
    Vector2 position;
    Rectangle rect;
} Food;

// Remove GameStateEnum since we're not using states anymore
typedef struct {
    Agent agents[POPULATION_SIZE];
    Food food[MAX_FOOD];
    int scores[POPULATION_SIZE];
    Action last_actions[POPULATION_SIZE];
    bool over;
    bool paused;
    float evolution_timer;
    unsigned int current_generation;
    NEAT_t* neat_populations[POPULATION_SIZE];
    long double* vision_inputs;
    int next_agent_id;
    int current_player_idx;
    unsigned int num_active_players;
} GameState;

// Add these function declarations after the struct definitions and before any function implementations
void init_agent(Agent* agent, int id);
void init_game(GameState* game);
void update_game(GameState* game);
void update_agent_stats(Agent* agent);
void encode_vision(GameState* game, int player_idx, long double* vision_output);
void spawn_food(Food* food);
void execute_action(GameState* game, int agent_idx, Action action);
void handle_agent_collision(GameState* game, int agent1_idx, int agent2_idx);
void eat_agent(Agent* predator, Agent* prey);
void kill_agent(GameState* game, int agent_idx);
void start_breeding(Agent* agent1, Agent* agent2);
void handle_breeding(GameState* game, int agent_idx);
void eat_food(Agent* agent);
void transfer_weights_with_mutation(NN_t* old_nn, NN_t* new_nn, int level);
void check_collisions(GameState* game, int agent_idx);
void update_agent(GameState* game, int agent_idx);
void draw_game_state(GameState* game);
void draw_stats(unsigned int generation, unsigned int current_player, 
                unsigned int total_players, float current_fitness, float best_fitness);
void update_game(GameState* game);
void apply_action(GameState* game, Action action);
void update_agent_physics(Agent* agent);
void init_agent(Agent* agent, int id);
void level_up(Agent* agent);
void init_food(Food* food);
void try_breed(GameState* game, int p1_idx, int p2_idx);
void cleanup_game_state(GameState* game);
void update_agent_size(Agent* agent);
void update_agent_color(Agent* agent);
void move_agent(GameState* game, Agent* agent, Action action);
bool check_collision(Rectangle rect1, Rectangle rect2);
void evolve_population(GameState* game);
void handle_breeding_completion(GameState* game, int agent_idx);
void evolve_agent(Agent* agent);
void transfer_weights(NN_t* old_nn, NN_t* new_nn);

// Add this helper function to get total input size
size_t get_total_input_size() {
    return (SCREEN_WIDTH * SCREEN_HEIGHT) + 7;  // Vision grid + 7 additional inputs
}

// Update encode_vision to handle all inputs
void encode_vision(GameState* game, int player_idx, long double* vision_output) {
    size_t vision_size = SCREEN_WIDTH * SCREEN_HEIGHT;
    Agent* current = &game->agents[player_idx];
    
    // 1. First encode the vision grid (as before)
    memset(vision_output, 0, vision_size * sizeof(long double));
    
    // Encode current agent (SELF)
    for (int y = 0; y < current->size; y++) {
        for (int x = 0; x < current->size; x++) {
            int pos_x = (int)current->position.x + x;
            int pos_y = (int)current->position.y + y;
            
            if (pos_x >= 0 && pos_x < SCREEN_WIDTH && 
                pos_y >= 0 && pos_y < SCREEN_HEIGHT) {
                vision_output[pos_y * SCREEN_WIDTH + pos_x] = VISION_SELF;
            }
        }
    }
    
    // Encode other agents
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (i == player_idx) continue;
        
        Agent* other = &game->agents[i];
        int vision_type = (other->parent_id == current->agent_id) ? 
                         VISION_OFFSPRING : VISION_OTHER;
        
        for (int y = 0; y < other->size; y++) {
            for (int x = 0; x < other->size; x++) {
                int pos_x = (int)other->position.x + x;
                int pos_y = (int)other->position.y + y;
                
                if (pos_x >= 0 && pos_x < SCREEN_WIDTH && 
                    pos_y >= 0 && pos_y < SCREEN_HEIGHT) {
                    vision_output[pos_y * SCREEN_WIDTH + pos_x] = vision_type;
                }
            }
        }
    }
    
    // Encode food
    for (int i = 0; i < MAX_FOOD; i++) {
        Food* food = &game->food[i];
        int pos_x = (int)food->position.x;
        int pos_y = (int)food->position.y;
        
        if (pos_x >= 0 && pos_x < SCREEN_WIDTH && 
            pos_y >= 0 && pos_y < SCREEN_HEIGHT) {
            vision_output[pos_y * SCREEN_WIDTH + pos_x] = VISION_FOOD;
        }
    }
    
    // 2. Add the additional inputs after the vision grid
    size_t additional_input_idx = vision_size;
    
    // Normalize values to range [0, 1] for neural network
    vision_output[additional_input_idx++] = current->is_breeding ? 1.0L : 0.0L;
    vision_output[additional_input_idx++] = (long double)current->num_offsprings / POPULATION_SIZE;
    vision_output[additional_input_idx++] = (long double)current->parent_id / POPULATION_SIZE;
    vision_output[additional_input_idx++] = (long double)current->agent_id / POPULATION_SIZE;
    vision_output[additional_input_idx++] = (long double)current->size; 
    vision_output[additional_input_idx++] = (long double)current->level; 
    vision_output[additional_input_idx] = (long double)current->total_xp;
}

// Update the neural network initialization to account for all inputs
void init_agent_brain(Agent* agent) {
    size_t total_inputs = get_total_input_size();
    agent->brain = NEAT_init(total_inputs, ACTION_COUNT, 1);  
    
    if (!agent->brain) {
        fprintf(stderr, "Failed to initialize agent brain\n");
        exit(1);
    }
}

// Helper function to process neural network output into an action
Action get_action_from_output(long double* outputs) {
    Action best_action = ACTION_NONE;
    long double max_value = -INFINITY;
    
    // Find action with highest output value
    for (int i = 0; i < ACTION_COUNT; i++) {
        if (outputs[i] > max_value) {
            max_value = outputs[i];
            best_action = (Action)i;
        }
    }
    
    return best_action;
}

void init_agent(Agent* agent, int id) {
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
    
    // Initialize brain
    agent->brain = NEAT_init(get_total_input_size(), ACTION_COUNT, 1);
}

void update_agent(GameState* game, int agent_idx) {
    Agent* agent = &game->agents[agent_idx];
    agent->time_alive += GetFrameTime();
    
    // Update vision and get next action
    encode_vision(game, agent_idx, game->vision_inputs);
    long double* outputs = NEAT_forward(agent->brain, game->vision_inputs);
    
    if (outputs) {
        // Find highest output (action)
        Action action = ACTION_NONE;
        long double max_output = outputs[0];
        for (int i = 1; i < ACTION_COUNT; i++) {
            if (outputs[i] > max_output) {
                max_output = outputs[i];
                action = i;
            }
        }
        
        // Execute action
        execute_action(game, agent_idx, action);
        game->last_actions[agent_idx] = action;
    }
    
    // Update breeding status
    if (agent->is_breeding) {
        agent->breeding_timer += GetFrameTime();
        if (agent->breeding_timer >= BREEDING_DURATION) {
            handle_breeding(game, agent_idx);
        }
    }
}

void update_agent_size(Agent* agent) {
    agent->size = agent->level + 1;  // Size is level + 1 (so level 0 = size 1)
    agent->rect = (Rectangle){
        agent->position.x,
        agent->position.y,
        agent->size,
        agent->size
    };
}

void update_agent_color(Agent* agent) {
    float xp_factor = (float)agent->total_xp / (float)(agent->level * XP_PER_LEVEL);
    float level_factor = (float)agent->level / 10.0f;  // Assuming max level is 10
    float size_factor = (float)agent->size / (float)(SCREEN_WIDTH / 10);
    
    agent->color = (Color){
        (unsigned char)(255 * xp_factor),     // Red component
        (unsigned char)(255 * level_factor),  // Green component
        (unsigned char)(255 * size_factor),   // Blue component
        255                                   // Alpha
    };
}

void level_up(Agent* agent) {
    while (agent->total_xp >= (agent->level + 1) * XP_PER_LEVEL) {
        agent->total_xp -= (agent->level + 1) * XP_PER_LEVEL;
        agent->level++;
        agent->size = agent->level + 1;
        
        // Update rectangle size
        agent->rect.width = agent->size;
        agent->rect.height = agent->size;
        
        // Update color based on new level and XP
        update_agent_color(agent);
        
        // Evolve the agent's neural network
        evolve_agent(agent);
        
        // Print evolution message
        printf("Agent %d evolved to level %d!\n", agent->agent_id, agent->level);
    }
}

// Add XP gain functions
void gain_xp_from_food(Agent* agent) {
    agent->total_xp += XP_FROM_FOOD;
    if (agent->total_xp >= (agent->level + 1) * XP_PER_LEVEL) {
        level_up(agent);
    }
}

void gain_xp_from_agent(Agent* agent, Agent* eaten_agent) {
    agent->total_xp += XP_FROM_AGENT;
    if (agent->total_xp >= (agent->level + 1) * XP_PER_LEVEL) {
        level_up(agent);
    }
}

void gain_xp_from_offspring(Agent* agent) {
    agent->total_xp += XP_FROM_OFFSPRING;
    if (agent->total_xp >= (agent->level + 1) * XP_PER_LEVEL) {
        level_up(agent);
    }
}

// Calculate fitness for NEAT evolution
float calculate_fitness(Agent* agent) {
    return (float)(
        agent->total_xp +
        agent->num_offsprings + 
        agent->num_eaten +
        agent->level 
    );
}

bool can_move_to(GameState* game, Agent* agent, Vector2 new_pos) {
    Rectangle new_rect = {
        new_pos.x,
        new_pos.y,
        agent->rect.width,
        agent->rect.height
    };
    
    // Check screen boundaries
    if (new_pos.x < 0 || new_pos.x + agent->size > SCREEN_WIDTH ||
        new_pos.y < 0 || new_pos.y + agent->size > SCREEN_HEIGHT) {
        return false;
    }
    
    // Check collision with other agents
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (game->agents[i].level >= 0 && &game->agents[i] != agent) {
            if (CheckCollisionRecs(new_rect, game->agents[i].rect)) {
                return false;
            }
        }
    }
    
    return true;
}

void move_agent(GameState* game, Agent* agent, Action action) {
    Vector2 new_pos = agent->position;
    float move_speed = 1.0f;  // Fixed movement speed
    
    switch(action) {
        case ACTION_MOVE_LEFT:
            new_pos.x -= move_speed;
            break;
        case ACTION_MOVE_RIGHT:
            new_pos.x += move_speed;
            break;
        case ACTION_MOVE_UP:
            new_pos.y -= move_speed;
            break;
        case ACTION_MOVE_DOWN:
            new_pos.y += move_speed;
            break;
        default:
            return;
    }
    
    if (can_move_to(game, agent, new_pos)) {
        agent->position = new_pos;
        agent->rect.x = new_pos.x;
        agent->rect.y = new_pos.y;
    }
}

bool check_collision(Rectangle rect1, Rectangle rect2) {
    return CheckCollisionRecs(rect1, rect2);
}

void update_game(GameState* game) {
    // Spawn new food if needed
    for (int i = 0; i < MAX_FOOD; i++) {
        if (game->food[i].rect.width == 0 && ((float)rand() / RAND_MAX) < FOOD_SPAWN_CHANCE) {
            spawn_food(&game->food[i]);
        }
    }
    
    // Update each agent
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (game->agents[i].level >= 0) {
            update_agent(game, i);
        }
    }
}

void check_collisions(GameState* game, int agent_idx) {
    Agent* agent = &game->agents[agent_idx];
    
    // Check food collisions
    for (int i = 0; i < MAX_FOOD; i++) {
        if (CheckCollisionRecs(agent->rect, game->food[i].rect)) {
            eat_food(agent);
            game->food[i].rect.width = 0;  // Mark food as eaten
        }
    }
    
    // Check agent collisions
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (i != agent_idx && game->agents[i].level >= 0) {
            if (CheckCollisionRecs(agent->rect, game->agents[i].rect)) {
                handle_agent_collision(game, agent_idx, i);
            }
        }
    }
}

void evolve_population(GameState* game) {
    // Calculate fitness for all agents
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (game->agents[i].level >= 0) {
            game->agents[i].brain->nodes[0]->fitness = calculate_fitness(&game->agents[i]);
        }
    }
    
    // Evolve each agent's neural network
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (game->agents[i].brain) {
            NEAT_evolve(game->agents[i].brain);
        }
    }
    
    game->current_generation++;
}

void init_game(GameState* game) {
    game->current_generation = 0;
    game->num_active_players = POPULATION_SIZE;
    game->next_agent_id = 0;
    game->paused = false;
    game->vision_inputs = malloc(get_total_input_size() * sizeof(long double));
    
    // Initialize agents
    for (int i = 0; i < POPULATION_SIZE; i++) {
        init_agent(&game->agents[i], game->next_agent_id++);
    }
    
    // Initialize food
    for (int i = 0; i < MAX_FOOD; i++) {
        spawn_food(&game->food[i]);
    }
}

void handle_breeding_completion(GameState* game, int agent_idx) {
    Agent* parent1 = &game->agents[agent_idx];
    
    // Find breeding partner
    int partner_idx = -1;
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (i != agent_idx && game->agents[i].is_breeding && 
            game->agents[i].breeding_timer <= 0) {
            partner_idx = i;
            break;
        }
    }
    
    if (partner_idx >= 0) {
        Agent* parent2 = &game->agents[partner_idx];
        
        // Find empty slot for offspring
        int offspring_idx = -1;
        for (int i = 0; i < POPULATION_SIZE; i++) {
            if (game->agents[i].level < 0) {
                offspring_idx = i;
                break;
            }
        }
        
        if (offspring_idx >= 0) {
            // Create offspring
            init_agent(&game->agents[offspring_idx], game->next_agent_id++);
            Agent* offspring = &game->agents[offspring_idx];
            
            // Set offspring properties
            offspring->level = (parent1->level + parent2->level) / 2;
            offspring->size = offspring->level + 1;
            offspring->parent_id = parent1->agent_id;
            
            // Crossover neural networks
            offspring->brain = NEAT_crossover(parent1->brain->nodes[0], 
                                           parent2->brain->nodes[0]);
            
            // Update parent stats
            parent1->num_offsprings++;
            parent2->num_offsprings++;
            parent1->total_xp += offspring->total_xp;
            parent2->total_xp += offspring->total_xp;
            
            // Reset breeding state
            parent1->is_breeding = false;
            parent2->is_breeding = false;
            game->num_active_players++;
        }
    }
    
    // Reset breeding state if no partner found
    parent1->is_breeding = false;
}

void evolve_agent(Agent* agent) {
    if (!agent->brain || !agent->brain->nodes[0]) return;
    
    Perceptron_t* old_brain = agent->brain->nodes[0];
    
    // Calculate new network dimensions based on level and maturity
    unsigned int base_neurons = 16;  // Base number of neurons
    unsigned int maturity_bonus = (unsigned int)(agent->time_alive / 60.0f);  // Bonus neurons based on survival time
    unsigned int level_bonus = agent->level * 4;  // Bonus neurons based on level
    unsigned int experience_bonus = (agent->num_eaten + agent->num_offsprings) * 2;  // Bonus based on experiences
    
    // Calculate total neurons for each hidden layer
    unsigned int total_neurons = base_neurons + maturity_bonus + level_bonus + experience_bonus;
    
    // Create new layer configuration with growing complexity
    size_t layers[] = {
        get_total_input_size(),     // Input layer
        total_neurons,              // First hidden layer (largest)
        total_neurons * 3/4,        // Second hidden layer (75% of first)
        total_neurons / 2,          // Third hidden layer (50% of first)
        total_neurons / 4,          // Fourth hidden layer (25% of first)
        ACTION_COUNT,               // Output layer
        0                          // Terminator
    };
    
    // Create activation functions array with increasing complexity
    ActivationFunctionType actFuncs[] = {
        agent->level > 15 ? LINEAR : RELU,     // Input processing
        agent->level > 10 ? TANH : RELU,       // Complex pattern recognition
        agent->level > 5 ? SIGMOID : RELU,     // Decision making
        RELU,                                  // Action selection
        SIGMOID                                // Output normalization
    };
    
    ActivationDerivativeType actDerivs[] = {
        agent->level > 15 ? LINEAR_DERIVATIVE : RELU_DERIVATIVE,
        agent->level > 10 ? TANH_DERIVATIVE : RELU_DERIVATIVE,
        agent->level > 5 ? SIGMOID_DERIVATIVE : RELU_DERIVATIVE,
        RELU_DERIVATIVE,
        SIGMOID_DERIVATIVE
    };
    
    // Adjust learning parameters based on maturity
    long double base_learning_rate = 0.01L;
    long double maturity_factor = 1.0L / (1.0L + agent->time_alive / 3600.0f);  // Decreases with age
    long double experience_factor = 1.0L / (1.0L + (agent->num_eaten + agent->num_offsprings) * 0.1L);
    long double learning_rate = base_learning_rate * maturity_factor * experience_factor;
    
    // Create new neural network with enhanced configuration
    NN_t* new_nn = NN_init(
        layers,
        actFuncs,
        actDerivs,
        agent->level > 10 ? CE : (agent->level > 5 ? HUBER : MSE),  // Loss function complexity increases with level
        agent->level > 10 ? CE_DERIVATIVE : (agent->level > 5 ? HUBER_DERIVATIVE : MSE_DERIVATIVE),
        agent->level > 8 ? L2 : L1,  // Switch to L2 regularization at higher levels
        agent->level > 12 ? ADAM : (agent->level > 6 ? RMSPROP : SGD),  // Optimizer complexity increases with level
        learning_rate
    );
    
    if (!new_nn) return;
    
    // Create new perceptron with the enhanced network
    Perceptron_t* new_brain = Perceptron_init(new_nn, old_brain->fitness, 0, NULL, true);
    if (!new_brain) {
        NN_destroy(new_nn);  // Using NN_free instead of NN_destroy
        return;
    }
    
    // Transfer and mutate weights
    transfer_weights_with_mutation(old_brain->nn, new_brain->nn, agent->level);
    
    // Update the agent's brain
    NEAT_destroy(agent->brain);
    agent->brain = NEAT_init(get_total_input_size(), ACTION_COUNT, 1);
    if (agent->brain) {
        if (agent->brain->nodes[0]) {
            Perceptron_destroy(agent->brain->nodes[0]);
        }
        agent->brain->nodes[0] = new_brain;
    }
    
    printf("Agent %d evolved! Level: %d, Neurons: %u, Time Alive: %.2f, Eaten: %d, Offspring: %d\n",
           agent->agent_id, agent->level, total_neurons, agent->time_alive, 
           agent->num_eaten, agent->num_offsprings);
}

void transfer_weights_with_mutation(NN_t* old_nn, NN_t* new_nn, int level) {
    if (!old_nn || !new_nn) return;
    
    // Copy weights with possible mutations
    for (size_t i = 0; i < old_nn->numLayers - 1; i++) {
        size_t weights_size = old_nn->layers[i] * old_nn->layers[i + 1];  // Using layerSizes instead of layers
        for (size_t j = 0; j < weights_size; j++) {
            // Higher level means less mutation
            float mutation_chance = 0.1f / (level + 1);
            if ((float)rand() / RAND_MAX < mutation_chance) {
                // Apply mutation
                new_nn->weights[i][j] = old_nn->weights[i][j] + 
                    ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
            } else {
                // Direct copy
                new_nn->weights[i][j] = old_nn->weights[i][j];
            }
        }
    }
}

void spawn_food(Food* food) {
    food->position.x = (float)(rand() % (SCREEN_WIDTH - FOOD_SIZE));
    food->position.y = (float)(rand() % (SCREEN_HEIGHT - FOOD_SIZE));
    food->rect = (Rectangle){
        food->position.x,
        food->position.y,
        FOOD_SIZE,
        FOOD_SIZE
    };
}

void execute_action(GameState* game, int agent_idx, Action action) {
    Agent* agent = &game->agents[agent_idx];
    float speed = MOVEMENT_SPEED;  // Changed from 1.0f to MOVEMENT_SPEED
    
    // Store previous position in case we need to revert
    float prev_x = agent->rect.x;
    float prev_y = agent->rect.y;
    
    // Apply movement based on action
    switch (action) {
        case ACTION_MOVE_LEFT:
            agent->rect.x -= speed;
            break;
        case ACTION_MOVE_RIGHT:
            agent->rect.x += speed;
            break;
        case ACTION_MOVE_UP:
            agent->rect.y -= speed;
            break;
        case ACTION_MOVE_DOWN:
            agent->rect.y += speed;
            break;
        default:
            break;
    }
    
    // Handle wall collisions by bouncing
    if (agent->rect.x <= 0) {
        agent->rect.x = speed;  // Move away from left wall
    } else if (agent->rect.x + agent->rect.width >= SCREEN_WIDTH) {
        agent->rect.x = SCREEN_WIDTH - agent->rect.width - speed;  // Move away from right wall
    }
    
    if (agent->rect.y <= 0) {
        agent->rect.y = speed;  // Move away from top wall
    } else if (agent->rect.y + agent->rect.height >= SCREEN_HEIGHT) {
        agent->rect.y = SCREEN_HEIGHT - agent->rect.height - speed;  // Move away from bottom wall
    }
    
    // Check collisions with other entities
    check_collisions(game, agent_idx);
    
}

void handle_agent_collision(GameState* game, int agent1_idx, int agent2_idx) {
    Agent* agent1 = &game->agents[agent1_idx];
    Agent* agent2 = &game->agents[agent2_idx];
    
    if (agent1->size > agent2->size) {
        eat_agent(agent1, agent2);
        kill_agent(game, agent2_idx);
    } else if (agent2->size > agent1->size) {
        eat_agent(agent2, agent1);
        kill_agent(game, agent1_idx);
    } else if (!agent1->is_breeding && !agent2->is_breeding) {
        start_breeding(agent1, agent2);
    }
}

void eat_food(Agent* agent) {
    agent->total_xp += XP_FROM_FOOD;
    if (agent->total_xp >= (agent->level + 1) * XP_PER_LEVEL) {
        level_up(agent);
    }
}

void eat_agent(Agent* predator, Agent* prey) {
    predator->total_xp += XP_FROM_AGENT;
    predator->num_eaten++;
    if (predator->total_xp >= (predator->level + 1) * XP_PER_LEVEL) {
        level_up(predator);
    }
    update_agent_color(predator);
}

void kill_agent(GameState* game, int agent_idx) {
    Agent* agent = &game->agents[agent_idx];
    agent->level = -1;  // Mark as dead
    agent->total_xp = 0;
    game->num_active_players--;
}

void start_breeding(Agent* agent1, Agent* agent2) {
    if (!agent1->is_breeding && !agent2->is_breeding) {
        agent1->is_breeding = true;
        agent2->is_breeding = true;
        agent1->breeding_timer = 0;
        agent2->breeding_timer = 0;
    }
}

void handle_breeding(GameState* game, int agent_idx) {
    Agent* parent = &game->agents[agent_idx];
    parent->is_breeding = false;
    parent->breeding_timer = 0;
    parent->num_offsprings++;
    
    // Find empty slot for offspring
    for (int i = 0; i < POPULATION_SIZE; i++) {
        if (game->agents[i].level < 0) {  // Found empty slot
            init_agent(&game->agents[i], game->next_agent_id++);
            game->agents[i].parent_id = parent->agent_id;
            game->num_active_players++;
            
            // Transfer neural network weights with mutation
            NEAT_t* old_brain = game->neat_populations[agent_idx];
            NEAT_t* new_brain = game->neat_populations[i];
            transfer_weights_with_mutation(old_brain->nodes[0]->nn, new_brain->nodes[0]->nn, parent->level);
            break;
        }
    }
}

int main(void) {
    // Initialize window
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Evolution Simulator");
    SetTargetFPS(FRAME_RATE);
    
    // Initialize random seed
    srand(time(NULL));
    
    // Initialize game state
    GameState game = {0};
    init_game(&game);
    
    system("clear");

    // Main game loop
    while (!WindowShouldClose()) {
        // Update
        if (IsKeyPressed(KEY_SPACE)) {
            game.paused = !game.paused;
        }
        
        if (!game.paused) {
            update_game(&game);
        }
        
        // Draw
        BeginDrawing();
        ClearBackground(BLACK);
        
        // Draw food
        for (int i = 0; i < MAX_FOOD; i++) {
            DrawRectangleRec(game.food[i].rect, GREEN);
        }
        
        // Draw agents
        for (int i = 0; i < POPULATION_SIZE; i++) {
            if (game.agents[i].level >= 0) {
                // Draw agent body
                DrawRectangleRec(game.agents[i].rect, game.agents[i].color);
                
                // Draw breeding indicator
                if (game.agents[i].is_breeding) {
                    DrawRectangleLinesEx(game.agents[i].rect, 2, PINK);
                }
                
                // Draw label above the agent
                const char* label = (i < MAX_GROUNDSKEEPERS) ? "GK" : "AG";
                Vector2 textPos = {
                    game.agents[i].rect.x,
                    game.agents[i].rect.y - LABEL_SIZE - 2
                };
                Color labelColor = (i < MAX_GROUNDSKEEPERS) ? RED : WHITE;
                DrawText(label, textPos.x, textPos.y, LABEL_SIZE, labelColor);
                
                // Draw level number
                DrawText(TextFormat("%d", game.agents[i].level), 
                    game.agents[i].rect.x + game.agents[i].rect.width + 2,
                    game.agents[i].rect.y,
                    LABEL_SIZE,
                    labelColor);
            }
        }
        
        // Draw UI
        DrawText(TextFormat("Generation: %d", game.current_generation), 10, 10, 20, WHITE);
        DrawText(TextFormat("Active Agents: %d", game.num_active_players), 10, 35, 20, WHITE);
        
        if (game.paused) {
            DrawText("PAUSED", SCREEN_WIDTH/2 - 50, 10, 20, WHITE);
        }
        
        EndDrawing();
    }
    
    // Cleanup
    free(game.vision_inputs);
    CloseWindow();
    
    return 0;
}
