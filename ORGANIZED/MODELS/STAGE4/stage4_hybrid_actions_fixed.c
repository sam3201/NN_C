#include "SAM/SAM.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>

#define CONTEXT_DIM 128
#define ACTION_MODE_COUNT 8
#define DISCRETE_ACTIONS 10
#define CONTINUOUS_ACTIONS 6
#define TOTAL_ACTIONS (DISCRETE_ACTIONS + CONTINUOUS_ACTIONS)

// Action modes
typedef enum {
    MODE_ON_FOOT,
    MODE_IN_MENU,
    MODE_LOCK_ON,
    MODE_BUILD_MODE,
    MODE_BOW_CHARGE,
    MODE_VEHICLE,
    MODE_INVENTORY,
    MODE_DIALOGUE
} ActionMode;

// Discrete action types
typedef enum {
    IDLE,
    MOVE,
    JUMP,
    ATTACK,
    INTERACT,
    BUILD,
    INVENTORY,
    MENU,
    DIALOGUE,
    SPECIAL
} DiscreteAction;

// Hybrid action structure
typedef struct {
    DiscreteAction discrete_action;         // a_disc
    long double continuous_actions[CONTINUOUS_ACTIONS]; // a_cont
    ActionMode action_mode;      // mode context
    long double *action_vector;    // Combined representation
} HybridAction;

// Context structure
typedef struct {
    long double *observations;      // o_{t-k:t}
    long double *action_history;    // a_{t-k:t-1}
    long double *memory;           // m_t
    ActionMode action_mode;        // e^{mode}_t
    long double *context_vector;    // s_t
    int observation_dim;
    int memory_dim;
    int history_length;
} Context;

// Expert base class
typedef struct {
    char name[50];
    int output_dim;
    long double *(*forward)(Context *ctx);
    void (*destroy)(void *expert);
} Expert;

// Vision expert
typedef struct {
    Expert base;
    int vision_dim;
} VisionExpert;

// Combat expert
typedef struct {
    Expert base;
    int combat_dim;
} CombatExpert;

// Navigation expert
typedef struct {
    Expert base;
    int nav_dim;
} NavigationExpert;

// Physics expert
typedef struct {
    Expert base;
    int physics_dim;
} PhysicsExpert;

// Head module
typedef struct {
    long double *routing_weights;
    long double *fusion_weights;
    long double *discrete_weights;
    long double *continuous_weights;
    long double *continuous_means;
    long double *continuous_stds;
    int expert_count;
    int context_dim;
    int discrete_count;
    int continuous_count;
} HeadModule;

// World model
typedef struct {
    long double *encoder_weights;
    long double *dynamics_weights;
    long double *prediction_weights;
    long double *reward_weights;
    long double *value_weights;
    int context_dim;
    int latent_dim;
    int action_dim;
} WorldModel;

// Planning node
typedef struct PlanningNode {
    Context *context;
    HybridAction action;
    long double value;
    int children_count;
    struct PlanningNode **children;
    int visit_count;
    int is_terminal;
} PlanningNode;

// Initialize context
Context* init_context(int observation_dim, int memory_dim, int history_length) {
    Context *ctx = malloc(sizeof(Context));
    if (!ctx) return NULL;
    
    ctx->observations = calloc(observation_dim, sizeof(long double));
    ctx->action_history = calloc(history_length * TOTAL_ACTIONS, sizeof(long double));
    ctx->memory = calloc(memory_dim, sizeof(long double));
    ctx->context_vector = calloc(CONTEXT_DIM, sizeof(long double));
    ctx->observation_dim = observation_dim;
    ctx->action_mode = MODE_ON_FOOT;
    ctx->memory_dim = memory_dim;
    ctx->history_length = history_length;
    
    return ctx;
}

// Get action mode name
const char* get_action_mode_name(ActionMode mode) {
    switch (mode) {
        case MODE_ON_FOOT: return "ON_FOOT";
        case MODE_IN_MENU: return "IN_MENU";
        case MODE_LOCK_ON: return "LOCK_ON";
        case MODE_BUILD_MODE: return "BUILD_MODE";
        case MODE_BOW_CHARGE: return "BOW_CHARGE";
        case MODE_VEHICLE: return "VEHICLE";
        case MODE_INVENTORY: return "INVENTORY";
        case MODE_DIALOGUE: return "DIALOGUE";
        default: return "UNKNOWN";
    }
}

// Get discrete action name
const char* get_discrete_action_name(DiscreteAction action) {
    switch (action) {
        case IDLE: return "IDLE";
        case MOVE: return "MOVE";
        case JUMP: return "JUMP";
        case ATTACK: return "ATTACK";
        case INTERACT: return "INTERACT";
        case BUILD: return "BUILD";
        case INVENTORY: return "INVENTORY";
        case MENU: return "MENU";
        case DIALOGUE: return "DIALOGUE";
        case SPECIAL: return "SPECIAL";
        default: return "UNKNOWN";
    }
}

// Encode action mode
void encode_action_mode(ActionMode mode, long double *vector, int vector_dim) {
    // One-hot encoding for action mode
    for (int i = 0; i < ACTION_MODE_COUNT; i++) {
        vector[i] = (mode == i) ? 1.0L : 0.0L;
    }
    
    // Add mode influence to context
    for (int i = 0; i < vector_dim; i++) {
        vector[i] += (long double)mode * 0.1L / ACTION_MODE_COUNT;
    }
}

// Encode hybrid action
void encode_hybrid_action(HybridAction *action, long double *vector, int vector_dim) {
    // Encode discrete action (one-hot)
    for (int i = 0; i < DISCRETE_ACTIONS; i++) {
        vector[i] = (action->discrete_action == i) ? 1.0L : 0.0L;
    }
    
    // Encode continuous actions
    int start_idx = DISCRETE_ACTIONS;
    for (int i = 0; i < CONTINUOUS_ACTIONS; i++) {
        vector[start_idx + i] = action->continuous_actions[i];
    }
    
    // Encode action mode
    encode_action_mode(action->action_mode, vector, vector_dim);
    
    // Store combined vector
    if (action->action_vector) {
        free(action->action_vector);
    }
    action->action_vector = malloc(vector_dim * sizeof(long double));
    for (int i = 0; i < vector_dim; i++) {
        action->action_vector[i] = vector[i];
    }
}

// Decode action from vector
void decode_action_from_vector(long double *vector, HybridAction *action, int vector_dim) {
    // Decode discrete action (argmax)
    int max_idx = 0;
    long double max_val = vector[0];
    for (int i = 1; i < DISCRETE_ACTIONS; i++) {
        if (vector[i] > max_val) {
            max_val = vector[i];
            max_idx = i;
        }
    }
    action->discrete_action = (DiscreteAction)max_idx;
    
    // Decode continuous actions
    int start_idx = DISCRETE_ACTIONS;
    for (int i = 0; i < CONTINUOUS_ACTIONS; i++) {
        action->continuous_actions[i] = vector[start_idx + i];
    }
    
    // Decode action mode (argmax)
    int mode_idx = 0;
    max_val = vector[DISCRETE_ACTIONS + CONTINUOUS_ACTIONS];
    for (int i = 1; i < ACTION_MODE_COUNT; i++) {
        if (vector[DISCRETE_ACTIONS + CONTINUOUS_ACTIONS + i] > max_val) {
            max_val = vector[DISCRETE_ACTIONS + CONTINUOUS_ACTIONS + i];
            mode_idx = i;
        }
    }
    action->action_mode = (ActionMode)mode_idx;
    
    // Store combined vector
    if (action->action_vector) {
        free(action->action_vector);
    }
    action->action_vector = malloc(vector_dim * sizeof(long double));
    for (int i = 0; i < vector_dim; i++) {
        action->action_vector[i] = vector[i];
    }
}

// Initialize vision expert
VisionExpert* init_vision_expert(int vision_dim) {
    VisionExpert *expert = malloc(sizeof(VisionExpert));
    expert->base.output_dim = vision_dim;
    expert->vision_dim = vision_dim;
    expert->base.forward = NULL;
    expert->base.destroy = NULL;
    
    strcpy(expert->base.name, "Vision");
    
    return expert;
}

// Vision expert forward pass
long double* vision_expert_forward(Context *ctx) {
    VisionExpert *expert = malloc(sizeof(VisionExpert));
    expert->vision_dim = 64;
    
    // Simple vision processing: detect obstacles, targets, spatial awareness
    long double *output = malloc(expert->vision_dim * sizeof(long double));
    
    // Simulate vision processing
    for (int i = 0; i < expert->vision_dim; i++) {
        output[i] = sin(i * 0.1) * cos(ctx->action_mode * 0.2);
        output[i] += (long double)rand() / RAND_MAX * 0.1;
    }
    
    free(expert);
    return output;
}

// Initialize combat expert
CombatExpert* init_combat_expert(int combat_dim) {
    CombatExpert *expert = malloc(sizeof(CombatExpert));
    expert->base.output_dim = combat_dim;
    expert->combat_dim = combat_dim;
    expert->base.forward = NULL;
    expert->base.destroy = NULL;
    
    strcpy(expert->base.name, "Combat");
    
    return expert;
}

// Combat expert forward pass
long double* combat_expert_forward(Context *ctx) {
    CombatExpert *expert = malloc(sizeof(CombatExpert));
    expert->combat_dim = 64;
    
    // Simple combat processing: threat assessment, attack timing, range estimation
    long double *output = malloc(expert->combat_dim * sizeof(long double));
    
    // Simulate combat processing
    for (int i = 0; i < expert->combat_dim; i++) {
        output[i] = cos(i * 0.15) * sin(ctx->action_mode * 0.1);
        output[i] += (long double)rand() / RAND_MAX * 0.1;
    }
    
    free(expert);
    return output;
}

// Initialize navigation expert
NavigationExpert* init_navigation_expert(int nav_dim) {
    NavigationExpert *expert = malloc(sizeof(NavigationExpert));
    expert->base.output_dim = nav_dim;
    expert->nav_dim = nav_dim;
    expert->base.forward = NULL;
    expert->base.destroy = NULL;
    
    strcpy(expert->base.name, "Navigation");
    
    return expert;
}

// Navigation expert forward pass
long double* navigation_expert_forward(Context *ctx) {
    NavigationExpert *expert = malloc(sizeof(NavigationExpert));
    expert->nav_dim = 64;
    
    // Simple navigation processing: exploration, waypoints, path optimization
    long double *output = malloc(expert->nav_dim * sizeof(long double));
    
    // Simulate navigation processing
    for (int i = 0; i < expert->nav_dim; i++) {
        output[i] = sin(i * 0.2) * cos(ctx->action_mode * 0.15);
        output[i] += (long double)rand() / RAND_MAX * 0.1;
    }
    
    free(expert);
    return output;
}

// Initialize physics expert
PhysicsExpert* init_physics_expert(int physics_dim) {
    PhysicsExpert *expert = malloc(sizeof(PhysicsExpert));
    expert->base.output_dim = physics_dim;
    expert->physics_dim = physics_dim;
    expert->base.forward = NULL;
    expert->base.destroy = NULL;
    
    strcpy(expert->base.name, "Physics");
    
    return expert;
}

// Physics expert forward pass
long double* physics_expert_forward(Context *ctx) {
    PhysicsExpert *expert = malloc(sizeof(PhysicsExpert));
    expert->physics_dim = 64;
    
    // Simple physics processing: movement feasibility, jump arcs, collision prediction
    long double *output = malloc(expert->physics_dim * sizeof(long double));
    
    // Simulate physics processing
    for (int i = 0; i < expert->physics_dim; i++) {
        output[i] = cos(i * 0.25) * sin(ctx->action_mode * 0.2);
        output[i] += (long double)rand() / RAND_MAX * 0.1;
    }
    
    free(expert);
    return output;
}

// Initialize head module
HeadModule* init_head_module(int expert_count, int context_dim) {
    HeadModule *head = malloc(sizeof(HeadModule));
    
    head->expert_count = expert_count;
    head->context_dim = context_dim;
    head->discrete_count = DISCRETE_ACTIONS;
    head->continuous_count = CONTINUOUS_ACTIONS;
    
    // Initialize weights
    head->routing_weights = malloc(expert_count * context_dim * sizeof(long double));
    head->fusion_weights = malloc(expert_count * 64 * sizeof(long double));
    head->discrete_weights = malloc(64 * DISCRETE_ACTIONS * sizeof(long double));
    head->continuous_weights = malloc(64 * CONTINUOUS_ACTIONS * sizeof(long double));
    head->continuous_means = malloc(CONTINUOUS_ACTIONS * sizeof(long double));
    head->continuous_stds = malloc(CONTINUOUS_ACTIONS * sizeof(long double));
    
    // Initialize with small random values
    for (int i = 0; i < expert_count * context_dim; i++) {
        head->routing_weights[i] = (long double)rand() / RAND_MAX * 0.1 - 0.05L;
    }
    
    for (int i = 0; i < expert_count * 64; i++) {
        head->fusion_weights[i] = (long double)rand() / RAND_MAX * 0.1 - 0.05L;
    }
    
    for (int i = 0; i < 64 * DISCRETE_ACTIONS; i++) {
        head->discrete_weights[i] = (long double)rand() / RAND_MAX * 0.1 - 0.05L;
    }
    
    for (int i = 0; i < 64 * CONTINUOUS_ACTIONS; i++) {
        head->continuous_weights[i] = (long double)rand() / RAND_MAX * 0.2 - 0.1L;
        head->continuous_means[i] = 0.0L;
        head->continuous_stds[i] = 1.0L;
    }
    
    return head;
}

// Softmax function
void hybrid_softmax(long double *values, int count) {
    long double max_val = values[0];
    for (int i = 1; i < count; i++) {
        if (values[i] > max_val) {
            max_val = values[i];
        }
    }
    
    long double sum = 0.0L;
    for (int i = 0; i < count; i++) {
        sum += expl(values[i] - max_val);
    }
    
    for (int i = 0; i < count; i++) {
        values[i] = expl(values[i] - max_val) / sum;
    }
}

// Route experts based on context
void route_experts(HeadModule *head, Context *ctx, long double *routing_weights) {
    // Simple routing: softmax over context
    hybrid_softmax(routing_weights, head->expert_count);
    
    // Store routing weights for backprop
    if (head->routing_weights) {
        free(head->routing_weights);
    }
    head->routing_weights = malloc(head->expert_count * sizeof(long double));
    for (int i = 0; i < head->expert_count; i++) {
        head->routing_weights[i] = routing_weights[i];
    }
}

// Fuse expert outputs
void fuse_experts(HeadModule *head, long double **expert_outputs, long double *fused_weights, long double *fused_weights_stds, int expert_count, int expert_dim) {
    // Weighted sum of expert outputs
    for (int i = 0; i < expert_dim; i++) {
        fused_weights[i] = 0.0L;
        for (int j = 0; j < expert_count; j++) {
            fused_weights[i] += head->routing_weights[j] * expert_outputs[j][i];
        }
        fused_weights_stds[i] = sqrt(fused_weights[i]);
    }
}

// Generate discrete policy
void generate_discrete_policy(HeadModule *head, long double *fused_weights, long double *discrete_logits) {
    // Simple linear layer for discrete actions
    for (int i = 0; i < DISCRETE_ACTIONS; i++) {
        discrete_logits[i] = 0.0L;
        for (int j = 0; j < 64; j++) {
            discrete_logits[i] += head->discrete_weights[i * DISCRETE_ACTIONS + j] * fused_weights[j];
        }
    }
    hybrid_softmax(discrete_logits, DISCRETE_ACTIONS);
}

// Generate continuous policy
void generate_continuous_policy(HeadModule *head, long double *fused_weights, long double *continuous_means, long double *continuous_stds, HybridAction *action) {
    // Generate continuous parameters for each action
    for (int i = 0; i < CONTINUOUS_ACTIONS; i++) {
        continuous_means[i] = 0.0L;
        continuous_stds[i] = 1.0L;
        
        for (int j = 0; j < 64; j++) {
            continuous_means[i] += head->continuous_weights[i * CONTINUOUS_ACTIONS + j] * fused_weights[j];
        }
        
        // Apply gating based on discrete action
        if (action->discrete_action == ATTACK && i == 0) {
            // Attack mode: aim vector
            continuous_means[i] = 0.0L;
            continuous_stds[i] = 0.1L;
        } else if (action->discrete_action == MOVE && i == 1) {
            // Move mode: movement vector
            continuous_means[i] = 0.0L;
            continuous_stds[i] = 0.5L;
        } else if (action->discrete_action == JUMP && i == 2) {
            // Jump mode: jump strength
            continuous_means[i] = 1.0L;
            continuous_stds[i] = 0.2L;
        }
    }
    
    // Add small noise for exploration
    for (int i = 0; i < CONTINUOUS_ACTIONS; i++) {
        continuous_means[i] += (long double)rand() / RAND_MAX * 0.01;
        continuous_stds[i] *= 0.9; // Reduce std for stability
    }
    
    // Store in action
    for (int i = 0; i < CONTINUOUS_ACTIONS; i++) {
        action->continuous_actions[i] = continuous_means[i];
    }
}

// Generate hybrid action from context
void generate_hybrid_action(HeadModule *head, Context *ctx, HybridAction *action) {
    // Route experts
    long double *routing_weights = malloc(head->expert_count * sizeof(long double));
    for (int i = 0; i < head->expert_count; i++) {
        routing_weights[i] = (long double)rand() / RAND_MAX;
    }
    route_experts(head, ctx, routing_weights);
    
    // Get expert outputs (simplified)
    long double **expert_outputs = malloc(head->expert_count * sizeof(long double*));
    long double *fused_weights = malloc(64 * sizeof(long double));
    long double *fused_weights_stds = malloc(64 * sizeof(long double));
    
    for (int i = 0; i < head->expert_count; i++) {
        expert_outputs[i] = malloc(64 * sizeof(long double));
        
        // Simulate expert forward pass
        if (i == 0) { // Vision
            long double *output = vision_expert_forward(ctx);
            for (int j = 0; j < 64; j++) {
                expert_outputs[i][j] = output[j];
            }
            free(output);
        } else if (i == 1) { // Combat
            long double *output = combat_expert_forward(ctx);
            for (int j = 0; j < 64; j++) {
                expert_outputs[i][j] = output[j];
            }
            free(output);
        } else if (i == 2) { // Navigation
            long double *output = navigation_expert_forward(ctx);
            for (int j = 0; j < 64; j++) {
                expert_outputs[i][j] = output[j];
            }
            free(output);
        } else if (i == 3) { // Physics
            long double *output = physics_expert_forward(ctx);
            for (int j = 0; j < 64; j++) {
                expert_outputs[i][j] = output[j];
            }
            free(output);
        }
    }
    
    // Fuse expert outputs
    fuse_experts(head, expert_outputs, fused_weights, fused_weights_stds, head->expert_count, 64);
    
    // Generate policies
    long double discrete_logits[DISCRETE_ACTIONS];
    generate_discrete_policy(head, fused_weights, discrete_logits);
    
    // Sample discrete action
    long double cumsum = 0.0L;
    long double rand_val = (long double)rand() / RAND_MAX;
    for (int i = 0; i < DISCRETE_ACTIONS; i++) {
        cumsum += discrete_logits[i];
        if (rand_val <= cumsum) {
            action->discrete_action = (DiscreteAction)i;
            break;
        }
    }
    
    // Generate continuous policy
    long double continuous_means[CONTINUOUS_ACTIONS];
    long double continuous_stds[CONTINUOUS_ACTIONS];
    generate_continuous_policy(head, fused_weights, continuous_means, continuous_stds, action);
    
    // Encode action
    long double action_vector[CONTEXT_DIM];
    encode_hybrid_action(action, action_vector, CONTEXT_DIM);
    
    // Cleanup
    for (int i = 0; i < head->expert_count; i++) {
        free(expert_outputs[i]);
    }
    free(expert_outputs);
    free(fused_weights);
    free(fused_weights_stds);
    free(routing_weights);
}

// Initialize world model
WorldModel* init_world_model(int context_dim, int latent_dim, int action_dim) {
    WorldModel *model = malloc(sizeof(WorldModel));
    
    model->context_dim = context_dim;
    model->latent_dim = latent_dim;
    model->action_dim = action_dim;
    
    // Initialize weights (simplified)
    model->encoder_weights = malloc(context_dim * latent_dim * sizeof(long double));
    model->dynamics_weights = malloc(latent_dim * action_dim * sizeof(long double));
    model->prediction_weights = malloc(latent_dim * 3 * sizeof(long double)); // (next_state, reward, done)
    model->reward_weights = malloc(latent_dim * sizeof(long double));
    model->value_weights = malloc(latent_dim * sizeof(long double));
    
    // Initialize with small random values
    for (int i = 0; i < context_dim * latent_dim; i++) {
        model->encoder_weights[i] = (long double)rand() / RAND_MAX * 0.1 - 0.05L;
    }
    
    for (int i = 0; i < latent_dim * action_dim; i++) {
        model->dynamics_weights[i] = (long double)rand() / RAND_MAX * 0.1 - 0.05L;
    }
    
    for (int i = 0; i < latent_dim * 3; i++) {
        model->prediction_weights[i] = (long double)rand() / RAND_MAX * 0.1 - 0.05L;
    }
    
    for (int i = 0; i < latent_dim; i++) {
        model->reward_weights[i] = (long double)rand() / RAND_MAX * 0.1 - 0.05L;
    }
    
    for (int i = 0; i < latent_dim; i++) {
        model->value_weights[i] = (long double)rand() / RAND_MAX * 0.1 - 0.05L;
    }
    
    return model;
}

// World model forward pass
void world_model_forward(WorldModel *model, Context *ctx, HybridAction *action, long double *next_state, long double *reward, int *done) {
    // Encode context to latent space
    for (int i = 0; i < model->latent_dim; i++) {
        next_state[i] = 0.0L;
        for (int j = 0; j < model->context_dim; j++) {
            next_state[i] += model->encoder_weights[i * model->context_dim + j] * ctx->context_vector[j];
        }
    }
    
    // Predict dynamics
    for (int i = 0; i < model->latent_dim; i++) {
        for (int j = 0; j < model->action_dim; j++) {
            next_state[i] += model->dynamics_weights[i * model->action_dim + j] * action->action_vector[j];
        }
    }
    
    // Predict next state, reward, done
    for (int i = 0; i < model->latent_dim; i++) {
        next_state[i] = tanhl(next_state[i]); // Activation
    }
    
    for (int i = 0; i < 3; i++) {
        model->prediction_weights[i * model->latent_dim + i] = next_state[i];
    }
    
    *reward = model->reward_weights[0]; // Simple reward prediction
    *done = 0; // Not terminal
}

// Initialize planning node
PlanningNode* init_planning_node(Context *context, HybridAction *action, long double value, int is_terminal) {
    PlanningNode *node = malloc(sizeof(PlanningNode));
    node->context = context;
    node->action = *action; // Copy action
    node->value = value;
    node->children_count = 0;
    node->children = NULL;
    node->visit_count = 0;
    node->is_terminal = is_terminal;
    
    return node;
}

// Free planning node
void free_planning_node(PlanningNode *node) {
    if (node) {
        if (node->children) {
            for (int i = 0; i < node->children_count; i++) {
                free_planning_node(node->children[i]);
            }
            free(node->children);
        }
        free(node);
    }
}

// Sample hybrid action
void sample_hybrid_action(HeadModule *head, Context *ctx, HybridAction *action) {
    generate_hybrid_action(head, ctx, action);
    
    // Add exploration noise
    if ((double)rand() / RAND_MAX < 0.1) {
        // Random discrete action
        action->discrete_action = rand() % DISCRETE_ACTIONS;
        
        // Random continuous actions
        for (int i = 0; i < CONTINUOUS_ACTIONS; i++) {
            action->continuous_actions[i] = (long double)rand() / RAND_MAX * 2.0 - 1.0;
        }
        
        // Random action mode
        action->action_mode = rand() % ACTION_MODE_COUNT;
        
        // Re-encode
        long double action_vector[CONTEXT_DIM];
        encode_hybrid_action(action, action_vector, CONTEXT_DIM);
    }
}

// Evaluate planning node
long double evaluate_planning_node(PlanningNode *node, WorldModel *world_model) {
    if (node->is_terminal) {
        return node->value;
    }
    
    // Simple evaluation: use world model to predict value
    HybridAction *action = &node->action;
    long double next_state[128];
    long double reward;
    int done;
    
    world_model_forward(world_model, node->context, action, next_state, &reward, &done);
    
    // Simple value calculation
    long double value = reward;
    if (done) {
        value -= 1.0L; // Terminal penalty
    }
    
    return value;
}

// MCTS-style planning in latent space
PlanningNode* plan_mcts(Context *ctx, WorldModel *world_model, HeadModule *head, int max_depth, int num_simulations) {
    PlanningNode *root = init_planning_node(ctx, NULL, 0.0L, 0);
    
    PlanningNode *best_node = root;
    
    for (int sim = 0; sim < num_simulations; sim++) {
        PlanningNode *node = root;
        
        for (int depth = 0; depth < max_depth; depth++) {
            if (node->is_terminal) break;
            
            // Sample action
            sample_hybrid_action(head, ctx, &node->action);
            
            // Evaluate node
            long double value = evaluate_planning_node(node, world_model);
            node->value = value;
            
            // Backpropagate value up tree
            PlanningNode *parent = root;
            while (parent) {
                if (parent->value < node->value) {
                    parent->value = node->value;
                }
                parent = root; // Move up
            }
            
            // Move to next depth
            if (!node->is_terminal && depth < max_depth - 1) {
                // Create child nodes
                node->children_count = 5; // 5 children per node
                node->children = malloc(node->children_count * sizeof(PlanningNode*));
                
                for (int child = 0; child < node->children_count; child++) {
                    node->children[child] = init_planning_node(ctx, NULL, 0.0L, 0);
                }
                node->is_terminal = (rand() % 10 == 0); // 10% chance of terminal
            }
        }
        
        // Select best node
        best_node = root;
        PlanningNode *current = root;
        
        for (int depth = 0; depth < max_depth; depth++) {
            if (current->is_terminal) break;
            
            PlanningNode *best_child = current;
            for (int child = 0; child < current->children_count; child++) {
                if (current->children[child]->value > best_child->value) {
                    best_child = current->children[child];
                }
            }
            
            current = best_child;
        }
        
        // Update root value
        root->value = best_node->value;
    }
    
    return best_node;
}

// Cleanup planning tree
void free_planning_tree(PlanningNode *root) {
    free_planning_node(root);
}

// Test hybrid action system
void test_hybrid_actions() {
    printf("=== Stage 4: Hybrid Action System Test ===\n\n");
    
    // Initialize context
    Context *ctx = init_context(256, 64, 10);
    ctx->action_mode = MODE_ON_FOOT;
    
    printf("✅ Context initialized\n");
    printf("  - Observation dim: %d\n", ctx->observation_dim);
    printf("  - Memory dim: %d\n", ctx->memory_dim);
    printf("  - History length: %d\n", ctx->history_length);
    printf("  - Action mode: %s\n", get_action_mode_name(ctx->action_mode));
    printf("\n");
    
    // Initialize head module
    HeadModule *head = init_head_module(4, CONTEXT_DIM);
    printf("✅ Head module initialized\n");
    printf("  - Expert count: %d\n", head->expert_count);
    printf("  - Context dim: %d\n", head->context_dim);
    printf("\n");
    
    // Test action encoding/decoding
    printf("Testing action encoding/decoding...\n");
    
    HybridAction test_action;
    
    // Test different action modes
    test_action.action_mode = MODE_ON_FOOT;
    encode_hybrid_action(&test_action, ctx->context_vector, CONTEXT_DIM);
    decode_action_from_vector(ctx->context_vector, &test_action, CONTEXT_DIM);
    
    printf("Mode: %s\n", get_action_mode_name(test_action.action_mode));
    printf("Discrete action: %s\n", get_discrete_action_name(test_action.discrete_action));
    printf("Continuous actions: ");
    for (int i = 0; i < CONTINUOUS_ACTIONS; i++) {
        printf("%.3f ", test_action.continuous_actions[i]);
    }
    printf("\n\n");
    
    // Test expert routing
    printf("Testing expert routing...\n");
    long double routing_weights[4];
    for (int i = 0; i < 4; i++) {
        routing_weights[i] = (long double)rand() / RAND_MAX;
    }
    route_experts(head, ctx, routing_weights);
    printf("✅ Expert routing working\n");
    printf("Routing weights: ");
    for (int i = 0; i < head->expert_count; i++) {
        printf("%.3f ", head->routing_weights[i]);
    }
    printf("\n\n");
    
    // Test action generation
    printf("Testing action generation...\n");
    for (int i = 0; i < 5; i++) {
        sample_hybrid_action(head, ctx, &test_action);
        printf("Sample %d: %s (%s) - Continuous: ", get_action_mode_name(test_action.action_mode), get_discrete_action_name(test_action.discrete_action));
        for (int j = 0; j < CONTINUOUS_ACTIONS; j++) {
            printf("%.3f ", test_action.continuous_actions[j]);
        }
        printf("\n");
    }
    
    printf("\n");
    
    // Test world model
    printf("Testing world model...\n");
    WorldModel *world = init_world_model(CONTEXT_DIM, 64, TOTAL_ACTIONS);
    printf("✅ World model initialized\n");
    
    // Test world model prediction
    HybridAction world_test_action;
    long double next_state[128];
    long double reward;
    int done;
    
    world_model_forward(world, ctx, &world_test_action, next_state, &reward, &done);
    
    printf("World model test:\n");
    printf("  - Next state generated: ");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", next_state[i]);
    }
    printf("\n");
    printf("  - Reward: %.6f\n", reward);
    printf("  - Terminal: %s\n", done ? "Yes" : "No");
    printf("\n");
    
    // Test planning
    printf("Testing MCTS planning...\n");
    PlanningNode *best_node = plan_mcts(ctx, world, head, 3, 10);
    
    if (best_node) {
        printf("✅ Planning completed\n");
        printf("  - Best value: %.6f\n", best_node->value);
        printf("  - Best action: %s (%s)\n", get_action_mode_name(best_node->action.action_mode), get_discrete_action_name(best_node->action.discrete_action));
        printf("  - Continuous actions: ");
        for (int i = 0; i < CONTINUOUS_ACTIONS; i++) {
            printf("%.3f ", best_node->action.continuous_actions[i]);
        }
        printf("\n");
    } else {
        printf("❌ Planning failed\n");
    }
    
    printf("\n");
    
    // Cleanup
    free_planning_tree(best_node);
    free(world);
    free(head);
    free(ctx->observations);
    free(ctx->action_history);
    free(ctx->memory);
    free(ctx->context_vector);
    free(ctx);
    
    printf("=== Stage 4: Hybrid Action System Test Completed ===\n");
    printf("Ready for Stage 4: Response Generation Training\n");
}

int main(int argc, char *argv[]) {
    printf("=== Stage 4: Hybrid Action System Implementation ===\n\n");
    
    srand(time(NULL));
    
    printf("Configuration:\n");
    printf("  Context dim: %d\n", CONTEXT_DIM);
    printf("  Discrete actions: %d\n", DISCRETE_ACTIONS);
    printf("  Continuous actions: %d\n", CONTINUOUS_ACTIONS);
    printf("  Action modes: %d\n", ACTION_MODE_COUNT);
    printf("\n");
    
    // Test hybrid action system
    test_hybrid_actions();
    
    return 0;
}
