#ifndef DISCRETE_ACTIONS_H
#define DISCRETE_ACTIONS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Discrete action space definitions
typedef enum {
    // Movement actions (31 bins: -1.0 to 1.0)
    ACTION_MOVE_LEFT_15 = 0,   // -1.0
    ACTION_MOVE_LEFT_14,       // -0.9
    ACTION_MOVE_LEFT_13,       // -0.8
    ACTION_MOVE_LEFT_12,       // -0.7
    ACTION_MOVE_LEFT_11,       // -0.6
    ACTION_MOVE_LEFT_10,       // -0.5
    ACTION_MOVE_LEFT_9,        // -0.4
    ACTION_MOVE_LEFT_8,        // -0.3
    ACTION_MOVE_LEFT_7,        // -0.2
    ACTION_MOVE_LEFT_6,        // -0.1
    ACTION_MOVE_LEFT_5,        // -0.0
    ACTION_MOVE_LEFT_4,        // 0.1
    ACTION_MOVE_LEFT_3,        // 0.2
    ACTION_MOVE_LEFT_2,        // 0.3
    ACTION_MOVE_LEFT_1,        // 0.4
    ACTION_MOVE_CENTER = 15,    // 0.0
    ACTION_MOVE_RIGHT_1,       // 0.6
    ACTION_MOVE_RIGHT_2,       // 0.7
    ACTION_MOVE_RIGHT_3,       // 0.8
    ACTION_MOVE_RIGHT_4,       // 0.9
    ACTION_MOVE_RIGHT_5,       // 1.0
    
    // Forward movement (11 bins: 0.0 to 1.0)
    ACTION_FORWARD_0 = 31,      // 0.0
    ACTION_FORWARD_1,           // 0.1
    ACTION_FORWARD_2,           // 0.2
    ACTION_FORWARD_3,           // 0.3
    ACTION_FORWARD_4,           // 0.4
    ACTION_FORWARD_5,           // 0.5
    ACTION_FORWARD_6,           // 0.6
    ACTION_FORWARD_7,           // 0.7
    ACTION_FORWARD_8,           // 0.8
    ACTION_FORWARD_9,           // 0.9
    ACTION_FORWARD_10,          // 1.0
    
    // Turn actions (31 bins: -1.0 to 1.0)
    ACTION_TURN_LEFT_15 = 42,    // -1.0
    ACTION_TURN_LEFT_14,        // -0.9
    ACTION_TURN_LEFT_13,        // -0.8
    ACTION_TURN_LEFT_12,        // -0.7
    ACTION_TURN_LEFT_11,        // -0.6
    ACTION_TURN_LEFT_10,        // -0.5
    ACTION_TURN_LEFT_9,         // -0.4
    ACTION_TURN_LEFT_8,         // -0.3
    ACTION_TURN_LEFT_7,         // -0.2
    ACTION_TURN_LEFT_6,         // -0.1
    ACTION_TURN_LEFT_5,         // 0.0
    ACTION_TURN_LEFT_4,         // 0.1
    ACTION_TURN_LEFT_3,         // 0.2
    ACTION_TURN_LEFT_2,         // 0.3
    ACTION_TURN_LEFT_1,         // 0.4
    ACTION_TURN_CENTER = 57,      // 0.0
    ACTION_TURN_RIGHT_1,        // 0.6
    ACTION_TURN_RIGHT_2,        // 0.7
    ACTION_TURN_RIGHT_3,        // 0.8
    ACTION_TURN_RIGHT_4,        // 0.9
    ACTION_TURN_RIGHT_5,        // 1.0
    
    // Attack actions (binary)
    ACTION_ATTACK_OFF = 73,
    ACTION_ATTACK_ON = 74,
    
    // Harvest actions (binary)
    ACTION_HARVEST_OFF = 75,
    ACTION_HARVEST_ON = 76,
    
    TOTAL_DISCRETE_ACTIONS = 77
} DiscreteAction;

// Action conversion functions
float action_to_move(int action_id);
float action_to_forward(int action_id);
float action_to_turn(int action_id);
int action_to_attack(int action_id);
int action_to_harvest(int action_id);

// Multi-discrete action packing
int pack_action(float move, float forward, float turn, int attack, int harvest);
void unpack_action(int packed_action, float *move, float *forward, float *turn, int *attack, int *harvest);

#ifdef __cplusplus
}
#endif

#endif // DISCRETE_ACTIONS_H
