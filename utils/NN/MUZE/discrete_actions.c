#include "discrete_actions.h"
#include <math.h>

// Convert discrete action ID to continuous value
float action_to_move(int action_id) {
    if (action_id >= ACTION_MOVE_LEFT_15 && action_id <= ACTION_MOVE_RIGHT_5) {
        return (float)(action_id - ACTION_MOVE_CENTER) / 15.0f;
    }
    return 0.0f;
}

float action_to_forward(int action_id) {
    if (action_id >= ACTION_FORWARD_0 && action_id <= ACTION_FORWARD_10) {
        return (float)(action_id - ACTION_FORWARD_0) / 10.0f;
    }
    return 0.0f;
}

float action_to_turn(int action_id) {
    if (action_id >= ACTION_TURN_LEFT_15 && action_id <= ACTION_TURN_RIGHT_5) {
        return (float)(action_id - ACTION_TURN_CENTER) / 15.0f;
    }
    return 0.0f;
}

int action_to_attack(int action_id) {
    return (action_id == ACTION_ATTACK_ON) ? 1 : 0;
}

int action_to_harvest(int action_id) {
    return (action_id == ACTION_HARVEST_ON) ? 1 : 0;
}

// Pack multiple continuous actions into a single discrete action ID
int pack_action(float move, float forward, float turn, int attack, int harvest) {
    // Clamp values to valid ranges
    move = fmaxf(-1.0f, fminf(1.0f, move));
    forward = fmaxf(0.0f, fminf(1.0f, forward));
    turn = fmaxf(-1.0f, fminf(1.0f, turn));
    
    // Convert to discrete bins
    int move_bin = (int)roundf((move + 1.0f) * 15.0f);
    int forward_bin = (int)roundf(forward * 10.0f);
    int turn_bin = (int)roundf((turn + 1.0f) * 15.0f);
    
    // Pack using base-N encoding (31 bins for move/turn, 11 for forward, 2 for attack, 2 for harvest)
    int packed = move_bin;  // 0-30
    packed = packed * 11 + forward_bin;  // 0-340
    packed = packed * 31 + turn_bin;    // 0-10540
    packed = packed * 2 + attack;     // 0-21081
    packed = packed * 2 + harvest;    // 0-42163
    
    return packed;
}

// Unpack single discrete action ID into multiple continuous actions
void unpack_action(int packed_action, float *move, float *forward, float *turn, int *attack, int *harvest) {
    // Unpack using base-N decoding
    *harvest = packed_action % 2;
    packed_action /= 2;
    *attack = packed_action % 2;
    packed_action /= 2;
    
    int turn_bin = packed_action % 31;
    packed_action /= 31;
    int forward_bin = packed_action % 11;
    packed_action /= 11;
    int move_bin = packed_action % 31;
    
    // Convert back to continuous values
    *move = (float)move_bin / 15.0f - 1.0f;
    *forward = (float)forward_bin / 10.0f;
    *turn = (float)turn_bin / 15.0f - 1.0f;
}
