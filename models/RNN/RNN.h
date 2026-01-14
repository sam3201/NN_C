#ifndef RNN_H
#define RNN_H

#include "NN.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// RNN layer types
typedef enum {
    RNN_SIMPLE = 0,        // Simple RNN
    RNN_LSTM = 1,          // Long Short-Term Memory
    RNN_GRU = 2,           // Gated Recurrent Unit
    RNN_BIDIRECTIONAL = 3, // Bidirectional RNN
    RNN_TYPE_COUNT = 4
} RNNType;

// RNN layer structure
typedef struct RNNLayer {
    RNNType type;
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    
    // Weight matrices
    long double *W_xh;      // Input to hidden weights
    long double *W_hh;      // Hidden to hidden weights
    long double *W_hy;      // Hidden to output weights
    
    // Bias vectors
    long double *b_h;       // Hidden bias
    long double *b_y;       // Output bias
    
    // LSTM specific gates
    long double *W_xf;      // Forget gate weights
    long double *W_xi;      // Input gate weights
    long double *W_xo;      // Output gate weights
    long double *W_xc;      // Cell gate weights
    
    long double *W_hf;      // Hidden forget gate weights
    long double *W_hi;      // Hidden input gate weights
    long double *W_ho;      // Hidden output gate weights
    long double *W_hc;      // Hidden cell gate weights
    
    long double *b_f;       // Forget gate bias
    long double *b_i;       // Input gate bias
    long double *b_o;       // Output gate bias
    long double *b_c;       // Cell gate bias
    
    // Hidden and cell states
    long double *hidden_state;
    long double *cell_state;
    long double *prev_hidden_state;
    long double *prev_cell_state;
    
    // Gradients
    long double *dW_xh, *dW_hh, *dW_hy;
    long double *db_h, *db_y;
    long double *dW_xf, *dW_xi, *dW_xo, *dW_xc;
    long double *dW_hf, *dW_hi, *dW_ho, *dW_hc;
    long double *db_f, *db_i, *db_o, *db_c;
    
    // Optimizer states
    long double *m_W_xh, *v_W_xh;
    long double *m_W_hh, *v_W_hh;
    long double *m_W_hy, *v_W_hy;
    long double *m_b_h, *v_b_h;
    long double *m_b_y, *v_b_y;
    
    // LSTM optimizer states
    long double *m_W_xf, *v_W_xf;
    long double *m_W_xi, *v_W_xi;
    long double *m_W_xo, *v_W_xo;
    long double *m_W_xc, *v_W_xc;
    long double *m_W_hf, *v_W_hf;
    long double *m_W_hi, *v_W_hi;
    long double *m_W_ho, *v_W_ho;
    long double *m_W_hc, *v_W_hc;
    long double *m_b_f, *v_b_f;
    long double *m_b_i, *v_b_i;
    long double *m_b_o, *v_b_o;
    long double *m_b_c, *v_b_c;
    
} RNNLayer;

// RNN network structure
typedef struct RNN {
    size_t num_layers;
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    size_t sequence_length;
    
    RNNLayer *layers;
    
    // Training parameters
    long double learning_rate;
    int time_step;
    
    // Activation functions
    ActivationFunctionType hidden_activation;
    ActivationFunctionType output_activation;
    
    // Loss function
    LossFunctionType loss_function;
    
    // Optimizer
    OptimizerType optimizer;
    
    // Training state
    bool training_mode;
    
} RNN_t;

// RNN creation and destruction
RNN_t *RNN_create(size_t input_size, size_t hidden_size, size_t output_size, 
                  size_t num_layers, RNNType type);
void RNN_destroy(RNN_t *rnn);

// RNN forward pass
long double *RNN_forward(RNN_t *rnn, const long double *inputs, size_t sequence_length);
long double *RNN_forward_sequence(RNN_t *rnn, const long double *sequence, size_t sequence_length);

// RNN backward pass
void RNN_backward(RNN_t *rnn, const long double *inputs, const long double *targets, 
                  size_t sequence_length);
void RNN_step(RNN_t *rnn);

// RNN state management
void RNN_reset_states(RNN_t *rnn);
void RNN_set_sequence_length(RNN_t *rnn, size_t sequence_length);

// RNN training utilities
void RNN_set_learning_rate(RNN_t *rnn, long double learning_rate);
void RNN_set_optimizer(RNN_t *rnn, OptimizerType optimizer);
void RNN_enable_training(RNN_t *rnn);
void RNN_disable_training(RNN_t *rnn);

// RNN utilities
void RNN_print_summary(RNN_t *rnn);
size_t RNN_get_parameter_count(RNN_t *rnn);

// Game-specific RNN functions for our game
RNN_t *RNN_create_game_network(size_t input_size, size_t hidden_size, size_t output_size);
long double *RNN_predict_game_state(RNN_t *rnn, const long double *game_state);
void RNN_train_game_sequence(RNN_t *rnn, const long double *sequence, const long double *targets, 
                            size_t sequence_length);

#ifdef __cplusplus
}
#endif

#endif // RNN_H
