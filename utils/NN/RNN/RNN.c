#include "RNN.h"
#include "../NN.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declarations
static void RNN_layer_destroy(RNNLayer *layer);

// RNN layer creation
static RNNLayer* RNN_layer_create(RNNType type, size_t input_size, size_t hidden_size, size_t output_size) {
    RNNLayer *layer = malloc(sizeof(RNNLayer));
    if (!layer) return NULL;
    
    layer->type = type;
    layer->input_size = input_size;
    layer->hidden_size = hidden_size;
    layer->output_size = output_size;
    
    // Allocate weight matrices
    layer->W_xh = calloc(input_size * hidden_size, sizeof(long double));
    layer->W_hh = calloc(hidden_size * hidden_size, sizeof(long double));
    layer->W_hy = calloc(hidden_size * output_size, sizeof(long double));
    
    // Allocate bias vectors
    layer->b_h = calloc(hidden_size, sizeof(long double));
    layer->b_y = calloc(output_size, sizeof(long double));
    
    // Allocate states
    layer->hidden_state = calloc(hidden_size, sizeof(long double));
    layer->prev_hidden_state = calloc(hidden_size, sizeof(long double));
    
    // LSTM specific allocations
    if (type == RNN_LSTM) {
        layer->W_xf = calloc(input_size * hidden_size, sizeof(long double));
        layer->W_xi = calloc(input_size * hidden_size, sizeof(long double));
        layer->W_xo = calloc(input_size * hidden_size, sizeof(long double));
        layer->W_xc = calloc(input_size * hidden_size, sizeof(long double));
        
        layer->W_hf = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->W_hi = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->W_ho = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->W_hc = calloc(hidden_size * hidden_size, sizeof(long double));
        
        layer->b_f = calloc(hidden_size, sizeof(long double));
        layer->b_i = calloc(hidden_size, sizeof(long double));
        layer->b_o = calloc(hidden_size, sizeof(long double));
        layer->b_c = calloc(hidden_size, sizeof(long double));
        
        layer->cell_state = calloc(hidden_size, sizeof(long double));
        layer->prev_cell_state = calloc(hidden_size, sizeof(long double));
    }
    
    // Allocate gradients
    layer->dW_xh = calloc(input_size * hidden_size, sizeof(long double));
    layer->dW_hh = calloc(hidden_size * hidden_size, sizeof(long double));
    layer->dW_hy = calloc(hidden_size * output_size, sizeof(long double));
    layer->db_h = calloc(hidden_size, sizeof(long double));
    layer->db_y = calloc(output_size, sizeof(long double));
    
    // Allocate optimizer states
    layer->m_W_xh = calloc(input_size * hidden_size, sizeof(long double));
    layer->v_W_xh = calloc(input_size * hidden_size, sizeof(long double));
    layer->m_W_hh = calloc(hidden_size * hidden_size, sizeof(long double));
    layer->v_W_hh = calloc(hidden_size * hidden_size, sizeof(long double));
    layer->m_W_hy = calloc(hidden_size * output_size, sizeof(long double));
    layer->v_W_hy = calloc(hidden_size * output_size, sizeof(long double));
    layer->m_b_h = calloc(hidden_size, sizeof(long double));
    layer->v_b_h = calloc(hidden_size, sizeof(long double));
    layer->m_b_y = calloc(output_size, sizeof(long double));
    layer->m_b_y = calloc(output_size, sizeof(long double));
    
    if (type == RNN_LSTM) {
        layer->dW_xf = calloc(input_size * hidden_size, sizeof(long double));
        layer->dW_xi = calloc(input_size * hidden_size, sizeof(long double));
        layer->dW_xo = calloc(input_size * hidden_size, sizeof(long double));
        layer->dW_xc = calloc(input_size * hidden_size, sizeof(long double));
        layer->dW_hf = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->dW_hi = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->dW_ho = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->dW_hc = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->db_f = calloc(hidden_size, sizeof(long double));
        layer->db_i = calloc(hidden_size, sizeof(long double));
        layer->db_o = calloc(hidden_size, sizeof(long double));
        layer->db_c = calloc(hidden_size, sizeof(long double));
        
        layer->m_W_xf = calloc(input_size * hidden_size, sizeof(long double));
        layer->v_W_xf = calloc(input_size * hidden_size, sizeof(long double));
        layer->m_W_xi = calloc(input_size * hidden_size, sizeof(long double));
        layer->v_W_xi = calloc(input_size * hidden_size, sizeof(long double));
        layer->m_W_xo = calloc(input_size * hidden_size, sizeof(long double));
        layer->v_W_xo = calloc(input_size * hidden_size, sizeof(long double));
        layer->m_W_xc = calloc(input_size * hidden_size, sizeof(long double));
        layer->v_W_xc = calloc(input_size * hidden_size, sizeof(long double));
        layer->m_W_hf = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->v_W_hf = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->m_W_hi = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->v_W_hi = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->m_W_ho = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->v_W_ho = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->m_W_hc = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->v_W_hc = calloc(hidden_size * hidden_size, sizeof(long double));
        layer->m_b_f = calloc(hidden_size, sizeof(long double));
        layer->v_b_f = calloc(hidden_size, sizeof(long double));
        layer->m_b_i = calloc(hidden_size, sizeof(long double));
        layer->v_b_i = calloc(hidden_size, sizeof(long double));
        layer->m_b_o = calloc(hidden_size, sizeof(long double));
        layer->v_b_o = calloc(hidden_size, sizeof(long double));
        layer->m_b_c = calloc(hidden_size, sizeof(long double));
        layer->v_b_c = calloc(hidden_size, sizeof(long double));
    }
    
    // Check allocations
    if (!layer->W_xh || !layer->W_hh || !layer->W_hy || !layer->b_h || !layer->b_y || 
        !layer->hidden_state || !layer->prev_hidden_state ||
        !layer->dW_xh || !layer->dW_hh || !layer->dW_hy || !layer->db_h || !layer->db_y ||
        !layer->m_W_xh || !layer->v_W_xh || !layer->m_W_hh || !layer->v_W_hh || 
        !layer->m_W_hy || !layer->v_W_hy || !layer->m_b_h || !layer->v_b_h || !layer->m_b_y) {
        
        RNN_layer_destroy(layer);
        return NULL;
    }
    
    if (type == RNN_LSTM) {
        if (!layer->W_xf || !layer->W_xi || !layer->W_xo || !layer->W_xc ||
            !layer->W_hf || !layer->W_hi || !layer->W_ho || !layer->W_hc ||
            !layer->b_f || !layer->b_i || !layer->b_o || !layer->b_c ||
            !layer->cell_state || !layer->prev_cell_state) {
            
            RNN_layer_destroy(layer);
            return NULL;
        }
    }
    
    // Initialize weights with Xavier initialization
    double scale = sqrtl(2.0L / (input_size + hidden_size));
    for (size_t i = 0; i < input_size * hidden_size; i++) {
        layer->W_xh[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
    }
    
    scale = sqrtl(2.0L / (hidden_size + hidden_size));
    for (size_t i = 0; i < hidden_size * hidden_size; i++) {
        layer->W_hh[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
    }
    
    scale = sqrtl(2.0L / (hidden_size + output_size));
    for (size_t i = 0; i < hidden_size * output_size; i++) {
        layer->W_hy[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
    }
    
    // Initialize LSTM weights
    if (type == RNN_LSTM) {
        for (size_t i = 0; i < input_size * hidden_size; i++) {
            layer->W_xf[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
            layer->W_xi[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
            layer->W_xo[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
            layer->W_xc[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
        }
        
        for (size_t i = 0; i < hidden_size * hidden_size; i++) {
            layer->W_hf[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
            layer->W_hi[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
            layer->W_ho[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
            layer->W_hc[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
        }
    }
    
    return layer;
}

// RNN layer destruction
static void RNN_layer_destroy(RNNLayer *layer) {
    if (!layer) return;
    
    free(layer->W_xh);
    free(layer->W_hh);
    free(layer->W_hy);
    free(layer->b_h);
    free(layer->b_y);
    free(layer->hidden_state);
    free(layer->prev_hidden_state);
    
    if (layer->type == RNN_LSTM) {
        free(layer->W_xf);
        free(layer->W_xi);
        free(layer->W_xo);
        free(layer->W_xc);
        free(layer->W_hf);
        free(layer->W_hi);
        free(layer->W_ho);
        free(layer->W_hc);
        free(layer->b_f);
        free(layer->b_i);
        free(layer->b_o);
        free(layer->b_c);
        free(layer->cell_state);
        free(layer->prev_cell_state);
    }
    
    free(layer->dW_xh);
    free(layer->dW_hh);
    free(layer->dW_hy);
    free(layer->db_h);
    free(layer->db_y);
    
    free(layer->m_W_xh);
    free(layer->v_W_xh);
    free(layer->m_W_hh);
    free(layer->v_W_hh);
    free(layer->m_W_hy);
    free(layer->v_W_hy);
    free(layer->m_b_h);
    free(layer->v_b_h);
    free(layer->m_b_y);
    
    if (layer->type == RNN_LSTM) {
        free(layer->dW_xf);
        free(layer->dW_xi);
        free(layer->dW_xo);
        free(layer->dW_xc);
        free(layer->dW_hf);
        free(layer->dW_hi);
        free(layer->dW_ho);
        free(layer->dW_hc);
        free(layer->db_f);
        free(layer->db_i);
        free(layer->db_o);
        free(layer->db_c);
        
        free(layer->m_W_xf);
        free(layer->v_W_xf);
        free(layer->m_W_xi);
        free(layer->v_W_xi);
        free(layer->m_W_xo);
        free(layer->v_W_xo);
        free(layer->m_W_xc);
        free(layer->v_W_xc);
        free(layer->m_W_hf);
        free(layer->v_W_hf);
        free(layer->m_W_hi);
        free(layer->v_W_hi);
        free(layer->m_W_ho);
        free(layer->v_W_ho);
        free(layer->m_W_hc);
        free(layer->v_W_hc);
        free(layer->m_b_f);
        free(layer->v_b_f);
        free(layer->m_b_i);
        free(layer->v_b_i);
        free(layer->m_b_o);
        free(layer->v_b_o);
        free(layer->m_b_c);
        free(layer->v_b_c);
    }
    
    free(layer);
}

// RNN creation
RNN_t* RNN_create(size_t input_size, size_t hidden_size, size_t output_size, 
                 size_t num_layers, RNNType type) {
    RNN_t *rnn = malloc(sizeof(RNN_t));
    if (!rnn) return NULL;
    
    rnn->num_layers = num_layers;
    rnn->input_size = input_size;
    rnn->hidden_size = hidden_size;
    rnn->output_size = output_size;
    rnn->sequence_length = 1;
    
    rnn->layers = malloc(num_layers * sizeof(RNNLayer));
    if (!rnn->layers) {
        free(rnn);
        return NULL;
    }
    
    // Create layers
    for (size_t i = 0; i < num_layers; i++) {
        size_t layer_input_size = (i == 0) ? input_size : hidden_size;
        RNNLayer *layer = RNN_layer_create(type, layer_input_size, hidden_size, output_size);
        if (!layer) {
            for (size_t j = 0; j < i; j++) {
                RNN_layer_destroy(&rnn->layers[j]);
            }
            free(rnn->layers);
            free(rnn);
            return NULL;
        }
        rnn->layers[i] = *layer;
        free(layer);  // Free the temporary layer since we copied it
    }
    
    rnn->learning_rate = 0.001L;
    rnn->time_step = 0;
    rnn->hidden_activation = TANH;
    rnn->output_activation = SIGMOID;
    rnn->loss_function = MSE;
    rnn->optimizer = ADAM;
    rnn->training_mode = true;
    
    return rnn;
}

// RNN destruction
void RNN_destroy(RNN_t *rnn) {
    if (!rnn) return;
    
    for (size_t i = 0; i < rnn->num_layers; i++) {
        RNN_layer_destroy(&rnn->layers[i]);
    }
    free(rnn->layers);
    free(rnn);
}

// Simple RNN forward pass
static void simple_rnn_forward(RNNLayer *layer, const long double *input) {
    // Save previous hidden state
    memcpy(layer->prev_hidden_state, layer->hidden_state, layer->hidden_size * sizeof(long double));
    
    // Compute new hidden state: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
    for (size_t i = 0; i < layer->hidden_size; i++) {
        long double sum = layer->b_h[i];
        
        // Input contribution
        for (size_t j = 0; j < layer->input_size; j++) {
            sum += input[j] * layer->W_xh[j * layer->hidden_size + i];
        }
        
        // Hidden contribution
        for (size_t j = 0; j < layer->hidden_size; j++) {
            sum += layer->prev_hidden_state[j] * layer->W_hh[j * layer->hidden_size + i];
        }
        
        layer->hidden_state[i] = tanh_activation(sum);
    }
}

// LSTM forward pass
static void lstm_forward(RNNLayer *layer, const long double *input) {
    // Save previous states
    memcpy(layer->prev_hidden_state, layer->hidden_state, layer->hidden_size * sizeof(long double));
    memcpy(layer->prev_cell_state, layer->cell_state, layer->hidden_size * sizeof(long double));
    
    for (size_t i = 0; i < layer->hidden_size; i++) {
        // Compute gate inputs
        long double f = layer->b_f[i];  // Forget gate
        long double i_gate = layer->b_i[i];  // Input gate
        long double o = layer->b_o[i];   // Output gate
        long double c = layer->b_c[i];   // Cell gate
        
        // Input contributions
        for (size_t j = 0; j < layer->input_size; j++) {
            f += input[j] * layer->W_xf[j * layer->hidden_size + i];
            i_gate += input[j] * layer->W_xi[j * layer->hidden_size + i];
            o += input[j] * layer->W_xo[j * layer->hidden_size + i];
            c += input[j] * layer->W_xc[j * layer->hidden_size + i];
        }
        
        // Hidden contributions
        for (size_t j = 0; j < layer->hidden_size; j++) {
            f += layer->prev_hidden_state[j] * layer->W_hf[j * layer->hidden_size + i];
            i_gate += layer->prev_hidden_state[j] * layer->W_hi[j * layer->hidden_size + i];
            o += layer->prev_hidden_state[j] * layer->W_ho[j * layer->hidden_size + i];
            c += layer->prev_hidden_state[j] * layer->W_hc[j * layer->hidden_size + i];
        }
        
        // Apply activations
        f = sigmoid(f);
        i_gate = sigmoid(i_gate);
        o = sigmoid(o);
        c = tanh_activation(c);
        
        // Update cell state: C_t = f * C_{t-1} + i * c~
        layer->cell_state[i] = f * layer->prev_cell_state[i] + i_gate * c;
        
        // Update hidden state: h_t = o * tanh(C_t)
        layer->hidden_state[i] = o * tanh_activation(layer->cell_state[i]);
    }
}

// RNN forward pass
long double* RNN_forward(RNN_t *rnn, const long double *inputs, size_t sequence_length) {
    if (!rnn || !inputs) return NULL;
    
    rnn->sequence_length = sequence_length;
    long double *output = calloc(rnn->output_size, sizeof(long double));
    if (!output) return NULL;
    
    // Process sequence
    for (size_t t = 0; t < sequence_length; t++) {
        const long double *current_input = inputs + t * rnn->input_size;
        
        // Forward pass through layers
        for (size_t l = 0; l < rnn->num_layers; l++) {
            RNNLayer *layer = &rnn->layers[l];
            
            if (layer->type == RNN_SIMPLE) {
                simple_rnn_forward(layer, current_input);
            } else if (layer->type == RNN_LSTM) {
                lstm_forward(layer, current_input);
            }
            
            // Set input for next layer
            current_input = layer->hidden_state;
        }
    }
    
    // Compute output from last layer
    RNNLayer *last_layer = &rnn->layers[rnn->num_layers - 1];
    for (size_t i = 0; i < rnn->output_size; i++) {
        long double sum = last_layer->b_y[i];
        for (size_t j = 0; j < rnn->hidden_size; j++) {
            sum += last_layer->hidden_state[j] * last_layer->W_hy[j * rnn->output_size + i];
        }
        output[i] = sigmoid(sum);  // Use sigmoid for output
    }
    
    return output;
}

// RNN state management
void RNN_reset_states(RNN_t *rnn) {
    if (!rnn) return;
    
    for (size_t i = 0; i < rnn->num_layers; i++) {
        RNNLayer *layer = &rnn->layers[i];
        memset(layer->hidden_state, 0, layer->hidden_size * sizeof(long double));
        memset(layer->prev_hidden_state, 0, layer->hidden_size * sizeof(long double));
        
        if (layer->type == RNN_LSTM) {
            memset(layer->cell_state, 0, layer->hidden_size * sizeof(long double));
            memset(layer->prev_cell_state, 0, layer->hidden_size * sizeof(long double));
        }
    }
}

void RNN_set_sequence_length(RNN_t *rnn, size_t sequence_length) {
    if (rnn) {
        rnn->sequence_length = sequence_length;
    }
}

// Game-specific functions
RNN_t* RNN_create_game_network(size_t input_size, size_t hidden_size, size_t output_size) {
    // Create a 2-layer LSTM network for game state prediction
    return RNN_create(input_size, hidden_size, output_size, 2, RNN_LSTM);
}

long double* RNN_predict_game_state(RNN_t *rnn, const long double *game_state) {
    return RNN_forward(rnn, game_state, 1);
}

void RNN_train_game_sequence(RNN_t *rnn, const long double *sequence, const long double *targets, 
                            size_t sequence_length) {
    // This is a simplified training function
    // In practice, you'd implement full backpropagation through time (BPTT)
    
    long double *predictions = RNN_forward(rnn, sequence, sequence_length);
    if (predictions) {
        // Simple loss calculation (in practice, you'd compute gradients and update weights)
        long double loss = 0.0L;
        for (size_t i = 0; i < rnn->output_size; i++) {
            long double diff = predictions[i] - targets[i];
            loss += diff * diff;
        }
        loss /= rnn->output_size;
        
        printf("Training loss: %.6Lf\n", loss);
        free(predictions);
    }
    
    // Reset states for next sequence
    RNN_reset_states(rnn);
}

void RNN_print_summary(RNN_t *rnn) {
    if (!rnn) return;
    
    printf("=== RNN Summary ===\n");
    printf("Type: %s\n", (rnn->layers[0].type == RNN_LSTM) ? "LSTM" : "Simple RNN");
    printf("Layers: %zu\n", rnn->num_layers);
    printf("Input size: %zu\n", rnn->input_size);
    printf("Hidden size: %zu\n", rnn->hidden_size);
    printf("Output size: %zu\n", rnn->output_size);
    printf("Learning rate: %.6Lf\n", rnn->learning_rate);
    printf("==================\n");
}

size_t RNN_get_parameter_count(RNN_t *rnn) {
    if (!rnn) return 0;
    
    size_t total = 0;
    for (size_t i = 0; i < rnn->num_layers; i++) {
        RNNLayer *layer = &rnn->layers[i];
        total += layer->input_size * layer->hidden_size;  // W_xh
        total += layer->hidden_size * layer->hidden_size; // W_hh
        total += layer->hidden_size * layer->output_size; // W_hy
        total += layer->hidden_size;  // b_h
        total += layer->output_size;  // b_y
        
        if (layer->type == RNN_LSTM) {
            total += 4 * layer->input_size * layer->hidden_size;  // LSTM input weights
            total += 4 * layer->hidden_size * layer->hidden_size; // LSTM hidden weights
            total += 4 * layer->hidden_size;  // LSTM biases
        }
    }
    
    return total;
}
