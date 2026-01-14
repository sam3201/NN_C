#include "observation_encoder.h"
#include <string.h>

// Initialize positional encodings
void observation_encoder_init_positional_encodings(ObservationEncoder* encoder) {
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            for (int i = 0; i < EMBED_DIM / 2; i++) {
                float pos = (float)(y * GRID_WIDTH + x) / (GRID_HEIGHT * GRID_WIDTH);
                float freq = pow(10000.0f, 2.0f * i / EMBED_DIM);
                encoder->pos_encodings[y][x].sin_pos[2 * i] = sinf(pos * freq);
                encoder->pos_encodings[y][x].cos_pos[2 * i] = cosf(pos * freq);
                encoder->pos_encodings[y][x].sin_pos[2 * i + 1] = sinf(pos * freq);
                encoder->pos_encodings[y][x].cos_pos[2 * i + 1] = cosf(pos * freq);
            }
        }
    }
}

// Embed raw features into continuous vectors
void embed_features(GridObservation* obs, float* embedded_grid, float (*feature_embeddings)[EMBED_DIM]) {
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            int idx = (y * GRID_WIDTH + x) * EMBED_DIM;
            
            // For each feature channel, add its embedding
            for (int c = 0; c < NUM_FEATURE_CHANNELS; c++) {
                float feature_value = obs->grid[y][x][c];
                for (int d = 0; d < EMBED_DIM; d++) {
                    embedded_grid[idx + d] += feature_value * feature_embeddings[c][d];
                }
            }
            
            // Add height and velocity information
            embedded_grid[idx + EMBED_DIM/4] += obs->height_map[y][x];
            embedded_grid[idx + EMBED_DIM/2] += obs->velocity_x[y][x];
            embedded_grid[idx + 3*EMBED_DIM/4] += obs->velocity_y[y][x];
        }
    }
}

// Add positional encoding to embedded grid
void add_positional_encoding(float* embedded_grid, PositionalEncoding (*pos_encodings)[GRID_WIDTH]) {
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            int idx = (y * GRID_WIDTH + x) * EMBED_DIM;
            for (int d = 0; d < EMBED_DIM; d++) {
                embedded_grid[idx + d] += pos_encodings[y][x].sin_pos[d];
                embedded_grid[idx + d] += pos_encodings[y][x].cos_pos[d];
            }
        }
    }
}

// Create observation encoder
ObservationEncoder* observation_encoder_create() {
    ObservationEncoder* encoder = malloc(sizeof(ObservationEncoder));
    if (!encoder) return NULL;
    
    // Initialize feature embeddings with random values
    for (int c = 0; c < NUM_FEATURE_CHANNELS; c++) {
        for (int d = 0; d < EMBED_DIM; d++) {
            encoder->feature_embeddings[c][d] = (float)(rand() % 1000 - 500) / 1000.0f;
        }
    }
    
    // Initialize positional encodings
    observation_encoder_init_positional_encodings(encoder);
    
    // Create CNN for spatial feature extraction
    encoder->conv_net = convnet_create(GRID_HEIGHT, GRID_WIDTH, EMBED_DIM, 0.001f);
    if (!encoder->conv_net) {
        free(encoder);
        return NULL;
    }
    
    // Add convolutional layers
    convnet_add_conv2d(encoder->conv_net, CONV_CHANNELS, 3, 3, 1, 1);
    convnet_add_relu(encoder->conv_net);
    convnet_add_conv2d(encoder->conv_net, CONV_CHANNELS, 3, 3, 1, 1);
    convnet_add_relu(encoder->conv_net);
    convnet_add_conv2d(encoder->conv_net, CONV_CHANNELS, 2, 2, 2, 1);
    convnet_add_relu(encoder->conv_net);
    
    // Flatten and add dense layers
    convnet_add_flatten(encoder->conv_net);
    convnet_add_dense(encoder->conv_net, CONV_CHANNELS * 8);
    convnet_add_relu(encoder->conv_net);
    
    // Create Transformer for global relational reasoning
    encoder->transformer = TRANSFORMER_init(CONV_CHANNELS * 8, NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS);
    if (!encoder->transformer) {
        convnet_free(encoder->conv_net);
        free(encoder);
        return NULL;
    }
    
    // Create aggregation MLP
    size_t layers[] = {CONV_CHANNELS * 8, LATENT_DIM * 2, LATENT_DIM};
    ActivationFunctionType actFuncs[] = {RELU, RELU, LINEAR};
    ActivationDerivativeType actDerivs[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    encoder->aggregation_mlp = NN_init(layers, actFuncs, actDerivs, MSE, MSE_DERIVATIVE, L1, SGD, 0.001);
    if (!encoder->aggregation_mlp) {
        TRANSFORMER_destroy(encoder->transformer);
        convnet_free(encoder->conv_net);
        free(encoder);
        return NULL;
    }
    
    // Initialize latent vector
    memset(encoder->latent_vector, 0, sizeof(float) * LATENT_DIM);
    
    return encoder;
}

// Encode observation to latent vector
void observation_encoder_encode(ObservationEncoder* encoder, GridObservation* obs, float* latent_output) {
    // Step 1: Embed features
    float embedded_grid[GRID_HEIGHT * GRID_WIDTH * EMBED_DIM];
    memset(embedded_grid, 0, sizeof(embedded_grid));
    embed_features(obs, embedded_grid, encoder->feature_embeddings);
    
    // Step 2: Add positional encoding
    add_positional_encoding(embedded_grid, encoder->pos_encodings);
    
    // Step 3: Pass through CNN
    const float* conv_output = convnet_forward(encoder->conv_net, (float*)embedded_grid);
    
    // Step 4: Prepare sequence for Transformer
    int seq_length = CONV_CHANNELS * 8;
    long double** transformer_input = malloc(seq_length * sizeof(long double*));
    for (int i = 0; i < seq_length; i++) {
        transformer_input[i] = malloc(sizeof(long double));
        transformer_input[i][0] = (long double)conv_output[i];
    }
    
    // Step 5: Pass through Transformer
    long double** transformer_output = TRANSFORMER_forward(encoder->transformer, transformer_input, seq_length);
    
    // Step 6: Aggregate to single latent vector
    float aggregated[CONV_CHANNELS * 8];
    for (int i = 0; i < seq_length; i++) {
        aggregated[i] = (float)transformer_output[i][0];
    }
    
    // Step 7: Pass through aggregation MLP
    long double mlp_input[CONV_CHANNELS * 8];
    for (int i = 0; i < seq_length; i++) {
        mlp_input[i] = (long double)aggregated[i];
    }
    
    long double* mlp_output = NN_forward(encoder->aggregation_mlp, mlp_input);
    
    // Copy to output
    for (int i = 0; i < LATENT_DIM; i++) {
        latent_output[i] = (float)mlp_output[i];
        encoder->latent_vector[i] = latent_output[i];
    }
    
    // Cleanup
    for (int i = 0; i < seq_length; i++) {
        free(transformer_input[i]);
    }
    free(transformer_input);
}

// Reset encoder state
void observation_encoder_reset(ObservationEncoder* encoder) {
    memset(encoder->latent_vector, 0, sizeof(float) * LATENT_DIM);
}

// Destroy observation encoder
void observation_encoder_destroy(ObservationEncoder* encoder) {
    if (!encoder) return;
    
    if (encoder->conv_net) convnet_free(encoder->conv_net);
    if (encoder->transformer) TRANSFORMER_destroy(encoder->transformer);
    if (encoder->aggregation_mlp) NN_destroy(encoder->aggregation_mlp);
    
    free(encoder);
}
