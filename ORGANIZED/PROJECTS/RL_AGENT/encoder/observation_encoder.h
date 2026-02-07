#ifndef OBSERVATION_ENCODER_H
#define OBSERVATION_ENCODER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../utils/NN/NN/NN.h"
#include "../../utils/NN/CONVOLUTION/CONVOLUTION.h"
#include "../../utils/NN/TRANSFORMER/TRANSFORMER.h"

// Observation grid dimensions
#define GRID_HEIGHT 32
#define GRID_WIDTH 32
#define NUM_FEATURE_CHANNELS 8
#define EMBED_DIM 64
#define CONV_CHANNELS 128
#define NUM_TRANSFORMER_LAYERS 4
#define NUM_ATTENTION_HEADS 8
#define LATENT_DIM 256

// Feature types for grid cells
typedef enum {
    FEATURE_EMPTY = 0,
    FEATURE_SELF_AGENT = 1,
    FEATURE_ALLY_AGENT = 2,
    FEATURE_ENEMY_AGENT = 3,
    FEATURE_PASSIVE_MOB = 4,
    FEATURE_HOSTILE_MOB = 5,
    FEATURE_RESOURCE = 6,
    FEATURE_STRUCTURE = 7,
    FEATURE_TERRAIN = 8
} FeatureType;

// Grid observation structure
typedef struct {
    float grid[GRID_HEIGHT][GRID_WIDTH][NUM_FEATURE_CHANNELS];
    float height_map[GRID_HEIGHT][GRID_WIDTH];
    float velocity_x[GRID_HEIGHT][GRID_WIDTH];
    float velocity_y[GRID_HEIGHT][GRID_WIDTH];
} GridObservation;

// Positional encoding
typedef struct {
    float sin_pos[EMBED_DIM];
    float cos_pos[EMBED_DIM];
} PositionalEncoding;

// Observation encoder state
typedef struct {
    // CNN components
    ConvNet* conv_net;
    
    // Transformer components
    Transformer_t* transformer;
    
    // Positional encodings
    PositionalEncoding pos_encodings[GRID_HEIGHT][GRID_WIDTH];
    
    // Output latent vector
    float latent_vector[LATENT_DIM];
    
    // Feature embedding matrix
    float feature_embeddings[NUM_FEATURE_CHANNELS][EMBED_DIM];
    
    // MLP for final aggregation
    NN_t* aggregation_mlp;
} ObservationEncoder;

// Function declarations
ObservationEncoder* observation_encoder_create();
void observation_encoder_destroy(ObservationEncoder* encoder);
void observation_encoder_encode(ObservationEncoder* encoder, GridObservation* obs, float* latent_output);
void observation_encoder_reset(ObservationEncoder* encoder);
void observation_encoder_init_positional_encodings(ObservationEncoder* encoder);
void embed_features(GridObservation* obs, float* embedded_grid, float (*feature_embeddings)[EMBED_DIM]);
void add_positional_encoding(float* embedded_grid, PositionalEncoding (*pos_encodings)[GRID_WIDTH]);

#endif // OBSERVATION_ENCODER_H
