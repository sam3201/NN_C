#include "GNN.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper functions
static long double relu_activation(long double x) {
    return x > 0.0L ? x : 0.0L;
}

static long double sigmoid(long double x) {
    return 1.0L / (1.0L + expl(-x));
}

// Graph Neural Network creation
GNN_t *GNN_create(size_t num_nodes, size_t feature_dim, size_t hidden_dim, size_t num_layers) {
    GNN_t *gnn = malloc(sizeof(GNN_t));
    if (!gnn) return NULL;
    
    gnn->num_nodes = num_nodes;
    gnn->feature_dim = feature_dim;
    gnn->hidden_dim = hidden_dim;
    gnn->num_layers = num_layers;
    gnn->learning_rate = 0.001L;
    gnn->num_iterations = 3;
    gnn->directed = false;
    
    // Create nodes
    gnn->nodes = malloc(num_nodes * sizeof(GNNNode));
    if (!gnn->nodes) {
        free(gnn);
        return NULL;
    }
    
    for (size_t i = 0; i < num_nodes; i++) {
        gnn->nodes[i].id = i;
        gnn->nodes[i].feature_dim = feature_dim;
        gnn->nodes[i].features = calloc(feature_dim, sizeof(long double));
        gnn->nodes[i].hidden_state = calloc(hidden_dim, sizeof(long double));
        gnn->nodes[i].num_neighbors = 0;
        gnn->neighbors = NULL;
        gnn->nodes[i].edge_weights = NULL;
    }
    
    // Create layers
    gnn->layers = malloc(num_layers * sizeof(GNNLayer));
    if (!gnn->layers) {
        free(gnn->nodes);
        free(gnn);
        return NULL;
    }
    
    for (size_t i = 0; i < num_layers; i++) {
        gnn->layers[i].input_dim = (i == 0) ? feature_dim : hidden_dim;
        gnn->layers[i].output_dim = hidden_dim;
        gnn->layers[i].activation = RELU;
        
        // Allocate weight matrices
        gnn->layers[i].W_transform = calloc(gnn->layers[i].input_dim * gnn->layers[i].output_dim, sizeof(long double));
        gnn->layers[i].W_message = calloc(gnn->layers[i].input_dim * gnn->layers[i].output_dim, sizeof(long double));
        gnn->layers[i].W_update = calloc(gnn->layers[i].input_dim * gnn->layers[i].output_dim, sizeof(long double));
        gnn->layers[i].b_transform = calloc(gnn->layers[i].output_dim, sizeof(long double));
        gnn->layers[i].b_message = calloc(gnn->layers[i].output_dim, sizeof(long double));
        gnn->layers[i].b_update = calloc(gnn->layers[i].output_dim, sizeof(long double));
        
        // Allocate gradients
        gnn->layers[i].grad_W_transform = calloc(gnn->layers[i].input_dim * gnn->layers[i].output_dim, sizeof(long double));
        gnn->layers[i].grad_W_message = calloc(gnn->layers[i].input_dim * gnn->layers[i].output_dim, sizeof(long double));
        gnn->layers[i].grad_W_update = calloc(gnn->layers[i].input_dim * gnn->layers[i].output_dim, sizeof(long double));
        gnn->layers[i].grad_b_transform = calloc(gnn->layers[i].output_dim, sizeof(long double));
        gnn->layers[i].grad_b_message = calloc(gnn->layers[i].output_dim, sizeof(long double));
        gnn->layers[i].grad_b_update = calloc(gnn->layers[i].output_dim, sizeof(long double));
        
        // Initialize weights with Xavier initialization
        double scale = sqrtl(2.0L / (gnn->layers[i].input_dim + gnn->layers[i].output_dim));
        for (size_t j = 0; j < gnn->layers[i].input_dim * gnn->layers[i].output_dim; j++) {
            gnn->layers[i].W_transform[j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
            gnn->layers[i].W_message[j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
            gnn->layers[i].W_update[j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * scale;
        }
        
        // Initialize biases to zero
        for (size_t j = 0; j < gnn->layers[i].output_dim; j++) {
            gnn->layers[i].b_transform[j] = 0.0L;
            gnn->layers[i].b_message[j] = 0.0L;
            gnn->layers[i].b_update[j] = 0.0L;
        }
    }
    
    // Allocate node outputs
    gnn->node_outputs = malloc(num_nodes * hidden_dim * sizeof(long double));
    if (!gnn->node_outputs) {
        for (size_t i = 0; i < num_layers; i++) {
            free(gnn->layers[i].W_transform);
            free(gnn->layers[i].W_message);
            free(gnn->layers[i].W_update);
            free(gnn->layers[i].b_transform);
            free(gnn->layers[i].b_message);
            free(gnn->layers[i].b_update);
            free(gnn->layers[i].grad_W_transform);
            free(gnn->layers[i].grad_W_message);
            free(gnn->layers[i].grad_W_update);
            free(gnn->layers[i].grad_b_transform);
            free(gnn->layers[i].grad_b_message);
            free(gnn->layers[i].grad_b_update);
        }
        free(gnn->layers);
        free(gnn->nodes);
        free(gnn);
        return NULL;
    }
    
    return gnn;
}

// Graph Neural Network destruction
void GNN_destroy(GNN_t *gnn) {
    if (!gnn) return;
    
    // Free nodes
    for (size_t i = 0; i < gnn->num_nodes; i++) {
        free(gnn->nodes[i].features);
        free(gnn->nodes[i].hidden_state);
        free(gnn->nodes[i].neighbors);
        free(gnn->nodes[i].edge_weights);
    }
    free(gnn->nodes);
    
    // Free layers
    for (size_t i = 0; i < gnn->num_layers; i++) {
        free(gnn->layers[i].W_transform);
        free(gnn->layers[i].W_message);
        free(gnn->layers[i].W_update);
        free(gnn->layers[i].b_transform);
        free(gnn->layers[i].b_message);
        free(gnn->layers[i].b_update);
        free(gnn->layers[i].grad_W_transform);
        free(gnn->layers[i].grad_W_message);
        free(gnn->layers[i].grad_W_update);
        free(gnn->layers[i].grad_b_transform);
        free(gnn->layers[i].grad_b_message);
        free(gnn->layers[i].grad_b_update);
    }
    free(gnn->layers);
    
    // Free outputs
    free(gnn->node_outputs);
    
    free(gnn);
}

// Add edge to graph
void GNN_add_edge(GNN_t *gnn, size_t node1, size_t node2, long double weight) {
    if (!gnn || node1 >= gnn->num_nodes || node2 >= gnn->num_nodes) return;
    
    // Add edge to node1's neighbor list
    size_t old_capacity = gnn->nodes[node1].num_neighbors;
    size_t new_capacity = old_capacity + 1;
    
    gnn->nodes[node1].neighbors = realloc(gnn->nodes[node1].neighbors, new_capacity * sizeof(size_t));
    gnn->nodes[node1].edge_weights = realloc(gnn->nodes[node1].edge_weights, new_capacity * sizeof(long double));
    
    gnn->nodes[node1].neighbors[old_capacity] = node2;
    gnn->nodes[node1].edge_weights[old_capacity] = weight;
    gnn->nodes[node1].num_neighbors++;
    
    // If undirected, add reverse edge
    if (!gnn->directed) {
        old_capacity = gnn->nodes[node2].num_neighbors;
        new_capacity = old_capacity + 1;
        
        gnn->nodes[node2].neighbors = realloc(gnn->nodes[node2].neighbors, new_capacity * sizeof(size_t));
        gnn->nodes[node2].edge_weights = realloc(gnn->nodes[node2].edge_weights, new_capacity * sizeof(long double));
        
        gnn->nodes[node2].neighbors[old_capacity] = node1;
        gnn->nodes[node2].edge_weights[old_capacity] = weight;
        gnn->nodes[node2].num_neighbors++;
    }
}

// Remove edge from graph
void GNN_remove_edge(GNN_t *gnn, size_t node1, size_t node2) {
    if (!gnn || node1 >= gnn->num_nodes || node2 >= gnn->num_nodes) return;
    
    // Find and remove edge from node1
    for (size_t i = 0; i < gnn->nodes[node1].num_neighbors; i++) {
        if (gnn->nodes[node1].neighbors[i] == node2) {
            // Shift remaining elements
            for (size_t j = i; j < gnn->nodes[node1].num_neighbors - 1; j++) {
                gnn->nodes[node1].neighbors[j] = gnn->nodes[node1].neighbors[j + 1];
                gnn->nodes[node1].edge_weights[j] = gnn->nodes[node1].edge_weights[j + 1];
            }
            gnn->nodes[node1].num_neighbors--;
            break;
        }
    }
    
    // If undirected, remove reverse edge
    if (!gnn->directed) {
        for (size_t i = 0; i < gnn->nodes[node2].num_neighbors; i++) {
            if (gnn->nodes[node2].neighbors[i] == node1) {
                // Shift remaining elements
                for (size_t j = i; j < gnn->nodes[node2].num_neighbors - 1; j++) {
                    gnn->nodes[node2].neighbors[j] = gnn->nodes[node2].neighbors[j + 1];
                    gnn->nodes[node2].edge_weights[j] = gnn->nodes[node2].edge_weights[j + 1];
                }
                gnn->nodes[node2].num_neighbors--;
                break;
            }
        }
    }
}

// Set node features
void GNN_set_node_features(GNN_t *gnn, size_t node_id, long double *features) {
    if (!gnn || node_id >= gnn->num_nodes) return;
    memcpy(gnn->nodes[node_id].features, features, gnn->feature_dim * sizeof(long double));
}

// Check if edge exists
bool GNN_has_edge(GNN_t *gnn, size_t node1, size_t node2) {
    if (!gnn || node1 >= gnn->num_nodes || node2 >= gnn->num_nodes) return false;
    
    for (size_t i = 0; i < gnn->nodes[node1].num_neighbors; i++) {
        if (gnn->nodes[node1].neighbors[i] == node2) {
            return true;
        }
    }
    return false;
}

// Get node degree
size_t GNN_get_degree(GNN_t *gnn, size_t node_id) {
    if (!gnn || node_id >= gnn->num_nodes) return 0;
    return gnn->nodes[node_id].num_neighbors;
}

// Set learning rate
void GNN_set_learning_rate(GNN_t *gnn, long double learning_rate) {
    if (gnn) {
        gnn->learning_rate = learning_rate;
    }
}

// Reset gradients
void GNN_reset_gradients(GNN_t *gnn) {
    if (!gnn) return;
    
    for (size_t i = 0; i < gnn->num_layers; i++) {
        memset(gnn->layers[i].grad_W_transform, 0, gnn->layers[i].input_dim * gnn->layers[i].output_dim * sizeof(long double));
        memset(gnn->layers[i].grad_W_message, 0, gnn->layers[i].input_dim * gnn->layers[i].output_dim * sizeof(long double));
        memset(gnn->layers[i].grad_W_update, 0, gnn->layers[i].input_dim * gnn->layers[i].output_dim * sizeof(long double));
        memset(gnn->layers[i].grad_b_transform, 0, gnn->layers[i].output_dim * sizeof(long double));
        memset(gnn->layers[i].grad_b_message, 0, gnn->layers[i].output_dim * sizeof(long double));
        memset(gnn->layers[i].grad_b_update, 0, gnn->layers[i].output_dim * sizeof(long double));
    }
}

// Update weights
void GNN_update_weights(GNN_t *gnn) {
    if (!gnn) return;
    
    for (size_t i = 0; i < gnn->num_layers; i++) {
        GNNLayer *layer = &gnn->layers[i];
        
        // Update weights with gradients
        for (size_t j = 0; j < layer->input_dim * layer->output_dim; j++) {
            layer->W_transform[j] -= gnn->learning_rate * layer->grad_W_transform[j];
            layer->W_message[j] -= gnn->learning_rate * layer->grad_W_message[j];
            layer->W_update[j] -= gnn->learning_rate * layer->grad_W_update[j];
        }
        
        for (size_t j = 0; j < layer->output_dim; j++) {
            layer->b_transform[j] -= gnn->learning_rate * layer->grad_b_transform[j];
            layer->b_message[j] -= gnn->learning_rate * layer->grad_b_message[j];
            layer->b_update[j] -= gnn->learning_rate * layer->grad_b_update[j];
        }
    }
}

// Get parameter count
size_t GNN_get_parameter_count(GNN_t *gnn) {
    if (!gnn) return 0;
    
    size_t total = 0;
    for (size_t i = 0; i < gnn->num_layers; i++) {
        GNNLayer *layer = &gnn->layers[i];
        total += layer->input_dim * layer->output_dim;  // W_transform
        total += layer->input_dim * layer->output_dim;  // W_message
        total += layer->input_dim * layer->output_dim;  // W_update
        total += layer->output_dim;              // b_transform
        total += layer->output_dim;              // b_message
        total += layer->output_dim;              // b_update
    }
    return total;
}

// Print summary
void GNN_print_summary(GNN_t *gnn) {
    if (!gnn) return;
    
    printf("=== GNN Summary ===\n");
    printf("Nodes: %zu\n", gnn->num_nodes);
    printf("Feature Dimension: %zu\n", gnn->feature_dim);
    printf("Hidden Dimension: %zu\n", gnn->hidden_dim);
    printf("Layers: %zu\n", gnn->num_layers);
    printf("Message Passing Iterations: %d\n", gnn->num_iterations);
    printf("Directed: %s\n", gnn->directed ? "Yes" : "No");
    printf("Learning Rate: %.6Lf\n", gnn->learning_rate);
    printf("Parameters: %zu\n", GNN_get_parameter_count(gnn));
    printf("==================\n");
}

// Create complete graph
void GNN_create_complete_graph(GNN_t *gnn) {
    if (!gnn) return;
    
    for (size_t i = 0; i < gnn->num_nodes; i++) {
        for (size_t j = 0; j < gnn->num_nodes; j++) {
            if (i != j) {
                GNN_add_edge(gnn, i, j, 1.0L);
            }
        }
    }
}

// Create line graph
void GNN_create_line_graph(GNN_t *gnn) {
    if (!gnn) return;
    
    for (size_t i = 0; i < gnn->num_nodes - 1; i++) {
        GNN_add_edge(gnn, i, i + 1, 1.0L);
    }
}

// Create star graph
void GNN_create_star_graph(GNN_t *gnn, size_t center_node) {
    if (!gnn || center_node >= gnn->num_nodes) return;
    
    for (size_t i = 0; i < gnn->num_nodes; i++) {
        if (i != center_node) {
            GNN_add_edge(gnn, center_node, i, 1.0L);
        }
    }
}

// Message passing function
static void message_passing(GNN_t *gnn, size_t layer_idx) {
    GNNLayer *layer = &gnn->layers[layer_idx];
    
    for (size_t node_id = 0; node_id < gnn->num_nodes; node_id++) {
        GNNNode *node = &gnn->nodes[node_id];
        
        // Aggregate messages from neighbors
        long double *message = calloc(layer->output_dim, sizeof(long double));
        for (size_t i = 0; i < node->num_neighbors; i++) {
            size_t neighbor_id = node->neighbors[i];
            GNNNode *neighbor = &gnn->nodes[neighbor_id];
            long double edge_weight = node->edge_weights[i];
            
            // Message passing: m_ij = W_message * h_j
            for (size_t j = 0; j < layer->input_dim; j++) {
                long double sum = 0.0L;
                for (size_t k = 0; k < layer->input_dim; k++) {
                    sum += neighbor->hidden_state[k] * layer->W_message[k * layer->output_dim + j];
                }
                message[j] += sum * edge_weight;
            }
        }
        
        // Update: h_i' = Update(h_i, m_i)
        for (size_t j = 0; j < layer->input_dim; j++) {
            long double sum = 0.0L;
            for (size_t k = 0; k < layer->input_dim; k++) {
                sum += node->features[k] * layer->W_transform[k * layer->output_dim + j];
            }
            message[j] += sum;
        }
        
        for (size_t j = 0; j < layer->output_dim; j++) {
            long double pre_activation = message[j] + layer->b_message[j];
            long double post_activation = relu_activation(pre_activation);
            node->hidden_state[j] = post_activation;
        }
        
        free(message);
    }
}

// Forward pass
long double *GNN_forward(GNN_t *gnn, long double **node_features) {
    if (!gnn || !node_features) return NULL;
    
    // Set node features
    for (size_t i = 0; i < gnn->num_nodes; i++) {
        GNN_set_node_features(gnn, i, node_features[i]);
    }
    
    // Message passing iterations
    for (int iter = 0; iter < gnn->num_iterations; iter++) {
        for (size_t layer_idx = 0; layer_idx < gnn->num_layers; layer_idx++) {
            message_passing(gnn, layer_idx);
        }
    }
    
    // Collect outputs
    long double *output = malloc(gnn->num_nodes * gnn->hidden_dim * sizeof(long double));
    if (!output) return NULL;
    
    for (size_t i = 0; i < gnn->num_nodes; i++) {
        for (size_t j = 0; j < gnn->hidden_dim; j++) {
            output[i * gnn->hidden_dim + j] = gnn->nodes[i].hidden_state[j];
        }
    }
    
    return output;
}

// Backward pass
void GNN_backward(GNN_t *gnn, long double **node_targets) {
    if (!gnn || !node_targets) return;
    
    // Reset gradients
    GNN_reset_gradients(gnn);
    
    // Compute loss gradients (simplified MSE)
    for (size_t i = 0; i < gnn->num_nodes; i++) {
        long double *output = &gnn->node_outputs[i * gnn->hidden_dim];
        long double *target = node_targets[i];
        
        for (size_t j = 0; j < gnn->hidden_dim; j++) {
            long double error = output[j] - target[j];
            // Backprop through final layer
            gnn->layers[gnn->num_layers - 1].grad_b_update[j] += error;
            gnn->layers[gnn->num_layers - 1].grad_W_update[j * gnn->hidden_dim + j] += error * gnn->nodes[i].hidden_state[j];
        }
    }
    
    // Backprop through layers
    for (int layer_idx = gnn->num_layers - 2; layer_idx >= 0; layer_idx--) {
        GNNLayer *layer = &gnn->layers[layer_idx];
        
        for (size_t node_id = 0; node_id < gnn->num_nodes; node_id++) {
            GNNNode *node = &gnn->nodes[node_id];
            
            // Backprop through update layer
            for (size_t j = 0; j < layer->output_dim; j++) {
                long double grad_output = gnn->layers[layer_idx + 1].grad_b_update[j];
                long double h_j = node->hidden_state[j];
                
                // Backprop through activation
                long double grad_pre_activation = grad_output * (h_j > 0.0L ? 1.0L : 0.0L);
                gnn->layers[layer_idx].grad_b_message[j] += grad_pre_activation;
                
                // Backprop through message passing
                for (size_t k = 0; k < layer->input_dim; k++) {
                    gnn->layers[layer_idx].grad_W_message[k * layer->output_dim + j] += grad_pre_activation * node->features[k];
                }
            }
            
            // Backprop through transform layer
            for (size_t j = 0; j < layer->input_dim; j++) {
                long double grad_hidden = 0.0L;
                for (size_t k = 0; k < layer->output_dim; k++) {
                    grad_hidden += gnn->layers[layer_idx].grad_W_update[j * layer->output_dim + k] * node->hidden_state[k];
                }
                gnn->layers[layer_idx].grad_W_transform[j * layer->output_dim + j] += grad_hidden * node->features[j];
                gnn->layers[layer_idx].grad_b_transform[j] += grad_hidden;
            }
        }
    }
    
    // Update weights
    GNN_update_weights(gnn);
}
