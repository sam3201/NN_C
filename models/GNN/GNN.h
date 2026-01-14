#ifndef GNN_H
#define GNN_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Graph Neural Network Node structure
typedef struct GNNNode {
  size_t id;
  size_t feature_dim;
  long double *features;
  long double *hidden_state;
  size_t num_neighbors;
  size_t *neighbors;
  long double *edge_weights;
} GNNNode;

// Graph Neural Network Layer structure
typedef struct GNNLayer {
  size_t input_dim;
  size_t output_dim;
  long double *W_transform;  // Node transformation matrix
  long double *W_message;    // Message passing matrix
  long double *W_update;     // Update matrix
  long double *b_transform;  // Transform bias
  long double *b_message;    // Message bias
  long double *b_update;     // Update bias
  long double *grad_W_transform;
  long double *grad_W_message;
  long double *grad_W_update;
  long double *grad_b_transform;
  long double *grad_b_message;
  long double *grad_b_update;
} GNNLayer;

// Graph Neural Network structure
typedef struct GNN {
  size_t num_nodes;
  size_t num_layers;
  size_t feature_dim;
  size_t hidden_dim;
  GNNNode *nodes;
  GNNLayer *layers;
  long double learning_rate;
  int num_iterations;  // Message passing iterations
  bool directed;
  long double **node_outputs;
} GNN_t;

// Graph Neural Network functions
GNN_t *GNN_create(size_t num_nodes, size_t feature_dim, size_t hidden_dim, size_t num_layers);
void GNN_destroy(GNN_t *gnn);
long double *GNN_forward(GNN_t *gnn, long double **node_features);
void GNN_backward(GNN_t *gnn, long double **node_targets);
void GNN_add_edge(GNN_t *gnn, size_t node1, size_t node2, long double weight);
void GNN_remove_edge(GNN_t *gnn, size_t node1, size_t node2);
void GNN_set_node_features(GNN_t *gnn, size_t node_id, long double *features);
void GNN_set_learning_rate(GNN_t *gnn, long double learning_rate);
void GNN_reset_gradients(GNN_t *gnn);
void GNN_update_weights(GNN_t *gnn);
size_t GNN_get_parameter_count(GNN_t *gnn);
void GNN_print_summary(GNN_t *gnn);

// Graph utilities
void GNN_create_complete_graph(GNN_t *gnn);
void GNN_create_line_graph(GNN_t *gnn);
void GNN_create_star_graph(GNN_t *gnn, size_t center_node);
bool GNN_has_edge(GNN_t *gnn, size_t node1, size_t node2);
size_t GNN_get_degree(GNN_t *gnn, size_t node_id);

#ifdef __cplusplus
}
#endif

#endif // GNN_H
