#ifndef KAN_H
#define KAN_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Kolmogorov-Arnold Network Layer structure
typedef struct KANLayer {
  size_t input_dim;
  size_t output_dim;
  long double *coefficients;   // Coefficients for each function
  long double *bias;
  long double *grad_coefficients;
  long double *grad_bias;
  int grid_size;  // Grid size for spline functions
  long double *grid_points;
} KANLayer;

// Kolmogorov-Arnold Network structure
typedef struct KAN {
  size_t num_layers;
  size_t input_dim;
  size_t output_dim;
  size_t hidden_dim;
  KANLayer *layers;
  long double learning_rate;
  int grid_size;  // Grid size for spline functions
  long double spline_scale;
  bool use_symbolic;
  int symbolic_threshold;
} KAN_t;

// Kolmogorov-Arnold Network functions
KAN_t *KAN_create(size_t input_dim, size_t hidden_dim, size_t output_dim, size_t num_layers);
void KAN_destroy(KAN_t *kan);
long double *KAN_forward(KAN_t *kan, long double *inputs);
void KAN_backward(KAN_t *kan, long double *targets);
void KAN_train_step(KAN_t *kan, long double *inputs, long double *targets);
void KAN_set_learning_rate(KAN_t *kan, long double learning_rate);
void KAN_set_grid_size(KAN_t *kan, int grid_size);
void KAN_enable_symbolic(KAN_t *kan, bool enable);
void KAN_set_symbolic_threshold(KAN_t *kan, int threshold);
size_t KAN_get_parameter_count(KAN_t *kan);
void KAN_print_summary(KAN_t *kan);

// Spline functions
long double kan_b_spline(long double x, long double *grid, int grid_size, int i);
long double kan_b_spline_derivative(long double x, long double *grid, int grid_size, int i);
void kan_update_spline_grid(KANLayer *layer, long double *inputs, size_t num_samples);

// Symbolic functions
void kan_prune_functions(KAN_t *kan);
void kan_symbolic_search(KAN_t *kan);
char *kan_get_formula(KAN_t *kan, size_t layer_idx, size_t output_idx);
bool kan_is_symbolic(KAN_t *kan, size_t layer_idx, size_t output_idx);

// Regularization
void kan_l1_regularization(KAN_t *kan, long double lambda);
void kan_entropy_regularization(KAN_t *kan, long double lambda);
void kan_simplify_functions(KAN_t *kan, long double threshold);

#ifdef __cplusplus
}
#endif

#endif // KAN_H
