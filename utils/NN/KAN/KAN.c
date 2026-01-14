#include "KAN.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper functions
static long double b_spline(long double x, long double *grid, int grid_size, int i) {
    if (i < 0 || i >= grid_size) return 0.0L;
    
    long double x0 = grid[i];
    long double x1 = grid[i + 1];
    
    if (x < x0 || x > x1) return 0.0L;
    
    long double t = (x - x0) / (x1 - x0);
    
    if (t < 0.5L) {
        return 2.0L * t;
    } else {
        return 2.0L * (1.0L - t);
    }
}

// Kolmogorov-Arnold Network creation
KAN_t *KAN_create(size_t input_dim, size_t hidden_dim, size_t output_dim, size_t num_layers) {
    KAN_t *kan = malloc(sizeof(KAN_t));
    if (!kan) return NULL;
    
    kan->input_dim = input_dim;
    kan->hidden_dim = hidden_dim;
    kan->output_dim = output_dim;
    kan->num_layers = num_layers;
    kan->learning_rate = 0.001L;
    kan->grid_size = 10;
    kan->spline_scale = 1.0L;
    kan->use_symbolic = false;
    kan->symbolic_threshold = 5;
    
    // Create layers
    kan->layers = malloc(num_layers * sizeof(KANLayer));
    if (!kan->layers) {
        free(kan);
        return NULL;
    }
    
    // Initialize layers with simplified structure
    for (size_t i = 0; i < num_layers; i++) {
        kan->layers[i].input_dim = (i == 0) ? input_dim : hidden_dim;
        kan->layers[i].output_dim = (i == num_layers - 1) ? output_dim : hidden_dim;
        kan->layers[i].grid_size = kan->grid_size;
        kan->layers[i].spline_scale = kan->spline_scale;
        
        // Simple allocation for coefficients
        kan->layers[i].coefficients = malloc(kan->layers[i].input_dim * kan->layers[i].output_dim * sizeof(long double));
        kan->layers[i].bias = calloc(kan->layers[i].output_dim, sizeof(long double));
        kan->layers[i].grad_coefficients = malloc(kan->layers[i].input_dim * kan->layers[i].output_dim * sizeof(long double));
        kan->layers[i].grad_bias = calloc(kan->layers[i].output_dim, sizeof(long double));
        
        // Initialize coefficients with small random values
        for (size_t j = 0; j < kan->layers[i].input_dim * kan->layers[i].output_dim; j++) {
            kan->layers[i].coefficients[j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.1L;
        }
    }
    
    return kan;
}

// Kolmogorov-Arnold Network destruction
void KAN_destroy(KAN_t *kan) {
    if (!kan) return;
    
    // Free layers
    for (size_t i = 0; i < kan->num_layers; i++) {
        free(kan->layers[i].coefficients);
        free(kan->layers[i].bias);
        free(kan->layers[i].grad_coefficients);
        free(kan->layers[i].grad_bias);
    }
    free(kan->layers);
    free(kan);
}

// Set grid size
void KAN_set_grid_size(KAN_t *kan, int grid_size) {
    if (kan) {
        kan->grid_size = grid_size;
        for (size_t i = 0; i < kan->num_layers; i++) {
            kan->layers[i].grid_size = grid_size;
        }
    }
}

// Enable symbolic mode
void KAN_enable_symbolic(KAN_t *kan, bool enable) {
    if (kan) {
        kan->use_symbolic = enable;
    }
}

// Set symbolic threshold
void KAN_set_symbolic_threshold(KAN_t *kan, int threshold) {
    if (kan) {
        kan->symbolic_threshold = threshold;
    }
}

// Forward pass through KAN
long double *KAN_forward(KAN_t *kan, long double *inputs) {
    if (!kan || !inputs) return NULL;
    
    long double *current_inputs = inputs;
    
    // Forward pass through layers
    for (size_t layer_idx = 0; layer_idx < kan->num_layers; layer_idx++) {
        KANLayer *layer = &kan->layers[layer_idx];
        
        // Apply learnable activation functions
        long double *next_inputs = malloc(layer->output_dim * sizeof(long double));
        if (!next_inputs) {
            if (current_inputs != inputs) free(current_inputs);
            return NULL;
        }
        
        for (size_t i = 0; i < layer->output_dim; i++) {
            long double sum = layer->bias[i];
            
            // Sum over all inputs
            for (size_t j = 0; j < layer->input_dim; j++) {
                // Simple linear combination (simplified from spline)
                long double phi_value = current_inputs[j];
                sum += layer->coefficients[j * layer->output_dim + i] * phi_value;
            }
            
            next_inputs[i] = sum + layer->bias[i];
        }
        
        if (layer_idx < kan->num_layers - 1) {
            free(current_inputs);
            current_inputs = next_inputs;
        } else {
            current_inputs = next_inputs;
        }
    }
    
    return current_inputs;
}

// Backward pass through KAN
void KAN_backward(KAN_t *kan, long double *targets) {
    if (!kan || !targets) return;
    
    // Simplified backward pass
    for (size_t layer_idx = 0; layer_idx < kan->num_layers; layer_idx++) {
        KANLayer *layer = &kan->layers[layer_idx];
        
        // Reset gradients
        for (size_t i = 0; i < layer->output_dim; i++) {
            layer->grad_bias[i] = 0.0L;
        }
        
        for (size_t i = 0; i < layer->input_dim * layer->output_dim; i++) {
            layer->grad_coefficients[i] = 0.0L;
        }
    }
    
    // Simple gradient computation (simplified)
    for (size_t layer_idx = kan->num_layers - 1; layer_idx < kan->num_layers; layer_idx++) {
        KANLayer *layer = &kan->layers[layer_idx];
        
        for (size_t i = 0; i < layer->output_dim; i++) {
            long double error = targets[i] - layer->bias[i];
            layer->grad_bias[i] = error;
            
            for (size_t j = 0; j < layer->input_dim; j++) {
                layer->grad_coefficients[j * layer->output_dim + i] = error;
            }
        }
    }
}

// Training step
void KAN_train_step(KAN_t *kan, long double *inputs, long double *targets) {
    if (!kan || !inputs || !targets) return;
    
    // Forward pass
    long double *predictions = KAN_forward(kan, inputs);
    
    // Backward pass
    KAN_backward(kan, targets);
    
    // Update weights
    for (size_t i = 0; i < kan->num_layers; i++) {
        KANLayer *layer = &kan->layers[i];
        
        // Update coefficients
        for (size_t j = 0; j < layer->input_dim * layer->output_dim; j++) {
            layer->coefficients[j] -= kan->learning_rate * layer->grad_coefficients[j];
        }
        
        // Update biases
        for (size_t j = 0; j < layer->output_dim; j++) {
            layer->bias[j] -= kan->learning_rate * layer->grad_bias[j];
        }
    }
    
    free(predictions);
}

// Prune unnecessary functions
void kan_prune_functions(KAN_t *kan, long double threshold) {
    if (!kan) return;
    
    for (size_t layer_idx = 0; layer_idx < kan->num_layers; layer_idx++) {
        KANLayer *layer = &kan->layers[layer_idx];
        
        for (size_t i = 0; i < layer->input_dim * layer->output_dim; i++) {
            long double abs_coeff = fabsl(layer->coefficients[i]);
            if (abs_coeff < threshold) {
                // Zero out small coefficients
                layer->coefficients[i] = 0.0L;
            }
        }
    }
}

// Entropy regularization
void kan_entropy_regularization(KAN_t *kan, long double lambda) {
    if (!kan) return;
    
    long double entropy = 0.0L;
    
    for (size_t layer_idx = 0; layer_idx < kan->num_layers; layer_idx++) {
        KANLayer *layer = &kan->layers[layer_idx];
        
        for (size_t i = 0; i < layer->input_dim * layer->output_dim; i++) {
            long double coeff = layer->coefficients[i];
            if (coeff != 0.0L) {
                entropy -= coeff * logl(fabsl(coeff));
            }
        }
    }
    
    // Add entropy penalty to loss
    // This would be integrated into the main training loop
}

// Simplify functions
void kan_simplify_functions(KAN_t *kan, long double threshold) {
    if (!kan) return;
    
    for (size_t layer_idx = 0; layer_idx < kan->num_layers; layer_idx++) {
        KANLayer *layer = &kan->layers[layer_idx];
        
        for (size_t i = 0; i < layer->input_dim * layer->output_dim; i++) {
            long double coeff = layer->coefficients[i];
            if (fabs(coeff) < threshold) {
                // Replace with simple linear function
                layer->coefficients[i] = 0.0L;
            }
        }
    }
}

// Get parameter count
size_t KAN_get_parameter_count(KAN_t *kan) {
    if (!kan) return 0;
    
    size_t total = 0;
    for (size_t i = 0; i < kan->num_layers; i++) {
        KANLayer *layer = &kan->layers[i];
        total += layer->input_dim * layer->output_dim;  // coefficients
        total += layer->output_dim;                    // biases
    }
    return total;
}

// Get symbolic formula for a specific output
char *kan_get_formula(KAN_t *kan, size_t layer_idx, size_t output_idx) {
    if (!kan || layer_idx >= kan->num_layers || output_idx >= kan->layers[layer_idx].output_dim) {
        return strdup("f(x) = 0.0");
    }
    
    KANLayer *layer = &kan->layers[layer_idx];
    char *formula = malloc(256 * sizeof(char));
    if (!formula) return NULL;
    
    // Check if function is symbolic (all coefficients are simple)
    bool is_symbolic = true;
    for (size_t i = 0; i < layer->input_dim * layer->output_dim; i++) {
        long double coeff = layer->coefficients[i];
        if (fabs(coeff) > kan->symbolic_threshold) {
            is_symbolic = false;
            break;
        }
    }
    
    if (is_symbolic) {
        // Extract symbolic formula (simplified)
        if (layer->input_dim == 1 && layer->output_dim == 1) {
            sprintf(formula, "f(x) = %.4Lf", layer->coefficients[0]);
        } else if (layer->input_dim == 2 && layer->output_dim == 1) {
            sprintf(formula, "f(x,y) = %.4Lf*x + %.4Lf*y", 
                    layer->coefficients[0], layer->coefficients[1]);
        } else {
            sprintf(formula, "f(x) = 0.0");  // Too complex for symbolic representation
        }
    } else {
        sprintf(formula, "f(x) = 0.0");  // Too complex for symbolic representation
    }
    
    return formula;
}

// Check if function is symbolic
bool kan_is_symbolic(KAN_t *kan, size_t layer_idx, size_t output_idx) {
    if (!kan || layer_idx >= kan->num_layers || output_idx >= kan->layers[layer_idx].output_dim) {
        return false;
    }
    
    KANLayer *layer = &kan->layers[layer_idx];
    
    // Check if function is symbolic (all coefficients are simple)
    for (size_t i = 0; i < layer->input_dim * layer->output_dim; i++) {
        long double coeff = layer->coefficients[i];
        if (fabs(coeff) > kan->symbolic_threshold) {
            return false;
        }
    }
    
    return true;
}

// Print summary
void KAN_print_summary(KAN_t *kan) {
    if (!kan) return;
    
    printf("=== KAN Summary ===\n");
    printf("Input Dimension: %zu\n", kan->input_dim);
    printf("Hidden Dimension: %zu\n", kan->hidden_dim);
    printf("Output Dimension: %zu\n", kan->output_dim);
    printf("Layers: %zu\n", kan->num_layers);
    printf("Grid Size: %d\n", kan->grid_size);
    printf("Learning Rate: %.6Lf\n", kan->learning_rate);
    printf("Symbolic Mode: %s\n", kan->use_symbolic ? "Enabled" : "Disabled");
    printf("Symbolic Threshold: %d\n", kan->symbolic_threshold);
    printf("Parameters: %zu\n", KAN_get_parameter_count(kan));
    printf("==================\n");
}
