#include "LLM_NEAT.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FF_DIM 256

// Helper function to create a basic neural network
NEAT_NN_t* create_basic_network(size_t input_dim, size_t output_dim) {
    NEAT_NN_t* nn = (NEAT_NN_t*)malloc(sizeof(NEAT_NN_t));
    if (!nn) return NULL;

    // Initialize network parameters
    nn->input_dim = input_dim;
    nn->output_dim = output_dim;
    nn->num_layers = 2;
    nn->learning_rate = 0.001L;
    nn->regularization_type = L2;

    // Create layer configuration
    size_t layers[] = {input_dim, output_dim, 0}; // Add 0 terminator
    ActivationFunctionType act_funcs[] = {RELU};
    ActivationDerivativeType act_derivs[] = {RELU_DERIVATIVE};

    // Initialize optimizer
    nn->optimizer = Optimizer_init(ADAM, NULL);
    if (!nn->optimizer) {
        free(nn);
        return NULL;
    }

    // Create the underlying neural network
    NN_t* network = NN_init(layers, act_funcs, act_derivs, 
                           MSE, MSE_DERIVATIVE, nn->learning_rate,
                           nn->regularization_type, nn->optimizer);
    
    if (!network) {
        Optimizer_destroy(nn->optimizer);
        free(nn);
        return NULL;
    }

    // Copy weights and biases
    nn->weights = network->weights[0];
    nn->biases = network->biases[0];
    nn->activation_functions = (ActivationFunctionType*)malloc(sizeof(ActivationFunctionType));
    nn->activation_derivatives = (ActivationDerivativeType*)malloc(sizeof(ActivationDerivativeType));
    
    if (!nn->activation_functions || !nn->activation_derivatives) {
        NN_destroy(network);
        Optimizer_destroy(nn->optimizer);
        free(nn);
        return NULL;
    }

    nn->activation_functions[0] = act_funcs[0];
    nn->activation_derivatives[0] = act_derivs[0];

    // Update optimizer's neural network pointer
    nn->optimizer->nn = network;

    return nn;
}

// Create multi-head attention layer
NEAT_MultiHeadAttention* create_multi_head_attention(size_t model_dim, size_t num_heads) {
    NEAT_MultiHeadAttention* mha = (NEAT_MultiHeadAttention*)malloc(sizeof(NEAT_MultiHeadAttention));
    if (!mha) return NULL;
    
    mha->model_dim = model_dim;
    mha->num_heads = num_heads;
    mha->head_dim = model_dim / num_heads;

    // Initialize layer configuration
    size_t layers[] = {model_dim, model_dim, 0}; // Add 0 terminator
    ActivationFunctionType act_funcs[] = {RELU};
    ActivationDerivativeType act_derivs[] = {RELU_DERIVATIVE};
    
    // Initialize optimizers for each projection
    Optimizer_t *q_opt = Optimizer_init(ADAM, NULL);
    Optimizer_t *k_opt = Optimizer_init(ADAM, NULL);
    Optimizer_t *v_opt = Optimizer_init(ADAM, NULL);
    Optimizer_t *o_opt = Optimizer_init(ADAM, NULL);

    if (!q_opt || !k_opt || !v_opt || !o_opt) {
        if (q_opt) Optimizer_destroy(q_opt);
        if (k_opt) Optimizer_destroy(k_opt);
        if (v_opt) Optimizer_destroy(v_opt);
        if (o_opt) Optimizer_destroy(o_opt);
        free(mha);
        return NULL;
    }
    
    // Create projection networks
    NN_t* q_nn = NN_init(layers, act_funcs, act_derivs, MSE, MSE_DERIVATIVE, 0.001L, L2, q_opt);
    NN_t* k_nn = NN_init(layers, act_funcs, act_derivs, MSE, MSE_DERIVATIVE, 0.001L, L2, k_opt);
    NN_t* v_nn = NN_init(layers, act_funcs, act_derivs, MSE, MSE_DERIVATIVE, 0.001L, L2, v_opt);
    NN_t* o_nn = NN_init(layers, act_funcs, act_derivs, MSE, MSE_DERIVATIVE, 0.001L, L2, o_opt);

    if (!q_nn || !k_nn || !v_nn || !o_nn) {
        if (q_nn) free(q_nn);
        if (k_nn) free(k_nn);
        if (v_nn) free(v_nn);
        if (o_nn) free(o_nn);
        Optimizer_destroy(q_opt);
        Optimizer_destroy(k_opt);
        Optimizer_destroy(v_opt);
        Optimizer_destroy(o_opt);
        free(mha);
        return NULL;
    }

    // Cast and assign networks
    mha->Q_proj = (NEAT_NN_t*)q_nn;
    mha->K_proj = (NEAT_NN_t*)k_nn;
    mha->V_proj = (NEAT_NN_t*)v_nn;
    mha->O_proj = (NEAT_NN_t*)o_nn;
    
    // Update optimizer neural network pointers
    q_opt->nn = q_nn;
    k_opt->nn = k_nn;
    v_opt->nn = v_nn;
    o_opt->nn = o_nn;
    
    return mha;
}

// Create feed-forward network
NEAT_FeedForward* create_feed_forward(size_t input_dim, size_t hidden_dim) {
    NEAT_FeedForward* ff = (NEAT_FeedForward*)malloc(sizeof(NEAT_FeedForward));
    if (!ff) return NULL;
    
    ff->input_dim = input_dim;
    ff->hidden_dim = hidden_dim;
    
    // Create layer configuration
    size_t layers[] = {input_dim, hidden_dim, input_dim, 0}; // Add 0 terminator
    ActivationFunctionType act_funcs[] = {RELU, RELU};
    ActivationDerivativeType act_derivs[] = {RELU_DERIVATIVE, RELU_DERIVATIVE};
    
    // Initialize optimizer
    Optimizer_t *optimizer = Optimizer_init(ADAM, NULL);
    if (!optimizer) {
        free(ff);
        return NULL;
    }
    
    // Create feed-forward network with optimizer
    NN_t* nn = NN_init(layers, act_funcs, act_derivs, MSE, MSE_DERIVATIVE, 0.001L, L2, optimizer);
    if (!nn) {
        Optimizer_destroy(optimizer);
        free(ff);
        return NULL;
    }
    
    // Update optimizer's neural network pointer and convert to NEAT_NN
    optimizer->nn = nn;
    ff->network = (NEAT_NN_t*)nn;  // Cast to NEAT_NN_t
    
    return ff;
}

// Create layer normalization
NEAT_LayerNorm* create_layer_norm(size_t dim, long double epsilon) {
    NEAT_LayerNorm *ln = (NEAT_LayerNorm*)malloc(sizeof(NEAT_LayerNorm));
    if (!ln) return NULL;
    
    ln->dim = dim;
    ln->epsilon = epsilon;
    ln->gamma = (long double*)calloc(dim, sizeof(long double));
    ln->beta = (long double*)calloc(dim, sizeof(long double));
    
    if (!ln->gamma || !ln->beta) {
        free_layer_norm(ln);
        return NULL;
    }
    
    // Initialize gamma to 1 and beta to 0
    for (size_t i = 0; i < dim; i++) {
        ln->gamma[i] = 1.0L;
        ln->beta[i] = 0.0L;
    }
    
    return ln;
}

// Create NEAT transformer
NEAT_Transformer* create_neat_transformer(size_t input_dim, size_t model_dim, size_t num_heads) {
    NEAT_Transformer* transformer = (NEAT_Transformer*)malloc(sizeof(NEAT_Transformer));
    if (!transformer) return NULL;

    transformer->input_dim = input_dim;
    transformer->model_dim = model_dim;
    transformer->num_heads = num_heads;

    // Create components
    transformer->self_attention = create_multi_head_attention(model_dim, num_heads);
    transformer->feed_forward = create_feed_forward(model_dim, FF_DIM);
    transformer->layer_norm1 = create_layer_norm(model_dim, 1e-5L);
    transformer->layer_norm2 = create_layer_norm(model_dim, 1e-5L);

    if (!transformer->self_attention || !transformer->feed_forward || 
        !transformer->layer_norm1 || !transformer->layer_norm2) {
        free_neat_transformer(transformer);
        return NULL;
    }

    return transformer;
}

NEAT_Transformer* neat_crossover(NEAT_Transformer* parent1, NEAT_Transformer* parent2) {
    if (!parent1 || !parent2) return NULL;

    // Create a new transformer with parent1's dimensions
    NEAT_Transformer* child = create_neat_transformer(
        parent1->input_dim, parent1->model_dim, parent1->num_heads
    );
    if (!child) return NULL;

    // Crossover attention mechanisms
    if (child->self_attention && parent1->self_attention && parent2->self_attention) {
        // Crossover weights from Q, K, V projections
        NN_t* child_q = (NN_t*)child->self_attention->Q_proj;
        NN_t* parent1_q = (NN_t*)parent1->self_attention->Q_proj;
        NN_t* parent2_q = (NN_t*)parent2->self_attention->Q_proj;
        
        for (size_t i = 0; i < child->model_dim; i++) {
            if (rand() % 2) {
                for (size_t j = 0; j < child_q->layers[0]; j++) {
                    child_q->weights[0][j] = parent1_q->weights[0][j];
                    ((NN_t*)child->self_attention->K_proj)->weights[0][j] = 
                        ((NN_t*)parent1->self_attention->K_proj)->weights[0][j];
                    ((NN_t*)child->self_attention->V_proj)->weights[0][j] = 
                        ((NN_t*)parent1->self_attention->V_proj)->weights[0][j];
                }
            } else {
                for (size_t j = 0; j < child_q->layers[0]; j++) {
                    child_q->weights[0][j] = parent2_q->weights[0][j];
                    ((NN_t*)child->self_attention->K_proj)->weights[0][j] = 
                        ((NN_t*)parent2->self_attention->K_proj)->weights[0][j];
                    ((NN_t*)child->self_attention->V_proj)->weights[0][j] = 
                        ((NN_t*)parent2->self_attention->V_proj)->weights[0][j];
                }
            }
        }
    }

    // Crossover feed-forward network
    if (child->feed_forward && parent1->feed_forward && parent2->feed_forward) {
        NN_t* child_ff = (NN_t*)child->feed_forward->network;
        NN_t* parent1_ff = (NN_t*)parent1->feed_forward->network;
        NN_t* parent2_ff = (NN_t*)parent2->feed_forward->network;

        // Perform crossover on network weights
        for (size_t i = 0; i < child->feed_forward->hidden_dim; i++) {
            if (rand() % 2) {
                for (size_t j = 0; j < child_ff->layers[0]; j++) {
                    child_ff->weights[0][j] = parent1_ff->weights[0][j];
                    child_ff->weights[1][j] = parent1_ff->weights[1][j];
                }
            } else {
                for (size_t j = 0; j < child_ff->layers[0]; j++) {
                    child_ff->weights[0][j] = parent2_ff->weights[0][j];
                    child_ff->weights[1][j] = parent2_ff->weights[1][j];
                }
            }
        }
    }

    // Crossover layer normalization parameters
    if (child->layer_norm1 && parent1->layer_norm1 && parent2->layer_norm1) {
        for (size_t i = 0; i < child->model_dim; i++) {
            if (rand() % 2) {
                child->layer_norm1->gamma[i] = parent1->layer_norm1->gamma[i];
                child->layer_norm1->beta[i] = parent1->layer_norm1->beta[i];
            } else {
                child->layer_norm1->gamma[i] = parent2->layer_norm1->gamma[i];
                child->layer_norm1->beta[i] = parent2->layer_norm1->beta[i];
            }
        }
    }

    return child;
}

void neat_mutate(NEAT_Transformer* transformer, long double mutation_rate) {
    if (!transformer) return;

    // Mutate self-attention weights and learning rates
    if (transformer->self_attention) {
        // Mutate learning rates with small probability
        if ((long double)rand() / RAND_MAX < mutation_rate * 0.1) {
            NN_t* q_nn = (NN_t*)transformer->self_attention->Q_proj;
            q_nn->learningRate *= 1.0L + ((long double)rand() / RAND_MAX - 0.5L) * 0.2L;  // ±10% change
        }

        // Mutate weights
        for (size_t i = 0; i < transformer->model_dim; i++) {
            if ((long double)rand() / RAND_MAX < mutation_rate) {
                ((NN_t*)transformer->self_attention->Q_proj)->weights[0][i] += 
                    ((long double)rand() / RAND_MAX - 0.5L) * 0.1L;
                ((NN_t*)transformer->self_attention->K_proj)->weights[0][i] += 
                    ((long double)rand() / RAND_MAX - 0.5L) * 0.1L;
                ((NN_t*)transformer->self_attention->V_proj)->weights[0][i] += 
                    ((long double)rand() / RAND_MAX - 0.5L) * 0.1L;
            }
        }
    }

    // Mutate feed-forward network
    if (transformer->feed_forward) {
        // Mutate learning rate
        if ((long double)rand() / RAND_MAX < mutation_rate * 0.1) {
            NN_t* ff_nn = (NN_t*)transformer->feed_forward->network;
            ff_nn->learningRate *= 1.0L + ((long double)rand() / RAND_MAX - 0.5L) * 0.2L;
        }

        // Mutate weights
        for (size_t i = 0; i < transformer->feed_forward->hidden_dim; i++) {
            if ((long double)rand() / RAND_MAX < mutation_rate) {
                ((NN_t*)transformer->feed_forward->network)->weights[0][i] += 
                    ((long double)rand() / RAND_MAX - 0.5L) * 0.1L;
                ((NN_t*)transformer->feed_forward->network)->weights[1][i] += 
                    ((long double)rand() / RAND_MAX - 0.5L) * 0.1L;
            }
        }
    }

    // Mutate layer normalization parameters
    if (transformer->layer_norm1) {
        for (size_t i = 0; i < transformer->model_dim; i++) {
            if ((long double)rand() / RAND_MAX < mutation_rate) {
                transformer->layer_norm1->gamma[i] += 
                    ((long double)rand() / RAND_MAX - 0.5L) * 0.1L;
                transformer->layer_norm1->beta[i] += 
                    ((long double)rand() / RAND_MAX - 0.5L) * 0.1L;
            }
        }
    }

    // Ensure learning rates stay within reasonable bounds
    const long double MIN_LR = 1e-6L;
    const long double MAX_LR = 1e-2L;
    
    if (transformer->self_attention) {
        NN_t* q_nn = (NN_t*)transformer->self_attention->Q_proj;
        q_nn->learningRate = fmaxl(MIN_LR, fminl(MAX_LR, q_nn->learningRate));
    }
    
    if (transformer->feed_forward) {
        NN_t* ff_nn = (NN_t*)transformer->feed_forward->network;
        ff_nn->learningRate = fmaxl(MIN_LR, fminl(MAX_LR, ff_nn->learningRate));
    }
}

// Topology history structure
typedef struct TopologyHistory {
    size_t num_attention_heads;
    size_t hidden_dim;
    long double best_fitness;
    size_t generations_without_improvement;
} TopologyHistory;

void add_attention_head(NEAT_Transformer* transformer) {
    if (!transformer || !transformer->self_attention) return;
    
    transformer->num_heads++;
    transformer->self_attention->num_heads = transformer->num_heads;
    transformer->self_attention->head_dim = transformer->model_dim / transformer->num_heads;
}

void remove_attention_head(NEAT_Transformer* transformer) {
    if (!transformer || !transformer->self_attention || transformer->num_heads <= 1) return;
    
    transformer->num_heads--;
    transformer->self_attention->num_heads = transformer->num_heads;
    transformer->self_attention->head_dim = transformer->model_dim / transformer->num_heads;
}

void grow_feedforward(NEAT_Transformer* transformer) {
    if (!transformer || !transformer->feed_forward) return;
    
    size_t new_hidden_dim = transformer->feed_forward->hidden_dim * 2;
    NEAT_FeedForward* new_ff = create_feed_forward(transformer->feed_forward->input_dim, new_hidden_dim);
    if (!new_ff) return;
    
    // Copy existing weights
    NN_t* old_nn = (NN_t*)transformer->feed_forward->network;
    NN_t* new_nn = (NN_t*)new_ff->network;
    
    for (size_t i = 0; i < transformer->feed_forward->hidden_dim; i++) {
        new_nn->weights[0][i] = old_nn->weights[0][i];
        new_nn->weights[1][i] = old_nn->weights[1][i];
        new_nn->biases[0][i] = old_nn->biases[0][i];
        new_nn->biases[1][i] = old_nn->biases[1][i];
    }
    
    // Initialize new weights randomly
    for (size_t i = transformer->feed_forward->hidden_dim; i < new_hidden_dim; i++) {
        new_nn->weights[0][i] = ((long double)rand() / RAND_MAX) * 2.0L - 1.0L;
        new_nn->weights[1][i] = ((long double)rand() / RAND_MAX) * 2.0L - 1.0L;
        new_nn->biases[0][i] = ((long double)rand() / RAND_MAX) * 2.0L - 1.0L;
        new_nn->biases[1][i] = ((long double)rand() / RAND_MAX) * 2.0L - 1.0L;
    }
    
    // Free old feed-forward and update
    free_feed_forward(transformer->feed_forward);
    transformer->feed_forward = new_ff;
}

void shrink_feedforward(NEAT_Transformer* transformer) {
    if (!transformer || !transformer->feed_forward || transformer->feed_forward->hidden_dim <= 1) return;
    
    size_t new_hidden_dim = transformer->feed_forward->hidden_dim / 2;
    NEAT_FeedForward* new_ff = create_feed_forward(transformer->feed_forward->input_dim, new_hidden_dim);
    if (!new_ff) return;
    
    // Copy subset of weights
    NN_t* old_nn = (NN_t*)transformer->feed_forward->network;
    NN_t* new_nn = (NN_t*)new_ff->network;
    
    for (size_t i = 0; i < new_hidden_dim; i++) {
        new_nn->weights[0][i] = old_nn->weights[0][i];
        new_nn->weights[1][i] = old_nn->weights[1][i];
        new_nn->biases[0][i] = old_nn->biases[0][i];
        new_nn->biases[1][i] = old_nn->biases[1][i];
    }
    
    // Free old feed-forward and update
    free_feed_forward(transformer->feed_forward);
    transformer->feed_forward = new_ff;
}

void update_topology(TopologyHistory* history, long double performance) {
    if (!history) return;
    
    if (performance > history->best_fitness) {
        history->best_fitness = performance;
        history->generations_without_improvement = 0;
    } else {
        history->generations_without_improvement++;
    }
}

void optimize_architecture(NEAT_Transformer* transformer, long double* input, long double* target) {
    if (!transformer) return;
    
    // Initialize topology history if not exists
    TopologyHistory history = {
        .num_attention_heads = transformer->num_heads,
        .hidden_dim = transformer->feed_forward->hidden_dim,
        .best_fitness = -INFINITY,
        .generations_without_improvement = 0
    };
    
    const size_t MAX_GENERATIONS_WITHOUT_IMPROVEMENT = 5;
    const size_t NUM_VARIANTS = 4;
    
    long double initial_fitness = calculate_fitness(transformer, input, target);
    history.best_fitness = initial_fitness;
    
    // Try different architectural variations
    for (size_t gen = 0; gen < MAX_GENERATIONS_WITHOUT_IMPROVEMENT; gen++) {
        long double prev_fitness = calculate_fitness(transformer, input, target);
        
        // Randomly choose between different architectural modifications
        switch (rand() % 4) {
            case 0:
                add_attention_head(transformer);
                if (calculate_fitness(transformer, input, target) <= prev_fitness) {
                    remove_attention_head(transformer);  // Revert if no improvement
                }
                break;
                
            case 1:
                grow_feedforward(transformer);
                if (calculate_fitness(transformer, input, target) <= prev_fitness) {
                    shrink_feedforward(transformer);  // Revert if no improvement
                }
                break;
                
            case 2:
                // Try different variants of the current architecture
                NEAT_Transformer* best_variant = NULL;
                long double best_variant_fitness = prev_fitness;
                
                for (size_t i = 0; i < NUM_VARIANTS; i++) {
                    NEAT_Transformer* variant = neat_crossover(transformer, transformer);
                    neat_mutate(variant, 0.1L);
                    
                    long double variant_fitness = calculate_fitness(variant, input, target);
                    if (variant_fitness > best_variant_fitness) {
                        if (best_variant) free_neat_transformer(best_variant);
                        best_variant = variant;
                        best_variant_fitness = variant_fitness;
                    } else {
                        free_neat_transformer(variant);
                    }
                }
                
                if (best_variant) {
                    free_neat_transformer(transformer);
                    transformer = best_variant;
                }
                break;
                
            case 3:
                // Try reducing complexity if performance is stagnant
                if (history.generations_without_improvement > MAX_GENERATIONS_WITHOUT_IMPROVEMENT / 2) {
                    if (transformer->num_heads > 1) remove_attention_head(transformer);
                    if (transformer->feed_forward->hidden_dim > transformer->model_dim) {
                        shrink_feedforward(transformer);
                    }
                }
                break;
        }
        
        update_topology(&history, calculate_fitness(transformer, input, target));
        
        if (history.generations_without_improvement >= MAX_GENERATIONS_WITHOUT_IMPROVEMENT) {
            break;
        }
    }
}

void calculate_fitness(NEAT_Transformer* transformer, long double* input, long double* target) {
    if (!transformer || !input || !target) return;
    
    // Forward pass
    long double* output = llm_neat_forward(transformer, input);
    if (!output) return;
    
    // Calculate MSE loss
    long double loss = 0.0L;
    size_t output_size = transformer->input_dim;  // Using input_dim as output_dim for now
    for (size_t i = 0; i < output_size; i++) {
        long double diff = output[i] - target[i];
        loss += diff * diff;
    }
    loss /= output_size;
    
    transformer->fitness = loss;
    free(output);
}

void NEAT_NN_copy(NEAT_NN_t* source, NEAT_NN_t* target) {
    if (!source || !target) return;
    
    // Copy weights
    size_t weight_size = source->input_dim * source->output_dim;
    memcpy(target->weights, source->weights, weight_size * sizeof(long double));
    
    // Copy biases
    size_t bias_size = source->output_dim;
    memcpy(target->biases, source->biases, bias_size * sizeof(long double));
}

void transformer_free(NEAT_Transformer* transformer) {
    if (transformer) {
        if (transformer->self_attention) free_multi_head_attention(transformer->self_attention);
        if (transformer->layer_norm1) free_layer_norm(transformer->layer_norm1);
        if (transformer->feed_forward) free_feed_forward(transformer->feed_forward);
        if (transformer->layer_norm2) free_layer_norm(transformer->layer_norm2);
        free(transformer);
    }
}

long double* llm_neat_forward(NEAT_Transformer* transformer, long double* input) {
    if (!transformer || !input) return NULL;

    size_t output_size = transformer->model_dim;
    long double* output = (long double*)malloc(output_size * sizeof(long double));
    if (!output) return NULL;

    // Forward pass through attention
    if (transformer->self_attention) {
        // Implement attention forward pass
        // For now, just copy input to output
        memcpy(output, input, output_size * sizeof(long double));
    }

    // Apply layer norm 1
    if (transformer->layer_norm1) {
        // Apply layer normalization
        for (size_t i = 0; i < output_size; i++) {
            output[i] = output[i] * transformer->layer_norm1->gamma[i] + transformer->layer_norm1->beta[i];
        }
    }

    // Forward through feed-forward
    if (transformer->feed_forward && transformer->feed_forward->network) {
        // Use the feed-forward network
        long double* ff_output = (long double*)malloc(output_size * sizeof(long double));
        if (ff_output) {
            // Forward pass through feed-forward network
            NN_forward(transformer->feed_forward->network, output);
            memcpy(ff_output, output, output_size * sizeof(long double));
            free(output);
            output = ff_output;
        }
    }

    // Apply layer norm 2
    if (transformer->layer_norm2) {
        // Apply layer normalization
        for (size_t i = 0; i < output_size; i++) {
            output[i] = output[i] * transformer->layer_norm2->gamma[i] + transformer->layer_norm2->beta[i];
        }
    }

    return output;
}

void llm_neat_backprop(NEAT_Transformer* transformer, long double* input, 
                      long double* grad_output, long double* grad_input) {
    if (!transformer || !input || !grad_output || !grad_input) return;
    
    // TODO: Implement backpropagation
    // This is a placeholder that just copies grad_output to grad_input
    memcpy(grad_input, grad_output, transformer->input_dim * sizeof(long double));
}

void llm_neat_update(NEAT_Transformer* transformer, long double learning_rate) {
    if (!transformer) return;
    
    // TODO: Implement parameter updates
}

void free_multi_head_attention(NEAT_MultiHeadAttention* mha) {
    if (mha) {
        if (mha->Q_proj) NN_free(mha->Q_proj);
        if (mha->K_proj) NN_free(mha->K_proj);
        if (mha->V_proj) NN_free(mha->V_proj);
        if (mha->O_proj) NN_free(mha->O_proj);
        free(mha);
    }
}

void free_feed_forward(NEAT_FeedForward* ff) {
    if (ff) {
        NN_free(ff->network);
        free(ff);
    }
}

void free_layer_norm(NEAT_LayerNorm* ln) {
    if (ln) {
        free(ln->gamma);
        free(ln->beta);
        free(ln);
    }
}

void NN_free(NEAT_NN_t* nn) {
    if (nn) {
        free(nn->weights);
        free(nn->biases);
        free(nn);
    }
}
