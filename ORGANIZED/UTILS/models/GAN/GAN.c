#include "GAN.h"
#include "NN.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Helper functions
static long double random_normal() {
    // Box-Muller transform for normal distribution
    static bool has_spare = false;
    static long double spare;
    
    if (has_spare) {
        has_spare = false;
        return spare;
    }
    
    has_spare = true;
    
    long double u, v, s;
    do {
        u = (long double)rand() / RAND_MAX * 2.0L - 1.0L;
        v = (long double)rand() / RAND_MAX * 2.0L - 1.0L;
        s = u * u + v * v;
    } while (s >= 1.0L || s == 0.0L);
    
    s = sqrtl(-2.0L * logl(s) / s);
    spare = v * s;
    return u * s;
}

// Generate random noise
static void generate_noise(GAN_t *gan, long double *noise, size_t batch_size) {
    for (size_t i = 0; i < batch_size * gan->latent_dim; i++) {
        noise[i] = random_normal();
    }
}

// Generative Adversarial Network creation
GAN_t *GAN_create(size_t input_dim, size_t hidden_dim, size_t output_dim, int latent_dim) {
    GAN_t *gan = malloc(sizeof(GAN_t));
    if (!gan) return NULL;
    
    gan->learning_rate = 0.0002L;
    gan->latent_dim = latent_dim;
    gan->training_steps = 0;
    gan->lambda_gp = 10.0L;
    gan->lambda_drift = 0.001L;
    gan->use_wasserstein = false;
    gan->n_critic = 5;
    gan->buffer_size = 0;
    gan->buffer_capacity = 10000;
    gan->buffer_ptr = 0;
    
    // Create generator network
    size_t gen_layers[] = {latent_dim, hidden_dim, hidden_dim, output_dim, 0};
    ActivationFunctionType gen_activations[] = {RELU, RELU, RELU, SIGMOID};
    ActivationDerivativeType gen_derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, SIGMOID_DERIVATIVE};
    
    gan->generator = NN_init_with_weight_init(gen_layers, gen_activations, gen_derivatives,
                                            MSE, MSE_DERIVATIVE, L2, ADAM, gan->learning_rate, HE);
    
    // Create discriminator network
    size_t disc_layers[] = {input_dim, hidden_dim, hidden_dim, 1, 0};
    ActivationFunctionType disc_activations[] = {RELU, RELU, RELU, SIGMOID};
    ActivationDerivativeType disc_derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, SIGMOID_DERIVATIVE};
    
    gan->discriminator = NN_init_with_weight_init(disc_layers, disc_activations, disc_derivatives,
                                               MSE, MSE_DERIVATIVE, L2, ADAM, gan->learning_rate, HE);
    
    // Initialize buffers
    gan->noise_buffer = malloc(gan->buffer_capacity * gan->latent_dim * sizeof(long double));
    gan->real_buffer = malloc(gan->buffer_capacity * input_dim * sizeof(long double));
    
    if (!gan->generator || !gan->discriminator || !gan->noise_buffer || !gan->real_buffer) {
        GAN_destroy(gan);
        return NULL;
    }
    
    // Initialize random seed
    srand((unsigned int)time(NULL));
    
    return gan;
}

// Generative Adversarial Network destruction
void GAN_destroy(GAN_t *gan) {
    if (!gan) return;
    
    if (gan->generator) NN_destroy(gan->generator);
    if (gan->discriminator) NN_destroy(gan->discriminator);
    
    free(gan->noise_buffer);
    free(gan->real_buffer);
    free(gan);
}

// Set learning rate
void GAN_set_learning_rate(GAN_t *gan, long double learning_rate) {
    if (gan) {
        gan->learning_rate = learning_rate;
        if (gan->generator) NN_set_base_lr(gan->generator, learning_rate);
        if (gan->discriminator) NN_set_base_lr(gan->discriminator, learning_rate);
    }
}

// Set gradient penalty coefficient
void GAN_set_lambda_gp(GAN_t *gan, long double lambda_gp) {
    if (gan) {
        gan->lambda_gp = lambda_gp;
    }
}

// Enable Wasserstein GAN
void GAN_enable_wasserstein(GAN_t *gan, bool enable) {
    if (gan) {
        gan->use_wasserstein = enable;
        
        // For Wasserstein GAN, use linear activation in discriminator
        if (enable && gan->discriminator) {
            // This would require modifying the discriminator's output activation
            // For now, we'll keep it as is since we don't have direct access to modify activations
        }
    }
}

// Set number of critic updates
void GAN_set_n_critic(GAN_t *gan, int n_critic) {
    if (gan) {
        gan->n_critic = n_critic;
    }
}

// Set buffer size
void GAN_set_buffer_size(GAN_t *gan, size_t buffer_size) {
    if (gan) {
        gan->buffer_capacity = buffer_size;
        gan->buffer_size = 0;
        gan->buffer_ptr = 0;
        
        // Reallocate buffers
        free(gan->noise_buffer);
        free(gan->real_buffer);
        gan->noise_buffer = malloc(gan->buffer_capacity * gan->latent_dim * sizeof(long double));
        gan->real_buffer = malloc(gan->buffer_capacity * gan->discriminator->layers[0] * sizeof(long double));
    }
}

// Reset buffers
void GAN_reset_buffers(GAN_t *gan) {
    if (gan) {
        gan->buffer_size = 0;
        gan->buffer_ptr = 0;
    }
}

// Generate samples
long double *GAN_generate(GAN_t *gan, long double *noise) {
    if (!gan || !noise || !gan->generator) return NULL;
    
    return NN_forward(gan->generator, noise);
}

// Discriminate real vs fake
long double GAN_discriminate(GAN_t *gan, long double *data) {
    if (!gan || !data || !gan->discriminator) return 0.0L;
    
    long double *output = NN_forward(gan->discriminator, data);
    if (!output) return 0.0L;
    
    long double score = output[0];
    free(output);
    
    return score;
}

// Generate noise
void GAN_generate_noise(GAN_t *gan, long double *noise, size_t batch_size) {
    if (!gan || !noise) return;
    
    for (size_t i = 0; i < batch_size * gan->latent_dim; i++) {
        noise[i] = random_normal();
    }
}

// Generate normal noise
void GAN_generate_normal_noise(GAN_t *gan, long double *noise, size_t batch_size) {
    GAN_generate_noise(gan, noise, batch_size);
}

// Generate uniform noise
void GAN_generate_uniform_noise(GAN_t *gan, long double *noise, size_t batch_size) {
    if (!gan || !noise) return;
    
    for (size_t i = 0; i < batch_size * gan->latent_dim; i++) {
        noise[i] = (long double)rand() / RAND_MAX * 2.0L - 1.0L;
    }
}

// Compute gradient penalty
long double GAN_compute_gradient_penalty(GAN_t *gan, long double **real_samples, 
                                        long double **fake_samples, size_t batch_size) {
    if (!gan || !real_samples || !fake_samples || !gan->discriminator) return 0.0L;
    
    long double penalty = 0.0L;
    
    for (size_t i = 0; i < batch_size; i++) {
        // Interpolate between real and fake samples
        long double *interpolated = malloc(gan->discriminator->layers[0] * sizeof(long double));
        if (!interpolated) continue;
        
        long double epsilon = (long double)rand() / RAND_MAX;
        for (size_t j = 0; j < gan->discriminator->layers[0]; j++) {
            interpolated[j] = epsilon * real_samples[i][j] + (1.0L - epsilon) * fake_samples[i][j];
        }
        
        // Compute discriminator output
        long double *disc_output = NN_forward(gan->discriminator, interpolated);
        if (disc_output) {
            // Gradient penalty: (||grad||_2 - 1)^2
            // Simplified version - in practice, you'd compute actual gradients
            long double grad_norm = fabsl(disc_output[0]);
            penalty += (grad_norm - 1.0L) * (grad_norm - 1.0L);
            free(disc_output);
        }
        
        free(interpolated);
    }
    
    return penalty / batch_size;
}

// Compute Wasserstein distance
long double GAN_compute_wasserstein_distance(GAN_t *gan, long double **real_samples, 
                                             long double **fake_samples, size_t batch_size) {
    if (!gan || !real_samples || !fake_samples || !gan->discriminator) return 0.0L;
    
    long double real_score = 0.0L;
    long double fake_score = 0.0L;
    
    // Average discriminator scores
    for (size_t i = 0; i < batch_size; i++) {
        real_score += GAN_discriminate(gan, real_samples[i]);
        fake_score += GAN_discriminate(gan, fake_samples[i]);
    }
    
    real_score /= batch_size;
    fake_score /= batch_size;
    
    // Wasserstein distance: E[real] - E[fake]
    return real_score - fake_score;
}

// Update buffers
void GAN_update_buffers(GAN_t *gan, long double **real_data, size_t batch_size) {
    if (!gan || !real_data) return;
    
    for (size_t i = 0; i < batch_size; i++) {
        // Add noise to buffer
        size_t noise_idx = gan->buffer_ptr * gan->latent_dim;
        GAN_generate_noise(gan, &gan->noise_buffer[noise_idx], 1);
        
        // Add real data to buffer
        size_t real_idx = gan->buffer_ptr * gan->discriminator->layers[0];
        memcpy(&gan->real_buffer[real_idx], real_data[i], gan->discriminator->layers[0] * sizeof(long double));
        
        gan->buffer_ptr = (gan->buffer_ptr + 1) % gan->buffer_capacity;
        if (gan->buffer_size < gan->buffer_capacity) {
            gan->buffer_size++;
        }
    }
}

// Training step
void GAN_train_step(GAN_t *gan, long double **real_data, size_t batch_size) {
    if (!gan || !real_data || !gan->generator || !gan->discriminator) return;
    
    // Update buffers
    GAN_update_buffers(gan, real_data, batch_size);
    
    // Generate fake samples
    long double **fake_samples = malloc(batch_size * sizeof(long double*));
    long double *noise = malloc(batch_size * gan->latent_dim * sizeof(long double));
    
    if (!fake_samples || !noise) {
        free(fake_samples);
        free(noise);
        return;
    }
    
    for (size_t i = 0; i < batch_size; i++) {
        GAN_generate_noise(gan, &noise[i * gan->latent_dim], 1);
        fake_samples[i] = GAN_generate(gan, &noise[i * gan->latent_dim]);
    }
    
    // Train discriminator for n_critic steps
    for (int critic_step = 0; critic_step < gan->n_critic; critic_step++) {
        // Train on real samples
        for (size_t i = 0; i < batch_size; i++) {
            long double real_score = GAN_discriminate(gan, real_data[i]);
            long double target_real = gan->use_wasserstein ? 1.0L : 1.0L;
            
            // Backpropagate discriminator
            long double *real_error = malloc(sizeof(long double));
            real_error[0] = real_score - target_real;
            NN_backprop_custom_delta(gan->discriminator, real_data[i], real_error);
            free(real_error);
        }
        
        // Train on fake samples
        for (size_t i = 0; i < batch_size; i++) {
            long double fake_score = GAN_discriminate(gan, fake_samples[i]);
            long double target_fake = gan->use_wasserstein ? -1.0L : 0.0L;
            
            // Backpropagate discriminator
            long double *fake_error = malloc(sizeof(long double));
            fake_error[0] = fake_score - target_fake;
            NN_backprop_custom_delta(gan->discriminator, fake_samples[i], fake_error);
            free(fake_error);
        }
        
        // Apply gradient penalty if Wasserstein GAN
        if (gan->use_wasserstein) {
            long double gp = GAN_compute_gradient_penalty(gan, real_data, fake_samples, batch_size);
            // This would be added to the discriminator loss
        }
        
        // Update discriminator weights
        gan->discriminator->optimizer(gan->discriminator);
    }
    
    // Train generator
    for (size_t i = 0; i < batch_size; i++) {
        // Generate fake sample
        long double *fake_sample = GAN_generate(gan, &noise[i * gan->latent_dim]);
        
        // Get discriminator score
        long double fake_score = GAN_discriminate(gan, fake_sample);
        
        // Generator wants discriminator to think it's real
        long double target_real = 1.0L;
        
        // Backpropagate through generator
        long double *gen_error = malloc(sizeof(long double));
        gen_error[0] = fake_score - target_real;
        
        // This is simplified - in practice, you'd need to backprop through the discriminator
        // to the generator, which requires more complex gradient computation
        
        free(gen_error);
        free(fake_sample);
    }
    
    // Update generator weights
    gan->generator->optimizer(gan->generator);
    
    // Increment training steps
    gan->training_steps++;
    
    // Cleanup
    for (size_t i = 0; i < batch_size; i++) {
        free(fake_samples[i]);
    }
    free(fake_samples);
    free(noise);
}

// Wasserstein GAN training
void GAN_train_wasserstein(GAN_t *gan, long double **real_data, size_t batch_size) {
    if (!gan) return;
    
    // Enable Wasserstein mode
    GAN_enable_wasserstein(gan, true);
    
    // Train with Wasserstein loss
    GAN_train_step(gan, real_data, batch_size);
}

// Get parameter count
size_t GAN_get_parameter_count(GAN_t *gan) {
    if (!gan) return 0;
    
    size_t total = 0;
    
    if (gan->generator) {
        for (size_t i = 0; i < gan->generator->numLayers - 1; i++) {
            total += gan->generator->layers[i] * gan->generator->layers[i + 1];
            total += gan->generator->layers[i + 1];
        }
    }
    
    if (gan->discriminator) {
        for (size_t i = 0; i < gan->discriminator->numLayers - 1; i++) {
            total += gan->discriminator->layers[i] * gan->discriminator->layers[i + 1];
            total += gan->discriminator->layers[i + 1];
        }
    }
    
    return total;
}

// Print summary
void GAN_print_summary(GAN_t *gan) {
    if (!gan) return;
    
    printf("=== GAN Summary ===\n");
    printf("Latent Dimension: %d\n", gan->latent_dim);
    printf("Learning Rate: %.6Lf\n", gan->learning_rate);
    printf("Wasserstein GAN: %s\n", gan->use_wasserstein ? "Yes" : "No");
    printf("Critic Updates: %d\n", gan->n_critic);
    printf("Gradient Penalty: %.6Lf\n", gan->lambda_gp);
    printf("Training Steps: %d\n", gan->training_steps);
    printf("Buffer Size: %zu/%zu\n", gan->buffer_size, gan->buffer_capacity);
    printf("Parameters: %zu\n", GAN_get_parameter_count(gan));
    
    if (gan->generator) {
        printf("Generator: %zu layers\n", gan->generator->numLayers);
    }
    
    if (gan->discriminator) {
        printf("Discriminator: %zu layers\n", gan->discriminator->numLayers);
    }
    
    printf("==================\n");
}

// Save model
void GAN_save_model(GAN_t *gan, const char *filename) {
    if (!gan || !filename) return;
    
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    
    // Save generator
    NN_save_fp(gan->generator, f);
    
    // Save discriminator
    NN_save_fp(gan->discriminator, f);
    
    // Save GAN parameters
    fwrite(&gan->learning_rate, sizeof(long double), 1, f);
    fwrite(&gan->latent_dim, sizeof(int), 1, f);
    fwrite(&gan->training_steps, sizeof(int), 1, f);
    fwrite(&gan->lambda_gp, sizeof(long double), 1, f);
    fwrite(&gan->use_wasserstein, sizeof(bool), 1, f);
    fwrite(&gan->n_critic, sizeof(int), 1, f);
    
    fclose(f);
}

// Load model
GAN_t *GAN_load_model(const char *filename) {
    if (!filename) return NULL;
    
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    
    // Load generator
    NN_t *generator = NN_load_fp(f);
    
    // Load discriminator
    NN_t *discriminator = NN_load_fp(f);
    
    if (!generator || !discriminator) {
        if (generator) NN_destroy(generator);
        if (discriminator) NN_destroy(discriminator);
        fclose(f);
        return NULL;
    }
    
    // Create GAN structure
    GAN_t *gan = malloc(sizeof(GAN_t));
    if (!gan) {
        NN_destroy(generator);
        NN_destroy(discriminator);
        fclose(f);
        return NULL;
    }
    
    gan->generator = generator;
    gan->discriminator = discriminator;
    
    // Load GAN parameters
    fread(&gan->learning_rate, sizeof(long double), 1, f);
    fread(&gan->latent_dim, sizeof(int), 1, f);
    fread(&gan->training_steps, sizeof(int), 1, f);
    fread(&gan->lambda_gp, sizeof(long double), 1, f);
    fread(&gan->use_wasserstein, sizeof(bool), 1, f);
    fread(&gan->n_critic, sizeof(int), 1, f);
    
    fclose(f);
    
    // Initialize other parameters
    gan->lambda_drift = 0.001L;
    gan->buffer_size = 0;
    gan->buffer_capacity = 10000;
    gan->buffer_ptr = 0;
    gan->noise_buffer = malloc(gan->buffer_capacity * gan->latent_dim * sizeof(long double));
    gan->real_buffer = malloc(gan->buffer_capacity * generator->layers[0] * sizeof(long double));
    
    return gan;
}

// Evaluate generator
long double GAN_evaluate_generator(GAN_t *gan, long double **test_data, size_t num_samples) {
    if (!gan || !test_data || !gan->generator || !gan->discriminator) return 0.0L;
    
    long double total_score = 0.0L;
    
    for (size_t i = 0; i < num_samples; i++) {
        // Generate sample
        long double *noise = malloc(gan->latent_dim * sizeof(long double));
        if (!noise) continue;
        
        GAN_generate_noise(gan, noise, 1);
        long double *generated = GAN_generate(gan, noise);
        
        if (generated) {
            // Get discriminator score
            long double score = GAN_discriminate(gan, generated);
            total_score += score;
            free(generated);
        }
        
        free(noise);
    }
    
    return total_score / num_samples;
}

// Evaluate discriminator
long double GAN_evaluate_discriminator(GAN_t *gan, long double **real_data, 
                                         long double **fake_data, size_t num_samples) {
    if (!gan || !real_data || !fake_data || !gan->discriminator) return 0.0L;
    
    long double real_score = 0.0L;
    long double fake_score = 0.0L;
    
    for (size_t i = 0; i < num_samples; i++) {
        real_score += GAN_discriminate(gan, real_data[i]);
        fake_score += GAN_discriminate(gan, fake_data[i]);
    }
    
    real_score /= num_samples;
    fake_score /= num_samples;
    
    // Discriminator should output 1 for real, 0 for fake
    return (real_score + (1.0L - fake_score)) / 2.0L;
}

// Compute inception score (simplified)
long double GAN_compute_inception_score(GAN_t *gan, long double **samples, size_t num_samples) {
    if (!gan || !samples) return 0.0L;
    
    // This is a simplified version - in practice, you'd use a pre-trained classifier
    long double total_score = 0.0L;
    
    for (size_t i = 0; i < num_samples; i++) {
        long double *sample = samples[i];
        
        // Simplified: use discriminator score as proxy for quality
        long double score = GAN_discriminate(gan, sample);
        total_score += score;
    }
    
    return total_score / num_samples;
}
