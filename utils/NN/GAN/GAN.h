#ifndef GAN_H
#define GAN_H

#include "NN.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Generative Adversarial Network structure
typedef struct GAN {
  NN_t *generator;
  NN_t *discriminator;
  long double learning_rate;
  int latent_dim;
  int training_steps;
  long double lambda_gp;  // Gradient penalty coefficient
  long double lambda_drift;  // Drift penalty coefficient
  bool use_wasserstein;  // Use Wasserstein GAN
  int n_critic;  // Number of critic updates per generator update
  long double **noise_buffer;  // Buffer for noise samples
  long double **real_buffer;   // Buffer for real samples
  size_t buffer_size;
  size_t buffer_capacity;
  int buffer_ptr;
} GAN_t;

// Generative Adversarial Network functions
GAN_t *GAN_create(size_t input_dim, size_t hidden_dim, size_t output_dim, int latent_dim);
void GAN_destroy(GAN_t *gan);
long double *GAN_generate(GAN_t *gan, long double *noise);
long double GAN_discriminate(GAN_t *gan, long double *data);
void GAN_train_step(GAN_t *gan, long double **real_data, size_t batch_size);
void GAN_train_wasserstein(GAN_t *gan, long double **real_data, size_t batch_size);
void GAN_set_learning_rate(GAN_t *gan, long double learning_rate);
void GAN_set_lambda_gp(GAN_t *gan, long double lambda_gp);
void GAN_enable_wasserstein(GAN_t *gan, bool enable);
void GAN_set_n_critic(GAN_t *gan, int n_critic);
void GAN_set_buffer_size(GAN_t *gan, size_t buffer_size);
void GAN_reset_buffers(GAN_t *gan);
size_t GAN_get_parameter_count(GAN_t *gan);
void GAN_print_summary(GAN_t *gan);

// Noise generation
void GAN_generate_noise(GAN_t *gan, long double *noise, size_t batch_size);
void GAN_generate_normal_noise(GAN_t *gan, long double *noise, size_t batch_size);
void GAN_generate_uniform_noise(GAN_t *gan, long double *noise, size_t batch_size);

// Training utilities
long double GAN_compute_gradient_penalty(GAN_t *gan, long double **real_samples, 
                                        long double **fake_samples, size_t batch_size);
long double GAN_compute_wasserstein_distance(GAN_t *gan, long double **real_samples, 
                                             long double **fake_samples, size_t batch_size);
void GAN_update_buffers(GAN_t *gan, long double **real_data, size_t batch_size);

// Evaluation
long double GAN_evaluate_generator(GAN_t *gan, long double **test_data, size_t num_samples);
long double GAN_evaluate_discriminator(GAN_t *gan, long double **real_data, 
                                         long double **fake_data, size_t num_samples);
long double GAN_compute_inception_score(GAN_t *gan, long double **samples, size_t num_samples);

// Model saving/loading
void GAN_save_model(GAN_t *gan, const char *filename);
GAN_t *GAN_load_model(const char *filename);

// Advanced GAN variants
typedef struct ConditionalGAN {
  GAN_t *gan;
  size_t condition_dim;
  NN_t *condition_encoder;
} ConditionalGAN_t;

ConditionalGAN_t *ConditionalGAN_create(size_t input_dim, size_t hidden_dim, 
                                        size_t output_dim, int latent_dim, size_t condition_dim);
void ConditionalGAN_destroy(ConditionalGAN_t *cgan);
long double *ConditionalGAN_generate(ConditionalGAN_t *cgan, long double *noise, 
                                       long double *condition);
void ConditionalGAN_train_step(ConditionalGAN_t *cgan, long double **real_data, 
                                 long double **conditions, size_t batch_size);

#ifdef __cplusplus
}
#endif

#endif // GAN_H
