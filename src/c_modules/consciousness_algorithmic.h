/*
 * Pure C Consciousness Module Header
 * Algorithmic consciousness implementation
 */

#ifndef CONSCIOUSNESS_ALGORITHMIC_H
#define CONSCIOUSNESS_ALGORITHMIC_H

#include <stddef.h>

// ================================
// CONSCIOUSNESS MODULE STRUCTURES
// ================================

typedef struct {
    size_t latent_dim;
    size_t action_dim;

    // Model parameters
    double *world_model;
    double *self_model;
    double *policy_model;
    double *resource_controller;

    // Statistics and metrics
    double *stats; // [lambda_world, lambda_self, lambda_cons, lambda_policy, lambda_compute, consciousness_score]
} ConsciousnessLossModule;

// ================================
// CONSCIOUSNESS MODULE API
// ================================

// Create consciousness module
ConsciousnessLossModule *consciousness_create(size_t latent_dim, size_t action_dim);

// Free consciousness module
void consciousness_free(ConsciousnessLossModule *module);

// Run consciousness optimization
double *consciousness_optimize(ConsciousnessLossModule *module,
                              double *z_t, double *a_t, double *z_next,
                              double *m_t, double *reward, int num_params, int epochs);

// Get consciousness statistics
double *consciousness_get_stats(ConsciousnessLossModule *module);

// ================================
// PYTHON COMPATIBILITY STRUCTURES
// ================================

typedef struct {
    ConsciousnessLossModule *module;
    double consciousness_score;
    int is_conscious;
    size_t latent_dim;
    size_t action_dim;
    double *stats; // Reference to module stats
} ConsciousnessModule;

#endif // CONSCIOUSNESS_ALGORITHMIC_H
