/*
 * SAM God Equation Core
 * Implementation - Pure C for maximum performance
 */

#include "sam_god_equation.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

SamGodEquation *sam_god_equation_create(size_t latent_dim, size_t num_terms) {
    SamGodEquation *ge = (SamGodEquation *)calloc(1, sizeof(SamGodEquation));
    if (!ge) return NULL;
    
    ge->psi_dim = latent_dim;
    ge->num_terms = num_terms > 0 ? num_terms : SAM_MAX_LOSS_TERMS;
    
    // Allocate arrays
    ge->psi = (double *)calloc(latent_dim, sizeof(double));
    ge->alpha = (double *)calloc(ge->num_terms, sizeof(double));
    ge->beta = (double *)calloc(ge->num_terms, sizeof(double));
    ge->gamma = (double *)calloc(ge->num_terms, sizeof(double));
    ge->delta = (double *)calloc(ge->num_terms, sizeof(double));
    ge->zeta = (double *)calloc(ge->num_terms, sizeof(double));
    ge->workspace = (double *)calloc(256, sizeof(double));
    ge->workspace_size = 256;
    
    // Initialize default K/U/O
    ge->K = 1.0;
    ge->U = 5.0;
    ge->O = 10.0;
    ge->omega = 0.5;
    ge->lambda = 0.0;
    
    // Default coefficients (from EpistemicSim)
    ge->alpha[0] = 0.05;   // discovery
    ge->beta[0] = 1.10;    // discovery scaling
    ge->gamma[0] = 0.02;   // maintenance
    ge->delta[0] = 1.00;    // burden scaling
    ge->zeta[0] = 0.01;     // contradiction penalty
    
    return ge;
}

void sam_god_equation_free(SamGodEquation *ge) {
    if (!ge) return;
    if (ge->psi) free(ge->psi);
    if (ge->alpha) free(ge->alpha);
    if (ge->beta) free(ge->beta);
    if (ge->gamma) free(ge->gamma);
    if (ge->delta) free(ge->delta);
    if (ge->zeta) free(ge->zeta);
    if (ge->workspace) free(ge->workspace);
    free(ge);
}

static double sigma_frontier(double U, double O) {
    double rho = 0.7;
    return (U + rho * O) / (1.0 + U + rho * O);
}

double sam_god_equation_contradiction(const SamGodEquation *ge) {
    return fmax(0.0, (ge->U + ge->O) / (1.0 + ge->K) - 1.0);
}

double sam_god_equation_compute(SamGodEquation *ge, double *telemetry, double dt) {
    if (!ge) return 0.0;
    
    double sigma = sigma_frontier(ge->U, ge->O);
    double contra = sam_god_equation_contradiction(ge);
    
    // Get efforts from telemetry if available
    double research = telemetry ? telemetry[0] : 0.5;
    double verify = telemetry ? telemetry[1] : 0.5;
    double morph = telemetry ? telemetry[2] : 0.2;
    
    // Discovery term
    double discovery = ge->alpha[0] * pow(ge->K, ge->beta[0]) * sigma * (0.5 + research);
    
    // Maintenance burden
    double burden = ge->gamma[0] * pow(ge->K, ge->delta[0]) * (1.2 - 0.7 * verify);
    
    // Contradiction penalty
    double contra_pen = ge->zeta[0] * pow(ge->K, ge->delta[0]) * contra;
    
    // Update K
    double dK = (discovery - burden - contra_pen) * dt;
    ge->K = fmax(0.0, ge->K + dK);
    
    // Update U (unknowns)
    double eta = 0.03;   // new unknowns created
    double mu = 1.0;
    double kappa = 0.04;  // resolution rate
    double created_U = eta * pow(fmax(ge->K, 1e-9), mu) * (0.4 + 0.6 * research) * dt;
    double resolved_U = kappa * ge->U * (0.3 + 0.7 * verify) * dt;
    ge->U = fmax(0.0, ge->U + created_U - resolved_U);
    
    // Update O (opacity)
    double xi = 0.02;   // new opacity created
    double nu = 1.0;
    double chi = 0.06;  // morphogenesis rate
    double created_O = xi * pow(fmax(ge->K, 1e-9), nu) * (0.5 + 0.5 * research) * dt;
    double morphed_O = chi * ge->O * (0.2 + 0.8 * morph) * dt;
    ge->O = fmax(0.0, ge->O + created_O - morphed_O);
    
    // Update omega based on coherence
    ge->omega = 1.0 - contra;
    
    return ge->K;
}

void sam_god_equation_update_kuo(SamGodEquation *ge, double dt, 
                                 double research_effort, double verify_effort, double morph_effort) {
    if (!ge) return;
    
    double dummy_telemetry[3] = {research_effort, verify_effort, morph_effort};
    sam_god_equation_compute(ge, dummy_telemetry, dt);
}

double sam_god_equation_task_term(SamGodEquation *ge, double *state) {
    return ge->K;
}

double sam_god_equation_gradient_term(SamGodEquation *ge, double *state, double dt) {
    return 0.0;  // Simplified
}

double sam_god_equation_coherence_term(SamGodEquation *ge) {
    return ge->omega;
}

double sam_god_equation_growth_term(SamGodEquation *ge) {
    return ge->O / (1.0 + ge->K);
}

double sam_god_equation_emergence_term(SamGodEquation *ge) {
    return ge->U / (1.0 + ge->K);
}

// Accessors
double sam_god_equation_get_K(const SamGodEquation *ge) { return ge ? ge->K : 0.0; }
double sam_god_equation_get_U(const SamGodEquation *ge) { return ge ? ge->U : 0.0; }
double sam_god_equation_get_O(const SamGodEquation *ge) { return ge ? ge->O : 0.0; }
double sam_god_equation_get_omega(const SamGodEquation *ge) { return ge ? ge->omega : 0.0; }

void sam_god_equation_set_coefficients(SamGodEquation *ge, double *alpha, double *beta, 
                                        double *gamma, double *delta, double *zeta) {
    if (!ge) return;
    if (alpha) memcpy(ge->alpha, alpha, ge->num_terms * sizeof(double));
    if (beta) memcpy(ge->beta, beta, ge->num_terms * sizeof(double));
    if (gamma) memcpy(ge->gamma, gamma, ge->num_terms * sizeof(double));
    if (delta) memcpy(ge->delta, delta, ge->num_terms * sizeof(double));
    if (zeta) memcpy(ge->zeta, zeta, ge->num_terms * sizeof(double));
}
