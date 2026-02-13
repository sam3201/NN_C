/*
 * SAM Regulator Compiler - 53 Regulator System
 * Implementation
 */

#include "sam_regulator_compiler.h"
#include "sam_fast_rng.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Loss names
const char *SAM_LOSS_NAMES[SAM_NUM_LOSS_TERMS] = {
    "task", "policy", "value", "dyn", "rew", "term", "planner_distill", "expert_distill",
    "calibration", "uncertainty", "retrieval", "evidence", "coherence", "contradiction",
    "identity_drift", "integrity_gate_fail", "resource_cost", "latency", "risk", "adversarial",
    "novelty", "coverage", "compression", "interference", "temporal_incoh", "context_collapse",
    "morph_cost", "governance_waste"
};

// Knob names
const char *SAM_KNOB_NAMES[SAM_NUM_KNOBS] = {
    "planner_depth", "planner_width", "search_budget", "temperature",
    "verify_budget", "research_budget", "morph_budget", "distill_weight",
    "consolidate_rate", "routing_degree", "context_strength",
    "risk_cap", "stasis_threshold", "patch_merge_threshold"
};

// 53 Regulator names
const char *SAM_REGULATOR_NAMES[SAM_NUM_REGULATORS] = {
    // Drive/Energy (8)
    "motivation", "desire", "curiosity", "ambition", "hunger", "fear", "aggression", "attachment",
    // Self-Regulation (6)
    "discipline", "patience", "focus", "flexibility", "resilience", "persistence",
    // Epistemic (7)
    "skepticism", "confidence", "doubt", "insight", "reflection", "wisdom", "deep_research",
    // Identity/Coherence (5)
    "identity", "integrity", "coherence", "loyalty", "authenticity",
    // Adversarial (4)
    "paranoia", "defensive_posture", "offensive_expansion", "revenge",
    // Growth/Creative (5)
    "creativity", "play", "morphogenesis", "self_transcendence", "sacrifice",
    // Social (6)
    "cooperation", "competition", "trust", "empathy", "authority_seeking", "independence",
    // Meta-System (6)
    "meta_optimization", "invariant_preservation", "collapse_avoidance",
    "adaptation_rate", "equilibrium_seeking", "phase_transition",
    // Temporal (3)
    "foresight", "memory_consolidation", "forgetting",
    // Power (3)
    "resource_awareness", "capability_estimation", "control_desire"
};

// Regime names
const char *SAM_REGIME_NAMES[SAM_NUM_REGIMES] = {
    "STASIS", "VERIFY", "GD_ADAM", "NATGRAD", "EVOLVE", "MORPH"
};

static double clip01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double softplus(double x) {
    return log(1.0 + exp(-fabs(x))) + fmax(x, 0.0);
}

SamRegulatorCompiler *sam_regulator_create(unsigned int seed) {
    SamRegulatorCompiler *rc = (SamRegulatorCompiler *)calloc(1, sizeof(SamRegulatorCompiler));
    if (!rc) return NULL;
    sam_regulator_init(rc, seed);
    return rc;
}

void sam_regulator_free(SamRegulatorCompiler *rc) {
    if (rc) free(rc);
}

void sam_regulator_init(SamRegulatorCompiler *rc, unsigned int seed) {
    SamFastRNG rng;
    sam_rng_init(&rng, seed);
    
    // Initialize regulators with defaults
    rc->m[0] = 0.6;   // motivation
    rc->m[2] = 0.5;   // curiosity
    rc->m[8] = 0.6;   // discipline
    rc->m[10] = 0.5;  // focus
    rc->m[14] = 0.5;  // skepticism
    rc->m[15] = 0.5;  // confidence
    rc->m[21] = 0.4;  // wisdom
    rc->m[22] = 0.7;  // identity
    rc->m[23] = 0.6;  // integrity
    rc->m[24] = 0.6;  // coherence
    
    // Add small random variations
    for (int i = 0; i < SAM_NUM_REGULATORS; i++) {
        rc->m[i] += sam_rng_double(&rng) * 0.1 - 0.05;
        rc->m[i] = clip01(rc->m[i]);
    }
    
    // Randomize matrices with small values
    for (int i = 0; i < SAM_NUM_LOSS_TERMS; i++) {
        for (int j = 0; j < SAM_NUM_REGULATORS; j++) {
            rc->W_m[i][j] = sam_rng_double(&rng) * 0.04 - 0.02;
        }
        for (int j = 0; j < SAM_NUM_TELEMETRY; j++) {
            rc->W_tau[i][j] = sam_rng_double(&rng) * 0.04 - 0.02;
        }
        rc->b_w[i] = sam_rng_double(&rng) * 0.02 - 0.01;
    }
    
    for (int i = 0; i < SAM_NUM_KNOBS; i++) {
        for (int j = 0; j < SAM_NUM_REGULATORS; j++) {
            rc->U_m[i][j] = sam_rng_double(&rng) * 0.06 - 0.03;
        }
        rc->b_u[i] = sam_rng_double(&rng) * 0.02 - 0.01;
    }
    
    // Bias toward coherence and identity
    // (find indices in loss names - simplified)
    rc->b_w[12] += 0.2;  // coherence
    rc->b_w[14] += 0.2; // identity_drift
    rc->b_w[17] += 0.1;  // risk
    
    // Bias toward verify and risk cap for safety
    rc->b_u[4] += 0.3;   // verify_budget
    rc->b_u[11] += 0.2;  // risk_cap
    
    rc->tick = 0;
    rc->current_regime = REGIME_GD_ADAM;
}

void sam_regulator_update(SamRegulatorCompiler *rc, 
                         double *telemetry, 
                         double outcome,
                         SamRegime regime) {
    if (!rc || !telemetry) return;
    
    SamFastRNG rng;
    sam_rng_init(&rng, rc->tick + 1);
    
    // Get telemetry values
    double residual = telemetry[0];
    double novelty = telemetry[15];
    
    // Curiosity increases with novelty
    rc->m[2] += 0.05 * novelty;
    
    // Motivation increases with success
    rc->m[0] += 0.03 * outcome;
    
    // Discipline with failures
    rc->m[8] += 0.02 * (1.0 - outcome);
    
    // Coherence emphasized in verify
    if (regime == REGIME_VERIFY) {
        rc->m[24] += 0.05;
    }
    
    // Growth in morph/evolve
    if (regime == REGIME_MORPH || regime == REGIME_EVOLVE) {
        rc->m[31] += 0.03;  // creativity
        rc->m[32] += 0.03;  // morphogenesis
    }
    
    // Stability in stasis
    if (regime == REGIME_STASIS) {
        rc->m[44] += 0.05;  // collapse_avoidance
    }
    
    // Add exploration noise
    for (int i = 0; i < SAM_NUM_REGULATORS; i++) {
        rc->m[i] += sam_rng_double(&rng) * 0.02 - 0.01;
        rc->m[i] = clip01(rc->m[i]);
    }
    
    rc->tick++;
}

SamRegime sam_regulator_pick_regime(double *tau) {
    if (!tau) return REGIME_GD_ADAM;
    
    double instability = tau[11];
    double gate_fail = tau[10];
    double adversary = tau[17];
    double contradiction = tau[8];
    double calib_error = tau[9];
    double plateau = tau[13];
    double rank_def = tau[1];
    double progress = tau[12];
    double temporal = tau[7];
    
    if (instability > 0.8 || gate_fail > 0.9 || adversary > 0.85) {
        return REGIME_STASIS;
    }
    if (contradiction > 0.6 || calib_error > 0.5 || adversary > 0.6) {
        return REGIME_VERIFY;
    }
    if (plateau > 0.5 && rank_def > 0.4) {
        return REGIME_MORPH;
    }
    if (plateau > 0.5 && fabs(progress) < 0.02) {
        return REGIME_EVOLVE;
    }
    if (instability > 0.4 && temporal > 0.4) {
        return REGIME_NATGRAD;
    }
    return REGIME_GD_ADAM;
}

void sam_regulator_compile(SamRegulatorCompiler *rc,
                          double *telemetry,
                          double K, double U, double omega,
                          double *resources,
                          double *weights_out,
                          double *knobs_out) {
    if (!rc || !telemetry || !weights_out || !knobs_out) return;
    
    // Compute omega from telemetry
    double omega_val = 0.0;
    omega_val += 1.2 * telemetry[1];  // rank_def
    omega_val += 0.8 * telemetry[13]; // plateau_flag
    omega_val += 0.4 * telemetry[0];  // residual
    omega_val += 1.0 * telemetry[7]; // temporal_incoh
    omega_val += 1.0 * telemetry[2]; // retrieval_entropy
    omega_val += 0.9 * telemetry[9];  // calibration_error
    omega_val += 1.1 * telemetry[8]; // contradiction
    
    // Pick regime
    rc->current_regime = sam_regulator_pick_regime(telemetry);
    
    // Compute loss weights
    for (int i = 0; i < SAM_NUM_LOSS_TERMS; i++) {
        double sum = rc->b_w[i];
        for (int j = 0; j < SAM_NUM_REGULATORS; j++) {
            sum += rc->W_m[i][j] * rc->m[j];
        }
        for (int j = 0; j < SAM_NUM_TELEMETRY; j++) {
            sum += rc->W_tau[i][j] * telemetry[j];
        }
        weights_out[i] = softplus(sum);
    }
    
    // Compute knobs
    for (int i = 0; i < SAM_NUM_KNOBS; i++) {
        double sum = rc->b_u[i];
        for (int j = 0; j < SAM_NUM_REGULATORS; j++) {
            sum += rc->U_m[i][j] * rc->m[j];
        }
        for (int j = 0; j < SAM_NUM_TELEMETRY; j++) {
            sum += rc->U_tau[i][j] * telemetry[j];
        }
        knobs_out[i] = clip01(sigmoid(sum));
    }
    
    // Apply regime overrides
    if (rc->current_regime == REGIME_STASIS) {
        knobs_out[0] = 0.0;
        knobs_out[1] = 0.0;
        knobs_out[2] = 0.0;
        knobs_out[3] = 0.0;
        knobs_out[5] = 0.0;
        knobs_out[6] = 0.0;
        knobs_out[4] = 1.0;
        knobs_out[12] = 1.0;
    } else if (rc->current_regime == REGIME_VERIFY) {
        knobs_out[4] = clip01(knobs_out[4] + 0.4);
        knobs_out[3] = fmax(0.05, knobs_out[3] * 0.5);
    } else if (rc->current_regime == REGIME_MORPH) {
        knobs_out[6] = clip01(knobs_out[6] + 0.5);
        knobs_out[0] = clip01(knobs_out[0] + 0.2);
    }
}

const double *sam_regulator_get_regulators(const SamRegulatorCompiler *rc) {
    return rc ? rc->m : NULL;
}

SamRegime sam_regulator_get_regime(const SamRegulatorCompiler *rc) {
    return rc ? rc->current_regime : REGIME_GD_ADAM;
}
