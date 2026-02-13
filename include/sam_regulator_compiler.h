/*
 * SAM Regulator Compiler - 53 Regulator System
 * Maps telemetry to loss weights, knobs, and regime selection
 */

#ifndef SAM_REGULATOR_COMPILER_H
#define SAM_REGULATOR_COMPILER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SAM_NUM_REGULATORS 53
#define SAM_NUM_LOSS_TERMS 28
#define SAM_NUM_KNOBS 14
#define SAM_NUM_TELEMETRY 18
#define SAM_NUM_REGIMES 6

typedef enum {
    REGIME_STASIS = 0,
    REGIME_VERIFY,
    REGIME_GD_ADAM,
    REGIME_NATGRAD,
    REGIME_EVOLVE,
    REGIME_MORPH
} SamRegime;

typedef struct {
    // 53 regulators
    double m[SAM_NUM_REGULATORS];
    
    // Matrices: (terms x regulators)
    double W_m[SAM_NUM_LOSS_TERMS][SAM_NUM_REGULATORS];
    double W_tau[SAM_NUM_LOSS_TERMS][SAM_NUM_TELEMETRY];
    double W_E[SAM_NUM_LOSS_TERMS][3];
    double W_r[SAM_NUM_LOSS_TERMS][8];
    double b_w[SAM_NUM_LOSS_TERMS];
    
    // Knob matrices
    double U_m[SAM_NUM_KNOBS][SAM_NUM_REGULATORS];
    double U_tau[SAM_NUM_KNOBS][SAM_NUM_TELEMETRY];
    double U_E[SAM_NUM_KNOBS][3];
    double U_r[SAM_NUM_KNOBS][8];
    double b_u[SAM_NUM_KNOBS];
    
    // State
    unsigned int tick;
    SamRegime current_regime;
} SamRegulatorCompiler;

SamRegulatorCompiler *sam_regulator_create(unsigned int seed);
void sam_regulator_free(SamRegulatorCompiler *rc);

// Initialize with default values
void sam_regulator_init(SamRegulatorCompiler *rc, unsigned int seed);

// Update regulators based on outcomes
void sam_regulator_update(SamRegulatorCompiler *rc, 
                          double *telemetry, 
                          double outcome,
                          SamRegime regime);

// Compile: m + telemetry + E + r -> weights + knobs
void sam_regulator_compile(SamRegulatorCompiler *rc,
                           double *telemetry,
                           double K, double U, double omega,
                           double *resources,
                           double *loss_weights_out,
                           double *knobs_out);

// Get regime from telemetry
SamRegime sam_regulator_pick_regime(double *telemetry);

// Accessors
const double *sam_regulator_get_regulators(const SamRegulatorCompiler *rc);
SamRegime sam_regulator_get_regime(const SamRegulatorCompiler *rc);

// Loss and knob names
extern const char *SAM_LOSS_NAMES[SAM_NUM_LOSS_TERMS];
extern const char *SAM_KNOB_NAMES[SAM_NUM_KNOBS];
extern const char *SAM_REGULATOR_NAMES[SAM_NUM_REGULATORS];
extern const char *SAM_REGIME_NAMES[SAM_NUM_REGIMES];

#ifdef __cplusplus
}
#endif

#endif // SAM_REGULATOR_COMPILER_H
