/*
 * SAM Telemetry Core - 53 Signal Collection System
 * High-performance telemetry collection and metric computation
 */

#ifndef SAM_TELEMETRY_CORE_H
#define SAM_TELEMETRY_CORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SAM_TELEMETRY_DIM 53

typedef struct {
    double values[SAM_TELEMETRY_DIM];
    unsigned int tick;
} SamTelemetry;

typedef struct {
    // Performance (5)
    double task_score;
    double tool_success_rate;
    double planner_value_gain;
    double latency_ms;
    double throughput;
    
    // Stability (5)
    double loss_variance;
    double gradient_norm;
    double weight_drift;
    double explosion_flag;
    double collapse_flag;
    
    // Identity (4)
    double anchor_similarity;
    double continuity;
    double self_coherence;
    double purpose_drift;
    
    // Uncertainty/Opacity (5)
    double retrieval_entropy;
    double prediction_entropy;
    double unknown_ratio;
    double confusion;
    double mystery;
    
    // Planning (4)
    double planner_friction;
    double depth_actual;
    double breadth_actual;
    double goal_drift;
    
    // Resources (4)
    double ram_usage;
    double compute_budget;
    double memory_budget;
    double energy;
    
    // Robustness (3)
    double contradiction_rate;
    double hallucination_rate;
    double calib_ece;
    
    // Pressure (7)
    double residual;
    double rank_deficit;
    double interference;
    double context_collapse;
    double compression_waste;
    double temporal_incoherence;
    double planner_pressure;
} SamTelemetryInput;

SamTelemetry *sam_telemetry_create(unsigned int seed);
void sam_telemetry_free(SamTelemetry *t);

int sam_telemetry_update(SamTelemetry *t, SamTelemetryInput *in);

double sam_telemetry_get(const SamTelemetry *t, int index);
const double *sam_telemetry_get_all(const SamTelemetry *t);
unsigned int sam_telemetry_get_tick(const SamTelemetry *t);

void sam_telemetry_compute_capacity(const SamTelemetry *t, double *out);
void sam_telemetry_compute_universality(const SamTelemetry *t, double *out);
void sam_telemetry_compute_innocence(const SamTelemetry *t, double *out);

void sam_telemetry_reset(SamTelemetry *t);

#ifdef __cplusplus
}
#endif

#endif // SAM_TELEMETRY_CORE_H
