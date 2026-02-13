/*
 * SAM Telemetry Core - 53 Signal Collection System
 * Implementation
 */

#include "sam_telemetry_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double clip(double val, double min, double max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

SamTelemetry *sam_telemetry_create(unsigned int seed) {
    SamTelemetry *t = (SamTelemetry *)calloc(1, sizeof(SamTelemetry));
    if (!t) return NULL;
    t->tick = 0;
    memset(t->values, 0, sizeof(t->values));
    return t;
}

void sam_telemetry_free(SamTelemetry *t) {
    if (t) free(t);
}

int sam_telemetry_update(SamTelemetry *t, SamTelemetryInput *in) {
    if (!t || !in) return -1;
    
    double *v = t->values;
    
    // Performance (0-4)
    v[0] = clip(in->task_score, 0.0, 1.0);
    v[1] = clip(in->tool_success_rate, 0.0, 1.0);
    v[2] = clip(in->planner_value_gain, 0.0, 1.0);
    v[3] = clip(in->latency_ms / 1000.0, 0.0, 1.0);
    v[4] = clip(in->throughput, 0.0, 1.0);
    
    // Stability (5-9)
    v[5] = clip(in->loss_variance, 0.0, 1.0);
    v[6] = clip(in->gradient_norm / 100.0, 0.0, 1.0);
    v[7] = clip(in->weight_drift, 0.0, 1.0);
    v[8] = clip(in->explosion_flag, 0.0, 1.0);
    v[9] = clip(in->collapse_flag, 0.0, 1.0);
    
    // Identity (10-13)
    v[10] = clip(in->anchor_similarity, 0.0, 1.0);
    v[11] = clip(in->continuity, 0.0, 1.0);
    v[12] = clip(in->self_coherence, 0.0, 1.0);
    v[13] = clip(in->purpose_drift, 0.0, 1.0);
    
    // Uncertainty/Opacity (14-18)
    v[14] = clip(in->retrieval_entropy, 0.0, 1.0);
    v[15] = clip(in->prediction_entropy, 0.0, 1.0);
    v[16] = clip(in->unknown_ratio, 0.0, 1.0);
    v[17] = clip(in->confusion, 0.0, 1.0);
    v[18] = clip(in->mystery, 0.0, 1.0);
    
    // Planning (19-22)
    v[19] = clip(in->planner_friction, 0.0, 1.0);
    v[20] = clip(in->depth_actual / 10.0, 0.0, 1.0);
    v[21] = clip(in->breadth_actual / 10.0, 0.0, 1.0);
    v[22] = clip(in->goal_drift, 0.0, 1.0);
    
    // Resources (23-26)
    v[23] = clip(in->ram_usage, 0.0, 1.0);
    v[24] = clip(in->compute_budget, 0.0, 1.0);
    v[25] = clip(in->memory_budget, 0.0, 1.0);
    v[26] = clip(in->energy, 0.0, 1.0);
    
    // Robustness (27-29)
    v[27] = clip(in->contradiction_rate, 0.0, 1.0);
    v[28] = clip(in->hallucination_rate, 0.0, 1.0);
    v[29] = clip(in->calib_ece, 0.0, 1.0);
    
    // Pressure (30-36)
    v[30] = clip(in->residual, 0.0, 1.0);
    v[31] = clip(in->rank_deficit, 0.0, 1.0);
    v[32] = clip(in->interference, 0.0, 1.0);
    v[33] = clip(in->context_collapse, 0.0, 1.0);
    v[34] = clip(in->compression_waste, 0.0, 1.0);
    v[35] = clip(in->temporal_incoherence, 0.0, 1.0);
    v[36] = clip(in->planner_pressure, 0.0, 1.0);
    
    t->tick++;
    return 0;
}

double sam_telemetry_get(const SamTelemetry *t, int index) {
    if (!t || index < 0 || index >= SAM_TELEMETRY_DIM) return 0.0;
    return t->values[index];
}

const double *sam_telemetry_get_all(const SamTelemetry *t) {
    return t ? t->values : NULL;
}

unsigned int sam_telemetry_get_tick(const SamTelemetry *t) {
    return t ? t->tick : 0;
}

void sam_telemetry_compute_capacity(const SamTelemetry *t, double *out) {
    if (!t || !out) return;
    double *v = (double *)t->values;
    
    double cap = 0.55 * v[0] + 0.25 * v[1] + 0.20 * v[2];
    double br = 1.0 - v[32];
    double sp = 1.0 - v[3];
    double rel = 1.0 - (0.4 * v[29] + 0.4 * v[28] + 0.2 * v[27]);
    
    *out = cap * br * sp * rel;
}

void sam_telemetry_compute_universality(const SamTelemetry *t, double *out) {
    if (!t || !out) return;
    double *v = (double *)t->values;
    
    double coh = 1.0 - (0.6 * v[27] + 0.4 * v[14]);
    double rel = 1.0 - v[29];
    double morph = 0.8;
    double id = v[10];
    
    *out = coh * rel * morph * id;
}

void sam_telemetry_compute_innocence(const SamTelemetry *t, double *out) {
    if (!t || !out) return;
    double *v = (double *)t->values;
    
    double capacity;
    sam_telemetry_compute_capacity(t, &capacity);
    
    double agency = v[19] + v[22];
    double irreversibility = v[8] + v[9];
    double verification = 0.5;
    
    double a = 2.0, b = 1.2, c = 1.0, d = 2.0, e = 1.5;
    double x = a - b * capacity - c * agency - d * irreversibility + e * verification;
    *out = 1.0 / (1.0 + exp(-x));
}

void sam_telemetry_reset(SamTelemetry *t) {
    if (t) {
        memset(t->values, 0, sizeof(t->values));
        t->tick = 0;
    }
}
