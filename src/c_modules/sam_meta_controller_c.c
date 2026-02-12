/*
 * SAM Meta-Controller - Pure C Core
 * Morphogenetic latency, pressure aggregation, growth primitives, invariants.
 */

#include "sam_meta_controller_c.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>

typedef struct {
    SubmodelLifecycle lifecycle;
    unsigned int step_count;
    double metrics; // Placeholder for aggregate performance metric
} SubmodelState;

// Full struct definition (as declared in the header)
struct SamMetaController {
    size_t latent_dim;
    size_t context_dim;
    size_t submodel_count;
    size_t submodel_max;
    size_t index_count;
    size_t routing_degree;
    size_t planner_depth;
    size_t archived_dim;

    double lambda;
    double lambda_threshold;
    double lambda_decay;

    double growth_budget;
    double growth_budget_max;

    double thresholds[8];
    double dominance_margin;
    unsigned int persistence_min;
    unsigned int cooldown_steps;
    double risk_max;
    unsigned int step;
    unsigned int last_growth_step;
    GrowthPrimitive last_selected;
    unsigned int last_selected_step;

    double *identity_anchor;
    size_t identity_dim;
    double *identity_vec;
    size_t identity_vec_dim;

    PressureState pressure;
    unsigned int rng;

    unsigned int invariant_violations;
    int last_violation;
    unsigned int primitive_failures[GP_REPARAM + 1];
    unsigned int primitive_blocked_until[GP_REPARAM + 1];

    // Growth diagnostics
    char last_growth_reason[256];
    int last_growth_attempt_successful; // 0 for false, 1 for true
    int growth_frozen; // 0 for false, 1 for true

    // Innocence Gate (Phase 4.4)
    double innocence;
    double innocence_threshold;

    // Submodel Lifecycle State
    SubmodelState *submodel_states;
};

SamMetaController *sam_meta_create(size_t latent_dim, size_t context_dim, size_t max_submodels, unsigned int seed) {
    SamMetaController *mc = (SamMetaController *)calloc(1, sizeof(SamMetaController));
    if (!mc) return NULL;
    mc->latent_dim = latent_dim;
    mc->context_dim = context_dim;
    mc->submodel_max = max_submodels > 0 ? max_submodels : 4;
    mc->submodel_count = 1;
    mc->index_count = 1;
    mc->routing_degree = 1;
    mc->planner_depth = 4;
    mc->archived_dim = 0;
    mc->lambda = 0.0;
    mc->lambda_threshold = 1.5;
    mc->lambda_decay = 0.99;
    mc->growth_budget = 0.0;
    mc->growth_budget_max = 4.0;
    for (int i = 0; i < 8; i++) {
        mc->thresholds[i] = 0.1;
    }
    mc->dominance_margin = 0.05;
    mc->persistence_min = 3;
    mc->cooldown_steps = 4;
    mc->risk_max = 0.85;
    mc->step = 0;
    mc->last_growth_step = 0;
    mc->last_selected = GP_NONE;
    mc->last_selected_step = 0;
    mc->identity_anchor = NULL;
    mc->identity_vec = NULL;
    mc->identity_dim = 0;
    mc->identity_vec_dim = 0;
    mc->rng = seed ? seed : 0xC0FFEEu;
    mc->invariant_violations = 0;
    mc->last_violation = 0;
    // New fields for growth diagnostics
    strncpy(mc->last_growth_reason, "None", sizeof(mc->last_growth_reason) - 1);
    mc->last_growth_reason[sizeof(mc->last_growth_reason) - 1] = '\0';
    mc->last_growth_attempt_successful = 0;
    mc->growth_frozen = 0;
    mc->innocence = 1.0;
    mc->innocence_threshold = 0.2;

    mc->submodel_states = (SubmodelState *)calloc(mc->submodel_max, sizeof(SubmodelState));
    if (!mc->submodel_states) {
        free(mc);
        return NULL;
    }
    // Initialize existing submodels to PLANNING state
    for (size_t i = 0; i < mc->submodel_count; i++) {
        mc->submodel_states[i].lifecycle = PDI_T_PLAN;
        mc->submodel_states[i].step_count = 0;
        mc->submodel_states[i].metrics = 0.5;
    }

    return mc;
}

void sam_meta_free(SamMetaController *mc) {
    if (!mc) return;
    free(mc->identity_anchor);
    free(mc->identity_vec);
    free(mc->submodel_states);
    free(mc);
}

// Static helper functions
static unsigned int rng_next(SamMetaController *mc) {
    mc->rng = mc->rng * 1664525u + 1013904223u;
    return mc->rng;
}

static double rand_unit(SamMetaController *mc) {
    return (rng_next(mc) >> 8) * (1.0 / 16777216.0);
}

static double dot(const double *a, const double *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static double norm(const double *a, size_t n) {
    return sqrt(dot(a, a, n));
}

static double cosine_sim(const double *a, const double *b, size_t n) {
    double na = norm(a, n);
    double nb = norm(b, n);
    if (na <= 1e-12 || nb <= 1e-12) return 0.0;
    return dot(a, b, n) / (na * nb);
}

double sam_meta_update_pressure(SamMetaController *mc,
                                double residual,
                                double rank_def,
                                double retrieval_entropy,
                                double interference,
                                double planner_friction,
                                double context_collapse,
                                double compression_waste,
                                double temporal_incoherence) {
    if (!mc) return 0.0;
    mc->step += 1;
    double values[8] = {residual, rank_def, retrieval_entropy, interference,
                        planner_friction, context_collapse, compression_waste, temporal_incoherence};
    double *persist = mc->pressure.persistence;
    for (int i = 0; i < 8; i++) {
        double v = values[i];
        persist[i] = (v > mc->thresholds[i]) ? (persist[i] + 1.0) : 0.0;
    }
    mc->pressure.residual = residual;
    mc->pressure.rank_def = rank_def;
    mc->pressure.retrieval_entropy = retrieval_entropy;
    mc->pressure.interference = interference;
    mc->pressure.planner_friction = planner_friction;
    mc->pressure.context_collapse = context_collapse;
    mc->pressure.compression_waste = compression_waste;
    mc->pressure.temporal_incoherence = temporal_incoherence;

    // Route-Aware Optimization Anchor (SAM 3.0 PREP)
    // Future: Calculate d(lambda)/d(route) to optimize information flow
    // between submodels based on causal graph topology.

    double weighted = 0.0;
    weighted += residual * 0.35;
    weighted += rank_def * 0.3;
    weighted += retrieval_entropy * 0.25;
    weighted += interference * 0.3;
    weighted += planner_friction * 0.2;
    weighted += context_collapse * 0.2;
    weighted += compression_waste * 0.2;
    weighted += temporal_incoherence * 0.25;

    mc->lambda = mc->lambda * mc->lambda_decay + weighted;
    return mc->lambda;
}

static int dominant_pressure(const SamMetaController *mc, double *out_value) {
    double values[8] = {
        mc->pressure.residual,
        mc->pressure.rank_def,
        mc->pressure.retrieval_entropy,
        mc->pressure.interference,
        mc->pressure.planner_friction,
        mc->pressure.context_collapse,
        mc->pressure.compression_waste,
        mc->pressure.temporal_incoherence
    };
    double maxv = values[0];
    int maxi = 0;
    double second = -1e9;
    for (int i = 1; i < 8; i++) {
        if (values[i] > maxv) {
            second = maxv;
            maxv = values[i];
            maxi = i;
        } else if (values[i] > second) {
            second = values[i];
        }
    }
    if (out_value) *out_value = maxv;
    // Lower dominance margin to 0.01 for more responsive growth
    if (maxv - second < 0.01) return -1;
    return maxi;
}

static int pressure_compensable(const SamMetaController *mc, int dominant_idx) {
    if (!mc) return 0;
    switch (dominant_idx) {
        case 4: // planner friction
            return (mc->pressure.residual > mc->thresholds[0]) ||
                   (mc->pressure.rank_def > mc->thresholds[1]);
        case 2: // retrieval entropy
            return (mc->pressure.compression_waste > mc->thresholds[6]);
        default:
            return 0;
    }
}

static double primitive_risk(const SamMetaController *mc, GrowthPrimitive primitive) {
    double budget_factor = mc ? (mc->growth_budget / (mc->growth_budget_max + 1e-6)) : 0.0;
    switch (primitive) {
        case GP_LATENT_EXPAND: return 0.7 + 0.15 * budget_factor;
        case GP_SUBMODEL_SPAWN: return 0.8 + 0.1 * budget_factor;
        case GP_INDEX_EXPAND: return 0.35 + 0.1 * budget_factor;
        case GP_ROUTING_INCREASE: return 0.45 + 0.1 * budget_factor;
        case GP_CONTEXT_EXPAND: return 0.55 + 0.1 * budget_factor;
        case GP_PLANNER_WIDEN: return 0.6 + 0.1 * budget_factor;
        case GP_CONSOLIDATE: return 0.25;
        case GP_REPARAM: return 0.65;
        case GP_NONE:
        default:
            return 1.0;
    }
}

static int primitive_blocked(const SamMetaController *mc, GrowthPrimitive primitive) {
    if (!mc || primitive < GP_NONE || primitive > GP_REPARAM) return 1;
    if (primitive == GP_NONE) return 1;
    return mc->primitive_blocked_until[primitive] > mc->step;
}

GrowthPrimitive sam_meta_select_primitive(SamMetaController *mc) {
    if (!mc) return GP_NONE;
    
    // Growth evaluation
    mc->last_growth_attempt_successful = 0; // Reset for this attempt
    strcpy(mc->last_growth_reason, "No pressure met criteria"); // Default reason

    if (mc->growth_frozen) {
        strcpy(mc->last_growth_reason, "Growth currently frozen");
        return GP_NONE;
    }
    if (mc->lambda < mc->lambda_threshold) {
        strcpy(mc->last_growth_reason, "Lambda below threshold");
        return GP_NONE;
    }
    
    // Innocence Gate (Phase 4.4)
    if (mc->innocence < mc->innocence_threshold) {
        strcpy(mc->last_growth_reason, "Innocence gate: System is 'guilty' (power > wisdom)");
        return GP_NONE;
    }

    if (mc->growth_budget >= mc->growth_budget_max) {
        mc->last_selected = GP_CONSOLIDATE;
        mc->last_selected_step = mc->step;
        strcpy(mc->last_growth_reason, "Growth budget exceeded, consolidating");
        return GP_CONSOLIDATE;
    }
    if (mc->step > mc->last_growth_step &&
        (mc->step - mc->last_growth_step) < mc->cooldown_steps) {
        strcpy(mc->last_growth_reason, "Growth cooldown period active");
        return GP_NONE;
    }

    double maxv = 0.0;
    int idx = dominant_pressure(mc, &maxv);
    if (idx < 0) {
        strcpy(mc->last_growth_reason, "No dominant pressure");
        return GP_NONE;
    }
    if (mc->pressure.persistence[idx] < (double)mc->persistence_min) {
        snprintf(mc->last_growth_reason, sizeof(mc->last_growth_reason),
                 "Dominant pressure persistence below min (%d/%d)",
                 (int)mc->pressure.persistence[idx], mc->persistence_min);
        return GP_NONE;
    }
    if (pressure_compensable(mc, idx)) {
        strcpy(mc->last_growth_reason, "Dominant pressure is compensable");
        return GP_NONE;
    }

    GrowthPrimitive candidate = GP_NONE;
    switch (idx) {
        case 0: candidate = GP_LATENT_EXPAND; strcpy(mc->last_growth_reason, "Latent expansion (residual)"); break;
        case 1: candidate = (rand_unit(mc) > 0.5) ? GP_LATENT_EXPAND : GP_REPARAM; strcpy(mc->last_growth_reason, "Latent expand/reparam (rank def)"); break;
        case 2: candidate = GP_INDEX_EXPAND; strcpy(mc->last_growth_reason, "Index expansion (retrieval entropy)"); break;
        case 3: candidate = GP_SUBMODEL_SPAWN; strcpy(mc->last_growth_reason, "Submodel spawn (interference)"); break;
        case 4: candidate = GP_PLANNER_WIDEN; strcpy(mc->last_growth_reason, "Planner widen (planner friction)"); break;
        case 5: candidate = GP_CONTEXT_EXPAND; strcpy(mc->last_growth_reason, "Context expansion (context collapse)"); break;
        case 6: candidate = GP_CONSOLIDATE; strcpy(mc->last_growth_reason, "Consolidate (compression waste)"); break;
        case 7: candidate = GP_REPARAM; strcpy(mc->last_growth_reason, "Reparameterize (temporal incoherence)"); break;
        default: candidate = GP_NONE; strcpy(mc->last_growth_reason, "Unknown primitive candidate"); break;
    }

    if (candidate == GP_NONE) return GP_NONE;
    if (primitive_blocked(mc, candidate)) {
        snprintf(mc->last_growth_reason, sizeof(mc->last_growth_reason),
                 "Primitive %d blocked until step %u", candidate, mc->primitive_blocked_until[candidate]);
        return GP_NONE;
    }
    if (primitive_risk(mc, candidate) > mc->risk_max) {
        snprintf(mc->last_growth_reason, sizeof(mc->last_growth_reason),
                 "Primitive %d risk too high (%.2f > %.2f)", candidate, primitive_risk(mc, candidate), mc->risk_max);
        return GP_NONE;
    }

    mc->last_selected = candidate;
    mc->last_selected_step = mc->step;
    mc->last_growth_attempt_successful = 1; // Mark success if a primitive was selected
    return candidate;
}

int sam_meta_apply_primitive(SamMetaController *mc, GrowthPrimitive primitive) {
    if (!mc) return 0;
    mc->last_growth_attempt_successful = 0; // Default to failure for application
    strcpy(mc->last_growth_reason, "Primitive application failed"); // Default reason

    if (primitive == GP_NONE) return 0;
    if (primitive != mc->last_selected || mc->last_selected_step != mc->step) {
        mc->invariant_violations += 1;
        mc->last_violation = 1; // growth causality violation
        return 0;
    }
    if (mc->step > mc->last_growth_step &&
        (mc->step - mc->last_growth_step) < mc->cooldown_steps) {
        mc->invariant_violations += 1;
        mc->last_violation = 2; // cooldown violation
        return 0;
    }
    switch (primitive) {
        case GP_LATENT_EXPAND:
            mc->latent_dim += 8;
            mc->growth_budget += 0.5;
            strcpy(mc->last_growth_reason, "Applied: Latent dimension expanded");
            break;
        case GP_SUBMODEL_SPAWN:
            if (mc->submodel_count < mc->submodel_max) {
                // Initialize new submodel state
                size_t new_idx = mc->submodel_count;
                if (mc->submodel_states) {
                    mc->submodel_states[new_idx].lifecycle = PDI_T_PLAN;
                    mc->submodel_states[new_idx].step_count = 0;
                    mc->submodel_states[new_idx].metrics = 0.5;
                }
                mc->submodel_count += 1;
                mc->growth_budget += 0.6;
                strcpy(mc->last_growth_reason, "Applied: Submodel spawned (PLAN state)");
            } else {
                strcpy(mc->last_growth_reason, "Primitive failed: Max submodels reached");
                return 0; // Failed to apply
            }
            break;
        case GP_INDEX_EXPAND:
            mc->index_count += 1;
            mc->growth_budget += 0.2;
            strcpy(mc->last_growth_reason, "Applied: Index expanded");
            break;
        case GP_ROUTING_INCREASE:
            mc->routing_degree += 1;
            mc->growth_budget += 0.2;
            strcpy(mc->last_growth_reason, "Applied: Routing degree increased");
            break;
        case GP_CONTEXT_EXPAND:
            mc->context_dim += 4;
            mc->growth_budget += 0.3;
            strcpy(mc->last_growth_reason, "Applied: Context expanded");
            break;
        case GP_PLANNER_WIDEN:
            mc->planner_depth += 2;
            mc->growth_budget += 0.2;
            strcpy(mc->last_growth_reason, "Applied: Planner widened");
            break;
        case GP_CONSOLIDATE:
            if (mc->latent_dim > 16) {
                mc->latent_dim -= 4;
                mc->archived_dim += 4;
                strcpy(mc->last_growth_reason, "Applied: Consolidated latent dimensions");
            } else {
                strcpy(mc->last_growth_reason, "Primitive failed: No dimensions to consolidate");
                return 0; // Failed to apply
            }
            if (mc->index_count > 1) mc->index_count -= 1;
            break;
        case GP_REPARAM:
            mc->lambda *= 0.85;
            strcpy(mc->last_growth_reason, "Applied: Reparameterized lambda");
            break;
        case GP_NONE:
        default:
            return 0;
    }
    mc->lambda *= 0.9;
    mc->last_growth_step = mc->step;
    mc->last_selected = GP_NONE; // Primitive consumed
    mc->last_growth_attempt_successful = 1; // Mark success if primitive applied
    return 1;
}

void sam_meta_set_policy_params(SamMetaController *mc,
                                const double *thresholds,
                                double dominance_margin,
                                unsigned int persistence_min,
                                unsigned int cooldown_steps,
                                double risk_max) {
    if (!mc) return;
    if (thresholds) {
        for (int i = 0; i < 8; i++) {
            mc->thresholds[i] = thresholds[i];
        }
    }
    if (dominance_margin > 0.0) mc->dominance_margin = dominance_margin;
    if (persistence_min > 0) mc->persistence_min = persistence_min;
    if (cooldown_steps > 0) mc->cooldown_steps = cooldown_steps;
    if (risk_max > 0.0) mc->risk_max = risk_max;
}

void sam_meta_get_policy_params(const SamMetaController *mc, SamMetaPolicyParams *out_params) {
    if (!mc || !out_params) return;
    for (int i = 0; i < 8; i++) {
        out_params->thresholds[i] = mc->thresholds[i];
    }
    out_params->dominance_margin = mc->dominance_margin;
    out_params->persistence_min = mc->persistence_min;
    out_params->cooldown_steps = mc->cooldown_steps;
    out_params->risk_max = mc->risk_max;
}

int sam_meta_record_growth_outcome(SamMetaController *mc, GrowthPrimitive primitive, int success) {
    if (!mc) return 0;
    if (primitive <= GP_NONE || primitive > GP_REPARAM) return 0;
    if (success) {
        if (mc->primitive_failures[primitive] > 0) {
            mc->primitive_failures[primitive] -= 1;
        }
        return 1;
    }
    mc->primitive_failures[primitive] += 1;
    // Freeze growth if a primitive consistently fails
    if (mc->primitive_failures[primitive] >= 3) {
        mc->primitive_blocked_until[primitive] = mc->step + (mc->cooldown_steps * 3);
        mc->primitive_failures[primitive] = 0;
        mc->growth_frozen = 1; // Freeze growth
        snprintf(mc->last_growth_reason, sizeof(mc->last_growth_reason),
                 "Growth frozen due to primitive %d repeated failures", primitive);
    }
    return 1;
}

int sam_meta_set_identity_anchor(SamMetaController *mc, const double *vec, size_t dim) {
    if (!mc || !vec || dim == 0) return 0;
    double *buf = (double *)calloc(dim, sizeof(double));
    if (!buf) return 0;
    memcpy(buf, vec, dim * sizeof(double));
    free(mc->identity_anchor);
    mc->identity_anchor = buf;
    mc->identity_dim = dim;
    if (mc->identity_vec && mc->identity_vec_dim != dim) {
        free(mc->identity_vec);
        mc->identity_vec = NULL;
        mc->identity_vec_dim = 0;
    }
    return 1;
}

int sam_meta_update_identity_vector(SamMetaController *mc, const double *vec, size_t dim) {
    if (!mc || !vec || dim == 0) return 0;
    if (mc->identity_dim > 0 && dim != mc->identity_dim) return 0;
    double *buf = (double *)calloc(dim, sizeof(double));
    if (!buf) return 0;
    memcpy(buf, vec, dim * sizeof(double));
    free(mc->identity_vec);
    mc->identity_vec = buf;
    mc->identity_vec_dim = dim;
    return 1;
}

int sam_meta_check_invariants(SamMetaController *mc, double *out_identity_similarity) {
    if (!mc) return 0;
    if (!mc->identity_anchor || !mc->identity_vec || mc->identity_dim == 0 || mc->identity_vec_dim == 0) {
        if (out_identity_similarity) *out_identity_similarity = 0.0;
        return 1;
    }
    if (mc->identity_vec_dim != mc->identity_dim) {
        if (out_identity_similarity) *out_identity_similarity = 0.0;
        return 0;
    }
    double sim = cosine_sim(mc->identity_anchor, mc->identity_vec, mc->identity_dim);
    if (out_identity_similarity) *out_identity_similarity = sim;
    if (sim < 0.8) return 0;
    return 1;
}

SamMetaInvariantState sam_meta_get_invariant_state(const SamMetaController *mc) {
    SamMetaInvariantState state;
    state.violations = mc ? mc->invariant_violations : 0;
    state.last_violation = mc ? mc->last_violation : 0;
    return state;
}

int sam_meta_evaluate_contract(SamMetaController *mc,
                               double baseline_worst_case,
                               double proposed_worst_case) {
    if (!mc) return 0;
    return proposed_worst_case > baseline_worst_case;
}

void sam_meta_set_innocence(SamMetaController *mc, double innocence, double threshold) {
    if (!mc) return;
    mc->innocence = innocence;
    mc->innocence_threshold = threshold;
}

size_t sam_meta_get_latent_dim(const SamMetaController *mc) { return mc ? mc->latent_dim : 0; }
size_t sam_meta_get_context_dim(const SamMetaController *mc) { return mc ? mc->context_dim : 0; }
size_t sam_meta_get_submodel_count(const SamMetaController *mc) { return mc ? mc->submodel_count : 0; }
size_t sam_meta_get_index_count(const SamMetaController *mc) { return mc ? mc->index_count : 0; }
size_t sam_meta_get_routing_degree(const SamMetaController *mc) { return mc ? mc->routing_degree : 0; }
size_t sam_meta_get_planner_depth(const SamMetaController *mc) { return mc ? mc->planner_depth : 0; }
double sam_meta_get_lambda(const SamMetaController *mc) { return mc ? mc->lambda : 0.0; }
double sam_meta_get_growth_budget(const SamMetaController *mc) { return mc ? mc->growth_budget : 0.0; }
size_t sam_meta_get_archived_dim(const SamMetaController *mc) { return mc ? mc->archived_dim : 0; }

double sam_meta_get_effective_rank(const SamMetaController *mc) {
    if (!mc) return 0.0;
    
    double values[8] = {
        mc->pressure.residual,
        mc->pressure.rank_def,
        mc->pressure.retrieval_entropy,
        mc->pressure.interference,
        mc->pressure.planner_friction,
        mc->pressure.context_collapse,
        mc->pressure.compression_waste,
        mc->pressure.temporal_incoherence
    };
    
    double sum = 0.0;
    for (int i = 0; i < 8; i++) sum += fabs(values[i]) + 1e-9;
    
    double entropy = 0.0;
    for (int i = 0; i < 8; i++) {
        double p = (fabs(values[i]) + 1e-9) / sum;
        entropy -= p * log(p);
    }
    
    return exp(entropy);
}

const char *sam_meta_get_last_growth_reason(const SamMetaController *mc) {
    return mc ? mc->last_growth_reason : "N/A";
}

int sam_meta_get_last_growth_attempt_successful(const SamMetaController *mc) {
    return mc ? mc->last_growth_attempt_successful : 0;
}

int sam_meta_get_growth_frozen(const SamMetaController *mc) {
    return mc ? mc->growth_frozen : 0;
}

GrowthPrimitive sam_meta_trigger_growth_evaluation(SamMetaController *mc) {
    if (!mc) return GP_NONE;
    // Force an immediate re-evaluation of growth primitives
    mc->growth_frozen = 0; // Unfreeze if it was frozen
    // Reset cooldown to allow re-evaluation, but don't clear last_growth_step fully
    if (mc->step < mc->cooldown_steps) { // Prevent underflow
         mc->last_growth_step = 0;
    } else {
        mc->last_growth_step = mc->step - mc->cooldown_steps;
    }
    return sam_meta_select_primitive(mc);
}

// Submodel Lifecycle (PDI-T)
SubmodelLifecycle sam_meta_get_submodel_lifecycle(const SamMetaController *mc, size_t submodel_idx) {
    if (!mc || !mc->submodel_states || submodel_idx >= mc->submodel_count) return PDI_T_NONE;
    return mc->submodel_states[submodel_idx].lifecycle;
}

int sam_meta_advance_submodel_lifecycle(SamMetaController *mc, size_t submodel_idx, int success) {
    if (!mc || !mc->submodel_states || submodel_idx >= mc->submodel_count) return 0;
    
    SubmodelState *state = &mc->submodel_states[submodel_idx];
    state->step_count++;

    if (!success) {
        // Failure logic: Retry plan if designing, retry design if implementing, etc.
        // For simplicity, if implementation/test fails, we go back to design.
        if (state->lifecycle == PDI_T_IMPLEMENT || state->lifecycle == PDI_T_TEST) {
            state->lifecycle = PDI_T_DESIGN;
            state->step_count = 0;
        }
        return 0; 
    }

    // Success transition logic
    switch (state->lifecycle) {
        case PDI_T_PLAN:
            state->lifecycle = PDI_T_DESIGN;
            break;
        case PDI_T_DESIGN:
            state->lifecycle = PDI_T_IMPLEMENT;
            break;
        case PDI_T_IMPLEMENT:
            state->lifecycle = PDI_T_TEST;
            break;
        case PDI_T_TEST:
            state->lifecycle = PDI_T_DEPLOYED;
            break;
        case PDI_T_DEPLOYED:
            // Stay deployed, but maybe update metrics
            state->metrics = fmin(1.0, state->metrics + 0.05);
            break;
        default:
            break;
    }
    
    // Reset step count on transition (except deployed)
    if (state->lifecycle != PDI_T_DEPLOYED) {
        state->step_count = 0;
    }
    return 1;
}

// ================================
// PYTHON BINDINGS
// ================================

static void capsule_destructor(PyObject *capsule) {
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    sam_meta_free(mc);
}

static PyObject *py_sam_meta_create(PyObject *self, PyObject *args) {
    (void)self;
    unsigned int seed = 0;
    unsigned long latent_dim = 64;
    unsigned long context_dim = 16;
    unsigned long max_submodels = 4;
    if (!PyArg_ParseTuple(args, "|kkkI", &latent_dim, &context_dim, &max_submodels, &seed)) {
        return NULL;
    }
    SamMetaController *mc = sam_meta_create(latent_dim, context_dim, max_submodels, seed);
    if (!mc) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create SamMetaController");
        return NULL;
    }
    return PyCapsule_New(mc, "SamMetaController", capsule_destructor);
}

static PyObject *py_sam_meta_update_pressure(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    double residual, rank_def, retrieval_entropy, interference;
    double planner_friction, context_collapse, compression_waste, temporal_incoherence;
    if (!PyArg_ParseTuple(args, "Odddddddd", &capsule,
                          &residual, &rank_def, &retrieval_entropy, &interference,
                          &planner_friction, &context_collapse, &compression_waste, &temporal_incoherence)) {
        return NULL;
    }
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    double lambda = sam_meta_update_pressure(mc, residual, rank_def, retrieval_entropy, interference,
                                             planner_friction, context_collapse, compression_waste, temporal_incoherence);
    return PyFloat_FromDouble(lambda);
}

static PyObject *py_sam_meta_select_primitive(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    GrowthPrimitive gp = sam_meta_select_primitive(mc);
    return PyLong_FromLong((long)gp);
}

static PyObject *py_sam_meta_apply_primitive(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    long primitive = 0;
    if (!PyArg_ParseTuple(args, "Ol", &capsule, &primitive)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    int ok = sam_meta_apply_primitive(mc, (GrowthPrimitive)primitive);
    return PyBool_FromLong(ok);
}

static PyObject *py_sam_meta_set_identity_anchor(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    PyObject *seq = NULL;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &seq)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    PyObject *fast = PySequence_Fast(seq, "identity_anchor must be a sequence");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    double *buf = (double *)calloc((size_t)n, sizeof(double));
    if (!buf) {
        Py_DECREF(fast);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
        double value = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            free(buf);
            Py_DECREF(fast);
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                PyErr_Format(PyExc_TypeError, "identity_anchor[%zd] must be a number", i);
            }
            return NULL;
        }
        buf[i] = value;
    }
    if (!sam_meta_set_identity_anchor(mc, buf, (size_t)n)) {
        free(buf);
        Py_DECREF(fast);
        PyErr_SetString(PyExc_RuntimeError, "Failed to set identity anchor");
        return NULL;
    }
    free(buf);
    Py_DECREF(fast);
    Py_RETURN_NONE;
}

static PyObject *py_sam_meta_update_identity_vector(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    PyObject *seq = NULL;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &seq)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    PyObject *fast = PySequence_Fast(seq, "identity_vec must be a sequence");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    double *buf = (double *)calloc((size_t)n, sizeof(double));
    if (!buf) {
        Py_DECREF(fast);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
        double value = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            free(buf);
            Py_DECREF(fast);
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                PyErr_Format(PyExc_TypeError, "identity_vec[%zd] must be a number", i);
            }
            return NULL;
        }
        buf[i] = value;
    }
    if (!sam_meta_update_identity_vector(mc, buf, (size_t)n)) {
        free(buf);
        Py_DECREF(fast);
        PyErr_SetString(PyExc_RuntimeError, "Failed to update identity vector");
        return NULL;
    }
    free(buf);
    Py_DECREF(fast);
    Py_RETURN_NONE;
}

static PyObject *py_sam_meta_check_invariants(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    double sim = 0.0;
    int ok = sam_meta_check_invariants(mc, &sim);
    return Py_BuildValue("{s:O,s:d}", "passed", ok ? Py_True : Py_False, "identity_similarity", sim);
}

static PyObject *py_sam_meta_estimate_pressures(PyObject *self, PyObject *args) {
    (void)self;
    double survival, agent_count, goal_count, goal_history, activity_age, learning_events;
    if (!PyArg_ParseTuple(args, "dddddd", &survival, &agent_count, &goal_count, &goal_history, &activity_age, &learning_events)) {
        return NULL;
    }
    double residual = fmin(1.0, fmax(0.0, 1.0 - survival));
    double rank_def = fmin(1.0, fmax(0.0, 1.0 - (agent_count / 10.0)));
    double retrieval_entropy = fmin(1.0, fmax(0.0, activity_age / 300.0));
    double interference = fmin(1.0, fmax(0.0, agent_count / 15.0));
    double planner_friction = fmin(1.0, fmax(0.0, learning_events / 50.0));
    double context_collapse = fmin(1.0, fmax(0.0, goal_count / 10.0));
    double compression_waste = fmin(1.0, fmax(0.0, goal_history / 50.0));
    double temporal_incoherence = fmin(1.0, fmax(0.0, fabs(sin(activity_age / 60.0))));

    return Py_BuildValue("{s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d}",
                         "residual", residual,
                         "rank_def", rank_def,
                         "retrieval_entropy", retrieval_entropy,
                         "interference", interference,
                         "planner_friction", planner_friction,
                         "context_collapse", context_collapse,
                         "compression_waste", compression_waste,
                         "temporal_incoherence", temporal_incoherence);
}

static PyObject *py_sam_meta_evaluate_contract(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    double baseline, proposed;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &baseline, &proposed)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    int ok = sam_meta_evaluate_contract(mc, baseline, proposed);
    return PyBool_FromLong(ok);
}

static PyObject *py_sam_meta_set_innocence(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    double innocence, threshold;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &innocence, &threshold)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    sam_meta_set_innocence(mc, innocence, threshold);
    Py_RETURN_NONE;
}

static PyObject *py_sam_meta_get_state(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    return Py_BuildValue("{s:k,s:k,s:k,s:k,s:k,s:d,s:d,s:k,s:d,s:d,s:k,s:k,s:d,s:k}",
                         "latent_dim", (unsigned long)sam_meta_get_latent_dim(mc),
                         "context_dim", (unsigned long)sam_meta_get_context_dim(mc),
                         "submodels", (unsigned long)sam_meta_get_submodel_count(mc),
                         "indices", (unsigned long)sam_meta_get_index_count(mc),
                         "planner_depth", (unsigned long)sam_meta_get_planner_depth(mc),
                         "lambda", sam_meta_get_lambda(mc),
                         "growth_budget", sam_meta_get_growth_budget(mc),
                         "archived_dim", (unsigned long)sam_meta_get_archived_dim(mc),
                         "lambda_threshold", mc->lambda_threshold,
                         "dominance_margin", mc->dominance_margin,
                         "persistence_min", (unsigned long)mc->persistence_min,
                         "cooldown_steps", (unsigned long)mc->cooldown_steps,
                         "risk_max", mc->risk_max,
                         "step", (unsigned long)mc->step);
}

static PyObject *py_sam_meta_set_policy_params(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    PyObject *seq = NULL;
    double dominance_margin = 0.0;
    unsigned long persistence_min = 0;
    unsigned long cooldown_steps = 0;
    double risk_max = 0.0;
    if (!PyArg_ParseTuple(args, "OOdkkd", &capsule, &seq, &dominance_margin, &persistence_min, &cooldown_steps, &risk_max)) {
        return NULL;
    }
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    double thresholds[8];
    double *threshold_ptr = NULL;
    if (seq != Py_None) {
        PyObject *fast = PySequence_Fast(seq, "thresholds must be a sequence or None");
        if (!fast) return NULL;
        Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
        if (n != 8) {
            Py_DECREF(fast);
            PyErr_SetString(PyExc_ValueError, "thresholds must have length 8");
            return NULL;
        }
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
            double value = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                Py_DECREF(fast);
                if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                    PyErr_Clear();
                    PyErr_Format(PyExc_TypeError, "thresholds[%zd] must be a number", i);
                }
                return NULL;
            }
            thresholds[i] = value;
        }
        threshold_ptr = thresholds;
        Py_DECREF(fast);
    }
    sam_meta_set_policy_params(mc, threshold_ptr, dominance_margin, (unsigned int)persistence_min,
                               (unsigned int)cooldown_steps, risk_max);
    Py_RETURN_NONE;
}

static PyObject *py_sam_meta_get_policy_params(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    SamMetaPolicyParams params;
    sam_meta_get_policy_params(mc, &params);
    PyObject *thresholds = PyList_New(8);
    for (int i = 0; i < 8; i++) {
        PyList_SetItem(thresholds, i, PyFloat_FromDouble(params.thresholds[i]));
    }
    return Py_BuildValue("{s:O,s:d,s:k,s:k,s:d}",
                         "thresholds", thresholds,
                         "dominance_margin", params.dominance_margin,
                         "persistence_min", (unsigned long)params.persistence_min,
                         "cooldown_steps", (unsigned long)params.cooldown_steps,
                         "risk_max", params.risk_max);
}

static PyObject *py_sam_meta_record_growth_outcome(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    long primitive = 0;
    int success = 0;
    if (!PyArg_ParseTuple(args, "Olp", &capsule, &primitive, &success)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    int ok = sam_meta_record_growth_outcome(mc, (GrowthPrimitive)primitive, success);
    return PyBool_FromLong(ok);
}

static PyObject *py_sam_meta_get_invariant_state(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    SamMetaInvariantState state = sam_meta_get_invariant_state(mc);
    return Py_BuildValue("{s:k,s:i}",
                         "violations", (unsigned long)state.violations,
                         "last_violation", state.last_violation);
}

static PyObject *py_sam_meta_trigger_growth_evaluation(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    GrowthPrimitive gp = sam_meta_trigger_growth_evaluation(mc);
    return PyLong_FromLong((long)gp);
}

static PyObject *py_sam_meta_get_submodel_lifecycle(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    unsigned long submodel_idx = 0;
    if (!PyArg_ParseTuple(args, "Ok", &capsule, &submodel_idx)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    SubmodelLifecycle lifecycle = sam_meta_get_submodel_lifecycle(mc, (size_t)submodel_idx);
    return PyLong_FromLong((long)lifecycle);
}

static PyObject *py_sam_meta_advance_submodel_lifecycle(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    unsigned long submodel_idx = 0;
    int success = 0;
    if (!PyArg_ParseTuple(args, "Okp", &capsule, &submodel_idx, &success)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    int ok = sam_meta_advance_submodel_lifecycle(mc, (size_t)submodel_idx, success);
    return PyBool_FromLong(ok);
}

static PyObject *py_sam_meta_get_growth_diagnostics(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    return Py_BuildValue("{s:s,s:i,s:i}",
                         "last_growth_reason", sam_meta_get_last_growth_reason(mc),
                         "last_growth_attempt_successful", sam_meta_get_last_growth_attempt_successful(mc),
                         "growth_frozen", sam_meta_get_growth_frozen(mc));
}

static PyObject *py_sam_meta_get_effective_rank(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    return PyFloat_FromDouble(sam_meta_get_effective_rank(mc));
}

static PyMethodDef MetaMethods[] = {
    {"create", py_sam_meta_create, METH_VARARGS, "Create SAM meta-controller"},
    {"estimate_pressures", py_sam_meta_estimate_pressures, METH_VARARGS, "Estimate pressure signals"},
    {"update_pressure", py_sam_meta_update_pressure, METH_VARARGS, "Update pressure signals"},
    {"select_primitive", py_sam_meta_select_primitive, METH_VARARGS, "Select growth primitive"},
    {"apply_primitive", py_sam_meta_apply_primitive, METH_VARARGS, "Apply growth primitive"},
    {"set_policy_params", py_sam_meta_set_policy_params, METH_VARARGS, "Set policy parameters"},
    {"get_policy_params", py_sam_meta_get_policy_params, METH_VARARGS, "Get policy parameters"},
    {"record_growth_outcome", py_sam_meta_record_growth_outcome, METH_VARARGS, "Record growth outcome"},
    {"set_identity_anchor", py_sam_meta_set_identity_anchor, METH_VARARGS, "Set identity anchor"},
    {"update_identity_vector", py_sam_meta_update_identity_vector, METH_VARARGS, "Update identity vector"},
    {"check_invariants", py_sam_meta_check_invariants, METH_VARARGS, "Check invariants"},
    {"get_invariant_state", py_sam_meta_get_invariant_state, METH_VARARGS, "Get invariant state"},
    {"evaluate_contract", py_sam_meta_evaluate_contract, METH_VARARGS, "Evaluate objective contract"},
    {"set_innocence", py_sam_meta_set_innocence, METH_VARARGS, "Set innocence gate parameters"},
    {"get_state", py_sam_meta_get_state, METH_VARARGS, "Get meta-controller state"},
    {"trigger_growth_evaluation", py_sam_meta_trigger_growth_evaluation, METH_VARARGS, "Trigger immediate growth evaluation"},
    {"get_growth_diagnostics", py_sam_meta_get_growth_diagnostics, METH_VARARGS, "Get growth diagnostic state"},
    {"get_effective_rank", py_sam_meta_get_effective_rank, METH_VARARGS, "Calculate effective rank of pressure signals"},
    {"get_submodel_lifecycle", py_sam_meta_get_submodel_lifecycle, METH_VARARGS, "Get submodel PDI-T lifecycle stage"},
    {"advance_submodel_lifecycle", py_sam_meta_advance_submodel_lifecycle, METH_VARARGS, "Advance submodel lifecycle"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef meta_module = {
    PyModuleDef_HEAD_INIT,
    "sam_meta_controller_c",
    "SAM Meta-Controller C Extension",
    -1,
    MetaMethods
};

PyMODINIT_FUNC PyInit_sam_meta_controller_c(void) {
    return PyModule_Create(&meta_module);
}
