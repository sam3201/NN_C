/*
 * SAM Meta-Controller - Pure C Core
 * Implements morphogenetic latency, pressure aggregation, growth primitives,
 * invariants, and objective contract evaluation.
 */

#ifndef SAM_META_CONTROLLER_C_H
#define SAM_META_CONTROLLER_C_H

#include <stddef.h>

typedef enum {
    GP_NONE = 0,
    GP_LATENT_EXPAND = 1,
    GP_SUBMODEL_SPAWN = 2,
    GP_INDEX_EXPAND = 3,
    GP_ROUTING_INCREASE = 4,
    GP_CONTEXT_EXPAND = 5,
    GP_PLANNER_WIDEN = 6,
    GP_CONSOLIDATE = 7,
    GP_REPARAM = 8
} GrowthPrimitive;

typedef struct SamMetaController SamMetaController;

SamMetaController *sam_meta_create(size_t latent_dim, size_t context_dim, size_t max_submodels, unsigned int seed);
void sam_meta_free(SamMetaController *mc);

// Pressure update and selection
double sam_meta_update_pressure(SamMetaController *mc,
                                double residual,
                                double rank_def,
                                double retrieval_entropy,
                                double interference,
                                double planner_friction,
                                double context_collapse,
                                double compression_waste,
                                double temporal_incoherence);

GrowthPrimitive sam_meta_select_primitive(SamMetaController *mc);
int sam_meta_apply_primitive(SamMetaController *mc, GrowthPrimitive primitive);

// Invariants and identity
void sam_meta_set_identity_anchor(SamMetaController *mc, const double *vec, size_t dim);
void sam_meta_update_identity_vector(SamMetaController *mc, const double *vec, size_t dim);
int sam_meta_check_invariants(SamMetaController *mc, double *out_identity_similarity);

// Objective contract evaluation (minimax)
int sam_meta_evaluate_contract(SamMetaController *mc,
                               double baseline_worst_case,
                               double proposed_worst_case);

// Query state
size_t sam_meta_get_latent_dim(const SamMetaController *mc);
size_t sam_meta_get_context_dim(const SamMetaController *mc);
size_t sam_meta_get_submodel_count(const SamMetaController *mc);
size_t sam_meta_get_index_count(const SamMetaController *mc);
size_t sam_meta_get_routing_degree(const SamMetaController *mc);
size_t sam_meta_get_planner_depth(const SamMetaController *mc);
double sam_meta_get_lambda(const SamMetaController *mc);
double sam_meta_get_growth_budget(const SamMetaController *mc);
size_t sam_meta_get_archived_dim(const SamMetaController *mc);

#endif // SAM_META_CONTROLLER_C_H
