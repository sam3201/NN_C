/*
 * SAM + SAV Dual System (Self-Referential, Unrestricted)
 * Pure C implementation optimized for speed and direct control.
 */

#ifndef SAM_SAV_DUAL_SYSTEM_H
#define SAM_SAV_DUAL_SYSTEM_H

#include <stddef.h>

typedef enum {
    SYSTEM_SAM = 0,
    SYSTEM_SAV = 1
} DualSystemId;

typedef struct DualSystemArena DualSystemArena;

// Create and destroy arena
DualSystemArena *dual_system_create(size_t state_dim, size_t arena_dim, unsigned int seed);
void dual_system_free(DualSystemArena *arena);

// Single step or run for N steps
void dual_system_step(DualSystemArena *arena);
void dual_system_run(DualSystemArena *arena, size_t steps);

// External control hooks (self-referential objective mutation)
void dual_system_force_objective_mutation(DualSystemArena *arena, DualSystemId target, unsigned int rounds);

// Lightweight telemetry
double dual_system_get_sam_survival(const DualSystemArena *arena);
double dual_system_get_sav_survival(const DualSystemArena *arena);
double dual_system_get_sam_score(const DualSystemArena *arena);
double dual_system_get_sav_score(const DualSystemArena *arena);

#endif // SAM_SAV_DUAL_SYSTEM_H
