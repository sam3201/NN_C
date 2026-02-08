/*
 * ANANKE Core - Pure C Adversarial System
 */

#ifndef ANANKE_CORE_C_H
#define ANANKE_CORE_C_H

#include <stddef.h>

typedef struct AnankeCore AnankeCore;

AnankeCore *ananke_create(unsigned int seed);
void ananke_free(AnankeCore *core);

// Step with SAM metrics (survival, capability, efficiency)
void ananke_step(AnankeCore *core, double sam_survival, double sam_capability, double sam_efficiency);

// Query status
double ananke_get_pressure(const AnankeCore *core);
double ananke_get_termination_probability(const AnankeCore *core);
double ananke_get_adversarial_intensity(const AnankeCore *core);
size_t ananke_get_scenario_count(const AnankeCore *core);

#endif // ANANKE_CORE_C_H
