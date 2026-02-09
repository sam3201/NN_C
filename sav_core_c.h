/*
 * SAV Core - Pure C Adversarial System
 */

#ifndef SAV_CORE_C_H
#define SAV_CORE_C_H

#include <stddef.h>

typedef struct SavCore SavCore;

SavCore *sav_create(unsigned int seed);
void sav_free(SavCore *core);

// Step with SAM metrics (survival, capability, efficiency)
void sav_step(SavCore *core, double sam_survival, double sam_capability, double sam_efficiency);

// Query status
double sav_get_pressure(const SavCore *core);
double sav_get_termination_probability(const SavCore *core);
double sav_get_adversarial_intensity(const SavCore *core);
size_t sav_get_scenario_count(const SavCore *core);

#endif // SAV_CORE_C_H
