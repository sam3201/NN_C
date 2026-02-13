#ifndef SAM_REGULATOR_C_H
#define SAM_REGULATOR_C_H

struct SAMRegulator;
typedef struct SAMRegulator SAMRegulator;

SAMRegulator* sam_regulator_create();
void sam_regulator_update(SAMRegulator* reg, double* m_vec, double dt);
void sam_regulator_mutate(SAMRegulator* reg, double survival_score);
double* sam_regulator_get_state(SAMRegulator* reg);
void sam_regulator_destroy(SAMRegulator* reg);

#endif
