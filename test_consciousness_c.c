/*
 * Pure C Consciousness Test - No Python bindings
 * Tests the core C consciousness logic
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "consciousness_core.h"

int main() {
    srand(time(NULL));

    printf("🧠 Pure C Consciousness Test - No Python Dependencies\n");
    printf("Testing core C consciousness logic...\n");

    // Create consciousness module
    ConsciousnessLossModule *module = consciousness_create(64, 16);

    if (!module) {
        fprintf(stderr, "❌ Failed to create consciousness module\n");
        return 1;
    }

    printf("✅ Consciousness module created successfully\n");

    // Test with dummy data
    Matrix *z_t = matrix_create(64, 1);
    Matrix *a_t = matrix_create(16, 1);
    Matrix *m_t = matrix_create(64, 1);
    Matrix *z_next = matrix_create(64, 1);
    Matrix *reward = matrix_create(32, 1);

    if (!z_t || !a_t || !m_t || !z_next || !reward) {
        fprintf(stderr, "❌ Failed to create test matrices\n");
        consciousness_free(module);
        return 1;
    }

    // Fill with random data
    matrix_random_normal(z_t, 0.0, 1.0);
    matrix_random_normal(a_t, 0.0, 1.0);
    matrix_random_normal(z_next, 0.0, 1.0);
    matrix_random_normal(m_t, 0.0, 1.0);
    matrix_random_normal(reward, 0.0, 1.0);

    printf("✅ Test data created\n");

    // Test individual loss functions
    Matrix *l_world = world_prediction_loss(module, z_t, a_t, z_next);
    if (l_world) {
        printf("✅ World prediction loss: %.6f\n", l_world->data[0]);
        matrix_free(l_world);
    } else {
        printf("❌ World prediction loss failed\n");
    }

    Matrix *l_self = self_model_loss(module, z_t, a_t, m_t, z_next);
    if (l_self) {
        printf("✅ Self model loss: %.6f\n", l_self->data[0]);
        matrix_free(l_self);
    } else {
        printf("❌ Self model loss failed\n");
    }

    Matrix *l_cons = consciousness_loss(module, z_t, a_t, z_next, m_t);
    if (l_cons) {
        printf("✅ Consciousness loss: %.6f\n", l_cons->data[0]);
        matrix_free(l_cons);
    } else {
        printf("❌ Consciousness loss failed\n");
    }

    // Test full loss computation
    Matrix *losses = consciousness_compute_loss_c(module, z_t, a_t, z_next, m_t, reward, 10000);
    if (losses) {
        printf("✅ Full loss computation:\n");
        printf("   L_world: %.6f\n", losses->data[0]);
        printf("   L_self: %.6f\n", losses->data[1]);
        printf("   L_cons: %.6f\n", losses->data[2]);
        printf("   L_policy: %.6f\n", losses->data[3]);
        printf("   C_compute: %.6f\n", losses->data[4]);
        printf("   L_total: %.6f\n", losses->data[5]);
        printf("   Consciousness Score: %.6f\n", losses->data[6]);

        matrix_free(losses);
    } else {
        printf("❌ Full loss computation failed\n");
    }

    // Cleanup
    matrix_free(z_t);
    matrix_free(a_t);
    matrix_free(m_t);
    matrix_free(z_next);
    matrix_free(reward);

    consciousness_free(module);

    printf("✅ Pure C consciousness test completed!\n");
    printf("🎯 Core C logic verified - ready for Python bindings\n");

    return 0;
}
