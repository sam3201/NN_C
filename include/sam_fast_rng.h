/*
 * Fast RNG - xorshift64* Implementation
 * Deterministic, high-performance random number generation
 * No external dependencies - pure C with standard library only
 */

#ifndef SAM_FAST_RNG_H
#define SAM_FAST_RNG_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t state;
} SamFastRNG;

// Initialize with seed (seed=0 becomes 1)
void sam_rng_init(SamFastRNG *rng, uint64_t seed);

// Generate next uint64
uint64_t sam_rng_next(SamFastRNG *rng);

// Random in [0, max)
uint64_t sam_rng_range(SamFastRNG *rng, uint64_t max);

// Random double in [0, 1)
double sam_rng_double(SamFastRNG *rng);

// Random double in [min, max)
double sam_rng_double_range(SamFastRNG *rng, double min, double max);

// Gaussian/Normal distribution (Box-Muller transform)
double sam_rng_gaussian(SamFastRNG *rng, double mean, double stddev);

// Bernoulli trial - returns 1 with probability p
int sam_rng_bernoulli(SamFastRNG *rng, double p);

// Shuffle array in-place (Fisher-Yates)
void sam_rng_shuffle(SamFastRNG *rng, void *arr, size_t n, size_t elem_size);

// Get seed from system time
uint64_t sam_rng_seed_from_time(void);

#ifdef __cplusplus
}
#endif

#endif // SAM_FAST_RNG_H
