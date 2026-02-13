#ifndef SAM_MEMORY_H
#define SAM_MEMORY_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_EPISODES 1000
#define MAX_SEMANTIC_ENTRIES 500
#define EMBEDDING_DIM 64

typedef struct {
    uint64_t id;
    float timestamp;
    float embedding[EMBEDDING_DIM];
    float salience;
    uint8_t encoded;
    char* summary;
} Episode;

typedef struct {
    uint64_t id;
    char* concept;
    float embedding[EMBEDDING_DIM];
    float strength;
    uint64_t episode_refs[10];
    int num_refs;
} SemanticEntry;

typedef struct {
    Episode episodes[MAX_EPISODES];
    int episode_count;
    int episode_index;
    
    SemanticEntry semantic[MAX_SEMANTIC_ENTRIES];
    int semantic_count;
    
    uint64_t next_episode_id;
    uint64_t next_semantic_id;
    
    float recency_weight;
    float salience_weight;
    float similarity_threshold;
} SamMemory;

void sam_memory_init(SamMemory* mem, uint64_t seed);
void sam_memory_destroy(SamMemory* mem);

int sam_memory_store_episode(SamMemory* mem, float* embedding, float salience, const char* summary);
int sam_memory_retrieve_episodes(SamMemory* mem, float* query_embedding, float* results, int max_results);
int sam_memory_store_semantic(SamMemory* mem, const char* concept, float* embedding, float strength);
int sam_memory_recall_semantic(SamMemory* mem, const char* concept, float* embedding);
void sam_memory_consolidate(SamMemory* mem, float threshold);

float sam_memory_compute_similarity(float* a, float* b, int dim);
void sam_memory_get_stats(SamMemory* mem, int* episode_count, int* semantic_count);

#ifdef __cplusplus
}
#endif

#endif
