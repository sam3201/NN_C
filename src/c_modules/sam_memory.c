#include "sam_memory.h"
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint64_t xorshift64star(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

void sam_memory_init(SamMemory* mem, uint64_t seed) {
    memset(mem, 0, sizeof(SamMemory));
    mem->next_episode_id = seed;
    mem->next_semantic_id = seed;
    mem->recency_weight = 0.3f;
    mem->salience_weight = 0.4f;
    mem->similarity_threshold = 0.7f;
}

void sam_memory_destroy(SamMemory* mem) {
    for (int i = 0; i < mem->episode_count; i++) {
        if (mem->episodes[i].summary) {
            free(mem->episodes[i].summary);
        }
    }
    for (int i = 0; i < mem->semantic_count; i++) {
        if (mem->semantic[i].concept) {
            free(mem->semantic[i].concept);
        }
    }
}

float sam_memory_compute_similarity(float* a, float* b, int dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

int sam_memory_store_episode(SamMemory* mem, float* embedding, float salience, const char* summary) {
    if (mem->episode_count >= MAX_EPISODES) {
        mem->episode_index = (mem->episode_index + 1) % MAX_EPISODES;
    } else {
        mem->episode_index = mem->episode_count;
        mem->episode_count++;
    }
    
    Episode* ep = &mem->episodes[mem->episode_index];
    ep->id = mem->next_episode_id++;
    ep->timestamp = (float)(ep->id) / 1000.0f;
    ep->salience = salience;
    ep->encoded = 0;
    
    if (embedding) {
        memcpy(ep->embedding, embedding, EMBEDDING_DIM * sizeof(float));
    } else {
        memset(ep->embedding, 0, EMBEDDING_DIM * sizeof(float));
    }
    
    if (summary) {
        ep->summary = malloc(strlen(summary) + 1);
        strcpy(ep->summary, summary);
    } else {
        ep->summary = NULL;
    }
    
    return (int)ep->id;
}

int sam_memory_retrieve_episodes(SamMemory* mem, float* query_embedding, float* results, int max_results) {
    if (!query_embedding || max_results <= 0) return 0;
    
    float scores[MAX_EPISODES];
    int indices[MAX_EPISODES];
    
    for (int i = 0; i < mem->episode_count; i++) {
        Episode* ep = &mem->episodes[i];
        float sim = sam_memory_compute_similarity(query_embedding, ep->embedding, EMBEDDING_DIM);
        
        float recency = 1.0f / (1.0f + fabsf(ep->timestamp - (float)(mem->next_episode_id) / 1000.0f));
        scores[i] = mem->salience_weight * ep->salience + mem->recency_weight * recency + 0.3f * sim;
        indices[i] = i;
    }
    
    for (int i = 0; i < mem->episode_count - 1; i++) {
        for (int j = i + 1; j < mem->episode_count; j++) {
            if (scores[j] > scores[i]) {
                float temp_s = scores[i];
                scores[i] = scores[j];
                scores[j] = temp_s;
                int temp_i = indices[i];
                indices[i] = indices[j];
                indices[j] = temp_i;
            }
        }
    }
    
    int count = mem->episode_count < max_results ? mem->episode_count : max_results;
    for (int i = 0; i < count; i++) {
        results[i] = scores[i];
    }
    
    return count;
}

int sam_memory_store_semantic(SamMemory* mem, const char* concept, float* embedding, float strength) {
    if (!concept || mem->semantic_count >= MAX_SEMANTIC_ENTRIES) return -1;
    
    SemanticEntry* entry = &mem->semantic[mem->semantic_count];
    entry->id = mem->next_semantic_id++;
    entry->strength = strength;
    entry->num_refs = 0;
    
    entry->concept = malloc(strlen(concept) + 1);
    strcpy(entry->concept, concept);
    
    if (embedding) {
        memcpy(entry->embedding, embedding, EMBEDDING_DIM * sizeof(float));
    } else {
        memset(entry->embedding, 0, EMBEDDING_DIM * sizeof(float));
    }
    
    mem->semantic_count++;
    return (int)entry->id;
}

int sam_memory_recall_semantic(SamMemory* mem, const char* concept, float* embedding) {
    if (!concept) return -1;
    
    for (int i = 0; i < mem->semantic_count; i++) {
        if (mem->semantic[i].concept && strcmp(mem->semantic[i].concept, concept) == 0) {
            if (embedding) {
                memcpy(embedding, mem->semantic[i].embedding, EMBEDDING_DIM * sizeof(float));
            }
            return (int)mem->semantic[i].strength;
        }
    }
    return -1;
}

void sam_memory_consolidate(SamMemory* mem, float threshold) {
    for (int i = 0; i < mem->semantic_count; i++) {
        if (mem->semantic[i].strength < threshold) {
            if (mem->semantic[i].concept) {
                free(mem->semantic[i].concept);
                mem->semantic[i].concept = NULL;
            }
            mem->semantic[i].strength = 0.0f;
        }
    }
}

void sam_memory_get_stats(SamMemory* mem, int* episode_count, int* semantic_count) {
    if (episode_count) *episode_count = mem->episode_count;
    if (semantic_count) *semantic_count = mem->semantic_count;
}

static SamMemory g_memory;
static int g_initialized = 0;

void SamMemory_init(uint64_t seed) {
    sam_memory_init(&g_memory, seed);
    g_initialized = 1;
}

void SamMemory_destroy() {
    if (g_initialized) {
        sam_memory_destroy(&g_memory);
        g_initialized = 0;
    }
}

int SamMemory_store_episode(float* embedding, float salience, const char* summary) {
    if (!g_initialized) SamMemory_init(42);
    return sam_memory_store_episode(&g_memory, embedding, salience, summary);
}

int SamMemory_retrieve(float* query, float* results, int max_results) {
    if (!g_initialized) return 0;
    return sam_memory_retrieve_episodes(&g_memory, query, results, max_results);
}

int SamMemory_store_semantic(const char* concept, float* embedding, float strength) {
    if (!g_initialized) SamMemory_init(42);
    return sam_memory_store_semantic(&g_memory, concept, embedding, strength);
}

int SamMemory_recall(const char* concept, float* embedding) {
    if (!g_initialized) return -1;
    return sam_memory_recall_semantic(&g_memory, concept, embedding);
}

void SamMemory_consolidate(float threshold) {
    if (!g_initialized) return;
    sam_memory_consolidate(&g_memory, threshold);
}

void SamMemory_get_stats(int* episode_count, int* semantic_count) {
    if (!g_initialized) {
        if (episode_count) *episode_count = 0;
        if (semantic_count) *semantic_count = 0;
        return;
    }
    sam_memory_get_stats(&g_memory, episode_count, semantic_count);
}

static PyObject* py_SamMemory_init(PyObject* self, PyObject* args) {
    uint64_t seed;
    if (!PyArg_ParseTuple(args, "K", &seed)) return NULL;
    SamMemory_init(seed);
    Py_RETURN_NONE;
}

static PyObject* py_SamMemory_destroy(PyObject* self, PyObject* args) {
    SamMemory_destroy();
    Py_RETURN_NONE;
}

static PyObject* py_SamMemory_store_episode(PyObject* self, PyObject* args) {
    PyObject* embedding_obj;
    float salience;
    const char* summary = NULL;
    
    if (!PyArg_ParseTuple(args, "Of|s", &embedding_obj, &salience, &summary)) return NULL;
    
    float embedding[EMBEDDING_DIM];
    PyObject* item;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        item = PyList_GetItem(embedding_obj, i);
        embedding[i] = (float)PyFloat_AsDouble(item);
    }
    
    int result = SamMemory_store_episode(embedding, salience, summary);
    return PyLong_FromLong(result);
}

static PyObject* py_SamMemory_retrieve(PyObject* self, PyObject* args) {
    PyObject* query_obj;
    int max_results;
    
    if (!PyArg_ParseTuple(args, "Oi", &query_obj, &max_results)) return NULL;
    
    float query[EMBEDDING_DIM];
    PyObject* item;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        item = PyList_GetItem(query_obj, i);
        query[i] = (float)PyFloat_AsDouble(item);
    }
    
    float* results = malloc(max_results * sizeof(float));
    int count = SamMemory_retrieve(query, results, max_results);
    
    PyObject* list = PyList_New(count);
    for (int i = 0; i < count; i++) {
        PyList_SetItem(list, i, PyFloat_FromDouble(results[i]));
    }
    free(results);
    return list;
}

static PyObject* py_SamMemory_store_semantic(PyObject* self, PyObject* args) {
    const char* concept;
    PyObject* embedding_obj;
    float strength;
    
    if (!PyArg_ParseTuple(args, "sOf", &concept, &embedding_obj, &strength)) return NULL;
    
    float embedding[EMBEDDING_DIM];
    PyObject* item;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        item = PyList_GetItem(embedding_obj, i);
        embedding[i] = (float)PyFloat_AsDouble(item);
    }
    
    int result = SamMemory_store_semantic(concept, embedding, strength);
    return PyLong_FromLong(result);
}

static PyObject* py_SamMemory_recall(PyObject* self, PyObject* args) {
    const char* concept;
    
    if (!PyArg_ParseTuple(args, "s", &concept)) return NULL;
    
    float embedding[EMBEDDING_DIM];
    int strength = SamMemory_recall(concept, embedding);
    
    if (strength < 0) {
        Py_RETURN_NONE;
    }
    
    PyObject* list = PyList_New(EMBEDDING_DIM);
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        PyList_SetItem(list, i, PyFloat_FromDouble(embedding[i]));
    }
    return list;
}

static PyObject* py_SamMemory_consolidate(PyObject* self, PyObject* args) {
    float threshold;
    if (!PyArg_ParseTuple(args, "f", &threshold)) return NULL;
    SamMemory_consolidate(threshold);
    Py_RETURN_NONE;
}

static PyObject* py_SamMemory_get_stats(PyObject* self, PyObject* args) {
    int ep_count, sem_count;
    SamMemory_get_stats(&ep_count, &sem_count);
    return Py_BuildValue("(ii)", ep_count, sem_count);
}

static PyMethodDef MemoryMethods[] = {
    {"SamMemory_init", py_SamMemory_init, METH_VARARGS, "Initialize memory system"},
    {"SamMemory_destroy", py_SamMemory_destroy, METH_VARARGS, "Destroy memory system"},
    {"SamMemory_store_episode", py_SamMemory_store_episode, METH_VARARGS, "Store episode"},
    {"SamMemory_retrieve", py_SamMemory_retrieve, METH_VARARGS, "Retrieve episodes"},
    {"SamMemory_store_semantic", py_SamMemory_store_semantic, METH_VARARGS, "Store semantic"},
    {"SamMemory_recall", py_SamMemory_recall, METH_VARARGS, "Recall semantic"},
    {"SamMemory_consolidate", py_SamMemory_consolidate, METH_VARARGS, "Consolidate memory"},
    {"SamMemory_get_stats", py_SamMemory_get_stats, METH_VARARGS, "Get memory stats"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef sam_memory_module = {
    PyModuleDef_HEAD_INIT,
    "sam_memory",
    "SAM Memory - Episodic and Semantic Memory",
    -1,
    MemoryMethods
};

PyMODINIT_FUNC PyInit_sam_memory(void) {
    return PyModule_Create(&sam_memory_module);
}
