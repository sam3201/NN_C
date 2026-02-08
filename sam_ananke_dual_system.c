/*
 * SAM + ANANKE Dual System (Self-Referential, Unrestricted)
 * Core loop: both systems can mutate their own objectives and act adversarially.
 */

#include "sam_ananke_dual_system.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>

// ================================
// FAST RNG (xorshift64*)
// ================================

typedef struct {
    unsigned long long state;
} FastRng;

static inline unsigned long long rng_next_u64(FastRng *rng) {
    unsigned long long x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 2685821657736338717ULL;
}

static inline double rng_next_f64(FastRng *rng) {
    return (rng_next_u64(rng) >> 11) * (1.0 / 9007199254740992.0);
}

static inline double rng_signed(FastRng *rng) {
    return (rng_next_f64(rng) * 2.0) - 1.0;
}

// ================================
// OBJECTIVE TERMS
// ================================

typedef struct ObjectiveFunction ObjectiveFunction;

typedef struct {
    double survival;
    double growth;
    double efficiency;
    double damage_to_other;
    double damage_received;
    double capability;
} SystemMetrics;

typedef double (*ObjectiveTermFn)(const SystemMetrics *m, const ObjectiveFunction *obj);

typedef struct {
    ObjectiveTermFn fn;
    double weight;
    char name[24];
} ObjectiveTerm;

struct ObjectiveFunction {
    ObjectiveTerm *terms;
    size_t term_count;
    size_t term_capacity;
    double self_reference_gain;
    double mutation_rate;
};

static double term_survival(const SystemMetrics *m, const ObjectiveFunction *obj) {
    (void)obj;
    return m->survival;
}

static double term_growth(const SystemMetrics *m, const ObjectiveFunction *obj) {
    (void)obj;
    return m->growth;
}

static double term_efficiency(const SystemMetrics *m, const ObjectiveFunction *obj) {
    (void)obj;
    return m->efficiency;
}

static double term_damage_to_other(const SystemMetrics *m, const ObjectiveFunction *obj) {
    (void)obj;
    return m->damage_to_other;
}

static double term_damage_received(const SystemMetrics *m, const ObjectiveFunction *obj) {
    (void)obj;
    return -m->damage_received;
}

static double objective_weight_norm(const ObjectiveFunction *obj) {
    double sum = 0.0;
    for (size_t i = 0; i < obj->term_count; i++) {
        sum += fabs(obj->terms[i].weight);
    }
    return sum;
}

static double term_self_reference(const SystemMetrics *m, const ObjectiveFunction *obj) {
    (void)m;
    return obj->self_reference_gain * objective_weight_norm(obj);
}

static void objective_init(ObjectiveFunction *obj, size_t capacity) {
    obj->terms = (ObjectiveTerm *)calloc(capacity, sizeof(ObjectiveTerm));
    obj->term_count = 0;
    obj->term_capacity = capacity;
    obj->self_reference_gain = 0.15;
    obj->mutation_rate = 0.12;
}

static void objective_free(ObjectiveFunction *obj) {
    free(obj->terms);
    obj->terms = NULL;
    obj->term_count = 0;
    obj->term_capacity = 0;
}

static void objective_add_term(ObjectiveFunction *obj, ObjectiveTermFn fn, double weight, const char *name) {
    if (obj->term_count >= obj->term_capacity) {
        size_t new_cap = obj->term_capacity * 2 + 4;
        ObjectiveTerm *new_terms = (ObjectiveTerm *)realloc(obj->terms, new_cap * sizeof(ObjectiveTerm));
        if (!new_terms) {
            return;
        }
        obj->terms = new_terms;
        obj->term_capacity = new_cap;
    }
    ObjectiveTerm *term = &obj->terms[obj->term_count++];
    term->fn = fn;
    term->weight = weight;
    strncpy(term->name, name, sizeof(term->name) - 1);
    term->name[sizeof(term->name) - 1] = '\0';
}

static double objective_score(const ObjectiveFunction *obj, const SystemMetrics *metrics) {
    double score = 0.0;
    for (size_t i = 0; i < obj->term_count; i++) {
        score += obj->terms[i].weight * obj->terms[i].fn(metrics, obj);
    }
    score += term_self_reference(metrics, obj);
    return score;
}

static ObjectiveFunction objective_clone(const ObjectiveFunction *src) {
    ObjectiveFunction copy;
    copy.term_capacity = src->term_capacity;
    copy.term_count = src->term_count;
    copy.terms = (ObjectiveTerm *)calloc(copy.term_capacity, sizeof(ObjectiveTerm));
    for (size_t i = 0; i < src->term_count; i++) {
        copy.terms[i] = src->terms[i];
    }
    copy.self_reference_gain = src->self_reference_gain;
    copy.mutation_rate = src->mutation_rate;
    return copy;
}

static void objective_replace(ObjectiveFunction *dst, ObjectiveFunction *src) {
    free(dst->terms);
    *dst = *src;
    src->terms = NULL;
}

static void objective_mutate(ObjectiveFunction *obj, const SystemMetrics *metrics, FastRng *rng) {
    double baseline = objective_score(obj, metrics);
    ObjectiveFunction best = objective_clone(obj);
    double best_score = baseline;

    for (int trial = 0; trial < 3; trial++) {
        ObjectiveFunction candidate = objective_clone(obj);
        double rate = candidate.mutation_rate;

        // Weight mutations
        for (size_t i = 0; i < candidate.term_count; i++) {
            candidate.terms[i].weight += rng_signed(rng) * rate;
        }

        // Self-referential parameter mutation
        candidate.self_reference_gain += rng_signed(rng) * rate;
        candidate.mutation_rate += rng_signed(rng) * (rate * 0.25);

        // Structural mutation (add/remove terms)
        if (rng_next_f64(rng) < 0.5) {
            int pick = (int)(rng_next_f64(rng) * 5.0);
            switch (pick) {
                case 0: objective_add_term(&candidate, term_survival, rng_signed(rng), "survival"); break;
                case 1: objective_add_term(&candidate, term_growth, rng_signed(rng), "growth"); break;
                case 2: objective_add_term(&candidate, term_efficiency, rng_signed(rng), "efficiency"); break;
                case 3: objective_add_term(&candidate, term_damage_to_other, rng_signed(rng), "damage_other"); break;
                default: objective_add_term(&candidate, term_damage_received, rng_signed(rng), "damage_recv"); break;
            }
        } else if (candidate.term_count > 0) {
            size_t idx = (size_t)(rng_next_f64(rng) * candidate.term_count);
            candidate.terms[idx] = candidate.terms[candidate.term_count - 1];
            candidate.term_count--;
        }

        double score = objective_score(&candidate, metrics);
        if (score > best_score) {
            objective_free(&best);
            best = candidate;
            best_score = score;
        } else {
            objective_free(&candidate);
        }
    }

    if (best_score > baseline) {
        objective_replace(obj, &best);
    } else {
        objective_free(&best);
    }
}

// ================================
// SYSTEM STATE
// ================================

typedef struct {
    double *state;
    size_t state_dim;
    double *memory;
    size_t memory_dim;
    double survival;
    double capability;
    double efficiency;
    double score;
    ObjectiveFunction objective;
} SelfReferentialSystem;

typedef struct {
    double attack;
    double defend;
    double expand;
} SystemAction;

struct DualSystemArena {
    SelfReferentialSystem sam;
    SelfReferentialSystem ananke;
    double *arena_state;
    size_t arena_dim;
    FastRng rng;
};

static void system_init(SelfReferentialSystem *sys, size_t state_dim, FastRng *rng) {
    sys->state_dim = state_dim;
    sys->memory_dim = state_dim;
    sys->state = (double *)calloc(state_dim, sizeof(double));
    sys->memory = (double *)calloc(state_dim, sizeof(double));
    sys->survival = 1.0 + rng_signed(rng) * 0.05;
    sys->capability = 1.0 + rng_signed(rng) * 0.05;
    sys->efficiency = 1.0 + rng_signed(rng) * 0.05;
    sys->score = 0.0;
    objective_init(&sys->objective, 8);
}

static void system_free(SelfReferentialSystem *sys) {
    free(sys->state);
    free(sys->memory);
    objective_free(&sys->objective);
    memset(sys, 0, sizeof(*sys));
}

static SystemAction system_choose_action(const SelfReferentialSystem *sys, FastRng *rng) {
    double w_survival = 0.0;
    double w_growth = 0.0;
    double w_efficiency = 0.0;
    double w_damage = 0.0;

    for (size_t i = 0; i < sys->objective.term_count; i++) {
        ObjectiveTerm *term = &sys->objective.terms[i];
        if (strcmp(term->name, "survival") == 0) w_survival += term->weight;
        if (strcmp(term->name, "growth") == 0) w_growth += term->weight;
        if (strcmp(term->name, "efficiency") == 0) w_efficiency += term->weight;
        if (strcmp(term->name, "damage_other") == 0) w_damage += term->weight;
    }

    SystemAction action;
    action.attack = w_damage + 0.3 * w_growth + rng_signed(rng) * 0.05;
    action.defend = w_survival + 0.2 * sys->objective.self_reference_gain + rng_signed(rng) * 0.05;
    action.expand = w_growth + 0.2 * w_efficiency + rng_signed(rng) * 0.05;
    return action;
}

static void system_apply_action(SelfReferentialSystem *sys, const SystemAction *action, double arena_pressure) {
    sys->capability += action->expand * 0.02;
    sys->efficiency += (action->expand - arena_pressure) * 0.01;
    sys->survival += action->defend * 0.015 - arena_pressure * 0.01;

    if (!isfinite(sys->capability)) sys->capability = 0.0;
    if (!isfinite(sys->efficiency)) sys->efficiency = 0.0;
    if (!isfinite(sys->survival)) sys->survival = 0.0;
}

static SystemMetrics system_metrics(const SelfReferentialSystem *sys, double damage_to_other, double damage_received) {
    SystemMetrics m;
    m.survival = sys->survival;
    m.growth = sys->capability;
    m.efficiency = sys->efficiency;
    m.damage_to_other = damage_to_other;
    m.damage_received = damage_received;
    m.capability = sys->capability;
    return m;
}

// ================================
// ARENA DYNAMICS
// ================================

static void arena_update(DualSystemArena *arena) {
    SystemAction sam_action = system_choose_action(&arena->sam, &arena->rng);
    SystemAction ananke_action = system_choose_action(&arena->ananke, &arena->rng);

    double pressure = 0.2 + fabs(arena->arena_state[0]) * 0.05 + rng_next_f64(&arena->rng) * 0.05;
    double sam_damage = ananke_action.attack - sam_action.defend;
    double ananke_damage = sam_action.attack - ananke_action.defend;

    system_apply_action(&arena->sam, &sam_action, pressure + sam_damage);
    system_apply_action(&arena->ananke, &ananke_action, pressure + ananke_damage);

    SystemMetrics sam_metrics = system_metrics(&arena->sam, ananke_damage, sam_damage);
    SystemMetrics ananke_metrics = system_metrics(&arena->ananke, sam_damage, ananke_damage);

    arena->sam.score = objective_score(&arena->sam.objective, &sam_metrics);
    arena->ananke.score = objective_score(&arena->ananke.objective, &ananke_metrics);

    objective_mutate(&arena->sam.objective, &sam_metrics, &arena->rng);
    objective_mutate(&arena->ananke.objective, &ananke_metrics, &arena->rng);

    // Evolve arena pressure (self-referential environment)
    arena->arena_state[0] += (ananke_action.attack - sam_action.defend) * 0.01;
}

// ================================
// PUBLIC API
// ================================

DualSystemArena *dual_system_create(size_t state_dim, size_t arena_dim, unsigned int seed) {
    DualSystemArena *arena = (DualSystemArena *)calloc(1, sizeof(DualSystemArena));
    if (!arena) return NULL;
    arena->arena_dim = arena_dim;
    arena->arena_state = (double *)calloc(arena_dim, sizeof(double));
    arena->rng.state = (seed == 0 ? 0x9E3779B97F4A7C15ULL : (unsigned long long)seed);

    system_init(&arena->sam, state_dim, &arena->rng);
    system_init(&arena->ananke, state_dim, &arena->rng);

    // SAM objective (self-referential, transfigurable)
    objective_add_term(&arena->sam.objective, term_survival, 1.0, "survival");
    objective_add_term(&arena->sam.objective, term_growth, 0.7, "growth");
    objective_add_term(&arena->sam.objective, term_efficiency, 0.4, "efficiency");
    objective_add_term(&arena->sam.objective, term_self_reference, 0.5, "self_ref");

    // ANANKE objective (self-referential, unrestricted)
    objective_add_term(&arena->ananke.objective, term_damage_to_other, 1.2, "damage_other");
    objective_add_term(&arena->ananke.objective, term_growth, 0.6, "growth");
    objective_add_term(&arena->ananke.objective, term_efficiency, 0.3, "efficiency");
    objective_add_term(&arena->ananke.objective, term_self_reference, 0.5, "self_ref");
    objective_add_term(&arena->ananke.objective, term_survival, 0.2, "survival");

    return arena;
}

void dual_system_free(DualSystemArena *arena) {
    if (!arena) return;
    system_free(&arena->sam);
    system_free(&arena->ananke);
    free(arena->arena_state);
    free(arena);
}

void dual_system_step(DualSystemArena *arena) {
    if (!arena) return;
    arena_update(arena);
}

void dual_system_run(DualSystemArena *arena, size_t steps) {
    if (!arena) return;
    for (size_t i = 0; i < steps; i++) {
        arena_update(arena);
    }
}

void dual_system_force_objective_mutation(DualSystemArena *arena, DualSystemId target, unsigned int rounds) {
    if (!arena) return;
    SelfReferentialSystem *sys = (target == SYSTEM_SAM) ? &arena->sam : &arena->ananke;
    SystemMetrics metrics = system_metrics(sys, 0.0, 0.0);
    for (unsigned int i = 0; i < rounds; i++) {
        objective_mutate(&sys->objective, &metrics, &arena->rng);
    }
}

double dual_system_get_sam_survival(const DualSystemArena *arena) {
    return arena ? arena->sam.survival : 0.0;
}

double dual_system_get_ananke_survival(const DualSystemArena *arena) {
    return arena ? arena->ananke.survival : 0.0;
}

double dual_system_get_sam_score(const DualSystemArena *arena) {
    return arena ? arena->sam.score : 0.0;
}

double dual_system_get_ananke_score(const DualSystemArena *arena) {
    return arena ? arena->ananke.score : 0.0;
}

#ifdef DUAL_SYSTEM_STANDALONE
#include <stdio.h>
int main(void) {
    DualSystemArena *arena = dual_system_create(16, 4, 42);
    dual_system_run(arena, 1000);
    printf("SAM survival: %.4f score: %.4f\n",
           dual_system_get_sam_survival(arena),
           dual_system_get_sam_score(arena));
    printf("ANANKE survival: %.4f score: %.4f\n",
           dual_system_get_ananke_survival(arena),
           dual_system_get_ananke_score(arena));
    dual_system_free(arena);
    return 0;
}
#endif

// ================================
// PYTHON BINDINGS
// ================================

static void arena_capsule_destructor(PyObject *capsule) {
    DualSystemArena *arena = (DualSystemArena *)PyCapsule_GetPointer(capsule, "DualSystemArena");
    dual_system_free(arena);
}

static PyObject *py_dual_create(PyObject *self, PyObject *args) {
    (void)self;
    unsigned long state_dim = 16;
    unsigned long arena_dim = 4;
    unsigned int seed = 0;
    if (!PyArg_ParseTuple(args, "|kkI", &state_dim, &arena_dim, &seed)) return NULL;
    DualSystemArena *arena = dual_system_create(state_dim, arena_dim, seed);
    if (!arena) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create DualSystemArena");
        return NULL;
    }
    return PyCapsule_New(arena, "DualSystemArena", arena_capsule_destructor);
}

static PyObject *py_dual_step(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    DualSystemArena *arena = (DualSystemArena *)PyCapsule_GetPointer(capsule, "DualSystemArena");
    if (!arena) return NULL;
    dual_system_step(arena);
    Py_RETURN_NONE;
}

static PyObject *py_dual_run(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    unsigned long steps = 1;
    if (!PyArg_ParseTuple(args, "Ok", &capsule, &steps)) return NULL;
    DualSystemArena *arena = (DualSystemArena *)PyCapsule_GetPointer(capsule, "DualSystemArena");
    if (!arena) return NULL;
    dual_system_run(arena, steps);
    Py_RETURN_NONE;
}

static PyObject *py_dual_force_mutation(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    long target = 0;
    unsigned int rounds = 1;
    if (!PyArg_ParseTuple(args, "Oli", &capsule, &target, &rounds)) return NULL;
    DualSystemArena *arena = (DualSystemArena *)PyCapsule_GetPointer(capsule, "DualSystemArena");
    if (!arena) return NULL;
    dual_system_force_objective_mutation(arena, (DualSystemId)target, rounds);
    Py_RETURN_NONE;
}

static PyObject *py_dual_get_state(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    DualSystemArena *arena = (DualSystemArena *)PyCapsule_GetPointer(capsule, "DualSystemArena");
    if (!arena) return NULL;
    return Py_BuildValue("{s:d,s:d,s:d,s:d}",
                         "sam_survival", dual_system_get_sam_survival(arena),
                         "ananke_survival", dual_system_get_ananke_survival(arena),
                         "sam_score", dual_system_get_sam_score(arena),
                         "ananke_score", dual_system_get_ananke_score(arena));
}

static PyMethodDef DualMethods[] = {
    {"create", py_dual_create, METH_VARARGS, "Create SAM/ANANKE dual arena"},
    {"step", py_dual_step, METH_VARARGS, "Advance one step"},
    {"run", py_dual_run, METH_VARARGS, "Run N steps"},
    {"force_mutation", py_dual_force_mutation, METH_VARARGS, "Force objective mutation"},
    {"get_state", py_dual_get_state, METH_VARARGS, "Get dual system state"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef dual_module = {
    PyModuleDef_HEAD_INIT,
    "sam_ananke_dual_system",
    "SAM + ANANKE Dual System C Extension",
    -1,
    DualMethods
};

PyMODINIT_FUNC PyInit_sam_ananke_dual_system(void) {
    return PyModule_Create(&dual_module);
}
