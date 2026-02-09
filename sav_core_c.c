/*
 * SAV Core - Pure C Adversarial System (Python Extension)
 */

#include "sav_core_c.h"

#include <Python.h>
#include <math.h>
#include <stdlib.h>

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

struct SavCore {
    double pressure;
    double adversarial_intensity;
    double termination_probability;
    size_t scenario_count;
    FastRng rng;
};

SavCore *sav_create(unsigned int seed) {
    SavCore *core = (SavCore *)calloc(1, sizeof(SavCore));
    if (!core) return NULL;
    core->pressure = 0.0;
    core->adversarial_intensity = 0.5;
    core->termination_probability = 0.0;
    core->scenario_count = 0;
    core->rng.state = seed ? seed : 0xBF58476D1CE4E5B9ULL;
    return core;
}

void sav_free(SavCore *core) {
    if (!core) return;
    free(core);
}

static double clamp(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

void sav_step(SavCore *core, double sam_survival, double sam_capability, double sam_efficiency) {
    if (!core) return;

    // Generate adversarial scenarios
    double scenario_entropy = rng_next_f64(&core->rng);
    double resource_starvation = (1.0 - sam_efficiency) * (0.3 + 0.7 * scenario_entropy);
    double edge_case = (sam_capability > 1.0 ? 0.2 : 0.4) * (0.5 + scenario_entropy);
    double overfit_attack = (sam_capability - sam_efficiency) * (0.4 + 0.6 * scenario_entropy);

    double pressure = resource_starvation + edge_case + overfit_attack;
    core->pressure = 0.9 * core->pressure + 0.1 * pressure;

    core->adversarial_intensity = clamp(0.5 + (1.0 - sam_survival) * 0.7, 0.3, 1.5);
    core->termination_probability = clamp(core->pressure * core->adversarial_intensity * (1.0 - sam_survival), 0.0, 1.0);

    core->scenario_count += 1;
}

double sav_get_pressure(const SavCore *core) { return core ? core->pressure : 0.0; }
double sav_get_termination_probability(const SavCore *core) { return core ? core->termination_probability : 0.0; }
double sav_get_adversarial_intensity(const SavCore *core) { return core ? core->adversarial_intensity : 0.0; }
size_t sav_get_scenario_count(const SavCore *core) { return core ? core->scenario_count : 0; }

// ================================
// PYTHON EXTENSION
// ================================

static void capsule_destructor(PyObject *capsule) {
    SavCore *core = (SavCore *)PyCapsule_GetPointer(capsule, "SavCore");
    sav_free(core);
}

static PyObject *py_create(PyObject *self, PyObject *args) {
    unsigned int seed = 0;
    if (!PyArg_ParseTuple(args, "I", &seed)) return NULL;
    SavCore *core = sav_create(seed);
    if (!core) return PyErr_NoMemory();
    return PyCapsule_New(core, "SavCore", capsule_destructor);
}

static PyObject *py_step(PyObject *self, PyObject *args) {
    PyObject *capsule = NULL;
    double survival, capability, efficiency;
    if (!PyArg_ParseTuple(args, "Oddd", &capsule, &survival, &capability, &efficiency)) return NULL;
    SavCore *core = (SavCore *)PyCapsule_GetPointer(capsule, "SavCore");
    sav_step(core, survival, capability, efficiency);
    Py_RETURN_NONE;
}

static PyObject *py_get_status(PyObject *self, PyObject *args) {
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SavCore *core = (SavCore *)PyCapsule_GetPointer(capsule, "SavCore");
    PyObject *dict = PyDict_New();
    PyDict_SetItemString(dict, "pressure", PyFloat_FromDouble(sav_get_pressure(core)));
    PyDict_SetItemString(dict, "termination_probability", PyFloat_FromDouble(sav_get_termination_probability(core)));
    PyDict_SetItemString(dict, "adversarial_intensity", PyFloat_FromDouble(sav_get_adversarial_intensity(core)));
    PyDict_SetItemString(dict, "scenario_count", PyLong_FromSize_t(sav_get_scenario_count(core)));
    return dict;
}

static PyMethodDef SavMethods[] = {
    {"create", py_create, METH_VARARGS, "Create SAV core"},
    {"step", py_step, METH_VARARGS, "Step SAV with SAM metrics"},
    {"get_status", py_get_status, METH_VARARGS, "Get SAV status"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef SavModule = {
    PyModuleDef_HEAD_INIT,
    "sav_core_c",
    "SAV Core C Extension",
    -1,
    SavMethods
};

PyMODINIT_FUNC PyInit_sav_core_c(void) {
    return PyModule_Create(&SavModule);
}
