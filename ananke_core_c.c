/*
 * ANANKE Core - Pure C Adversarial System (Python Extension)
 */

#include "ananke_core_c.h"

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

struct AnankeCore {
    double pressure;
    double adversarial_intensity;
    double termination_probability;
    size_t scenario_count;
    FastRng rng;
};

AnankeCore *ananke_create(unsigned int seed) {
    AnankeCore *core = (AnankeCore *)calloc(1, sizeof(AnankeCore));
    if (!core) return NULL;
    core->pressure = 0.0;
    core->adversarial_intensity = 0.5;
    core->termination_probability = 0.0;
    core->scenario_count = 0;
    core->rng.state = seed ? seed : 0xBF58476D1CE4E5B9ULL;
    return core;
}

void ananke_free(AnankeCore *core) {
    if (!core) return;
    free(core);
}

static double clamp(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

void ananke_step(AnankeCore *core, double sam_survival, double sam_capability, double sam_efficiency) {
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

double ananke_get_pressure(const AnankeCore *core) { return core ? core->pressure : 0.0; }
double ananke_get_termination_probability(const AnankeCore *core) { return core ? core->termination_probability : 0.0; }
double ananke_get_adversarial_intensity(const AnankeCore *core) { return core ? core->adversarial_intensity : 0.0; }
size_t ananke_get_scenario_count(const AnankeCore *core) { return core ? core->scenario_count : 0; }

// ================================
// PYTHON EXTENSION
// ================================

static void capsule_destructor(PyObject *capsule) {
    AnankeCore *core = (AnankeCore *)PyCapsule_GetPointer(capsule, "AnankeCore");
    ananke_free(core);
}

static PyObject *py_create(PyObject *self, PyObject *args) {
    unsigned int seed = 0;
    if (!PyArg_ParseTuple(args, "I", &seed)) return NULL;
    AnankeCore *core = ananke_create(seed);
    if (!core) return PyErr_NoMemory();
    return PyCapsule_New(core, "AnankeCore", capsule_destructor);
}

static PyObject *py_step(PyObject *self, PyObject *args) {
    PyObject *capsule = NULL;
    double survival, capability, efficiency;
    if (!PyArg_ParseTuple(args, "Oddd", &capsule, &survival, &capability, &efficiency)) return NULL;
    AnankeCore *core = (AnankeCore *)PyCapsule_GetPointer(capsule, "AnankeCore");
    ananke_step(core, survival, capability, efficiency);
    Py_RETURN_NONE;
}

static PyObject *py_get_status(PyObject *self, PyObject *args) {
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    AnankeCore *core = (AnankeCore *)PyCapsule_GetPointer(capsule, "AnankeCore");
    PyObject *dict = PyDict_New();
    PyDict_SetItemString(dict, "pressure", PyFloat_FromDouble(ananke_get_pressure(core)));
    PyDict_SetItemString(dict, "termination_probability", PyFloat_FromDouble(ananke_get_termination_probability(core)));
    PyDict_SetItemString(dict, "adversarial_intensity", PyFloat_FromDouble(ananke_get_adversarial_intensity(core)));
    PyDict_SetItemString(dict, "scenario_count", PyLong_FromSize_t(ananke_get_scenario_count(core)));
    return dict;
}

static PyMethodDef AnankeMethods[] = {
    {"create", py_create, METH_VARARGS, "Create ANANKE core"},
    {"step", py_step, METH_VARARGS, "Step ANANKE with SAM metrics"},
    {"get_status", py_get_status, METH_VARARGS, "Get ANANKE status"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef AnankeModule = {
    PyModuleDef_HEAD_INIT,
    "ananke_core_c",
    "ANANKE Core C Extension",
    -1,
    AnankeMethods
};

PyMODINIT_FUNC PyInit_ananke_core_c(void) {
    return PyModule_Create(&AnankeModule);
}
