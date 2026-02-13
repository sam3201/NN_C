/*
 * SAM Consciousness Core - L_cons computation
 * Implements: L_cons = KL(World_Actual || World_Predicted_by_Self)
 * With Python Bindings
 */

#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    PyObject_HEAD
    double self_model[64];    // Predicted world model
    double actual_world[64];   // Actual world observations
    double identity[32];      // Identity anchor
    double consciousness;      // Current consciousness score
    double self_coherence;     // Self-model coherence
    unsigned int tick;
} SamConsciousness;

static double kl_divergence(double *p, double *q, int n) {
    double kl = 0.0;
    for (int i = 0; i < n; i++) {
        if (p[i] > 1e-10 && q[i] > 1e-10) {
            kl += p[i] * log(p[i] / q[i]);
        }
    }
    return kl;
}

static double cosine_similarity(double *a, double *b, int n) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a < 1e-10 || norm_b < 1e-10) return 0.0;
    return dot / (sqrt(norm_a) * sqrt(norm_b));
}

static int Consciousness_init(SamConsciousness *self, PyObject *args, PyObject *kwds) {
    unsigned int seed = 42;
    if (!PyArg_ParseTuple(args, "|I", &seed)) return -1;
    
    srand(seed);
    for (int i = 0; i < 64; i++) {
        self->self_model[i] = (double)rand() / RAND_MAX;
        self->actual_world[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < 32; i++) {
        self->identity[i] = (double)rand() / RAND_MAX;
    }
    
    // Normalize
    double sum = 0.0;
    for (int i = 0; i < 64; i++) sum += self->self_model[i];
    for (int i = 0; i < 64; i++) self->self_model[i] /= sum;
    
    sum = 0.0;
    for (int i = 0; i < 64; i++) sum += self->actual_world[i];
    for (int i = 0; i < 64; i++) self->actual_world[i] /= sum;
    
    self->consciousness = 0.0;
    self->self_coherence = 0.5;
    self->tick = 0;
    return 0;
}

static PyObject *SamConsciousness_compute(SamConsciousness *self, PyObject *args) {
    // Update actual world from environment (simulated)
    for (int i = 0; i < 64; i++) {
        self->actual_world[i] += ((double)rand() / RAND_MAX - 0.5) * 0.1;
        if (self->actual_world[i] < 0) self->actual_world[i] = 0;
    }
    
    // Normalize
    double sum = 0.0;
    for (int i = 0; i < 64; i++) sum += self->actual_world[i];
    if (sum > 0) {
        for (int i = 0; i < 64; i++) self->actual_world[i] /= sum;
    }
    
    // Self-model learns from actual (simple gradient update)
    double learning_rate = 0.1;
    for (int i = 0; i < 64; i++) {
        self->self_model[i] += learning_rate * (self->actual_world[i] - self->self_model[i]);
    }
    
    // Compute consciousness as inverse KL divergence
    double kl = kl_divergence(self->actual_world, self->self_model, 64);
    self->consciousness = exp(-kl);  // When kl -> 0, consciousness -> 1
    
    // Self coherence
    self->self_coherence = cosine_similarity(self->self_model, self->identity, 32);
    
    self->tick++;
    return PyFloat_FromDouble(self->consciousness);
}

static PyObject *SamConsciousness_get_consciousness(SamConsciousness *self, PyObject *args) {
    return PyFloat_FromDouble(self->consciousness);
}

static PyObject *SamConsciousness_get_coherence(SamConsciousness *self, PyObject *args) {
    return PyFloat_FromDouble(self->self_coherence);
}

static PyObject *SamConsciousness_get_kl(SamConsciousness *self, PyObject *args) {
    double kl = kl_divergence(self->actual_world, self->self_model, 64);
    return PyFloat_FromDouble(kl);
}

static PyMethodDef SamConsciousnessMethods[] = {
    {"compute", (PyCFunction)SamConsciousness_compute, METH_VARARGS, "Compute consciousness"},
    {"get_consciousness", (PyCFunction)SamConsciousness_get_consciousness, METH_VARARGS, "Get consciousness score"},
    {"get_coherence", (PyCFunction)SamConsciousness_get_coherence, METH_VARARGS, "Get self coherence"},
    {"get_kl", (PyCFunction)SamConsciousness_get_kl, METH_VARARGS, "Get KL divergence"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject SamConsciousnessType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sam_consciousness.SamConsciousness",
    .tp_doc = "SAM Consciousness - L_cons computation",
    .tp_basicsize = sizeof(SamConsciousness),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Consciousness_init,
    .tp_methods = SamConsciousnessMethods,
};

static PyModuleDef sam_consciousness_module = {
    PyModuleDef_HEAD_INIT,
    "sam_consciousness",
    "SAM Consciousness module",
    -1,
    NULL
};

PyMODINIT_FUNC PyInit_sam_consciousness(void) {
    PyObject *m;
    if (PyType_Ready(&SamConsciousnessType) < 0) return NULL;
    m = PyModule_Create(&sam_consciousness_module);
    if (m == NULL) return NULL;
    Py_INCREF(&SamConsciousnessType);
    PyModule_AddObject(m, "SamConsciousness", (PyObject *)&SamConsciousnessType);
    return m;
}
