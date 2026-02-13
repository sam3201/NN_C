/*
 * SAM Fast RNG - xorshift64* with Python bindings
 * High-performance random number generation
 */

#include <Python.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

typedef struct {
    PyObject_HEAD
    uint64_t state;
} SamFastRNG;

static int SamFastRNG_init(SamFastRNG *self, PyObject *args, PyObject *kwds) {
    unsigned int seed = 42;
    if (!PyArg_ParseTuple(args, "|I", &seed)) return -1;
    if (seed == 0) seed = 1;
    self->state = seed;
    return 0;
}

static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static PyObject *SamFastRNG_next(SamFastRNG *self, PyObject *args) {
    return PyLong_FromUnsignedLongLong(rng_next(&self->state));
}

static PyObject *SamFastRNG_double(SamFastRNG *self, PyObject *args) {
    return PyFloat_FromDouble((double)(rng_next(&self->state) >> 11) * (1.0 / 9007199254740992.0));
}

static PyObject *SamFastRNG_range(SamFastRNG *self, PyObject *args) {
    unsigned long max;
    if (!PyArg_ParseTuple(args, "k", &max)) return NULL;
    if (max == 0) return PyLong_FromLong(0);
    uint64_t threshold = (uint64_t)(-max) % max;
    uint64_t r;
    do { r = rng_next(&self->state); } while (r < threshold);
    return PyLong_FromUnsignedLongLong(r % max);
}

static PyObject *SamFastRNG_gaussian(SamFastRNG *self, PyObject *args) {
    double mean, stddev;
    if (!PyArg_ParseTuple(args, "dd", &mean, &stddev)) return NULL;
    double u1 = (double)(rng_next(&self->state) >> 11) * (1.0 / 9007199254740992.0);
    double u2 = (double)(rng_next(&self->state) >> 11) * (1.0 / 9007199254740992.0);
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return PyFloat_FromDouble(mean + stddev * z0);
}

static PyMethodDef SamFastRNGMethods[] = {
    {"next", (PyCFunction)SamFastRNG_next, METH_VARARGS, "Next random uint64"},
    {"double", (PyCFunction)SamFastRNG_double, METH_VARARGS, "Next random double [0,1)"},
    {"range", (PyCFunction)SamFastRNG_range, METH_VARARGS, "Random in [0, max)"},
    {"gaussian", (PyCFunction)SamFastRNG_gaussian, METH_VARARGS, "Gaussian with mean and stddev"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject SamFastRNGType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sam_fast_rng.SamFastRNG",
    .tp_doc = "Fast xorshift64* RNG",
    .tp_basicsize = sizeof(SamFastRNG),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)SamFastRNG_init,
    .tp_methods = SamFastRNGMethods,
};

static PyModuleDef sam_fast_rng_module = {
    PyModuleDef_HEAD_INIT,
    "sam_fast_rng",
    "Fast xorshift64* RNG module for SAM-D",
    -1,
    NULL
};

PyMODINIT_FUNC PyInit_sam_fast_rng(void) {
    PyObject *m;
    if (PyType_Ready(&SamFastRNGType) < 0) return NULL;
    m = PyModule_Create(&sam_fast_rng_module);
    if (m == NULL) return NULL;
    Py_INCREF(&SamFastRNGType);
    if (PyModule_AddObject(m, "SamFastRNG", (PyObject *)&SamFastRNGType) < 0) {
        Py_DECREF(&SamFastRNGType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
