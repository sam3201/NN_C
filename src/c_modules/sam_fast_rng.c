/*
 * Fast RNG - xorshift64* Implementation
 * With Python bindings
 */

#include <Python.h>
#include "sam_fast_rng.h"
#include <math.h>
#include <string.h>
#include <time.h>

static PyMethodDef SamFastRNGMethods[] = {
    {NULL, NULL, 0, NULL}
};

typedef struct {
    PyObject_HEAD
    SamFastRNG rng;
} SamFastRNGObject;

static int SamFastRNG_init(SamFastRNGObject *self, PyObject *args, PyObject *kwds) {
    unsigned int seed = 42;
    if (!PyArg_ParseTuple(args, "|I", &seed)) return -1;
    sam_rng_init(&self->rng, seed);
    return 0;
}

static PyObject *SamFastRNG_next(SamFastRNGObject *self, PyObject *Py_UNUSED(ignored)) {
    return PyLong_FromUnsignedLongLong(sam_rng_next(&self->rng));
}

static PyObject *SamFastRNG_double(SamFastRNGObject *self, PyObject *Py_UNUSED(ignored)) {
    return PyFloat_FromDouble(sam_rng_double(&self->rng));
}

static PyObject *SamFastRNG_range(SamFastRNGObject *self, PyObject *args) {
    unsigned long max;
    if (!PyArg_ParseTuple(args, "k", &max)) return NULL;
    return PyLong_FromUnsignedLongLong(sam_rng_range(&self->rng, max));
}

static PyObject *SamFastRNG_gaussian(SamFastRNGObject *self, PyObject *args) {
    double mean, stddev;
    if (!PyArg_ParseTuple(args, "dd", &mean, &stddev)) return NULL;
    return PyFloat_FromDouble(sam_rng_gaussian(&self->rng, mean, stddev));
}

static PyMethodDef SamFastRNGObjectMethods[] = {
    {"next", (PyCFunction)SamFastRNG_next, METH_NOARGS, "Next random uint64"},
    {"double", (PyCFunction)SamFastRNG_double, METH_NOARGS, "Next random double [0,1)"},
    {"range", METH_VARARGS, "Random in [0, max)"},
    {"gaussian", METH_VARARGS, "Gaussian with mean and stddev"},
    {NULL}
};

static PyTypeObject SamFastRNGType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sam_fast_rng.SamFastRNG",
    .tp_doc = "Fast xorshift64* RNG",
    .tp_basicsize = sizeof(SamFastRNGObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)SamFastRNG_init,
    .tp_methods = SamFastRNGObjectMethods,
};

static PyModuleDef sam_fast_rng_module = {
    PyModuleDef_HEAD_INIT,
    "sam_fast_rng",
    "Fast xorshift64* RNG module",
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
