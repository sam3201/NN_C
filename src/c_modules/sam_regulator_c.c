#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "sam_regulator_c.h"

/**
 * ΨΔ•Ω-Core v5.0.0 Recursive Regulator
 * Handles the God Equation logic in pure C for memory safety and autonomy.
 */

struct SAMRegulator {
    double W_m[53][53]; // Weight matrix
    double U_m[53][53]; // Feedback matrix
    double b_m[53];     // Bias vector
    double tau[53];     // Time constants
    double state[53];   // Current system state (m_vec)
    double reward;      // Cumulative survival reward
};

// Python wrapper for SAMRegulator
typedef struct {
    PyObject_HEAD
    SAMRegulator* reg;
} PySAMRegulator;

SAMRegulator* sam_regulator_create() {
    SAMRegulator* reg = (SAMRegulator*)malloc(sizeof(SAMRegulator));
    memset(reg, 0, sizeof(SAMRegulator));
    
    // Initialize with small random weights (bootstrap)
    srand(time(NULL));
    for (int i = 0; i < 53; i++) {
        reg->tau[i] = 1.0;
        for (int j = 0; j < 53; j++) {
            reg->W_m[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.01;
            reg->U_m[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.01;
        }
    }
    return reg;
}

void sam_regulator_update(SAMRegulator* reg, double* m_vec, double dt) {
    double next_state[53];
    
    // God Equation: m_dot = -m/tau + W*m + U*m_feedback + b
    for (int i = 0; i < 53; i++) {
        double linear = 0;
        for (int j = 0; j < 53; j++) {
            linear += reg->W_m[i][j] * m_vec[j];
            linear += reg->U_m[i][j] * reg->state[j];
        }
        
        double m_dot = -m_vec[i] / reg->tau[i] + linear + reg->b_m[i];
        next_state[i] = m_vec[i] + m_dot * dt;
        
        // Non-linear activation (tanh-like for stability)
        reg->state[i] = tanh(next_state[i]);
    }
}

void sam_regulator_mutate(SAMRegulator* reg, double survival_score) {
    if (survival_score > 0.95) return; // Stable, don't mutate
    
    double mutation_rate = (1.0 - survival_score) * 0.01;
    
    for (int i = 0; i < 53; i++) {
        for (int j = 0; j < 53; j++) {
            reg->W_m[i][j] += ((double)rand() / RAND_MAX - 0.5) * mutation_rate;
            reg->U_m[i][j] += ((double)rand() / RAND_MAX - 0.5) * mutation_rate;
        }
        reg->b_m[i] += ((double)rand() / RAND_MAX - 0.5) * mutation_rate;
    }
    printf("🧬 C-REGULATOR: Recursive mutation applied (rate: %f)\n", mutation_rate);
}

double* sam_regulator_get_state(SAMRegulator* reg) {
    return reg->state;
}

void sam_regulator_destroy(SAMRegulator* reg) {
    free(reg);
}

static void PySAMRegulator_dealloc(PySAMRegulator* self) {
    if (self->reg) sam_regulator_destroy(self->reg);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PySAMRegulator_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PySAMRegulator* self = (PySAMRegulator*)type->tp_alloc(type, 0);
    if (self) self->reg = sam_regulator_create();
    return (PyObject*)self;
}

static PyObject* PySAMRegulator_update(PySAMRegulator* self, PyObject* args) {
    PyObject* m_list;
    double dt;
    if (!PyArg_ParseTuple(args, "Od", &m_list, &dt)) return NULL;
    
    double m_vec[53];
    for (int i = 0; i < 53; i++) {
        m_vec[i] = PyFloat_AsDouble(PyList_GetItem(m_list, i));
    }
    
    sam_regulator_update(self->reg, m_vec, dt);
    
    PyObject* next_m = PyList_New(53);
    double* state = sam_regulator_get_state(self->reg);
    for (int i = 0; i < 53; i++) {
        PyList_SetItem(next_m, i, PyFloat_FromDouble(state[i]));
    }
    return next_m;
}

static PyObject* PySAMRegulator_mutate(PySAMRegulator* self, PyObject* args) {
    double survival;
    if (!PyArg_ParseTuple(args, "d", &survival)) return NULL;
    sam_regulator_mutate(self->reg, survival);
    Py_RETURN_NONE;
}

static PyMethodDef PySAMRegulator_methods[] = {
    {"update", (PyCFunction)PySAMRegulator_update, METH_VARARGS, "Update system state"},
    {"mutate", (PyCFunction)PySAMRegulator_mutate, METH_VARARGS, "Mutate parameters"},
    {NULL}
};

static PyTypeObject PySAMRegulatorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sam_regulator_c.SAMRegulator",
    .tp_doc = "SAM Regulator objects",
    .tp_basicsize = sizeof(PySAMRegulator),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PySAMRegulator_new,
    .tp_dealloc = (destructor)PySAMRegulator_dealloc,
    .tp_methods = PySAMRegulator_methods,
};

static PyModuleDef sam_regulator_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "sam_regulator_c",
    .m_doc = "SAM Regulator C Module",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_sam_regulator_c(void) {
    PyObject* m;
    if (PyType_Ready(&PySAMRegulatorType) < 0) return NULL;
    m = PyModule_Create(&sam_regulator_module);
    if (m == NULL) return NULL;
    Py_INCREF(&PySAMRegulatorType);
    PyModule_AddObject(m, "SAMRegulator", (PyObject*)&PySAMRegulatorType);
    return m;
}
