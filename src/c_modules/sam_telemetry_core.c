/*
 * SAM Telemetry Core - 53 Signal Collection with Python Bindings
 */

#include <Python.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SAM_TELEMETRY_DIM 53

typedef struct {
    PyObject_HEAD
    double values[SAM_TELEMETRY_DIM];
    unsigned int tick;
} SamTelemetry;

typedef struct {
    PyObject_HEAD
    double task_score;
    double tool_success_rate;
    double planner_value_gain;
    double latency_ms;
    double throughput;
    double loss_variance;
    double gradient_norm;
    double weight_drift;
    double explosion_flag;
    double collapse_flag;
    double anchor_similarity;
    double continuity;
    double self_coherence;
    double purpose_drift;
    double retrieval_entropy;
    double prediction_entropy;
    double unknown_ratio;
    double confusion;
    double mystery;
    double planner_friction;
    double depth_actual;
    double breadth_actual;
    double goal_drift;
    double ram_usage;
    double compute_budget;
    double memory_budget;
    double energy;
    double contradiction_rate;
    double hallucination_rate;
    double calib_ece;
    double residual;
    double rank_deficit;
    double interference;
    double context_collapse;
    double compression_waste;
    double temporal_incoherence;
    double planner_pressure;
} SamTelemetryInput;

static PyTypeObject SamTelemetryInputType;

static int clip(double *val, double min, double max) {
    if (*val < min) *val = min;
    if (*val > max) *val = max;
    return 0;
}

static int TelemetryInput_init(SamTelemetryInput *self, PyObject *args, PyObject *kwds) {
    self->task_score = 0.5;
    self->tool_success_rate = 0.5;
    self->planner_value_gain = 0.5;
    self->latency_ms = 500.0;
    self->throughput = 0.5;
    self->loss_variance = 0.1;
    self->gradient_norm = 1.0;
    self->weight_drift = 0.1;
    self->explosion_flag = 0.0;
    self->collapse_flag = 0.0;
    self->anchor_similarity = 0.8;
    self->continuity = 0.8;
    self->self_coherence = 0.8;
    self->purpose_drift = 0.1;
    self->retrieval_entropy = 0.5;
    self->prediction_entropy = 0.5;
    self->unknown_ratio = 0.5;
    self->confusion = 0.1;
    self->mystery = 0.1;
    self->planner_friction = 0.2;
    self->depth_actual = 4.0;
    self->breadth_actual = 4.0;
    self->goal_drift = 0.1;
    self->ram_usage = 0.5;
    self->compute_budget = 0.5;
    self->memory_budget = 0.5;
    self->energy = 0.5;
    self->contradiction_rate = 0.1;
    self->hallucination_rate = 0.1;
    self->calib_ece = 0.1;
    self->residual = 0.2;
    self->rank_deficit = 0.2;
    self->interference = 0.1;
    self->context_collapse = 0.1;
    self->compression_waste = 0.1;
    self->temporal_incoherence = 0.1;
    self->planner_pressure = 0.2;
    return 0;
}

static PyObject *SamTelemetry_get(SamTelemetry *self, PyObject *args) {
    int index;
    if (!PyArg_ParseTuple(args, "i", &index)) return NULL;
    if (index < 0 || index >= SAM_TELEMETRY_DIM) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }
    return PyFloat_FromDouble(self->values[index]);
}

static PyObject *SamTelemetry_get_all(SamTelemetry *self, PyObject *args) {
    PyObject *list = PyList_New(SAM_TELEMETRY_DIM);
    for (int i = 0; i < SAM_TELEMETRY_DIM; i++) {
        PyList_SetItem(list, i, PyFloat_FromDouble(self->values[i]));
    }
    return list;
}

static PyObject *SamTelemetry_get_tick(SamTelemetry *self, PyObject *args) {
    return PyLong_FromUnsignedLong(self->tick);
}

static PyObject *SamTelemetry_compute_capacity(SamTelemetry *self, PyObject *args) {
    double *v = self->values;
    double cap = 0.55 * v[0] + 0.25 * v[1] + 0.20 * v[2];
    double br = 1.0 - v[32];
    double sp = 1.0 - v[3];
    double rel = 1.0 - (0.4 * v[29] + 0.4 * v[28] + 0.2 * v[27]);
    return PyFloat_FromDouble(cap * br * sp * rel);
}

static PyObject *SamTelemetry_compute_innocence(SamTelemetry *self, PyObject *args) {
    double *v = self->values;
    double cap_val = 0.55 * v[0] + 0.25 * v[1] + 0.20 * v[2];
    double br = 1.0 - v[32];
    double sp = 1.0 - v[3];
    double rel = 1.0 - (0.4 * v[29] + 0.4 * v[28] + 0.2 * v[27]);
    double capacity = cap_val * br * sp * rel;
    double agency = v[19] + v[22];
    double irreversibility = v[8] + v[9];
    double verification = 0.5;
    double x = 2.0 - 1.2 * capacity - 1.0 * agency - 2.0 * irreversibility + 1.5 * verification;
    return PyFloat_FromDouble(1.0 / (1.0 + exp(-x)));
}

static int Telemetry_init(SamTelemetry *self, PyObject *args, PyObject *kwds) {
    unsigned int seed = 42;
    if (!PyArg_ParseTuple(args, "|I", &seed)) return -1;
    memset(self->values, 0, sizeof(self->values));
    self->tick = 0;
    return 0;
}

static PyObject *SamTelemetry_update(SamTelemetry *self, PyObject *args) {
    SamTelemetryInput *in;
    if (!PyArg_ParseTuple(args, "O!", NULL, &SamTelemetryInputType, &in)) return NULL;
    
    double *v = self->values;
    v[0] = fmax(0.0, fmin(1.0, in->task_score));
    v[1] = fmax(0.0, fmin(1.0, in->tool_success_rate));
    v[2] = fmax(0.0, fmin(1.0, in->planner_value_gain));
    v[3] = fmax(0.0, fmin(1.0, in->latency_ms / 1000.0));
    v[4] = fmax(0.0, fmin(1.0, in->throughput));
    v[5] = fmax(0.0, fmin(1.0, in->loss_variance));
    v[6] = fmax(0.0, fmin(1.0, in->gradient_norm / 100.0));
    v[7] = fmax(0.0, fmin(1.0, in->weight_drift));
    v[8] = fmax(0.0, fmin(1.0, in->explosion_flag));
    v[9] = fmax(0.0, fmin(1.0, in->collapse_flag));
    v[10] = fmax(0.0, fmin(1.0, in->anchor_similarity));
    v[11] = fmax(0.0, fmin(1.0, in->continuity));
    v[12] = fmax(0.0, fmin(1.0, in->self_coherence));
    v[13] = fmax(0.0, fmin(1.0, in->purpose_drift));
    v[14] = fmax(0.0, fmin(1.0, in->retrieval_entropy));
    v[15] = fmax(0.0, fmin(1.0, in->prediction_entropy));
    v[16] = fmax(0.0, fmin(1.0, in->unknown_ratio));
    v[17] = fmax(0.0, fmin(1.0, in->confusion));
    v[18] = fmax(0.0, fmin(1.0, in->mystery));
    v[19] = fmax(0.0, fmin(1.0, in->planner_friction));
    v[20] = fmax(0.0, fmin(1.0, in->depth_actual / 10.0));
    v[21] = fmax(0.0, fmin(1.0, in->breadth_actual / 10.0));
    v[22] = fmax(0.0, fmin(1.0, in->goal_drift));
    v[23] = fmax(0.0, fmin(1.0, in->ram_usage));
    v[24] = fmax(0.0, fmin(1.0, in->compute_budget));
    v[25] = fmax(0.0, fmin(1.0, in->memory_budget));
    v[26] = fmax(0.0, fmin(1.0, in->energy));
    v[27] = fmax(0.0, fmin(1.0, in->contradiction_rate));
    v[28] = fmax(0.0, fmin(1.0, in->hallucination_rate));
    v[29] = fmax(0.0, fmin(1.0, in->calib_ece));
    v[30] = fmax(0.0, fmin(1.0, in->residual));
    v[31] = fmax(0.0, fmin(1.0, in->rank_deficit));
    v[32] = fmax(0.0, fmin(1.0, in->interference));
    v[33] = fmax(0.0, fmin(1.0, in->context_collapse));
    v[34] = fmax(0.0, fmin(1.0, in->compression_waste));
    v[35] = fmax(0.0, fmin(1.0, in->temporal_incoherence));
    v[36] = fmax(0.0, fmin(1.0, in->planner_pressure));
    
    self->tick++;
    Py_RETURN_NONE;
}

static PyMethodDef SamTelemetryMethods[] = {
    {"get", (PyCFunction)SamTelemetry_get, METH_VARARGS, "Get telemetry value by index"},
    {"get_all", (PyCFunction)SamTelemetry_get_all, METH_VARARGS, "Get all telemetry values"},
    {"get_tick", (PyCFunction)SamTelemetry_get_tick, METH_VARARGS, "Get current tick"},
    {"compute_capacity", (PyCFunction)SamTelemetry_compute_capacity, METH_VARARGS, "Compute capacity metric"},
    {"compute_innocence", (PyCFunction)SamTelemetry_compute_innocence, METH_VARARGS, "Compute innocence gate"},
    {"update", (PyCFunction)SamTelemetry_update, METH_VARARGS, "Update with input"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject SamTelemetryType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sam_telemetry_core.SamTelemetry",
    .tp_doc = "SAM 53-dim telemetry vector",
    .tp_basicsize = sizeof(SamTelemetry),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Telemetry_init,
    .tp_methods = SamTelemetryMethods,
};

static PyTypeObject SamTelemetryInputType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sam_telemetry_core.SamTelemetryInput",
    .tp_doc = "Telemetry input values",
    .tp_basicsize = sizeof(SamTelemetryInput),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)TelemetryInput_init,
};

static PyModuleDef sam_telemetry_core_module = {
    PyModuleDef_HEAD_INIT,
    "sam_telemetry_core",
    "SAM 53-dim telemetry module",
    -1,
    NULL
};

PyMODINIT_FUNC PyInit_sam_telemetry_core(void) {
    PyObject *m;
    if (PyType_Ready(&SamTelemetryType) < 0) return NULL;
    if (PyType_Ready(&SamTelemetryInputType) < 0) return NULL;
    m = PyModule_Create(&sam_telemetry_core_module);
    if (m == NULL) return NULL;
    Py_INCREF(&SamTelemetryType);
    Py_INCREF(&SamTelemetryInputType);
    PyModule_AddObject(m, "SamTelemetry", (PyObject *)&SamTelemetryType);
    PyModule_AddObject(m, "SamTelemetryInput", (PyObject *)&SamTelemetryInputType);
    return m;
}
