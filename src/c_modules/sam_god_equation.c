/*
 * SAM God Equation Core - with Python Bindings
 */

#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    PyObject_HEAD
    double K;
    double U;
    double O;
    double omega;
    double alpha;
    double beta;
    double gamma;
    double delta;
    double zeta;
} SamGodEquation;

static double sigma_frontier(double U, double O) {
    double rho = 0.7;
    return (U + rho * O) / (1.0 + U + rho * O);
}

static double contradiction(double K, double U, double O) {
    return fmax(0.0, (U + O) / (1.0 + K) - 1.0);
}

static int GodEquation_init(SamGodEquation *self, PyObject *args, PyObject *kwds) {
    self->K = 1.0;
    self->U = 5.0;
    self->O = 10.0;
    self->omega = 0.5;
    self->alpha = 0.05;
    self->beta = 1.10;
    self->gamma = 0.02;
    self->delta = 1.00;
    self->zeta = 0.01;
    return 0;
}

static PyObject *SamGodEquation_compute(SamGodEquation *self, PyObject *args) {
    double research = 0.5, verify = 0.5, morph = 0.2;
    double dt = 1.0;
    PyArg_ParseTuple(args, "|ddd", &research, &verify, &morph);
    
    double sigma = sigma_frontier(self->U, self->O);
    double contra = contradiction(self->K, self->U, self->O);
    
    double discovery = self->alpha * pow(self->K, self->beta) * sigma * (0.5 + research);
    double burden = self->gamma * pow(self->K, self->delta) * (1.2 - 0.7 * verify);
    double contra_pen = self->zeta * pow(self->K, self->delta) * contra;
    
    double dK = (discovery - burden - contra_pen) * dt;
    self->K = fmax(0.0, self->K + dK);
    
    double eta = 0.03, mu = 1.0, kappa = 0.04;
    double created_U = eta * pow(fmax(self->K, 1e-9), mu) * (0.4 + 0.6 * research) * dt;
    double resolved_U = kappa * self->U * (0.3 + 0.7 * verify) * dt;
    self->U = fmax(0.0, self->U + created_U - resolved_U);
    
    double xi = 0.02, nu = 1.0, chi = 0.06;
    double created_O = xi * pow(fmax(self->K, 1e-9), nu) * (0.5 + 0.5 * research) * dt;
    double morphed_O = chi * self->O * (0.2 + 0.8 * morph) * dt;
    self->O = fmax(0.0, self->O + created_O - morphed_O);
    
    self->omega = 1.0 - contra;
    
    return PyFloat_FromDouble(self->K);
}

static PyObject *SamGodEquation_get_K(SamGodEquation *self, PyObject *args) {
    return PyFloat_FromDouble(self->K);
}

static PyObject *SamGodEquation_get_U(SamGodEquation *self, PyObject *args) {
    return PyFloat_FromDouble(self->U);
}

static PyObject *SamGodEquation_get_O(SamGodEquation *self, PyObject *args) {
    return PyFloat_FromDouble(self->O);
}

static PyObject *SamGodEquation_get_omega(SamGodEquation *self, PyObject *args) {
    return PyFloat_FromDouble(self->omega);
}

static PyObject *SamGodEquation_contradiction(SamGodEquation *self, PyObject *args) {
    return PyFloat_FromDouble(contradiction(self->K, self->U, self->O));
}

static PyMethodDef SamGodEquationMethods[] = {
    {"compute", (PyCFunction)SamGodEquation_compute, METH_VARARGS, "Step the equation"},
    {"get_K", (PyCFunction)SamGodEquation_get_K, METH_VARARGS, "Get K"},
    {"get_U", (PyCFunction)SamGodEquation_get_U, METH_VARARGS, "Get U"},
    {"get_O", (PyCFunction)SamGodEquation_get_O, METH_VARARGS, "Get O"},
    {"get_omega", (PyCFunction)SamGodEquation_get_omega, METH_VARARGS, "Get omega"},
    {"contradiction", (PyCFunction)SamGodEquation_contradiction, METH_VARARGS, "Get contradiction"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject SamGodEquationType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sam_god_equation.SamGodEquation",
    .tp_doc = "SAM God Equation K/U/O dynamics",
    .tp_basicsize = sizeof(SamGodEquation),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)GodEquation_init,
    .tp_methods = SamGodEquationMethods,
};

static PyModuleDef sam_god_equation_module = {
    PyModuleDef_HEAD_INIT,
    "sam_god_equation",
    "SAM God Equation module",
    -1,
    NULL
};

PyMODINIT_FUNC PyInit_sam_god_equation(void) {
    PyObject *m;
    if (PyType_Ready(&SamGodEquationType) < 0) return NULL;
    m = PyModule_Create(&sam_god_equation_module);
    if (m == NULL) return NULL;
    Py_INCREF(&SamGodEquationType);
    PyModule_AddObject(m, "SamGodEquation", (PyObject *)&SamGodEquationType);
    return m;
}
