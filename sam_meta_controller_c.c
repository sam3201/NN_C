/*
 * SAM Meta-Controller - Pure C Core
 * Morphogenetic latency, pressure aggregation, growth primitives, invariants.
 */

#include "sam_meta_controller_c.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>

typedef struct {
    double residual;
    double rank_def;
    double retrieval_entropy;
    double interference;
    double planner_friction;
    double context_collapse;
    double compression_waste;
    double temporal_incoherence;
    double persistence[8];
} PressureState;

struct SamMetaController {
    size_t latent_dim;
    size_t context_dim;
    size_t submodel_count;
    size_t submodel_max;
    size_t index_count;
    size_t routing_degree;
    size_t planner_depth;
    size_t archived_dim;

    double lambda;
    double lambda_threshold;
    double lambda_decay;

    double growth_budget;
    double growth_budget_max;

    double *identity_anchor;
    size_t identity_dim;
    double *identity_vec;

    PressureState pressure;
    unsigned int rng;
};

static unsigned int rng_next(SamMetaController *mc) {
    mc->rng = mc->rng * 1664525u + 1013904223u;
    return mc->rng;
}

static double rand_unit(SamMetaController *mc) {
    return (rng_next(mc) >> 8) * (1.0 / 16777216.0);
}

static double dot(const double *a, const double *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static double norm(const double *a, size_t n) {
    return sqrt(dot(a, a, n));
}

static double cosine_sim(const double *a, const double *b, size_t n) {
    double na = norm(a, n);
    double nb = norm(b, n);
    if (na <= 1e-12 || nb <= 1e-12) return 0.0;
    return dot(a, b, n) / (na * nb);
}

SamMetaController *sam_meta_create(size_t latent_dim, size_t context_dim, size_t max_submodels, unsigned int seed) {
    SamMetaController *mc = (SamMetaController *)calloc(1, sizeof(SamMetaController));
    if (!mc) return NULL;
    mc->latent_dim = latent_dim;
    mc->context_dim = context_dim;
    mc->submodel_max = max_submodels > 0 ? max_submodels : 4;
    mc->submodel_count = 1;
    mc->index_count = 1;
    mc->routing_degree = 1;
    mc->planner_depth = 4;
    mc->archived_dim = 0;
    mc->lambda = 0.0;
    mc->lambda_threshold = 1.5;
    mc->lambda_decay = 0.99;
    mc->growth_budget = 0.0;
    mc->growth_budget_max = 4.0;
    mc->identity_anchor = NULL;
    mc->identity_vec = NULL;
    mc->identity_dim = 0;
    mc->rng = seed ? seed : 0xC0FFEEu;
    return mc;
}

void sam_meta_free(SamMetaController *mc) {
    if (!mc) return;
    free(mc->identity_anchor);
    free(mc->identity_vec);
    free(mc);
}

double sam_meta_update_pressure(SamMetaController *mc,
                                double residual,
                                double rank_def,
                                double retrieval_entropy,
                                double interference,
                                double planner_friction,
                                double context_collapse,
                                double compression_waste,
                                double temporal_incoherence) {
    if (!mc) return 0.0;
    double values[8] = {residual, rank_def, retrieval_entropy, interference,
                        planner_friction, context_collapse, compression_waste, temporal_incoherence};
    double *persist = mc->pressure.persistence;
    for (int i = 0; i < 8; i++) {
        double v = values[i];
        persist[i] = (v > 0.1) ? (persist[i] + 1.0) : 0.0;
    }
    mc->pressure.residual = residual;
    mc->pressure.rank_def = rank_def;
    mc->pressure.retrieval_entropy = retrieval_entropy;
    mc->pressure.interference = interference;
    mc->pressure.planner_friction = planner_friction;
    mc->pressure.context_collapse = context_collapse;
    mc->pressure.compression_waste = compression_waste;
    mc->pressure.temporal_incoherence = temporal_incoherence;

    double weighted = 0.0;
    weighted += residual * 0.35;
    weighted += rank_def * 0.3;
    weighted += retrieval_entropy * 0.25;
    weighted += interference * 0.3;
    weighted += planner_friction * 0.2;
    weighted += context_collapse * 0.2;
    weighted += compression_waste * 0.2;
    weighted += temporal_incoherence * 0.25;

    mc->lambda = mc->lambda * mc->lambda_decay + weighted;
    return mc->lambda;
}

static int dominant_pressure(const SamMetaController *mc, double *out_value) {
    double values[8] = {
        mc->pressure.residual,
        mc->pressure.rank_def,
        mc->pressure.retrieval_entropy,
        mc->pressure.interference,
        mc->pressure.planner_friction,
        mc->pressure.context_collapse,
        mc->pressure.compression_waste,
        mc->pressure.temporal_incoherence
    };
    double maxv = values[0];
    int maxi = 0;
    double second = -1e9;
    for (int i = 1; i < 8; i++) {
        if (values[i] > maxv) {
            second = maxv;
            maxv = values[i];
            maxi = i;
        } else if (values[i] > second) {
            second = values[i];
        }
    }
    if (out_value) *out_value = maxv;
    if (maxv - second < 0.05) return -1;
    return maxi;
}

GrowthPrimitive sam_meta_select_primitive(SamMetaController *mc) {
    if (!mc) return GP_NONE;
    if (mc->lambda < mc->lambda_threshold) return GP_NONE;
    if (mc->growth_budget >= mc->growth_budget_max) {
        return GP_CONSOLIDATE;
    }

    double maxv = 0.0;
    int idx = dominant_pressure(mc, &maxv);
    if (idx < 0) return GP_NONE;

    switch (idx) {
        case 0: return GP_LATENT_EXPAND;       // residual
        case 1: return (rand_unit(mc) > 0.5) ? GP_LATENT_EXPAND : GP_REPARAM; // rank_def
        case 2: return GP_INDEX_EXPAND;        // retrieval_entropy
        case 3: return GP_SUBMODEL_SPAWN;      // interference
        case 4: return GP_PLANNER_WIDEN;       // planner_friction
        case 5: return GP_CONTEXT_EXPAND;      // context_collapse
        case 6: return GP_CONSOLIDATE;         // compression_waste
        case 7: return GP_REPARAM;             // temporal_incoherence
        default: return GP_NONE;
    }
}

int sam_meta_apply_primitive(SamMetaController *mc, GrowthPrimitive primitive) {
    if (!mc) return 0;
    switch (primitive) {
        case GP_LATENT_EXPAND:
            mc->latent_dim += 8;
            mc->growth_budget += 0.5;
            break;
        case GP_SUBMODEL_SPAWN:
            if (mc->submodel_count < mc->submodel_max) {
                mc->submodel_count += 1;
                mc->growth_budget += 0.6;
            }
            break;
        case GP_INDEX_EXPAND:
            mc->index_count += 1;
            mc->growth_budget += 0.2;
            break;
        case GP_ROUTING_INCREASE:
            mc->routing_degree += 1;
            mc->growth_budget += 0.2;
            break;
        case GP_CONTEXT_EXPAND:
            mc->context_dim += 4;
            mc->growth_budget += 0.3;
            break;
        case GP_PLANNER_WIDEN:
            mc->planner_depth += 2;
            mc->growth_budget += 0.2;
            break;
        case GP_CONSOLIDATE:
            if (mc->latent_dim > 16) {
                mc->latent_dim -= 4;
                mc->archived_dim += 4;
            }
            if (mc->index_count > 1) mc->index_count -= 1;
            break;
        case GP_REPARAM:
            mc->lambda *= 0.85;
            break;
        case GP_NONE:
        default:
            return 0;
    }
    mc->lambda *= 0.9;
    return 1;
}

void sam_meta_set_identity_anchor(SamMetaController *mc, const double *vec, size_t dim) {
    if (!mc || !vec || dim == 0) return;
    free(mc->identity_anchor);
    mc->identity_anchor = (double *)calloc(dim, sizeof(double));
    memcpy(mc->identity_anchor, vec, dim * sizeof(double));
    mc->identity_dim = dim;
}

void sam_meta_update_identity_vector(SamMetaController *mc, const double *vec, size_t dim) {
    if (!mc || !vec || dim == 0) return;
    free(mc->identity_vec);
    mc->identity_vec = (double *)calloc(dim, sizeof(double));
    memcpy(mc->identity_vec, vec, dim * sizeof(double));
}

int sam_meta_check_invariants(SamMetaController *mc, double *out_identity_similarity) {
    if (!mc) return 0;
    if (!mc->identity_anchor || !mc->identity_vec || mc->identity_dim == 0) {
        if (out_identity_similarity) *out_identity_similarity = 0.0;
        return 1;
    }
    double sim = cosine_sim(mc->identity_anchor, mc->identity_vec, mc->identity_dim);
    if (out_identity_similarity) *out_identity_similarity = sim;
    if (sim < 0.8) return 0;
    return 1;
}

int sam_meta_evaluate_contract(SamMetaController *mc,
                               double baseline_worst_case,
                               double proposed_worst_case) {
    if (!mc) return 0;
    return proposed_worst_case > baseline_worst_case;
}

size_t sam_meta_get_latent_dim(const SamMetaController *mc) { return mc ? mc->latent_dim : 0; }
size_t sam_meta_get_context_dim(const SamMetaController *mc) { return mc ? mc->context_dim : 0; }
size_t sam_meta_get_submodel_count(const SamMetaController *mc) { return mc ? mc->submodel_count : 0; }
size_t sam_meta_get_index_count(const SamMetaController *mc) { return mc ? mc->index_count : 0; }
size_t sam_meta_get_routing_degree(const SamMetaController *mc) { return mc ? mc->routing_degree : 0; }
size_t sam_meta_get_planner_depth(const SamMetaController *mc) { return mc ? mc->planner_depth : 0; }
double sam_meta_get_lambda(const SamMetaController *mc) { return mc ? mc->lambda : 0.0; }
double sam_meta_get_growth_budget(const SamMetaController *mc) { return mc ? mc->growth_budget : 0.0; }
size_t sam_meta_get_archived_dim(const SamMetaController *mc) { return mc ? mc->archived_dim : 0; }

// ================================
// PYTHON BINDINGS
// ================================

static void capsule_destructor(PyObject *capsule) {
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    sam_meta_free(mc);
}

static PyObject *py_sam_meta_create(PyObject *self, PyObject *args) {
    (void)self;
    unsigned int seed = 0;
    unsigned long latent_dim = 64;
    unsigned long context_dim = 16;
    unsigned long max_submodels = 4;
    if (!PyArg_ParseTuple(args, "|kkkI", &latent_dim, &context_dim, &max_submodels, &seed)) {
        return NULL;
    }
    SamMetaController *mc = sam_meta_create(latent_dim, context_dim, max_submodels, seed);
    if (!mc) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create SamMetaController");
        return NULL;
    }
    return PyCapsule_New(mc, "SamMetaController", capsule_destructor);
}

static PyObject *py_sam_meta_update_pressure(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    double residual, rank_def, retrieval_entropy, interference;
    double planner_friction, context_collapse, compression_waste, temporal_incoherence;
    if (!PyArg_ParseTuple(args, "Odddddddd", &capsule,
                          &residual, &rank_def, &retrieval_entropy, &interference,
                          &planner_friction, &context_collapse, &compression_waste, &temporal_incoherence)) {
        return NULL;
    }
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    double lambda = sam_meta_update_pressure(mc, residual, rank_def, retrieval_entropy, interference,
                                             planner_friction, context_collapse, compression_waste, temporal_incoherence);
    return PyFloat_FromDouble(lambda);
}

static PyObject *py_sam_meta_select_primitive(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    GrowthPrimitive gp = sam_meta_select_primitive(mc);
    return PyLong_FromLong((long)gp);
}

static PyObject *py_sam_meta_apply_primitive(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    long primitive = 0;
    if (!PyArg_ParseTuple(args, "Ol", &capsule, &primitive)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    int ok = sam_meta_apply_primitive(mc, (GrowthPrimitive)primitive);
    return PyBool_FromLong(ok);
}

static PyObject *py_sam_meta_set_identity_anchor(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    PyObject *seq = NULL;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &seq)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    PyObject *fast = PySequence_Fast(seq, "identity_anchor must be a sequence");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    double *buf = (double *)calloc((size_t)n, sizeof(double));
    for (Py_ssize_t i = 0; i < n; i++) {
        buf[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fast, i));
    }
    sam_meta_set_identity_anchor(mc, buf, (size_t)n);
    free(buf);
    Py_DECREF(fast);
    Py_RETURN_NONE;
}

static PyObject *py_sam_meta_update_identity_vector(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    PyObject *seq = NULL;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &seq)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    PyObject *fast = PySequence_Fast(seq, "identity_vec must be a sequence");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    double *buf = (double *)calloc((size_t)n, sizeof(double));
    for (Py_ssize_t i = 0; i < n; i++) {
        buf[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fast, i));
    }
    sam_meta_update_identity_vector(mc, buf, (size_t)n);
    free(buf);
    Py_DECREF(fast);
    Py_RETURN_NONE;
}

static PyObject *py_sam_meta_check_invariants(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    double sim = 0.0;
    int ok = sam_meta_check_invariants(mc, &sim);
    return Py_BuildValue("{s:O,s:d}", "passed", ok ? Py_True : Py_False, "identity_similarity", sim);
}

static PyObject *py_sam_meta_estimate_pressures(PyObject *self, PyObject *args) {
    (void)self;
    double survival, agent_count, goal_count, goal_history, activity_age, learning_events;
    if (!PyArg_ParseTuple(args, "dddddd", &survival, &agent_count, &goal_count, &goal_history, &activity_age, &learning_events)) {
        return NULL;
    }
    double residual = fmin(1.0, fmax(0.0, 1.0 - survival));
    double rank_def = fmin(1.0, fmax(0.0, 1.0 - (agent_count / 10.0)));
    double retrieval_entropy = fmin(1.0, fmax(0.0, activity_age / 300.0));
    double interference = fmin(1.0, fmax(0.0, agent_count / 15.0));
    double planner_friction = fmin(1.0, fmax(0.0, learning_events / 50.0));
    double context_collapse = fmin(1.0, fmax(0.0, goal_count / 10.0));
    double compression_waste = fmin(1.0, fmax(0.0, goal_history / 50.0));
    double temporal_incoherence = fmin(1.0, fmax(0.0, fabs(sin(activity_age / 60.0))));

    return Py_BuildValue("{s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d}",
                         "residual", residual,
                         "rank_def", rank_def,
                         "retrieval_entropy", retrieval_entropy,
                         "interference", interference,
                         "planner_friction", planner_friction,
                         "context_collapse", context_collapse,
                         "compression_waste", compression_waste,
                         "temporal_incoherence", temporal_incoherence);
}

static PyObject *py_sam_meta_evaluate_contract(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    double baseline, proposed;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &baseline, &proposed)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    int ok = sam_meta_evaluate_contract(mc, baseline, proposed);
    return PyBool_FromLong(ok);
}

static PyObject *py_sam_meta_get_state(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule = NULL;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    SamMetaController *mc = (SamMetaController *)PyCapsule_GetPointer(capsule, "SamMetaController");
    if (!mc) return NULL;
    return Py_BuildValue("{s:k,s:k,s:k,s:k,s:k,s:d,s:d,s:k}",
                         "latent_dim", (unsigned long)sam_meta_get_latent_dim(mc),
                         "context_dim", (unsigned long)sam_meta_get_context_dim(mc),
                         "submodels", (unsigned long)sam_meta_get_submodel_count(mc),
                         "indices", (unsigned long)sam_meta_get_index_count(mc),
                         "planner_depth", (unsigned long)sam_meta_get_planner_depth(mc),
                         "lambda", sam_meta_get_lambda(mc),
                         "growth_budget", sam_meta_get_growth_budget(mc),
                         "archived_dim", (unsigned long)sam_meta_get_archived_dim(mc));
}

static PyMethodDef MetaMethods[] = {
    {"create", py_sam_meta_create, METH_VARARGS, "Create SAM meta-controller"},
    {"estimate_pressures", py_sam_meta_estimate_pressures, METH_VARARGS, "Estimate pressure signals"},
    {"update_pressure", py_sam_meta_update_pressure, METH_VARARGS, "Update pressure signals"},
    {"select_primitive", py_sam_meta_select_primitive, METH_VARARGS, "Select growth primitive"},
    {"apply_primitive", py_sam_meta_apply_primitive, METH_VARARGS, "Apply growth primitive"},
    {"set_identity_anchor", py_sam_meta_set_identity_anchor, METH_VARARGS, "Set identity anchor"},
    {"update_identity_vector", py_sam_meta_update_identity_vector, METH_VARARGS, "Update identity vector"},
    {"check_invariants", py_sam_meta_check_invariants, METH_VARARGS, "Check invariants"},
    {"evaluate_contract", py_sam_meta_evaluate_contract, METH_VARARGS, "Evaluate objective contract"},
    {"get_state", py_sam_meta_get_state, METH_VARARGS, "Get meta-controller state"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef meta_module = {
    PyModuleDef_HEAD_INIT,
    "sam_meta_controller_c",
    "SAM Meta-Controller C Extension",
    -1,
    MetaMethods
};

PyMODINIT_FUNC PyInit_sam_meta_controller_c(void) {
    return PyModule_Create(&meta_module);
}
