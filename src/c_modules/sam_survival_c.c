/*
 * SAM Survival and Goal Management C Library
 * Performance-critical functions for survival evaluation and task scheduling
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Python.h>

// ================================
// SURVIVAL PRIORITY CALCULATION
// ================================

/*
 * Calculate survival-based priority score for task execution
 * Priority = (base_priority + critical_bonus + confidence_factor + success_factor) - risk_penalty
 */
double calculate_survival_priority(double base_priority, int is_critical,
                                 double confidence_score, double success_rate) {
    double critical_bonus = is_critical ? 0.3 : 0.0;
    double confidence_factor = confidence_score;
    double success_factor = success_rate;

    // Risk penalty for low confidence
    double risk_penalty = (1.0 - confidence_score) * 0.2;

    double survival_score = (base_priority + critical_bonus + confidence_factor + success_factor) - risk_penalty;

    // Clamp to 0-1 range
    return fmax(0.0, fmin(1.0, survival_score));
}

/*
 * Batch calculate survival priorities for multiple tasks
 * Returns array of priority scores
 */
double* calculate_batch_survival_priorities(double* base_priorities, int* critical_flags,
                                          double* confidence_scores, double* success_rates,
                                          int num_tasks) {
    double* priorities = (double*)malloc(num_tasks * sizeof(double));

    for (int i = 0; i < num_tasks; i++) {
        priorities[i] = calculate_survival_priority(
            base_priorities[i],
            critical_flags[i],
            confidence_scores[i],
            success_rates[i]
        );
    }

    return priorities;
}

// ================================
// SURVIVAL ACTION EVALUATION
// ================================

/*
 * Evaluate an action for survival impact
 * Returns: survival_impact, optionality_impact, risk_level, confidence
 */
typedef struct {
    double survival_impact;
    double optionality_impact;
    double risk_level;
    double confidence;
} ActionEvaluation;

ActionEvaluation evaluate_action_impact(const char* action, double context_risk_level,
                                      double current_survival_score) {
    ActionEvaluation result;

    // Simplified action evaluation (would be more complex in real implementation)
    // Risk actions reduce survival, safe actions increase it
    if (strstr(action, "backup") || strstr(action, "safety") || strstr(action, "recovery")) {
        result.survival_impact = 0.2;  // Positive survival impact
        result.optionality_impact = 0.3;  // Increases options
        result.risk_level = 0.1;  // Low risk
    } else if (strstr(action, "explore") || strstr(action, "learn")) {
        result.survival_impact = 0.1;
        result.optionality_impact = 0.4;  // Learning increases future options
        result.risk_level = context_risk_level * 0.8;  // Moderate risk
    } else if (strstr(action, "update") || strstr(action, "patch")) {
        result.survival_impact = 0.05;
        result.optionality_impact = 0.2;
        result.risk_level = 0.6;  // Higher risk for modifications
    } else {
        // Default conservative evaluation
        result.survival_impact = -0.1;  // Slight negative impact for unknown actions
        result.optionality_impact = 0.0;
        result.risk_level = 0.5;
    }

    // Confidence based on current survival score and context
    result.confidence = fmax(0.3, current_survival_score * 0.8 + 0.2);

    return result;
}

/*
 * Determine if an action should be taken based on survival criteria
 */
int should_execute_action(double survival_impact, double optionality_impact,
                         double risk_level, double confidence,
                         double confidence_threshold, double risk_tolerance) {

    // Must meet confidence threshold
    if (confidence < confidence_threshold) {
        return 0;
    }

    // Risk must be within tolerance
    if (risk_level > risk_tolerance) {
        return 0;
    }

    // Must have positive survival impact or significant optionality gain
    if (survival_impact < 0.0 && optionality_impact < 0.2) {
        return 0;
    }

    return 1;  // Action approved
}

// ================================
// GOAL DEPENDENCY RESOLUTION
// ================================

/*
 * Check if a task can be executed (all dependencies met)
 * Returns 1 if executable, 0 if not
 */
int check_task_dependencies(int task_id, int* dependency_matrix, int* completion_status,
                           int num_tasks) {
    for (int dep = 0; dep < num_tasks; dep++) {
        if (dependency_matrix[task_id * num_tasks + dep] && !completion_status[dep]) {
            return 0;  // Dependency not met
        }
    }
    return 1;  // All dependencies met
}

/*
 * Find the optimal task to execute next based on survival priorities
 * Returns task index or -1 if no executable tasks
 */
int select_optimal_task(double* priorities, int* critical_flags, int* executable_flags,
                       int num_tasks, int max_concurrent_tasks, int* active_tasks) {

    int active_count = 0;
    for (int i = 0; i < num_tasks; i++) {
        if (active_tasks[i]) active_count++;
    }

    if (active_count >= max_concurrent_tasks) {
        return -1;  // At capacity
    }

    double best_priority = -1.0;
    int best_task = -1;

    for (int i = 0; i < num_tasks; i++) {
        if (!executable_flags[i] || active_tasks[i]) {
            continue;  // Not executable or already active
        }

        // Prioritize critical tasks
        double adjusted_priority = priorities[i];
        if (critical_flags[i]) {
            adjusted_priority += 0.5;  // Critical bonus
        }

        if (adjusted_priority > best_priority) {
            best_priority = adjusted_priority;
            best_task = i;
        }
    }

    return best_task;
}

// ================================
// THREAT ASSESSMENT
// ================================

/*
 * Assess current threats to system survival
 * Returns threat levels: high, medium, low
 */
typedef struct {
    int high_threat;
    int medium_threat;
    int low_threat;
} ThreatAssessment;

ThreatAssessment assess_survival_threats(double consciousness_score, int pending_critical_tasks,
                                       int recent_failures, double resource_usage) {

    ThreatAssessment threats = {0, 0, 0};

    // High threats - immediate danger
    if (consciousness_score < 0.3) {
        threats.high_threat = 1;
    }
    if (resource_usage > 0.9) {  // >90% resource usage
        threats.high_threat = 1;
    }

    // Medium threats - concerning but not critical
    if (pending_critical_tasks > 2) {
        threats.medium_threat = 1;
    }
    if (recent_failures > 3) {
        threats.medium_threat = 1;
    }

    // Low threats - monitor but not urgent
    if (consciousness_score < 0.6) {
        threats.low_threat = 1;
    }
    if (pending_critical_tasks > 0) {
        threats.low_threat = 1;
    }

    return threats;
}

// ================================
// PYTHON BINDINGS
// ================================

static PyObject* py_calculate_survival_priority(PyObject* self, PyObject* args) {
    double base_priority, confidence_score, success_rate;
    int is_critical;

    if (!PyArg_ParseTuple(args, "didd", &base_priority, &is_critical,
                         &confidence_score, &success_rate)) {
        return NULL;
    }

    double result = calculate_survival_priority(base_priority, is_critical,
                                              confidence_score, success_rate);

    return PyFloat_FromDouble(result);
}

static PyObject* py_evaluate_action_impact(PyObject* self, PyObject* args) {
    const char* action;
    double context_risk_level, current_survival_score;

    if (!PyArg_ParseTuple(args, "sdd", &action, &context_risk_level, &current_survival_score)) {
        return NULL;
    }

    ActionEvaluation result = evaluate_action_impact(action, context_risk_level, current_survival_score);

    return Py_BuildValue("{s:d,s:d,s:d,s:d}",
                        "survival_impact", result.survival_impact,
                        "optionality_impact", result.optionality_impact,
                        "risk_level", result.risk_level,
                        "confidence", result.confidence);
}

static PyObject* py_should_execute_action(PyObject* self, PyObject* args) {
    double survival_impact, optionality_impact, risk_level, confidence;
    double confidence_threshold, risk_tolerance;

    if (!PyArg_ParseTuple(args, "dddd", &survival_impact, &optionality_impact,
                         &risk_level, &confidence)) {
        // Use default thresholds if not provided
        confidence_threshold = 0.6;
        risk_tolerance = 0.3;
    } else if (!PyArg_ParseTuple(args, "dddddd", &survival_impact, &optionality_impact,
                                &risk_level, &confidence, &confidence_threshold, &risk_tolerance)) {
        return NULL;
    }

    int result = should_execute_action(survival_impact, optionality_impact,
                                     risk_level, confidence,
                                     confidence_threshold, risk_tolerance);

    return PyBool_FromLong(result);
}

static PyObject* py_check_task_dependencies(PyObject* self, PyObject* args) {
    PyObject* dep_matrix_obj, *completion_obj;
    int task_id, num_tasks;

    if (!PyArg_ParseTuple(args, "iOOi", &task_id, &dep_matrix_obj, &completion_obj, &num_tasks)) {
        return NULL;
    }

    // Convert Python lists to C arrays (simplified)
    int* dependency_matrix = (int*)malloc(num_tasks * num_tasks * sizeof(int));
    int* completion_status = (int*)malloc(num_tasks * sizeof(int));

    // Fill arrays from Python objects (would need proper conversion in real implementation)
    memset(dependency_matrix, 0, num_tasks * num_tasks * sizeof(int));
    memset(completion_status, 0, num_tasks * sizeof(int));

    int result = check_task_dependencies(task_id, dependency_matrix, completion_status, num_tasks);

    free(dependency_matrix);
    free(completion_status);

    return PyBool_FromLong(result);
}

static PyObject* py_assess_survival_threats(PyObject* self, PyObject* args) {
    double consciousness_score, resource_usage;
    int pending_critical_tasks, recent_failures;

    if (!PyArg_ParseTuple(args, "diid", &consciousness_score, &pending_critical_tasks,
                         &recent_failures, &resource_usage)) {
        return NULL;
    }

    ThreatAssessment threats = assess_survival_threats(consciousness_score, pending_critical_tasks,
                                                      recent_failures, resource_usage);

    return Py_BuildValue("{s:i,s:i,s:i}",
                        "high_threat", threats.high_threat,
                        "medium_threat", threats.medium_threat,
                        "low_threat", threats.low_threat);
}

// Module method table
static PyMethodDef SurvivalMethods[] = {
    {"calculate_survival_priority", py_calculate_survival_priority, METH_VARARGS, "Calculate task survival priority"},
    {"evaluate_action_impact", py_evaluate_action_impact, METH_VARARGS, "Evaluate action for survival impact"},
    {"should_execute_action", py_should_execute_action, METH_VARARGS, "Determine if action should be executed"},
    {"check_task_dependencies", py_check_task_dependencies, METH_VARARGS, "Check if task dependencies are met"},
    {"assess_survival_threats", py_assess_survival_threats, METH_VARARGS, "Assess current survival threats"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef survival_module = {
    PyModuleDef_HEAD_INIT,
    "sam_survival_c",
    "SAM Survival and Goal Management C Library",
    -1,
    SurvivalMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_sam_survival_c(void) {
    return PyModule_Create(&survival_module);
}
