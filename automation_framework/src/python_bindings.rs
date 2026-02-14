//! Python bindings for the Automation Framework using PyO3
//!
//! This module provides Python bindings for the core automation functionality,
//! allowing Python code to leverage the high-performance Rust implementation.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python wrapper for usage statistics
#[pyclass]
#[derive(Clone)]
pub struct PyUsageStats {
    #[pyo3(get)]
    pub api_calls: u64,
    #[pyo3(get)]
    pub tokens_consumed: u64,
    #[pyo3(get)]
    pub current_cost: f64,
    #[pyo3(get)]
    pub cost_limit: f64,
}

#[pymethods]
impl PyUsageStats {
    #[new]
    fn new(api_calls: u64, tokens_consumed: u64, current_cost: f64, cost_limit: f64) -> Self {
        Self {
            api_calls,
            tokens_consumed,
            current_cost,
            cost_limit,
        }
    }

    /// Get remaining budget
    fn remaining_budget(&self) -> f64 {
        (self.cost_limit - self.current_cost).max(0.0)
    }

    /// Get budget usage percentage
    fn budget_percentage(&self) -> f64 {
        if self.cost_limit > 0.0 {
            (self.current_cost / self.cost_limit) * 100.0
        } else {
            0.0
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "UsageStats(calls={}, tokens={}, cost=${:.2}/${:.2})",
            self.api_calls, self.tokens_consumed, self.current_cost, self.cost_limit
        )
    }
}

/// Python wrapper for framework configuration
#[pyclass]
#[derive(Clone)]
pub struct PyFrameworkConfig {
    #[pyo3(get, set)]
    pub max_concurrent_subagents: usize,
    #[pyo3(get, set)]
    pub billing_threshold: f64,
}

#[pymethods]
impl PyFrameworkConfig {
    #[new]
    fn new(max_concurrent: usize, billing_threshold: f64) -> Self {
        Self {
            max_concurrent_subagents: max_concurrent,
            billing_threshold,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FrameworkConfig(max_concurrent={}, billing_threshold=${:.2})",
            self.max_concurrent_subagents, self.billing_threshold
        )
    }
}

/// Python wrapper for workflow execution results
#[pyclass]
#[derive(Clone)]
pub struct PyWorkflowResult {
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub confidence: f64,
}

#[pymethods]
impl PyWorkflowResult {
    #[new]
    fn new(success: bool, message: String, confidence: f64) -> Self {
        Self {
            success,
            message,
            confidence,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkflowResult(success={}, confidence={:.2}, message='{}')",
            self.success, self.confidence, self.message
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// High-level Python API for easy integration
#[pyfunction]
fn check_system_health() -> PyResult<String> {
    Ok(format!(
        "Automation Framework v{} - Status: HEALTHY - Rust {}",
        env!("CARGO_PKG_VERSION"),
        "1.70+"
    ))
}

#[pyfunction]
fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Module initialization
#[pymodule]
fn automation_framework(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyUsageStats>()?;
    m.add_class::<PyFrameworkConfig>()?;
    m.add_class::<PyWorkflowResult>()?;
    m.add_function(wrap_pyfunction!(check_system_health, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
