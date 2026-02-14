//! Error types for automation framework

use thiserror::Error;

pub type Result<T> = std::result::Result<T, AutomationError>;

#[derive(Error, Debug)]
pub enum AutomationError {
    #[error("Resource quota exceeded: {resource} ({current}/{limit})")]
    QuotaExceeded {
        resource: String,
        current: u64,
        limit: u64,
    },

    #[error("Budget exceeded: ${current:.2} / ${limit:.2}")]
    BudgetExceeded { current: f64, limit: f64 },

    #[error("Governance rejected workflow: {reason}")]
    GovernanceRejected { reason: String },

    #[error("Subagent execution failed: {message}")]
    SubagentFailed { message: String },

    #[error("Race condition detected: {conflict_type} on resource '{resource}'")]
    RaceCondition {
        conflict_type: String,
        resource: String,
    },

    #[error("Workflow validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Git error: {0}")]
    Git(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Task panicked: {message}")]
    TaskPanicked { message: String },

    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    #[error("Unknown error: {message}")]
    Unknown { message: String },

    #[error("External service error: {0}")]
    ExternalService(String),
}

impl From<git2::Error> for AutomationError {
    fn from(err: git2::Error) -> Self {
        AutomationError::Git(err.message().to_string())
    }
}

impl From<walkdir::Error> for AutomationError {
    fn from(err: walkdir::Error) -> Self {
        AutomationError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            err.to_string(),
        ))
    }
}

impl AutomationError {
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            AutomationError::QuotaExceeded { .. }
                | AutomationError::BudgetExceeded { .. }
                | AutomationError::GovernanceRejected { .. }
        )
    }

    pub fn requires_retry(&self) -> bool {
        matches!(
            self,
            AutomationError::SubagentFailed { .. } | AutomationError::TaskPanicked { .. }
        )
    }
}
