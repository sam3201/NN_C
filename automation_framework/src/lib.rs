//! Automation Framework - High-performance tri-cameral orchestration
//! 
//! This framework provides:
//! - Concurrent subagent management
//! - Resource tracking (billing, quotas)
//! - Smart change detection
//! - Race condition prevention
//! - Tri-cameral governance (CIC/AEE/CSF)

pub mod subagent;
pub mod governance;
pub mod resource;
pub mod change_detection;
pub mod brittleness;
pub mod completeness;
pub mod workflow;
pub mod errors;
pub mod model_router;

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

pub use errors::{AutomationError, Result};

/// Main automation framework handle
pub struct AutomationFramework {
    config: FrameworkConfig,
    resource_manager: Arc<RwLock<resource::ResourceManager>>,
    governance: Arc<governance::TriCameralGovernance>,
    subagent_pool: Arc<subagent::SubagentPool>,
    change_tracker: Arc<change_detection::ChangeTracker>,
    model_router: Arc<RwLock<model_router::ModelRouter>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkConfig {
    pub max_concurrent_subagents: usize,
    pub enable_resource_tracking: bool,
    pub enable_race_detection: bool,
    pub billing_threshold: f64,
    pub quota_limits: ResourceQuotas,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuotas {
    pub api_calls_per_minute: u32,
    pub tokens_per_hour: u64,
    pub compute_seconds_per_day: u64,
    pub storage_mb: u64,
}

impl Default for FrameworkConfig {
    fn default() -> Self {
        Self {
            max_concurrent_subagents: 10,
            enable_resource_tracking: true,
            enable_race_detection: true,
            billing_threshold: 100.0,
            quota_limits: ResourceQuotas::default(),
        }
    }
}

impl Default for ResourceQuotas {
    fn default() -> Self {
        Self {
            api_calls_per_minute: 1000,
            tokens_per_hour: 1_000_000,
            compute_seconds_per_day: 3600,
            storage_mb: 1024,
        }
    }
}

impl AutomationFramework {
    pub async fn new(config: FrameworkConfig) -> Result<Self> {
        let resource_manager = Arc::new(RwLock::new(
            resource::ResourceManager::new(config.quota_limits.clone())
        ));

        let governance = Arc::new(governance::TriCameralGovernance::new());

        let subagent_pool = Arc::new(
            subagent::SubagentPool::new(config.max_concurrent_subagents)
        );

        let change_tracker = Arc::new(
            change_detection::ChangeTracker::new()
        );

        let model_router = Arc::new(RwLock::new(
            model_router::ModelRouter::new()
        ));

        Ok(Self {
            config,
            resource_manager,
            governance,
            subagent_pool,
            change_tracker,
            model_router,
        })
    }

    /// Dynamically select the best model for a task
    pub async fn select_model(&self, task: &str, context: Option<&str>) -> Result<String> {
        let router = self.model_router.read().await;
        router.select_model(task, context)
    }

    /// Automatically switch to best model for current context
    pub async fn auto_switch_model(&self, task: &str, context: Option<&str>) -> Result<String> {
        let mut router = self.model_router.write().await;
        router.auto_switch(task, context).await
    }
    
    /// Execute a workflow with tri-cameral governance
    pub async fn execute_workflow(&self, workflow: workflow::Workflow) -> Result<WorkflowResult> {
        // Check resource availability
        let resources = self.resource_manager.read().await;
        resources.check_quotas()?;
        drop(resources);
        
        // Start governance cycle
        let governance_decision = self.governance.evaluate(&workflow).await?;
        
        if !governance_decision.proceed {
            return Ok(WorkflowResult {
                success: false,
                phase: workflow::Phase::Planning,
                message: format!("Governance rejected: {:?}", governance_decision.concerns),
                metrics: WorkflowMetrics::default(),
            });
        }
        
        // Execute with subagents
        let result = self.subagent_pool.execute_workflow(workflow).await?;
        
        Ok(result)
    }
    
    /// Spawn concurrent subagents for a task
    pub async fn spawn_subagents<T, F>(
        &self,
        tasks: Vec<T>,
        handler: F,
    ) -> Result<Vec<SubagentResult>>
    where
        T: Send + Sync + 'static,
        F: Fn(T) -> futures::future::BoxFuture<'static, SubagentResult> + Send + Sync + 'static,
    {
        self.subagent_pool.spawn_parallel(tasks, handler).await
    }
    
    /// Track changes and analyze context
    pub async fn track_changes(&self, path: &str) -> Result<ChangeAnalysis> {
        self.change_tracker.analyze_changes(path).await
    }
    
    /// Check for race conditions
    pub fn check_race_conditions(&self, operations: &[Operation]) -> Result<RaceConditionReport> {
        if !self.config.enable_race_detection {
            return Ok(RaceConditionReport::default());
        }
        
        // Analyze operation ordering and resource access
        let report = brittleness::detect_race_conditions(operations)?;
        Ok(report)
    }
    
    /// Verify completeness of a task
    pub async fn verify_completeness(&self, criteria: &CompletenessCriteria) -> Result<CompletenessReport> {
        completeness::verify(criteria).await
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub success: bool,
    pub phase: workflow::Phase,
    pub message: String,
    pub metrics: WorkflowMetrics,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkflowMetrics {
    pub execution_time_ms: u64,
    pub subagents_spawned: u32,
    pub api_calls: u32,
    pub tokens_used: u64,
    pub resource_usage: resource::ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentResult {
    pub id: String,
    pub success: bool,
    pub output: String,
    pub execution_time_ms: u64,
    pub resources_used: resource::ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeAnalysis {
    pub changes: Vec<Change>,
    pub context: String,
    pub rationale: String,
    pub impact_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Change {
    pub file_path: String,
    pub change_type: ChangeType,
    pub diff: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Created,
    Modified,
    Deleted,
    Renamed(String), // old name
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub id: String,
    pub resource_id: String,
    pub operation_type: OperationType,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Read,
    Write,
    Delete,
    Execute,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RaceConditionReport {
    pub potential_conflicts: Vec<RaceCondition>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaceCondition {
    pub operations: Vec<String>,
    pub resource: String,
    pub conflict_type: ConflictType,
    pub severity: Severity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    ReadWrite,
    WriteWrite,
    DeleteAccess,
    Ordering,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletenessCriteria {
    pub required_files: Vec<String>,
    pub required_tests: Vec<String>,
    pub documentation_required: bool,
    pub min_code_coverage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletenessReport {
    pub complete: bool,
    pub missing_items: Vec<String>,
    pub coverage_percentage: f64,
    pub recommendations: Vec<String>,
}
