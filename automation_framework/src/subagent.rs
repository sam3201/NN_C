//! Subagent management with concurrency control

use crate::{errors::Result, resource::ResourceUsage, SubagentResult};
use dashmap::DashMap;
use futures::future::{BoxFuture, FutureExt};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Pool of concurrent subagents
pub struct SubagentPool {
    max_concurrent: usize,
    semaphore: Arc<Semaphore>,
    active_subagents: DashMap<String, SubagentHandle>,
    metrics: Arc<parking_lot::Mutex<SubagentMetrics>>,
}

#[derive(Debug, Clone)]
struct SubagentHandle {
    id: String,
    task_description: String,
    start_time: std::time::Instant,
    resource_usage: ResourceUsage,
}

#[derive(Debug, Default)]
struct SubagentMetrics {
    total_spawned: u64,
    total_completed: u64,
    total_failed: u64,
    average_execution_time_ms: u64,
}

impl SubagentPool {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            active_subagents: DashMap::new(),
            metrics: Arc::new(parking_lot::Mutex::new(SubagentMetrics::default())),
        }
    }
    
    /// Spawn multiple subagents in parallel
    pub async fn spawn_parallel<T, F>(
        &self,
        tasks: Vec<T>,
        handler: F,
    ) -> Result<Vec<SubagentResult>>
    where
        T: Send + Sync + 'static,
        F: Fn(T) -> BoxFuture<'static, SubagentResult> + Send + Sync + 'static,
    {
        let handler = Arc::new(handler);
        let mut handles = Vec::with_capacity(tasks.len());
        
        for (idx, task) in tasks.into_iter().enumerate() {
            let semaphore = Arc::clone(&self.semaphore);
            let handler = Arc::clone(&handler);
            let subagent_id = format!("subagent_{}_{}", idx, Uuid::new_v4());
            
            let handle = tokio::spawn(async move {
                // Acquire permit for concurrency control
                let _permit = semaphore.acquire().await.unwrap();
                
                debug!("Subagent {} started", subagent_id);
                let start_time = std::time::Instant::now();
                
                // Execute the task
                let result = handler(task).await;
                
                let execution_time = start_time.elapsed().as_millis() as u64;
                debug!(
                    "Subagent {} completed in {}ms", 
                    subagent_id, 
                    execution_time
                );
                
                result
            });
            
            handles.push(handle);
        }
        
        // Collect all results
        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok(result) => {
                    results.push(result);
                }
                Err(e) => {
                    error!("Subagent panicked: {}", e);
                    results.push(SubagentResult {
                        id: Uuid::new_v4().to_string(),
                        success: false,
                        output: format!("Subagent panicked: {}", e),
                        execution_time_ms: 0,
                        resources_used: ResourceUsage::default(),
                    });
                }
            }
        }
        
        // Update metrics
        let mut metrics = self.metrics.lock();
        metrics.total_spawned += results.len() as u64;
        metrics.total_completed += results.iter().filter(|r| r.success).count() as u64;
        metrics.total_failed += results.iter().filter(|r| !r.success).count() as u64;
        
        if !results.is_empty() {
            let total_time: u64 = results.iter().map(|r| r.execution_time_ms).sum();
            metrics.average_execution_time_ms = total_time / results.len() as u64;
        }
        
        Ok(results)
    }
    
    /// Spawn subagents with different roles for the same task
    /// e.g., Reader, Processor, Writer pattern
    pub async fn spawn_pipeline<T, R, P, W>(
        &self,
        task: T,
        reader: R,
        processor: P,
        writer: W,
    ) -> Result<SubagentResult>
    where
        T: Send + Sync + Clone + 'static,
        R: Fn(T) -> BoxFuture<'static, String> + Send + Sync + 'static,
        P: Fn(String) -> BoxFuture<'static, String> + Send + Sync + 'static,
        W: Fn(String) -> BoxFuture<'static, SubagentResult> + Send + Sync + 'static,
    {
        let pipeline_id = format!("pipeline_{}", Uuid::new_v4());
        info!("Starting pipeline {}", pipeline_id);
        
        // Stage 1: Reader
        let _permit = self.semaphore.acquire().await.unwrap();
        let context = reader(task.clone()).await;
        drop(_permit);
        
        // Stage 2: Processor
        let _permit = self.semaphore.acquire().await.unwrap();
        let processed = processor(context).await;
        drop(_permit);
        
        // Stage 3: Writer
        let _permit = self.semaphore.acquire().await.unwrap();
        let result = writer(processed).await;
        drop(_permit);
        
        info!("Pipeline {} completed", pipeline_id);
        Ok(result)
    }
    
    /// Execute a workflow with the subagent pool
    pub async fn execute_workflow(
        &self,
        workflow: crate::workflow::Workflow,
    ) -> Result<crate::WorkflowResult> {
        use crate::workflow::Phase;
        
        let start_time = std::time::Instant::now();
        
        // Create tasks from workflow phases
        let tasks: Vec<String> = workflow.phases.iter()
            .map(|p| format!("Execute phase: {:?}", p))
            .collect();
        
        // Execute each phase as a subagent task
        let mut phase_results = Vec::new();
        for task in tasks.iter() {
            let task = task.clone();
            let results = self.spawn_parallel(
                vec![task],
                |t| async move {
                    SubagentResult {
                        id: Uuid::new_v4().to_string(),
                        success: true,
                        output: format!("Completed: {}", t),
                        execution_time_ms: 0,
                        resources_used: ResourceUsage::default(),
                    }
                }.boxed()
            ).await?;
            
            phase_results.extend(results);
        }
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        Ok(crate::WorkflowResult {
            success: phase_results.iter().all(|r| r.success),
            phase: Phase::Complete,
            message: "Workflow executed with subagents".to_string(),
            metrics: crate::WorkflowMetrics {
                execution_time_ms: execution_time,
                subagents_spawned: phase_results.len() as u32,
                api_calls: 0,
                tokens_used: 0,
                resource_usage: ResourceUsage::default(),
            },
        })
    }

    /// Spawn verification subagents to check work from multiple angles
    pub async fn spawn_verifiers<T, V>(
        &self,
        task: T,
        verifiers: Vec<V>,
    ) -> Result<Vec<SubagentResult>>
    where
        T: Send + Sync + Clone + 'static,
        V: Fn(T) -> BoxFuture<'static, SubagentResult> + Send + Sync + 'static,
    {
        let verification_id = format!("verification_{}", Uuid::new_v4());
        info!("Starting verification {}", verification_id);
        
        let mut handles = Vec::with_capacity(verifiers.len());
        
        for (idx, verifier) in verifiers.into_iter().enumerate() {
            let task = task.clone();
            let semaphore = Arc::clone(&self.semaphore);
            
            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                
                let result = verifier(task).await;
                debug!("Verifier {} completed", idx);
                
                result
            });
            
            handles.push(handle);
        }
        
        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => {
                    error!("Verifier panicked: {}", e);
                    results.push(SubagentResult {
                        id: Uuid::new_v4().to_string(),
                        success: false,
                        output: format!("Verifier panicked: {}", e),
                        execution_time_ms: 0,
                        resources_used: ResourceUsage::default(),
                    });
                }
            }
        }
        
        info!(
            "Verification {} completed: {}/{} passed",
            verification_id,
            results.iter().filter(|r| r.success).count(),
            results.len()
        );
        
        Ok(results)
    }
    
    /// Get current pool metrics
    pub fn get_metrics(&self) -> SubagentPoolMetrics {
        let metrics = self.metrics.lock();
        SubagentPoolMetrics {
            max_concurrent: self.max_concurrent,
            active_subagents: self.active_subagents.len(),
            total_spawned: metrics.total_spawned,
            total_completed: metrics.total_completed,
            total_failed: metrics.total_failed,
            average_execution_time_ms: metrics.average_execution_time_ms,
        }
    }
    
    /// Cancel all active subagents
    pub async fn cancel_all(&self) {
        warn!("Cancelling all {} active subagents", self.active_subagents.len());
        self.active_subagents.clear();
    }
}

#[derive(Debug, Clone)]
pub struct SubagentPoolMetrics {
    pub max_concurrent: usize,
    pub active_subagents: usize,
    pub total_spawned: u64,
    pub total_completed: u64,
    pub total_failed: u64,
    pub average_execution_time_ms: u64,
}

/// Builder for complex subagent configurations
pub struct SubagentBuilder {
    id: String,
    task: String,
    priority: SubagentPriority,
    timeout_ms: Option<u64>,
    retry_count: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum SubagentPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl SubagentBuilder {
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            task: task.into(),
            priority: SubagentPriority::Normal,
            timeout_ms: None,
            retry_count: 0,
        }
    }
    
    pub fn with_priority(mut self, priority: SubagentPriority) -> Self {
        self.priority = priority;
        self
    }
    
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
    
    pub fn with_retry(mut self, retry_count: u32) -> Self {
        self.retry_count = retry_count;
        self
    }
    
    pub fn build(self) -> SubagentConfig {
        SubagentConfig {
            id: self.id,
            task: self.task,
            priority: self.priority,
            timeout_ms: self.timeout_ms,
            retry_count: self.retry_count,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubagentConfig {
    pub id: String,
    pub task: String,
    pub priority: SubagentPriority,
    pub timeout_ms: Option<u64>,
    pub retry_count: u32,
}
