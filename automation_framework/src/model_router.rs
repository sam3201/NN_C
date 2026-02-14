//! Dynamic Model Router - Automatically selects best AI model for each task

use crate::errors::{AutomationError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info, warn};

/// Model router that dynamically selects the best AI model for each task
pub struct ModelRouter {
    models: HashMap<String, ModelProfile>,
    task_analyzer: TaskAnalyzer,
    performance_tracker: PerformanceTracker,
    cost_optimizer: CostOptimizer,
    current_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub name: String,
    pub provider: ModelProvider,
    pub capabilities: ModelCapabilities,
    pub cost_per_1k_tokens: f64,
    pub latency_ms: u64,
    pub context_window: usize,
    pub reliability_score: f64, // 0.0 to 1.0
    pub specialty: Vec<TaskType>,
    pub quota_remaining: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelProvider {
    Anthropic,
    OpenAI,
    Local, // e.g., Ollama
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub reasoning: f64,       // 0.0 to 1.0
    pub coding: f64,
    pub creativity: f64,
    pub analysis: f64,
    pub long_context: f64,
    pub speed: f64,
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
    Reasoning,
    Coding,
    Analysis,
    Creative,
    LongContext,
    QuickResponse,
    SafetyCritical,
    MultiModal,
}

#[derive(Debug, Clone)]
pub struct TaskCharacteristics {
    pub task_type: TaskType,
    pub complexity: f64,      // 0.0 to 1.0
    pub context_size: usize,  // in tokens
    pub safety_critical: bool,
    pub time_sensitive: bool,
    pub budget_constraint: Option<f64>,
    pub required_reliability: f64, // 0.0 to 1.0
}

pub struct TaskAnalyzer;

impl TaskAnalyzer {
    pub fn new() -> Self {
        Self
    }
    
    pub fn analyze(&self, task_description: &str, context: Option<&str>) -> TaskCharacteristics {
        // Analyze task to determine characteristics
        let mut task_type = TaskType::Analysis;
        let mut complexity: f64 = 0.5;
        let context_size = context.map(|c| c.len()).unwrap_or(0);
        let mut safety_critical = false;
        let mut time_sensitive = false;
        
        // Detect task type from keywords
        let desc_lower = task_description.to_lowercase();
        
        if desc_lower.contains("code") || desc_lower.contains("implement") || desc_lower.contains("function") {
            task_type = TaskType::Coding;
            complexity = 0.7;
        } else if desc_lower.contains("analyze") || desc_lower.contains("review") || desc_lower.contains("check") {
            task_type = TaskType::Analysis;
            complexity = 0.6;
        } else if desc_lower.contains("design") || desc_lower.contains("create") || desc_lower.contains("innovate") {
            task_type = TaskType::Creative;
            complexity = 0.8;
        } else if desc_lower.contains("reason") || desc_lower.contains("logic") || desc_lower.contains("deduce") {
            task_type = TaskType::Reasoning;
            complexity = 0.9;
        }
        
        // Detect safety-critical tasks
        if desc_lower.contains("security") || desc_lower.contains("safety") || desc_lower.contains("invariant") {
            safety_critical = true;
            complexity = 0.95;
        }
        
        // Detect time-sensitive tasks
        if desc_lower.contains("quick") || desc_lower.contains("fast") || desc_lower.contains("urgent") {
            time_sensitive = true;
        }
        
        // Adjust complexity based on context size
        if context_size > 100_000 {
            complexity = (complexity + 0.2).min(1.0);
        }
        
        TaskCharacteristics {
            task_type,
            complexity,
            context_size,
            safety_critical,
            time_sensitive,
            budget_constraint: None,
            required_reliability: if safety_critical { 0.95 } else { 0.8 },
        }
    }
}

pub struct PerformanceTracker {
    model_performance: HashMap<String, ModelPerformance>,
    call_count: AtomicU64,
}

#[derive(Debug, Clone, Default)]
pub struct ModelPerformance {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub avg_latency_ms: f64,
    pub avg_cost_per_call: f64,
    pub user_satisfaction: f64, // Based on retries, corrections, etc.
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            model_performance: HashMap::new(),
            call_count: AtomicU64::new(0),
        }
    }
    
    pub fn record_usage(&mut self, model: &str, latency_ms: u64, cost: f64, success: bool) {
        let perf = self.model_performance.entry(model.to_string()).or_default();
        perf.total_calls += 1;
        if success {
            perf.successful_calls += 1;
        }
        
        // Update rolling averages
        let n = perf.total_calls as f64;
        perf.avg_latency_ms = (perf.avg_latency_ms * (n - 1.0) + latency_ms as f64) / n;
        perf.avg_cost_per_call = (perf.avg_cost_per_call * (n - 1.0) + cost) / n;
        
        self.call_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn get_model_score(&self, model: &str) -> f64 {
        if let Some(perf) = self.model_performance.get(model) {
            let reliability = if perf.total_calls > 0 {
                perf.successful_calls as f64 / perf.total_calls as f64
            } else {
                0.5
            };
            
            // Score based on reliability and inverse cost
            let cost_factor = 1.0 / (1.0 + perf.avg_cost_per_call);
            let speed_factor = 1.0 / (1.0 + perf.avg_latency_ms / 1000.0);
            
            reliability * 0.5 + cost_factor * 0.25 + speed_factor * 0.25
        } else {
            0.5 // Unknown model gets neutral score
        }
    }
}

pub struct CostOptimizer {
    daily_budget: f64,
    current_spend: AtomicU64, // in cents
    cost_thresholds: CostThresholds,
}

#[derive(Debug, Clone)]
pub struct CostThresholds {
    pub low_cost_max: f64,      // Use cheapest models
    pub medium_cost_max: f64,   // Balance cost/quality
    pub high_cost_threshold: f64, // Use best models regardless of cost
}

impl Default for CostThresholds {
    fn default() -> Self {
        Self {
            low_cost_max: 10.0,
            medium_cost_max: 50.0,
            high_cost_threshold: 80.0,
        }
    }
}

impl CostOptimizer {
    pub fn new(daily_budget: f64) -> Self {
        Self {
            daily_budget,
            current_spend: AtomicU64::new(0),
            cost_thresholds: CostThresholds::default(),
        }
    }
    
    pub fn get_cost_tier(&self) -> CostTier {
        let spend = self.current_spend.load(Ordering::Relaxed) as f64 / 100.0;
        let percentage = spend / self.daily_budget;
        
        if percentage < 0.3 {
            CostTier::Low
        } else if percentage < 0.7 {
            CostTier::Medium
        } else {
            CostTier::High
        }
    }
    
    pub fn record_cost(&self, cost_cents: u64) {
        self.current_spend.fetch_add(cost_cents, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CostTier {
    Low,    // Prioritize cheap models
    Medium, // Balance cost and quality
    High,   // Prioritize quality
}

impl ModelRouter {
    pub fn new() -> Self {
        let mut models = HashMap::new();
        
        // Register default models
        models.insert(
            "claude-3-5-sonnet".to_string(),
            ModelProfile {
                name: "Claude 3.5 Sonnet".to_string(),
                provider: ModelProvider::Anthropic,
                capabilities: ModelCapabilities {
                    reasoning: 0.95,
                    coding: 0.92,
                    creativity: 0.88,
                    analysis: 0.94,
                    long_context: 0.95,
                    speed: 0.75,
                },
                cost_per_1k_tokens: 0.003,
                latency_ms: 800,
                context_window: 200_000,
                reliability_score: 0.97,
                specialty: vec![TaskType::Reasoning, TaskType::Analysis, TaskType::Coding, TaskType::SafetyCritical],
                quota_remaining: None,
            }
        );
        
        models.insert(
            "claude-3-haiku".to_string(),
            ModelProfile {
                name: "Claude 3 Haiku".to_string(),
                provider: ModelProvider::Anthropic,
                capabilities: ModelCapabilities {
                    reasoning: 0.80,
                    coding: 0.78,
                    creativity: 0.75,
                    analysis: 0.82,
                    long_context: 0.70,
                    speed: 0.95,
                },
                cost_per_1k_tokens: 0.00025,
                latency_ms: 300,
                context_window: 200_000,
                reliability_score: 0.90,
                specialty: vec![TaskType::QuickResponse, TaskType::Analysis],
                quota_remaining: None,
            }
        );
        
        models.insert(
            "gpt-4".to_string(),
            ModelProfile {
                name: "GPT-4".to_string(),
                provider: ModelProvider::OpenAI,
                capabilities: ModelCapabilities {
                    reasoning: 0.93,
                    coding: 0.90,
                    creativity: 0.90,
                    analysis: 0.92,
                    long_context: 0.85,
                    speed: 0.70,
                },
                cost_per_1k_tokens: 0.03,
                latency_ms: 1200,
                context_window: 8_000,
                reliability_score: 0.95,
                specialty: vec![TaskType::Reasoning, TaskType::Coding, TaskType::Creative],
                quota_remaining: None,
            }
        );
        
        models.insert(
            "local-llm".to_string(),
            ModelProfile {
                name: "Local LLM (Ollama)".to_string(),
                provider: ModelProvider::Local,
                capabilities: ModelCapabilities {
                    reasoning: 0.70,
                    coding: 0.75,
                    creativity: 0.65,
                    analysis: 0.72,
                    long_context: 0.60,
                    speed: 0.85,
                },
                cost_per_1k_tokens: 0.0,
                latency_ms: 500,
                context_window: 4_000,
                reliability_score: 0.80,
                specialty: vec![TaskType::QuickResponse, TaskType::Coding],
                quota_remaining: None,
            }
        );
        
        // Kimi K2.5 - FREE model (priority: FREE MAX)
        models.insert(
            "kimi-k2.5-flash".to_string(),
            ModelProfile {
                name: "Kimi K2.5 Flash".to_string(),
                provider: ModelProvider::Custom("moonshot".to_string()),
                capabilities: ModelCapabilities {
                    reasoning: 0.88,
                    coding: 0.85,
                    creativity: 0.82,
                    analysis: 0.86,
                    long_context: 0.90,
                    speed: 0.80,
                },
                cost_per_1k_tokens: 0.0,  // FREE!
                latency_ms: 600,
                context_window: 128_000,
                reliability_score: 0.92,
                specialty: vec![TaskType::Reasoning, TaskType::Coding, TaskType::Analysis, TaskType::LongContext],
                quota_remaining: None,
            }
        );
        
        // Kimi K2.5 Vision (if available)
        models.insert(
            "kimi-k2.5-vision".to_string(),
            ModelProfile {
                name: "Kimi K2.5 Vision".to_string(),
                provider: ModelProvider::Custom("moonshot".to_string()),
                capabilities: ModelCapabilities {
                    reasoning: 0.90,
                    coding: 0.87,
                    creativity: 0.85,
                    analysis: 0.88,
                    long_context: 0.90,
                    speed: 0.75,
                },
                cost_per_1k_tokens: 0.0,  // FREE!
                latency_ms: 800,
                context_window: 128_000,
                reliability_score: 0.90,
                specialty: vec![TaskType::Reasoning, TaskType::Coding, TaskType::Analysis, TaskType::LongContext, TaskType::MultiModal],
                quota_remaining: None,
            }
        );
        
        Self {
            models,
            task_analyzer: TaskAnalyzer::new(),
            performance_tracker: PerformanceTracker::new(),
            cost_optimizer: CostOptimizer::new(100.0),
            current_model: None,
        }
    }
    
    /// Dynamically select the best model for a task
    pub fn select_model(&self, task: &str, context: Option<&str>) -> Result<String> {
        let characteristics = self.task_analyzer.analyze(task, context);
        
        debug!("Task characteristics: {:?}", characteristics);
        
        // Score each model for this task
        let mut model_scores: Vec<(String, f64)> = self.models
            .iter()
            .filter(|(_, profile)| self.is_model_available(profile))
            .map(|(name, profile)| {
                let score = self.score_model_for_task(profile, &characteristics);
                (name.clone(), score)
            })
            .collect();
        
        // Sort by score (descending)
        model_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        if let Some((best_model, score)) = model_scores.first() {
            info!("Selected model: {} (score: {:.2})", best_model, score);
            Ok(best_model.clone())
        } else {
            Err(AutomationError::ConfigError {
                message: "No suitable model found".to_string(),
            })
        }
    }
    
    /// Automatically switch to best model for current context
    pub async fn auto_switch(&mut self, task: &str, context: Option<&str>) -> Result<String> {
        let new_model = self.select_model(task, context)?;
        
        if self.current_model.as_ref() != Some(&new_model) {
            if let Some(old) = &self.current_model {
                info!("Switching model: {} -> {}", old, new_model);
            } else {
                info!("Initializing with model: {}", new_model);
            }
            self.current_model = Some(new_model.clone());
        }
        
        Ok(new_model)
    }
    
    /// Check if a model is available (has quota, etc.)
    fn is_model_available(&self, profile: &ModelProfile) -> bool {
        if let Some(quota) = profile.quota_remaining {
            quota > 0
        } else {
            true
        }
    }
    
    /// Score a model for a specific task
    fn score_model_for_task(&self, profile: &ModelProfile, task: &TaskCharacteristics) -> f64 {
        let mut score = 0.0;
        
        // Capability match (40%)
        let capability_score = match task.task_type {
            TaskType::Reasoning => profile.capabilities.reasoning,
            TaskType::Coding => profile.capabilities.coding,
            TaskType::Analysis => profile.capabilities.analysis,
            TaskType::Creative => profile.capabilities.creativity,
            TaskType::LongContext => profile.capabilities.long_context,
            TaskType::QuickResponse => profile.capabilities.speed,
            TaskType::SafetyCritical => profile.reliability_score,
            TaskType::MultiModal => profile.capabilities.analysis,
        };
        score += capability_score * 0.40;
        
        // Specialty match (20%)
        if profile.specialty.contains(&task.task_type) {
            score += 0.20;
        }
        
        // Context window fit (15%)
        if task.context_size <= profile.context_window {
            score += 0.15;
        } else {
            score += 0.05; // Penalize if context doesn't fit
        }
        
        // Cost optimization based on current spend (15%)
        // PRIORITY: FREE models ($0) always get max score
        let cost_tier = self.cost_optimizer.get_cost_tier();
        
        let cost_score = if profile.cost_per_1k_tokens == 0.0 {
            // FREE model - always max score regardless of tier!
            1.0
        } else {
            match cost_tier {
                CostTier::Low => 1.0 - (profile.cost_per_1k_tokens * 100.0).min(1.0),
                CostTier::Medium => 0.7 - (profile.cost_per_1k_tokens * 50.0).min(0.7),
                CostTier::High => 1.0 - (profile.cost_per_1k_tokens * 10.0).min(1.0),
            }
        };
        score += cost_score.max(0.0) * 0.15;
        
        // Historical performance (10%)
        let perf_score = self.performance_tracker.get_model_score(&profile.name);
        score += perf_score * 0.10;
        
        // Safety critical bonus
        if task.safety_critical && profile.reliability_score > 0.95 {
            score += 0.10;
        }
        
        // Time sensitive bonus for fast models
        if task.time_sensitive && profile.capabilities.speed > 0.8 {
            score += 0.05;
        }
        
        score
    }
    
    /// Record model usage for performance tracking
    pub fn record_usage(&mut self, model: &str, latency_ms: u64, cost: f64, success: bool) {
        self.performance_tracker.record_usage(model, latency_ms, cost, success);
        self.cost_optimizer.record_cost((cost * 100.0) as u64);
    }
    
    /// Get current model statistics
    pub fn get_stats(&self) -> RouterStats {
        RouterStats {
            current_model: self.current_model.clone(),
            available_models: self.models.len(),
            total_calls: self.performance_tracker.call_count.load(Ordering::Relaxed),
            cost_tier: format!("{:?}", self.cost_optimizer.get_cost_tier()),
            model_scores: self.models.keys().cloned().collect(),
        }
    }
    
    /// Add a new model to the router
    pub fn register_model(&mut self, name: String, profile: ModelProfile) {
        self.models.insert(name, profile);
    }
}

#[derive(Debug, Clone)]
pub struct RouterStats {
    pub current_model: Option<String>,
    pub available_models: usize,
    pub total_calls: u64,
    pub cost_tier: String,
    pub model_scores: Vec<String>,
}

/// Convenience function for quick model selection
pub fn select_best_model(task: &str) -> Result<String> {
    let router = ModelRouter::new();
    router.select_model(task, None)
}
