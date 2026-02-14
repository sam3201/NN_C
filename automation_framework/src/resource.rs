//! Resource management with billing and quota tracking

use crate::errors::{AutomationError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, error, info, warn};

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Alert types for resource monitoring
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertType {
    BudgetThreshold { percentage: f64 },
    QuotaThreshold { resource: String, percentage: f64 },
    RateLimitWarning { resource: String, current: u64, limit: u64 },
    FreeModelExhausted,
    DailyLimitReached,
}

/// Alert notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAlert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Alert callback type
pub type AlertCallback = Box<dyn Fn(&ResourceAlert) + Send + Sync>;

/// Alert manager for handling resource notifications
pub struct AlertManager {
    callbacks: Vec<AlertCallback>,
    webhook_url: Option<String>,
    alert_history: Vec<ResourceAlert>,
    suppression_window_minutes: i64,
}

impl Clone for AlertManager {
    fn clone(&self) -> Self {
        Self {
            callbacks: Vec::new(), // Callbacks can't be cloned, start fresh
            webhook_url: self.webhook_url.clone(),
            alert_history: self.alert_history.clone(),
            suppression_window_minutes: self.suppression_window_minutes,
        }
    }
}

impl std::fmt::Debug for AlertManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlertManager")
            .field("webhook_url", &self.webhook_url)
            .field("alert_history_count", &self.alert_history.len())
            .field("suppression_window_minutes", &self.suppression_window_minutes)
            .finish()
    }
}

impl AlertManager {
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
            webhook_url: std::env::var("AUTOMATION_ALERT_WEBHOOK").ok(),
            alert_history: Vec::new(),
            suppression_window_minutes: 5,
        }
    }

    /// Register an alert callback
    pub fn on_alert<F>(&mut self, callback: F)
    where
        F: Fn(&ResourceAlert) + Send + Sync + 'static,
    {
        self.callbacks.push(Box::new(callback));
    }

    /// Set webhook URL for alerts
    pub fn set_webhook(&mut self, url: String) {
        self.webhook_url = Some(url);
    }

    /// Send an alert through all channels
    pub async fn send_alert(&mut self, alert: ResourceAlert) {
        // Check if similar alert was sent recently (suppression)
        if self.is_suppressed(&alert) {
            return;
        }

        // Log the alert
        match alert.severity {
            AlertSeverity::Critical => error!("{}", alert.message),
            AlertSeverity::Warning => warn!("{}", alert.message),
            AlertSeverity::Info => info!("{}", alert.message),
        }

        // Execute callbacks
        for callback in &self.callbacks {
            callback(&alert);
        }

        // Send webhook if configured
        if let Some(ref webhook) = self.webhook_url {
            self.send_webhook(webhook, &alert).await;
        }

        // Store in history
        self.alert_history.push(alert);
    }

    /// Check if similar alert should be suppressed
    fn is_suppressed(&self, alert: &ResourceAlert) -> bool {
        let now = Utc::now();
        let window = chrono::Duration::minutes(self.suppression_window_minutes);

        self.alert_history.iter().any(|hist| {
            hist.alert_type == alert.alert_type
                && (now - hist.timestamp) < window
        })
    }

    /// Send alert to webhook
    async fn send_webhook(&self, url: &str, alert: &ResourceAlert) {
        let client = reqwest::Client::new();
        let payload = serde_json::json!({
            "severity": alert.severity,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "metadata": alert.metadata,
        });

        match client.post(url).json(&payload).send().await {
            Ok(_) => debug!("Alert webhook sent successfully"),
            Err(e) => warn!("Failed to send alert webhook: {}", e),
        }
    }

    /// Get alert history
    pub fn get_history(&self) -> &[ResourceAlert] {
        &self.alert_history
    }

    /// Clear old alerts from history
    pub fn clear_old_alerts(&mut self, older_than_hours: i64) {
        let cutoff = Utc::now() - chrono::Duration::hours(older_than_hours);
        self.alert_history.retain(|alert| alert.timestamp > cutoff);
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource quota limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuotas {
    pub api_calls_per_minute: u32,
    pub tokens_per_hour: u64,
    pub compute_seconds_per_day: u64,
    pub storage_mb: u64,
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

/// Resource manager for tracking usage, billing, and quotas
pub struct ResourceManager {
    quotas: ResourceQuotas,
    usage: ResourceUsage,
    billing: BillingTracker,
    quota_window_start: DateTime<Utc>,
    alert_manager: AlertManager,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    #[serde(skip)]
    pub api_calls: AtomicU64,
    #[serde(skip)]
    pub tokens_consumed: AtomicU64,
    #[serde(skip)]
    pub compute_seconds: AtomicU64,
    #[serde(skip)]
    pub storage_mb_used: AtomicU64,
    pub timestamp: DateTime<Utc>,
}

impl Clone for ResourceUsage {
    fn clone(&self) -> Self {
        Self {
            api_calls: AtomicU64::new(self.api_calls.load(Ordering::Relaxed)),
            tokens_consumed: AtomicU64::new(self.tokens_consumed.load(Ordering::Relaxed)),
            compute_seconds: AtomicU64::new(self.compute_seconds.load(Ordering::Relaxed)),
            storage_mb_used: AtomicU64::new(self.storage_mb_used.load(Ordering::Relaxed)),
            timestamp: self.timestamp,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingTracker {
    pub current_cost: f64,
    pub cost_limit: f64,
    pub hourly_usage: HashMap<String, f64>,
    pub alerts_sent: Vec<String>,
}

impl ResourceManager {
    pub fn new(quotas: ResourceQuotas) -> Self {
        Self {
            quotas,
            usage: ResourceUsage::default(),
            billing: BillingTracker::new(),
            quota_window_start: Utc::now(),
            alert_manager: AlertManager::new(),
        }
    }

    /// Get mutable reference to alert manager
    pub fn alert_manager_mut(&mut self) -> &mut AlertManager {
        &mut self.alert_manager
    }

    /// Register an alert callback
    pub fn on_alert<F>(&mut self, callback: F)
    where
        F: Fn(&ResourceAlert) + Send + Sync + 'static,
    {
        self.alert_manager.on_alert(callback);
    }

    /// Check if current usage is within quotas
    pub fn check_quotas(&self) -> Result<()> {
        let api_calls = self.usage.api_calls.load(Ordering::Relaxed);
        let tokens = self.usage.tokens_consumed.load(Ordering::Relaxed);
        let compute = self.usage.compute_seconds.load(Ordering::Relaxed);
        let storage = self.usage.storage_mb_used.load(Ordering::Relaxed);

        // Check API call rate (per minute)
        if api_calls > self.quotas.api_calls_per_minute as u64 {
            return Err(AutomationError::QuotaExceeded {
                resource: "api_calls_per_minute".to_string(),
                current: api_calls,
                limit: self.quotas.api_calls_per_minute as u64,
            });
        }

        // Check token usage (per hour)
        if tokens > self.quotas.tokens_per_hour {
            return Err(AutomationError::QuotaExceeded {
                resource: "tokens_per_hour".to_string(),
                current: tokens,
                limit: self.quotas.tokens_per_hour,
            });
        }

        // Check compute time (per day)
        if compute > self.quotas.compute_seconds_per_day {
            return Err(AutomationError::QuotaExceeded {
                resource: "compute_seconds_per_day".to_string(),
                current: compute,
                limit: self.quotas.compute_seconds_per_day,
            });
        }

        // Check storage
        if storage > self.quotas.storage_mb {
            return Err(AutomationError::QuotaExceeded {
                resource: "storage_mb".to_string(),
                current: storage,
                limit: self.quotas.storage_mb,
            });
        }

        // Check billing
        if self.billing.current_cost > self.billing.cost_limit {
            return Err(AutomationError::BudgetExceeded {
                current: self.billing.current_cost,
                limit: self.billing.cost_limit,
            });
        }

        debug!("All quotas within limits");
        Ok(())
    }

    /// Record resource usage
    pub fn record_usage(&mut self, usage: &ResourceUsage) {
        self.usage
            .api_calls
            .fetch_add(usage.api_calls.load(Ordering::Relaxed), Ordering::Relaxed);
        self.usage.tokens_consumed.fetch_add(
            usage.tokens_consumed.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.usage.compute_seconds.fetch_add(
            usage.compute_seconds.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        self.usage.storage_mb_used.fetch_add(
            usage.storage_mb_used.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );

        // Update billing
        let cost = self.calculate_cost(usage);
        self.billing.record_cost(cost);

        // Check for alerts
        self.check_billing_alerts();
    }

    /// Record a single API call
    pub fn record_api_call(&self) {
        self.usage.api_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Record token usage
    pub fn record_tokens(&self, tokens: u64) {
        self.usage
            .tokens_consumed
            .fetch_add(tokens, Ordering::Relaxed);
    }

    /// Record compute time
    pub fn record_compute(&self, seconds: u64) {
        self.usage
            .compute_seconds
            .fetch_add(seconds, Ordering::Relaxed);
    }

    /// Calculate cost for usage
    fn calculate_cost(&self, usage: &ResourceUsage) -> f64 {
        // Pricing model (example rates)
        const COST_PER_API_CALL: f64 = 0.001;
        const COST_PER_1K_TOKENS: f64 = 0.002;
        const COST_PER_COMPUTE_SECOND: f64 = 0.0001;
        const COST_PER_MB_STORAGE: f64 = 0.00001;

        let api_cost = usage.api_calls.load(Ordering::Relaxed) as f64 * COST_PER_API_CALL;
        let token_cost =
            (usage.tokens_consumed.load(Ordering::Relaxed) as f64 / 1000.0) * COST_PER_1K_TOKENS;
        let compute_cost =
            usage.compute_seconds.load(Ordering::Relaxed) as f64 * COST_PER_COMPUTE_SECOND;
        let storage_cost =
            usage.storage_mb_used.load(Ordering::Relaxed) as f64 * COST_PER_MB_STORAGE;

        api_cost + token_cost + compute_cost + storage_cost
    }

    /// Check and send billing alerts (async version)
    pub async fn check_billing_alerts_async(&mut self) {
        let usage_percentage = self.billing.current_cost / self.billing.cost_limit;

        if usage_percentage > 0.9 && !self.billing.alerts_sent.contains(&"90_percent".to_string()) {
            let alert = ResourceAlert {
                alert_type: AlertType::BudgetThreshold { percentage: 90.0 },
                severity: AlertSeverity::Critical,
                message: format!(
                    "Billing CRITICAL: 90% of budget used (${:.2} / ${:.2})",
                    self.billing.current_cost, self.billing.cost_limit
                ),
                timestamp: Utc::now(),
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("current_cost".to_string(), self.billing.current_cost.to_string());
                    m.insert("cost_limit".to_string(), self.billing.cost_limit.to_string());
                    m.insert("percentage".to_string(), "90".to_string());
                    m
                },
            };
            self.alert_manager.send_alert(alert).await;
            self.billing.alerts_sent.push("90_percent".to_string());
        }

        if usage_percentage > 0.75 && !self.billing.alerts_sent.contains(&"75_percent".to_string()) {
            let alert = ResourceAlert {
                alert_type: AlertType::BudgetThreshold { percentage: 75.0 },
                severity: AlertSeverity::Warning,
                message: format!(
                    "Billing WARNING: 75% of budget used (${:.2} / ${:.2})",
                    self.billing.current_cost, self.billing.cost_limit
                ),
                timestamp: Utc::now(),
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("current_cost".to_string(), self.billing.current_cost.to_string());
                    m.insert("cost_limit".to_string(), self.billing.cost_limit.to_string());
                    m.insert("percentage".to_string(), "75".to_string());
                    m
                },
            };
            self.alert_manager.send_alert(alert).await;
            self.billing.alerts_sent.push("75_percent".to_string());
        }

        if usage_percentage > 0.5 && !self.billing.alerts_sent.contains(&"50_percent".to_string()) {
            let alert = ResourceAlert {
                alert_type: AlertType::BudgetThreshold { percentage: 50.0 },
                severity: AlertSeverity::Info,
                message: format!(
                    "Billing INFO: 50% of budget used (${:.2} / ${:.2})",
                    self.billing.current_cost, self.billing.cost_limit
                ),
                timestamp: Utc::now(),
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("current_cost".to_string(), self.billing.current_cost.to_string());
                    m.insert("cost_limit".to_string(), self.billing.cost_limit.to_string());
                    m.insert("percentage".to_string(), "50".to_string());
                    m
                },
            };
            self.alert_manager.send_alert(alert).await;
            self.billing.alerts_sent.push("50_percent".to_string());
        }
    }

    /// Check and send billing alerts (sync version for compatibility)
    fn check_billing_alerts(&mut self) {
        // Fire and forget async check
        let runtime = tokio::runtime::Handle::try_current();
        if let Ok(rt) = runtime {
            let mut cloned = self.alert_manager.clone();
            let usage_percentage = self.billing.current_cost / self.billing.cost_limit;
            let current_cost = self.billing.current_cost;
            let cost_limit = self.billing.cost_limit;
            
            if usage_percentage > 0.9 && !self.billing.alerts_sent.contains(&"90_percent".to_string()) {
                rt.spawn(async move {
                    let alert = ResourceAlert {
                        alert_type: AlertType::BudgetThreshold { percentage: 90.0 },
                        severity: AlertSeverity::Critical,
                        message: format!(
                            "Billing CRITICAL: 90% of budget used (${:.2} / ${:.2})",
                            current_cost, cost_limit
                        ),
                        timestamp: Utc::now(),
                        metadata: HashMap::new(),
                    };
                    cloned.send_alert(alert).await;
                });
                self.billing.alerts_sent.push("90_percent".to_string());
            }
        }
    }

    /// Get current usage statistics
    pub fn get_usage_stats(&self) -> UsageStats {
        UsageStats {
            api_calls: self.usage.api_calls.load(Ordering::Relaxed),
            tokens_consumed: self.usage.tokens_consumed.load(Ordering::Relaxed),
            compute_seconds: self.usage.compute_seconds.load(Ordering::Relaxed),
            storage_mb: self.usage.storage_mb_used.load(Ordering::Relaxed),
            current_cost: self.billing.current_cost,
            cost_limit: self.billing.cost_limit,
            quota_percentage: self.calculate_quota_percentage(),
        }
    }

    fn calculate_quota_percentage(&self) -> HashMap<String, f64> {
        let mut percentages = HashMap::new();

        percentages.insert(
            "api_calls".to_string(),
            (self.usage.api_calls.load(Ordering::Relaxed) as f64
                / self.quotas.api_calls_per_minute as f64)
                * 100.0,
        );
        percentages.insert(
            "tokens".to_string(),
            (self.usage.tokens_consumed.load(Ordering::Relaxed) as f64
                / self.quotas.tokens_per_hour as f64)
                * 100.0,
        );
        percentages.insert(
            "compute".to_string(),
            (self.usage.compute_seconds.load(Ordering::Relaxed) as f64
                / self.quotas.compute_seconds_per_day as f64)
                * 100.0,
        );
        percentages.insert(
            "storage".to_string(),
            (self.usage.storage_mb_used.load(Ordering::Relaxed) as f64
                / self.quotas.storage_mb as f64)
                * 100.0,
        );
        percentages.insert(
            "budget".to_string(),
            (self.billing.current_cost / self.billing.cost_limit) * 100.0,
        );

        percentages
    }

    /// Reset usage counters (e.g., at end of billing period)
    pub fn reset_usage(&mut self) {
        self.usage = ResourceUsage::default();
        self.billing.reset_hourly_usage();
        self.quota_window_start = Utc::now();
        info!("Resource usage counters reset");
    }
}

impl BillingTracker {
    fn new() -> Self {
        Self {
            current_cost: 0.0,
            cost_limit: 100.0, // Default $100 limit
            hourly_usage: HashMap::new(),
            alerts_sent: Vec::new(),
        }
    }

    fn record_cost(&mut self, cost: f64) {
        self.current_cost += cost;

        let hour_key = Utc::now().format("%Y-%m-%d-%H").to_string();
        *self.hourly_usage.entry(hour_key).or_insert(0.0) += cost;
    }

    fn reset_hourly_usage(&mut self) {
        self.hourly_usage.clear();
        self.alerts_sent.clear();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    pub api_calls: u64,
    pub tokens_consumed: u64,
    pub compute_seconds: u64,
    pub storage_mb: u64,
    pub current_cost: f64,
    pub cost_limit: f64,
    pub quota_percentage: HashMap<String, f64>,
}

/// Smart resource allocator for subagents
pub struct ResourceAllocator {
    available_budget: f64,
    priority_weights: HashMap<String, f64>,
}

impl ResourceAllocator {
    pub fn new(budget: f64) -> Self {
        Self {
            available_budget: budget,
            priority_weights: HashMap::new(),
        }
    }

    /// Allocate resources to a subagent based on priority and estimated cost
    pub fn allocate(
        &mut self,
        subagent_id: &str,
        priority: f64,
        estimated_cost: f64,
    ) -> Allocation {
        if estimated_cost > self.available_budget {
            return Allocation {
                approved: false,
                allocated_budget: 0.0,
                reason: "Insufficient budget".to_string(),
            };
        }

        // Weight by priority (higher priority gets more allocation)
        let weighted_cost = estimated_cost * (1.0 + (1.0 - priority) * 0.5);

        if weighted_cost <= self.available_budget {
            self.available_budget -= weighted_cost;

            Allocation {
                approved: true,
                allocated_budget: weighted_cost,
                reason: format!("Allocated with priority weighting: {:.2}", priority),
            }
        } else {
            Allocation {
                approved: false,
                allocated_budget: 0.0,
                reason: "Cost exceeds available budget after priority weighting".to_string(),
            }
        }
    }

    /// Release allocated resources back to pool
    pub fn release(&mut self, amount: f64) {
        self.available_budget += amount;
    }
}

#[derive(Debug, Clone)]
pub struct Allocation {
    pub approved: bool,
    pub allocated_budget: f64,
    pub reason: String,
}
