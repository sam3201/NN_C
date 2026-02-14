//! Resource management with billing and quota tracking

use crate::errors::{AutomationError, Result};
use crate::ResourceQuotas;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info, warn};

/// Resource manager for tracking usage, billing, and quotas
pub struct ResourceManager {
    quotas: ResourceQuotas,
    usage: ResourceUsage,
    billing: BillingTracker,
    quota_window_start: DateTime<Utc>,
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
        }
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

    /// Check and send billing alerts
    fn check_billing_alerts(&mut self) {
        let usage_percentage = self.billing.current_cost / self.billing.cost_limit;

        if usage_percentage > 0.9 && !self.billing.alerts_sent.contains(&"90_percent".to_string()) {
            warn!(
                "Billing alert: 90% of budget used (${:.2} / ${:.2})",
                self.billing.current_cost, self.billing.cost_limit
            );
            self.billing.alerts_sent.push("90_percent".to_string());
        }

        if usage_percentage > 0.75 && !self.billing.alerts_sent.contains(&"75_percent".to_string())
        {
            info!(
                "Billing notice: 75% of budget used (${:.2} / ${:.2})",
                self.billing.current_cost, self.billing.cost_limit
            );
            self.billing.alerts_sent.push("75_percent".to_string());
        }

        if usage_percentage > 0.5 && !self.billing.alerts_sent.contains(&"50_percent".to_string()) {
            info!(
                "Billing notice: 50% of budget used (${:.2} / ${:.2})",
                self.billing.current_cost, self.billing.cost_limit
            );
            self.billing.alerts_sent.push("50_percent".to_string());
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
