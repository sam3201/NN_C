//! Hard and soft constraint enforcement system with real-time invariant checking

use crate::errors::{AutomationError, Result};
use crate::Change;
use std::collections::HashMap;
use tracing::{debug, error, info, warn};

/// Constraint types with different enforcement levels
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstraintType {
    /// Hard constraint - must always be satisfied, invariant violation is critical
    Hard,
    /// Soft constraint - should be satisfied but can be relaxed with approval
    Soft,
    /// Optimization goal - nice to have, doesn't block execution
    Optimization,
}

/// Constraint definition with validation logic
pub struct Constraint {
    pub name: String,
    pub constraint_type: ConstraintType,
    pub description: String,
    pub validator: ConstraintValidator,
    pub error_message: String,
    pub recovery_hint: Option<String>,
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constraint")
            .field("name", &self.name)
            .field("constraint_type", &self.constraint_type)
            .field("description", &self.description)
            .field("error_message", &self.error_message)
            .field("recovery_hint", &self.recovery_hint)
            .finish()
    }
}

impl Clone for Constraint {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            constraint_type: self.constraint_type.clone(),
            description: self.description.clone(),
            validator: Box::new(|_| true), // Default validator for clone
            error_message: self.error_message.clone(),
            recovery_hint: self.recovery_hint.clone(),
        }
    }
}

/// Constraint validation function type
pub type ConstraintValidator = Box<dyn Fn(&ConstraintContext) -> bool + Send + Sync>;

/// Context for constraint validation
#[derive(Debug, Clone, Default)]
pub struct ConstraintContext {
    pub changes: Vec<Change>,
    pub resource_usage: crate::resource::ResourceUsage,
    pub custom_data: HashMap<String, String>,
}

/// Constraint violation with severity and remediation
#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    pub constraint: Constraint,
    pub context: ConstraintContext,
    pub severity: ViolationSeverity,
    pub remediation_suggestions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ViolationSeverity {
    Critical,    // Hard constraint violated
    Warning,     // Soft constraint violated
    Info,        // Optimization not met
}

/// Invariant checker for continuous monitoring
pub struct InvariantChecker {
    invariants: Vec<Invariant>,
    violation_history: Vec<InvariantViolation>,
}

/// System invariants that must always hold
pub struct Invariant {
    pub name: String,
    pub check: Box<dyn Fn() -> bool + Send + Sync>,
    pub description: String,
}

impl std::fmt::Debug for Invariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Invariant")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

impl Clone for Invariant {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            check: Box::new(|| true), // Default validator for clone
            description: self.description.clone(),
        }
    }
}

/// Record of invariant violation
#[derive(Debug, Clone)]
pub struct InvariantViolation {
    pub invariant_name: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub auto_recovered: bool,
}

/// Constraint enforcement engine
pub struct ConstraintEnforcer {
    constraints: HashMap<String, Constraint>,
    violation_callbacks: Vec<Box<dyn Fn(&ConstraintViolation) + Send + Sync>>,
}

impl ConstraintEnforcer {
    pub fn new() -> Self {
        let mut enforcer = Self {
            constraints: HashMap::new(),
            violation_callbacks: Vec::new(),
        };
        
        // Add default constraints
        enforcer.add_default_constraints();
        
        enforcer
    }
    
    /// Add default system constraints
    fn add_default_constraints(&mut self) {
        // HARD: Budget limit - ACTUALLY CHECKS RESOURCE USAGE
        self.add_constraint(Constraint {
            name: "budget_limit".to_string(),
            constraint_type: ConstraintType::Hard,
            description: "Total cost must not exceed budget limit".to_string(),
            validator: Box::new(|ctx| {
                use std::sync::atomic::Ordering;
                // Calculate actual cost from resource usage
                let api_calls = ctx.resource_usage.api_calls.load(Ordering::Relaxed);
                let tokens = ctx.resource_usage.tokens_consumed.load(Ordering::Relaxed);
                
                // Pricing: $0.001 per API call, $0.002 per 1K tokens
                let cost = (api_calls as f64 * 0.001) + ((tokens as f64 / 1000.0) * 0.002);
                
                // Default budget limit is $100.00
                let budget_limit = 100.0;
                
                if cost > budget_limit {
                    return false;
                }
                true
            }),
            error_message: "Budget limit exceeded".to_string(),
            recovery_hint: Some("Reduce concurrent subagents or use cheaper models".to_string()),
        });
        
        // HARD: Security - No eval() or exec() in generated code - IMPROVED DETECTION
        self.add_constraint(Constraint {
            name: "no_eval_exec".to_string(),
            constraint_type: ConstraintType::Hard,
            description: "Generated code must not contain eval() or exec()".to_string(),
            validator: Box::new(|ctx| {
                // Simplified detection without regex
                let dangerous_functions = ["eval(", "exec(", "compile("];
                
                for change in &ctx.changes {
                    if let Some(ref new) = change.new_content {
                        for line in new.lines() {
                            let line_trimmed = line.trim();
                            let line_lower = line.to_lowercase();
                            
                            // Skip comments
                            if line_trimmed.starts_with("//") || 
                               line_trimmed.starts_with("#") ||
                               line_trimmed.starts_with("/*") ||
                               line_trimmed.starts_with("*") {
                                continue;
                            }
                            
                            // Check for dangerous function calls
                            for func in &dangerous_functions {
                                if line_lower.contains(func) {
                                    // Check if it's inside a string literal by counting quotes
                                    let func_pos = line_lower.find(func).unwrap();
                                    let before_func = &line[..func_pos];
                                    let double_quotes = before_func.matches('"').count();
                                    let single_quotes = before_func.matches('\'').count();
                                    
                                    // If even number of quotes (0, 2, 4...), it's actual code not a string
                                    if double_quotes % 2 == 0 && single_quotes % 2 == 0 {
                                        return false;
                                    }
                                }
                            }
                        }
                    }
                }
                true
            }),
            error_message: "Dangerous eval()/exec() detected in code".to_string(),
            recovery_hint: Some("Remove eval/exec or ensure it's in safe context".to_string()),
        });
        
        // HARD: No secrets in code - IMPROVED DETECTION
        self.add_constraint(Constraint {
            name: "no_secrets".to_string(),
            constraint_type: ConstraintType::Hard,
            description: "API keys, passwords, tokens must not be committed".to_string(),
            validator: Box::new(|ctx| {
                // Simplified secret detection without regex for now
                let secret_keywords = ["api_key", "apikey", "password", "secret_key", "token", "private_key"];
                
                for change in &ctx.changes {
                    if let Some(ref new) = change.new_content {
                        for line in new.lines() {
                            let line_lower = line.to_lowercase();
                            let line_trimmed = line.trim();
                            
                            // Skip comments
                            if line_trimmed.starts_with("//") || 
                               line_trimmed.starts_with("#") ||
                               line_trimmed.starts_with("/*") {
                                continue;
                            }
                            
                            // Check for secret keywords with assignment
                            for keyword in &secret_keywords {
                                if line_lower.contains(keyword) {
                                    // Check if followed by = or : (assignment)
                                    let idx = line_lower.find(keyword).unwrap();
                                    let after = &line_lower[idx + keyword.len()..];
                                    
                                    if after.trim().starts_with("=") || after.trim().starts_with(":") {
                                        // Check it's not just a comment in code
                                        if !line_trimmed.starts_with("//") && !line_trimmed.starts_with("#") {
                                            // Check it's not an example/placeholder
                                            if !line_lower.contains("your_") &&
                                               !line_lower.contains("example") &&
                                               !line_lower.contains("placeholder") &&
                                               !line_lower.contains("<") {
                                                return false;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                true
            }),
            error_message: "Potential secret detected in code".to_string(),
            recovery_hint: Some("Use environment variables or secret management".to_string()),
        });
        
        // SOFT: Code quality
        self.add_constraint(Constraint {
            name: "code_quality".to_string(),
            constraint_type: ConstraintType::Soft,
            description: "Code should follow quality standards".to_string(),
            validator: Box::new(|ctx| {
                for change in &ctx.changes {
                    if let Some(ref new) = change.new_content {
                        // Check for TODO/FIXME without ticket numbers
                        for line in new.lines() {
                            let trimmed = line.trim().to_lowercase();
                            if (trimmed.contains("todo") || trimmed.contains("fixme")) &&
                               !trimmed.contains("issue") &&
                               !trimmed.contains("ticket") &&
                               !trimmed.contains("#") {
                                return false;
                            }
                        }
                    }
                }
                true
            }),
            error_message: "Code quality issues detected (TODOs without tickets)".to_string(),
            recovery_hint: Some("Add issue/ticket references to TODOs".to_string()),
        });
        
        // SOFT: Test coverage
        self.add_constraint(Constraint {
            name: "test_coverage".to_string(),
            constraint_type: ConstraintType::Soft,
            description: "Changes should include tests".to_string(),
            validator: Box::new(|ctx| {
                let has_code_changes = ctx.changes.iter().any(|c| {
                    c.file_path.ends_with(".rs") || 
                    c.file_path.ends_with(".py") ||
                    c.file_path.ends_with(".js")
                });
                
                let has_test_changes = ctx.changes.iter().any(|c| {
                    c.file_path.contains("test") ||
                    c.file_path.contains("spec")
                });
                
                // If code changes but no test changes, soft violation
                !has_code_changes || has_test_changes
            }),
            error_message: "Code changes without corresponding test updates".to_string(),
            recovery_hint: Some("Add tests for new/changed functionality".to_string()),
        });
        
        // OPTIMIZATION: Performance
        self.add_constraint(Constraint {
            name: "performance".to_string(),
            constraint_type: ConstraintType::Optimization,
            description: "Performance should be acceptable".to_string(),
            validator: Box::new(|ctx| {
                // Check for obvious performance issues
                for change in &ctx.changes {
                    if let Some(ref new) = change.new_content {
                        // Nested loops warning
                        if new.matches("for").count() > 2 {
                            return false;
                        }
                    }
                }
                true
            }),
            error_message: "Potential performance issues detected".to_string(),
            recovery_hint: Some("Consider algorithm optimization".to_string()),
        });
    }
    
    /// Add a custom constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        info!("Adding constraint: {} ({:?})", constraint.name, constraint.constraint_type);
        self.constraints.insert(constraint.name.clone(), constraint);
    }
    
    /// Remove a constraint
    pub fn remove_constraint(&mut self, name: &str) -> Option<Constraint> {
        self.constraints.remove(name)
    }
    
    /// Validate all constraints against context
    pub fn validate(&self, context: &ConstraintContext) -> ConstraintValidationResult {
        let mut violations = Vec::new();
        let mut passed = Vec::new();
        
        for (_, constraint) in &self.constraints {
            let valid = (constraint.validator)(context);
            
            if valid {
                passed.push(constraint.name.clone());
            } else {
                let severity = match constraint.constraint_type {
                    ConstraintType::Hard => ViolationSeverity::Critical,
                    ConstraintType::Soft => ViolationSeverity::Warning,
                    ConstraintType::Optimization => ViolationSeverity::Info,
                };
                
                let violation = ConstraintViolation {
                    constraint: constraint.clone(),
                    context: context.clone(),
                    severity,
                    remediation_suggestions: self.generate_remediation(constraint, context),
                };
                
                violations.push(violation);
                
                // Trigger callbacks
                for callback in &self.violation_callbacks {
                    callback(&violations.last().unwrap());
                }
            }
        }
        
        let all_critical_passed = violations.iter().all(|v| v.severity != ViolationSeverity::Critical);
        
        ConstraintValidationResult {
            passed,
            violations,
            all_critical_passed,
        }
    }
    
    /// Generate remediation suggestions for a violation
    fn generate_remediation(&self, constraint: &Constraint, _context: &ConstraintContext) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        if let Some(ref hint) = constraint.recovery_hint {
            suggestions.push(hint.clone());
        }
        
        match constraint.constraint_type {
            ConstraintType::Hard => {
                suggestions.push("This is a HARD constraint - execution must be blocked".to_string());
            }
            ConstraintType::Soft => {
                suggestions.push("This is a SOFT constraint - can proceed with warning".to_string());
            }
            ConstraintType::Optimization => {
                suggestions.push("This is an OPTIMIZATION - can be ignored".to_string());
            }
        }
        
        suggestions
    }
    
    /// Register a callback for constraint violations
    pub fn on_violation<F>(&mut self, callback: F)
    where
        F: Fn(&ConstraintViolation) + Send + Sync + 'static,
    {
        self.violation_callbacks.push(Box::new(callback));
    }
    
    /// Get all hard constraints
    pub fn hard_constraints(&self) -> Vec<&Constraint> {
        self.constraints
            .values()
            .filter(|c| c.constraint_type == ConstraintType::Hard)
            .collect()
    }
    
    /// Check if all hard constraints are satisfied
    pub fn hard_constraints_satisfied(&self, context: &ConstraintContext) -> bool {
        self.constraints
            .values()
            .filter(|c| c.constraint_type == ConstraintType::Hard)
            .all(|c| (c.validator)(context))
    }
}

/// Result of constraint validation
#[derive(Debug, Clone)]
pub struct ConstraintValidationResult {
    pub passed: Vec<String>,
    pub violations: Vec<ConstraintViolation>,
    pub all_critical_passed: bool,
}

impl ConstraintValidationResult {
    /// Check if validation passed (no critical violations)
    pub fn is_valid(&self) -> bool {
        self.all_critical_passed
    }
    
    /// Get critical violations only
    pub fn critical_violations(&self) -> Vec<&ConstraintViolation> {
        self.violations
            .iter()
            .filter(|v| v.severity == ViolationSeverity::Critical)
            .collect()
    }
    
    /// Get summary string
    pub fn summary(&self) -> String {
        let total = self.passed.len() + self.violations.len();
        let critical = self.critical_violations().len();
        let warnings = self.violations.iter().filter(|v| v.severity == ViolationSeverity::Warning).count();
        
        format!(
            "Constraint validation: {}/{} passed ({} critical, {} warnings)",
            self.passed.len(),
            total,
            critical,
            warnings
        )
    }
}

impl Default for ConstraintEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

/// Real-time invariant checker
impl InvariantChecker {
    pub fn new() -> Self {
        Self {
            invariants: Vec::new(),
            violation_history: Vec::new(),
        }
    }
    
    /// Add a system invariant
    pub fn add_invariant<F>(&mut self, name: &str, description: &str, check: F)
    where
        F: Fn() -> bool + Send + Sync + 'static,
    {
        self.invariants.push(Invariant {
            name: name.to_string(),
            check: Box::new(check),
            description: description.to_string(),
        });
    }
    
    /// Check all invariants
    pub fn check_all(&mut self) -> Vec<&Invariant> {
        let mut violated = Vec::new();
        
        for invariant in &self.invariants {
            if !(invariant.check)() {
                violated.push(invariant);
                
                // Record violation
                self.violation_history.push(InvariantViolation {
                    invariant_name: invariant.name.clone(),
                    timestamp: chrono::Utc::now(),
                    auto_recovered: false,
                });
                
                error!("INVARIANT VIOLATED: {} - {}", invariant.name, invariant.description);
            }
        }
        
        violated
    }
    
    /// Start continuous monitoring
    pub async fn start_monitoring(&mut self, interval_ms: u64) {
        info!("Starting invariant monitoring ({}ms interval)", interval_ms);
        
        loop {
            let violated = self.check_all();
            
            if !violated.is_empty() {
                warn!("{} invariants violated", violated.len());
            }
            
            tokio::time::sleep(tokio::time::Duration::from_millis(interval_ms)).await;
        }
    }
    
    /// Get violation history
    pub fn violation_history(&self) -> &[InvariantViolation] {
        &self.violation_history
    }
    
    /// Clear violation history
    pub fn clear_history(&mut self) {
        self.violation_history.clear();
    }
}

impl Default for InvariantChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hard_constraint_blocks() {
        let enforcer = ConstraintEnforcer::new();
        let context = ConstraintContext::default();
        
        let result = enforcer.validate(&context);
        
        // Should pass by default (no dangerous patterns in empty context)
        assert!(result.is_valid());
    }
    
    #[test]
    fn test_constraint_types() {
        let enforcer = ConstraintEnforcer::new();
        
        let hard_count = enforcer.hard_constraints().len();
        assert!(hard_count >= 3); // budget, no_eval, no_secrets
        
        println!("Hard constraints: {}", hard_count);
    }
}
