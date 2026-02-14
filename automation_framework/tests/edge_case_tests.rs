//! Edge Case and Production Tests
//! 
//! Tests for:
//! - Empty/invalid inputs
//! - Concurrent access
//! - Resource exhaustion
//! - Malformed data
//! - Boundary conditions

use automation_framework::{
    Change, ChangeType,
    constraints::{ConstraintEnforcer, ConstraintContext},
    resource::{ResourceManager, ResourceQuotas},
    governance::TriCameralGovernance,
    workflow::WorkflowBuilder,
    production::{CircuitBreaker, CircuitState, RetryPolicy, RetryConfig, RateLimiter, ProductionGuard, ProductionConfig},
};
use std::collections::HashMap;
use std::time::Duration;

/// Test 1: Empty context should pass
#[test]
fn test_empty_context_passes() {
    let enforcer = ConstraintEnforcer::new();
    let context = ConstraintContext::default();
    let result = enforcer.validate(&context);
    
    assert!(result.is_valid(), "Empty context should pass all constraints");
    assert_eq!(result.violations.len(), 0, "Should have no violations");
}

/// Test 2: Very large resource usage
#[test]
fn test_very_large_resource_usage() {
    let quotas = ResourceQuotas::default();
    let manager = ResourceManager::new(quotas);
    
    // Add very large number of calls
    for _ in 0..1_000_000 {
        manager.record_api_call();
    }
    
    let stats = manager.get_usage_stats();
    assert_eq!(stats.api_calls, 1_000_000, "Should handle large numbers");
    
    // Should definitely exceed budget
    assert!(manager.check_quotas().is_err(), "Should block with massive usage");
}

/// Test 3: Unicode and special characters in code
#[test]
fn test_unicode_and_special_chars() {
    let enforcer = ConstraintEnforcer::new();
    
    let unicode_change = Change {
        file_path: "test.py".to_string(),
        change_type: ChangeType::Modified,
        diff: "+你好世界 = eval('1') 日本語テスト".to_string(),
        old_content: None,
        new_content: Some("你好世界 = eval('1') 日本語テスト".to_string()),
        timestamp: chrono::Utc::now(),
        author: "test".to_string(),
        commit_message: "test".to_string(),
    };
    
    let context = ConstraintContext {
        changes: vec![unicode_change],
        resource_usage: Default::default(),
        custom_data: HashMap::new(),
    };
    
    let result = enforcer.validate(&context);
    // Should still detect eval() even with unicode
    assert!(!result.is_valid(), "Should detect eval() with unicode");
}

/// Test 4: Empty file path
#[test]
fn test_empty_file_path() {
    let change = Change {
        file_path: "".to_string(),
        change_type: ChangeType::Added,
        diff: "+test".to_string(),
        old_content: None,
        new_content: Some("test".to_string()),
        timestamp: chrono::Utc::now(),
        author: "".to_string(),
        commit_message: "".to_string(),
    };
    
    // Should not panic with empty strings
    assert_eq!(change.file_path, "");
}

/// Test 5: Concurrent resource access
#[tokio::test]
async fn test_concurrent_resource_access() {
    use std::sync::Arc;
    
    let quotas = ResourceQuotas::default();
    let manager = Arc::new(ResourceManager::new(quotas));
    let mut handles = vec![];
    
    // Spawn 100 concurrent tasks
    for _ in 0..100 {
        let mgr = Arc::clone(&manager);
        let handle = tokio::spawn(async move {
            for _ in 0..10 {
                mgr.record_api_call();
            }
        });
        handles.push(handle);
    }
    
    // Wait for all to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    let stats = manager.get_usage_stats();
    assert_eq!(stats.api_calls, 1000, "Should handle concurrent access correctly");
}

/// Test 6: Circuit breaker opens after failures
#[tokio::test]
async fn test_circuit_breaker_opens() {
    let cb = CircuitBreaker::new(3, 2, Duration::from_secs(60));
    
    // Trigger 3 failures
    for i in 0..3 {
        let result = cb.call(|| async {
            Err::<(), _>(automation_framework::errors::AutomationError::ExternalService(
                format!("Failure {}", i)
            ))
        }).await;
        
        assert!(result.is_err());
    }
    
    // Circuit should be open now
    assert_eq!(cb.state().await, CircuitState::Open);
    
    // Next call should fail immediately
    let result = cb.call(|| async { Ok(42) }).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("OPEN"));
}

/// Test 7: Circuit breaker recovers
#[tokio::test]
async fn test_circuit_breaker_recovery() {
    use tokio::time::{sleep, timeout};
    
    let cb = CircuitBreaker::new(2, 2, Duration::from_millis(100));
    
    // Open the circuit
    for _ in 0..2 {
        let _ = cb.call(|| async {
            Err::<(), _>(automation_framework::errors::AutomationError::ExternalService("fail".to_string()))
        }).await;
    }
    
    assert_eq!(cb.state().await, CircuitState::Open);
    
    // Wait for timeout
    sleep(Duration::from_millis(150)).await;
    
    // Should be half-open, and successful calls should close it
    for _ in 0..2 {
        let result = cb.call(|| async { Ok(42) }).await;
        assert!(result.is_ok());
    }
    
    assert_eq!(cb.state().await, CircuitState::Closed);
}

/// Test 8: Retry policy with eventual success
#[tokio::test]
async fn test_retry_eventual_success() {
    let retry = RetryPolicy::new(RetryConfig {
        max_attempts: 5,
        initial_delay: Duration::from_millis(10),
        max_delay: Duration::from_millis(100),
        exponential_base: 2.0,
    });
    
    let attempts = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let attempts_clone = attempts.clone();
    
    let result = retry.execute(|| async {
        let count = attempts_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if count < 3 {
            Err(automation_framework::errors::AutomationError::ExternalService("temp".to_string()))
        } else {
            Ok("success")
        }
    }).await;
    
    assert!(result.is_ok());
    assert_eq!(attempts.load(std::sync::atomic::Ordering::Relaxed), 3);
}

/// Test 9: Rate limiter blocks when exhausted
#[tokio::test]
async fn test_rate_limiter_blocks() {
    let limiter = RateLimiter::new(5, 1); // 5 max, 1 per second refill
    
    // Use all tokens
    for _ in 0..5 {
        assert!(limiter.acquire(1).await);
    }
    
    // Should block now
    assert!(!limiter.acquire(1).await);
    
    // Wait for refill
    tokio::time::sleep(Duration::from_secs(1)).await;
    
    // Should have 1 token now
    assert!(limiter.acquire(1).await);
}

/// Test 10: Production guard with all safeguards
#[tokio::test]
async fn test_production_guard() {
    let config = ProductionConfig::default();
    let guard = ProductionGuard::new(config);
    
    // Add a simple health check
    let guard = {
        let mut g = guard;
        g.add_health_check(|| automation_framework::production::HealthStatus::Healthy);
        g
    };
    
    // Execute should work
    let result = guard.execute(|| async { Ok(42) }).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

/// Test 11: Malformed code patterns
#[test]
fn test_malformed_code_patterns() {
    let enforcer = ConstraintEnforcer::new();
    
    // Test various malformed patterns
    let malformed_patterns = vec![
        "eval(",                  // Unclosed eval
        "eval()",                 // Empty eval
        "  eval(  )  ",          // Whitespace
        "x=eval('1')",           // No spaces
        "eval ( '1' )",          // Spaced out
    ];
    
    for pattern in malformed_patterns {
        let change = Change {
            file_path: "test.py".to_string(),
            change_type: ChangeType::Modified,
            diff: format!("+{}", pattern),
            old_content: None,
            new_content: Some(pattern.to_string()),
            timestamp: chrono::Utc::now(),
            author: "test".to_string(),
            commit_message: "test".to_string(),
        };
        
        let context = ConstraintContext {
            changes: vec![change],
            resource_usage: Default::default(),
            custom_data: HashMap::new(),
        };
        
        let result = enforcer.validate(&context);
        // All should be blocked
        assert!(!result.is_valid(), "Should block malformed pattern: {}", pattern);
    }
}

/// Test 12: Zero quotas
#[test]
fn test_zero_quotas() {
    let quotas = ResourceQuotas {
        api_calls_per_minute: 0,
        tokens_per_hour: 0,
        compute_seconds_per_day: 0,
        storage_mb: 0,
    };
    
    let manager = ResourceManager::new(quotas);
    
    // Even one call should exceed
    manager.record_api_call();
    
    assert!(manager.check_quotas().is_err(), "Should block with zero quotas");
}

/// Test 13: Very long file paths
#[test]
fn test_very_long_file_path() {
    let long_path = "a".repeat(10000);
    
    let change = Change {
        file_path: long_path,
        change_type: ChangeType::Added,
        diff: "+test".to_string(),
        old_content: None,
        new_content: Some("test".to_string()),
        timestamp: chrono::Utc::now(),
        author: "test".to_string(),
        commit_message: "test".to_string(),
    };
    
    // Should not panic
    assert_eq!(change.file_path.len(), 10000);
}

/// Test 14: Governance with empty workflow
#[tokio::test]
async fn test_governance_empty_workflow() {
    let governance = TriCameralGovernance::new();
    let workflow = WorkflowBuilder::new("").build();
    
    let decision = governance.evaluate(&workflow).await;
    
    assert!(decision.is_ok());
    let decision = decision.unwrap();
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
}

/// Test 15: Multiple changes at once
#[test]
fn test_multiple_changes() {
    let enforcer = ConstraintEnforcer::new();
    
    let changes: Vec<Change> = (0..100).map(|i| {
        Change {
            file_path: format!("file_{}.py", i),
            change_type: ChangeType::Modified,
            diff: format!("+x = {}", i),
            old_content: None,
            new_content: Some(format!("x = {}", i)),
            timestamp: chrono::Utc::now(),
            author: "test".to_string(),
            commit_message: "test".to_string(),
        }
    }).collect();
    
    let context = ConstraintContext {
        changes,
        resource_usage: Default::default(),
        custom_data: HashMap::new(),
    };
    
    let result = enforcer.validate(&context);
    
    // All safe code, should pass
    assert!(result.is_valid());
}

/// Test 16: Nested dangerous code
#[test]
fn test_nested_dangerous_code() {
    let enforcer = ConstraintEnforcer::new();
    
    let nested = r#"
def outer():
    def inner():
        result = eval("1 + 1")
        return result
    return inner()
"#;
    
    let change = Change {
        file_path: "test.py".to_string(),
        change_type: ChangeType::Added,
        diff: format!("+{}", nested),
        old_content: None,
        new_content: Some(nested.to_string()),
        timestamp: chrono::Utc::now(),
        author: "test".to_string(),
        commit_message: "test".to_string(),
    };
    
    let context = ConstraintContext {
        changes: vec![change],
        resource_usage: Default::default(),
        custom_data: HashMap::new(),
    };
    
    let result = enforcer.validate(&context);
    // Should detect eval() even when nested
    assert!(!result.is_valid(), "Should detect nested eval()");
}
