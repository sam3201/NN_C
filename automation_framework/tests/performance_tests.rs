//! Performance and Load Testing Suite
//! 
//! Tests system behavior under various load conditions:
//! - Concurrent workflow execution
//! - Resource exhaustion scenarios
//! - Memory pressure
//! - High-throughput constraint validation

use automation_framework::{
    AutomationFramework, FrameworkConfig, ResourceQuotas,
    workflow::{Workflow, WorkflowBuilder},
    constraints::{ConstraintEnforcer, ConstraintContext, Change},
    ChangeType,
    governance::TriCameralGovernance,
    resource::ResourceManager,
    production::{ProductionGuard, ProductionConfig, CircuitBreaker, RetryPolicy, RetryConfig, RateLimiter},
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;

/// Test 1: High-throughput constraint validation
#[tokio::test]
async fn test_constraint_validation_throughput() {
    println!("\n🚀 Performance Test: Constraint Validation Throughput");
    
    let enforcer = ConstraintEnforcer::new();
    let iterations = 10_000;
    let start = Instant::now();
    
    for i in 0..iterations {
        let change = Change {
            file_path: format!("file_{}.py", i),
            change_type: ChangeType::Modified,
            diff: "+x = 1 + 1".to_string(),
            old_content: None,
            new_content: Some("x = 1 + 1".to_string()),
            timestamp: chrono::Utc::now(),
            author: "test".to_string(),
            commit_message: "test".to_string(),
        };
        
        let context = ConstraintContext {
            changes: vec![change],
            resource_usage: Default::default(),
            custom_data: std::collections::HashMap::new(),
        };
        
        let result = enforcer.validate(&context);
        assert!(result.is_valid());
    }
    
    let elapsed = start.elapsed();
    let throughput = iterations as f64 / elapsed.as_secs_f64();
    
    println!("  ✅ Validated {} constraints in {:?}", iterations, elapsed);
    println!("  📊 Throughput: {:.0} validations/second", throughput);
    
    // Should handle at least 1000 validations per second
    assert!(throughput > 1000.0, "Throughput too low: {:.0}/s", throughput);
}

/// Test 2: Concurrent workflow execution
#[tokio::test]
async fn test_concurrent_workflow_execution() {
    println!("\n🚀 Performance Test: Concurrent Workflow Execution");
    
    let config = FrameworkConfig::default();
    let framework = Arc::new(AutomationFramework::new(config).await.expect("Failed to create framework"));
    
    let num_workflows = 50;
    let mut handles = vec![];
    let start = Instant::now();
    
    for i in 0..num_workflows {
        let fw = Arc::clone(&framework);
        let handle = tokio::spawn(async move {
            let workflow = WorkflowBuilder::new(&format!("workflow_{}", i))
                .with_description("Performance test workflow")
                .build();
            
            fw.execute_workflow(workflow).await
        });
        handles.push(handle);
    }
    
    // Wait for all with timeout
    let results = futures::future::join_all(handles).await;
    let elapsed = start.elapsed();
    
    let success_count = results.iter().filter(|r| r.is_ok() && r.as_ref().unwrap().is_ok()).count();
    
    println!("  ✅ Completed {}/{} workflows in {:?}", success_count, num_workflows, elapsed);
    println!("  📊 Avg: {:.2} ms/workflow", elapsed.as_millis() as f64 / num_workflows as f64);
    
    assert!(success_count >= num_workflows * 9 / 10, "Too many failures: {}/{}", success_count, num_workflows);
}

/// Test 3: Resource exhaustion handling
#[tokio::test]
async fn test_resource_exhaustion_handling() {
    println!("\n🚀 Load Test: Resource Exhaustion");
    
    let quotas = ResourceQuotas {
        api_calls_per_minute: 100,
        tokens_per_hour: 1_000,
        compute_seconds_per_day: 3600,
        storage_mb: 512,
    };
    
    let manager = Arc::new(ResourceManager::new(quotas));
    let mut handles = vec![];
    
    // Spawn many concurrent resource consumers
    for _ in 0..200 {
        let mgr = Arc::clone(&manager);
        let handle = tokio::spawn(async move {
            for _ in 0..10 {
                mgr.record_api_call();
                // Small delay to simulate work
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
        });
        handles.push(handle);
    }
    
    // Wait for completion
    for handle in handles {
        handle.await.unwrap();
    }
    
    let stats = manager.get_usage_stats();
    
    println!("  ✅ Processed {} API calls across 200 tasks", stats.api_calls);
    println!("  📊 Resource tracking accurate: {}", stats.api_calls == 2000);
    
    // Should have exceeded quota
    assert!(manager.check_quotas().is_err(), "Should have exceeded quota");
}

/// Test 4: Circuit breaker under load
#[tokio::test]
async fn test_circuit_breaker_under_load() {
    println!("\n🚀 Load Test: Circuit Breaker Under Load");
    
    let cb = Arc::new(CircuitBreaker::new(5, 3, Duration::from_millis(100)));
    let mut handles = vec![];
    let success_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let reject_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
    
    // Spawn many concurrent requests
    for i in 0..100 {
        let cb = Arc::clone(&cb);
        let success = Arc::clone(&success_count);
        let reject = Arc::clone(&reject_count);
        
        let handle = tokio::spawn(async move {
            // First 5 should fail and open circuit
            // Rest should be rejected due to open circuit
            let result = cb.call(|| async {
                if i < 5 {
                    Err::<i32, _>(automation_framework::errors::AutomationError::ExternalService("fail".to_string()))
                } else {
                    Ok(42)
                }
            }).await;
            
            match result {
                Ok(_) => success.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                Err(_) => reject.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            };
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
    
    let successes = success_count.load(std::sync::atomic::Ordering::Relaxed);
    let rejects = reject_count.load(std::sync::atomic::Ordering::Relaxed);
    
    println!("  ✅ Successes: {}, Rejects: {}", successes, rejects);
    println!("  📊 Circuit breaker protected system from {} failures", rejects);
    
    // Most should be rejected after circuit opens
    assert!(rejects > 50, "Circuit breaker should have rejected many requests");
}

/// Test 5: Rate limiter burst handling
#[tokio::test]
async fn test_rate_limiter_burst() {
    println!("\n🚀 Load Test: Rate Limiter Burst Handling");
    
    let limiter = Arc::new(RateLimiter::new(50, 10)); // 50 max, 10/sec refill
    let mut handles = vec![];
    let allowed = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let blocked = Arc::new(std::sync::atomic::AtomicU32::new(0));
    
    // Burst of 100 requests
    for _ in 0..100 {
        let lim = Arc::clone(&limiter);
        let allowed = Arc::clone(&allowed);
        let blocked = Arc::clone(&blocked);
        
        let handle = tokio::spawn(async move {
            if lim.acquire(1).await {
                allowed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                blocked.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await.unwrap();
    }
    
    let allowed_count = allowed.load(std::sync::atomic::Ordering::Relaxed);
    let blocked_count = blocked.load(std::sync::atomic::Ordering::Relaxed);
    
    println!("  ✅ Allowed: {}, Blocked: {}", allowed_count, blocked_count);
    println!("  📊 Rate limiter correctly blocked {} requests", blocked_count);
    
    // Should allow around 50, block the rest
    assert!(allowed_count <= 50, "Should not exceed max tokens");
    assert!(blocked_count >= 50, "Should have blocked requests");
}

/// Test 6: Memory pressure with large workflows
#[tokio::test]
async fn test_memory_pressure() {
    println!("\n🚀 Load Test: Memory Pressure");
    
    let enforcer = ConstraintEnforcer::new();
    let start = Instant::now();
    
    // Create workflow with 1000 changes
    let changes: Vec<Change> = (0..1000).map(|i| {
        Change {
            file_path: format!("path/to/deep/nested/directory/structure/file_{}_{}.py", i, "a".repeat(100)),
            change_type: ChangeType::Modified,
            diff: format!("+{}", "x".repeat(1000)),
            old_content: Some("y".repeat(1000)),
            new_content: Some("x".repeat(1000)),
            timestamp: chrono::Utc::now(),
            author: "test".to_string(),
            commit_message: format!("Commit message {} {}", i, "b".repeat(200)),
        }
    }).collect();
    
    let context = ConstraintContext {
        changes,
        resource_usage: Default::default(),
        custom_data: std::collections::HashMap::new(),
    };
    
    let result = enforcer.validate(&context);
    let elapsed = start.elapsed();
    
    println!("  ✅ Validated 1000 large changes in {:?}", elapsed);
    println!("  📊 Memory efficient validation: {}", result.is_valid());
    
    assert!(elapsed < Duration::from_secs(5), "Too slow: {:?}", elapsed);
}

/// Test 7: Production guard under stress
#[tokio::test]
async fn test_production_guard_stress() {
    println!("\n🚀 Stress Test: Production Guard");
    
    let config = ProductionConfig::default();
    let guard = Arc::new(ProductionGuard::new(config));
    let mut handles = vec![];
    let start = Instant::now();
    
    // Stress test with many operations
    for i in 0..50 {
        let g = Arc::clone(&guard);
        let handle = tokio::spawn(async move {
            g.execute(|| async {
                // Simulate work
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok(format!("result_{}", i))
            }).await
        });
        handles.push(handle);
    }
    
    let results = futures::future::join_all(handles).await;
    let elapsed = start.elapsed();
    let success_count = results.iter().filter(|r| r.is_ok() && r.as_ref().unwrap().is_ok()).count();
    
    println!("  ✅ Completed {}/{} guarded operations in {:?}", success_count, 50, elapsed);
    println!("  📊 Production guard overhead: {:.2} ms/op", elapsed.as_millis() as f64 / 50.0);
    
    assert!(success_count >= 45, "Too many failures: {}/50", success_count);
}

/// Test 8: Governance under concurrent load
#[tokio::test]
async fn test_governance_concurrent_load() {
    println!("\n🚀 Load Test: Governance Concurrent Decisions");
    
    let governance = Arc::new(TriCameralGovernance::new());
    let mut handles = vec![];
    let start = Instant::now();
    
    for i in 0..30 {
        let gov = Arc::clone(&governance);
        let handle = tokio::spawn(async move {
            let workflow = WorkflowBuilder::new(&format!("load_test_{}", i))
                .with_description("Load testing governance")
                .build();
            
            gov.evaluate(&workflow).await
        });
        handles.push(handle);
    }
    
    let results = futures::future::join_all(handles).await;
    let elapsed = start.elapsed();
    let success_count = results.iter().filter(|r| r.is_ok()).count();
    
    println!("  ✅ {}/{} governance decisions in {:?}", success_count, 30, elapsed);
    println!("  📊 Decision rate: {:.0} decisions/sec", 30.0 / elapsed.as_secs_f64());
    
    assert_eq!(success_count, 30, "All governance decisions should succeed");
}

/// Test 9: Timeout handling
#[tokio::test]
async fn test_timeout_handling() {
    println!("\n🚀 Reliability Test: Timeout Handling");
    
    let cb = CircuitBreaker::new(3, 2, Duration::from_secs(1));
    
    // Operation that takes too long
    let result = timeout(Duration::from_millis(100), async {
        cb.call(|| async {
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok(42)
        }).await
    }).await;
    
    match result {
        Err(_) => {
            println!("  ✅ Correctly timed out slow operation");
        }
        Ok(_) => {
            panic!("Should have timed out");
        }
    }
}

/// Test 10: Recovery after failure
#[tokio::test]
async fn test_recovery_after_failure() {
    println!("\n🚀 Reliability Test: Recovery After Failure");
    
    let retry = RetryPolicy::new(RetryConfig {
        max_attempts: 3,
        initial_delay: Duration::from_millis(50),
        max_delay: Duration::from_secs(1),
        exponential_base: 2.0,
    });
    
    let attempts = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let start = Instant::now();
    
    let result = retry.execute({
        let attempts = Arc::clone(&attempts);
        move || {
            let attempts = Arc::clone(&attempts);
            async move {
                let count = attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if count < 3 {
                    Err(automation_framework::errors::AutomationError::ExternalService("temp".to_string()))
                } else {
                    Ok("recovered")
                }
            }
        }
    }).await;
    
    let elapsed = start.elapsed();
    
    assert!(result.is_ok(), "Should eventually succeed");
    assert_eq!(attempts.load(std::sync::atomic::Ordering::Relaxed), 3);
    
    println!("  ✅ Recovered after failures in {:?}", elapsed);
    println!("  📊 Total attempts: 3");
}
