//! ACTUAL Validation Tests - Testing Real Behavior
//! These tests verify the fixes work correctly

use automation_framework::{
    Change, ChangeType,
    constraints::{ConstraintEnforcer, ConstraintContext},
    resource::{ResourceManager, ResourceQuotas},
    governance::TriCameralGovernance,
    workflow::WorkflowBuilder,
    brittleness::BrittlenessAnalyzer,
};
use std::collections::HashMap;

/// Test 1: Constraint ACTUALLY detects dangerous code
#[test]
fn test_constraint_blocks_eval() {
    println!("\n🔍 Test 1: Constraint Detection");
    
    let enforcer = ConstraintEnforcer::new();
    
    // Test with ACTUAL dangerous code
    let dangerous_change = Change {
        file_path: "test.py".to_string(),
        change_type: ChangeType::Modified,
        diff: "+result = eval('1 + 1')".to_string(),
        old_content: None,
        new_content: Some("result = eval('1 + 1')".to_string()),
        timestamp: chrono::Utc::now(),
        author: "test".to_string(),
        commit_message: "test".to_string(),
    };
    
    let context = ConstraintContext {
        changes: vec![dangerous_change],
        resource_usage: Default::default(),
        custom_data: HashMap::new(),
    };
    
    let result = enforcer.validate(&context);
    
    // Should ACTUALLY block this
    let blocked = !result.is_valid();
    println!("  {}: eval() detection - {}", 
        if blocked { "✅" } else { "❌" },
        if blocked { "BLOCKED" } else { "ALLOWED (BUG!)" }
    );
    
    assert!(blocked, "Should block code containing eval()");
}

/// Test 2: Constraint ALLOWS safe code
#[test]
fn test_constraint_allows_safe_code() {
    println!("\n🔍 Test 2: Safe Code Passes");
    
    let enforcer = ConstraintEnforcer::new();
    
    let safe_change = Change {
        file_path: "test.py".to_string(),
        change_type: ChangeType::Modified,
        diff: "+x = 1 + 1".to_string(),
        old_content: None,
        new_content: Some("x = 1 + 1".to_string()),
        timestamp: chrono::Utc::now(),
        author: "test".to_string(),
        commit_message: "test".to_string(),
    };
    
    let context = ConstraintContext {
        changes: vec![safe_change],
        resource_usage: Default::default(),
        custom_data: HashMap::new(),
    };
    
    let result = enforcer.validate(&context);
    
    let passed = result.is_valid();
    println!("  {}: Safe code - {}",
        if passed { "✅" } else { "❌" },
        if passed { "ALLOWED" } else { "BLOCKED (FALSE POSITIVE!)" }
    );
    
    assert!(passed, "Should allow safe code");
}

/// Test 3: Constraint ALLOWS comments about eval
#[test]
fn test_constraint_allows_eval_comments() {
    println!("\n🔍 Test 3: Comment Detection");
    
    let enforcer = ConstraintEnforcer::new();
    
    let comment_change = Change {
        file_path: "test.py".to_string(),
        change_type: ChangeType::Modified,
        diff: "+# Don't use eval() in production".to_string(),
        old_content: None,
        new_content: Some("# Don't use eval() in production".to_string()),
        timestamp: chrono::Utc::now(),
        author: "test".to_string(),
        commit_message: "test".to_string(),
    };
    
    let context = ConstraintContext {
        changes: vec![comment_change],
        resource_usage: Default::default(),
        custom_data: HashMap::new(),
    };
    
    let result = enforcer.validate(&context);
    
    let passed = result.is_valid();
    println!("  {}: Comment with eval() - {}",
        if passed { "✅" } else { "❌" },
        if passed { "ALLOWED (correct)" } else { "BLOCKED (false positive!)" }
    );
    
    assert!(passed, "Should allow comments mentioning eval()");
}

/// Test 4: Resource quota ACTUALLY enforced
#[test]
fn test_quota_actually_enforced() {
    println!("\n🔍 Test 4: Quota Enforcement");
    
    let quotas = ResourceQuotas {
        api_calls_per_minute: 3, // Very low for testing
        tokens_per_hour: 1000,
        compute_seconds_per_day: 3600,
        storage_mb: 512,
    };
    
    let manager = ResourceManager::new(quotas);
    
    // Record 3 calls (at limit)
    manager.record_api_call();
    manager.record_api_call();
    manager.record_api_call();
    
    let at_limit_ok = manager.check_quotas().is_ok();
    println!("  {}: At limit (3/3) - {}",
        if at_limit_ok { "✅" } else { "❌" },
        if at_limit_ok { "PASSES" } else { "BLOCKED" }
    );
    assert!(at_limit_ok, "Should pass at exactly the limit");
    
    // Exceed limit
    manager.record_api_call();
    
    let over_limit_blocked = manager.check_quotas().is_err();
    println!("  {}: Over limit (4/3) - {}",
        if over_limit_blocked { "✅" } else { "❌" },
        if over_limit_blocked { "BLOCKED (correct)" } else { "PASSES (BUG!)" }
    );
    assert!(over_limit_blocked, "Should block when over limit");
}

/// Test 5: Budget ACTUALLY enforced
#[test]
fn test_budget_actually_enforced() {
    println!("\n🔍 Test 5: Budget Enforcement");
    
    let quotas = ResourceQuotas::default();
    let manager = ResourceManager::new(quotas);
    
    // Record usage that exceeds $100 budget
    // Cost = calls * 0.001 + tokens/1000 * 0.002
    // We need > $100, so: calls > 100,000 OR tokens > 50,000,000
    // Let's use API calls: 100,001 calls = $100.001
    
    for _ in 0..100_001 {
        manager.record_api_call();
    }
    
    let blocked = manager.check_quotas().is_err();
    println!("  {}: Budget exceeded (100,001 calls) - {}",
        if blocked { "✅" } else { "❌" },
        if blocked { "BLOCKED (correct)" } else { "PASSES (BUG!)" }
    );
    
    assert!(blocked, "Should block when budget exceeded");
}

/// Test 6: Resource tracking ACCURACY
#[test]
fn test_resource_tracking_accuracy() {
    println!("\n🔍 Test 6: Resource Tracking");
    
    let quotas = ResourceQuotas::default();
    let manager = ResourceManager::new(quotas);
    
    // Record exact amounts
    manager.record_api_call();
    manager.record_api_call();
    manager.record_api_call();
    manager.record_tokens(100);
    manager.record_tokens(200);
    
    let stats = manager.get_usage_stats();
    
    let api_correct = stats.api_calls == 3;
    let tokens_correct = stats.tokens_consumed == 300;
    
    println!("  {}: API calls = {} (expected 3)",
        if api_correct { "✅" } else { "❌" },
        stats.api_calls
    );
    println!("  {}: Tokens = {} (expected 300)",
        if tokens_correct { "✅" } else { "❌" },
        stats.tokens_consumed
    );
    
    assert_eq!(stats.api_calls, 3, "API calls should be exactly 3");
    assert_eq!(stats.tokens_consumed, 300, "Tokens should be exactly 300");
}

/// Test 7: Governance produces valid decisions
#[tokio::test]
async fn test_governance_produces_decisions() {
    println!("\n🔍 Test 7: Governance Decisions");
    
    let governance = TriCameralGovernance::new();
    let workflow = WorkflowBuilder::new("test_workflow")
        .with_description("Test workflow")
        .build();
    
    let decision = governance.evaluate(&workflow).await
        .expect("Governance should evaluate");
    
    // Check confidence is valid
    let confidence_valid = decision.confidence >= 0.0 && decision.confidence <= 1.0;
    println!("  {}: Confidence = {:.2} (valid range)",
        if confidence_valid { "✅" } else { "❌" },
        decision.confidence
    );
    assert!(confidence_valid, "Confidence should be between 0 and 1");
    
    // Check all branches voted
    let all_voted = !decision.cic_vote.reasoning.is_empty() 
        && !decision.aee_vote.reasoning.is_empty()
        && !decision.csf_vote.reasoning.is_empty();
    println!("  {}: All branches voted", if all_voted { "✅" } else { "❌" });
    assert!(all_voted, "All branches should provide reasoning");
    
    // Check decision is reasonable
    let reasonable = decision.confidence > 0.2;
    println!("  {}: Decision confidence reasonable", if reasonable { "✅" } else { "❌" });
    assert!(reasonable, "Confidence should be reasonable");
}

/// Test 8: Brittleness score validity
#[test]
fn test_brittleness_validity() {
    println!("\n🔍 Test 8: Brittleness Score");
    
    let analyzer = BrittlenessAnalyzer::new();
    let score = analyzer.get_brittleness_score();
    
    let score_valid = score >= 0.0 && score <= 1.0;
    println!("  {}: Score = {:.2} (valid range)",
        if score_valid { "✅" } else { "❌" },
        score
    );
    assert!(score_valid, "Score should be between 0 and 1");
    
    let is_stable = score < 0.5;
    println!("  {}: Empty system marked as stable ({} < 0.5)",
        if is_stable { "✅" } else { "❌" },
        score
    );
    assert!(is_stable, "Empty system should have low brittleness");
}

/// Test 9: Constraint summary accuracy
#[test]
fn test_constraint_summary() {
    println!("\n🔍 Test 9: Constraint Summary");
    
    let enforcer = ConstraintEnforcer::new();
    let context = ConstraintContext::default();
    
    let result = enforcer.validate(&context);
    let summary = result.summary();
    
    let has_info = summary.contains("passed") || summary.contains("failed");
    println!("  {}: Summary contains status info", if has_info { "✅" } else { "❌" });
    assert!(has_info, "Summary should contain pass/fail info");
    
    let valid = result.is_valid();
    println!("  {}: Empty context passes ({} violations)",
        if valid { "✅" } else { "❌" },
        result.violations.len()
    );
    assert!(valid, "Empty context should pass");
}

/// Test 10: Secrets detection
#[test]
fn test_secrets_detection() {
    println!("\n🔍 Test 10: Secrets Detection");
    
    let enforcer = ConstraintEnforcer::new();
    
    // Test with API key
    let secret_change = Change {
        file_path: "config.py".to_string(),
        change_type: ChangeType::Modified,
        diff: "+api_key = 'sk-1234567890abcdef'".to_string(),
        old_content: None,
        new_content: Some("api_key = 'sk-1234567890abcdef'".to_string()),
        timestamp: chrono::Utc::now(),
        author: "test".to_string(),
        commit_message: "test".to_string(),
    };
    
    let context = ConstraintContext {
        changes: vec![secret_change],
        resource_usage: Default::default(),
        custom_data: HashMap::new(),
    };
    
    let result = enforcer.validate(&context);
    let blocked = !result.is_valid();
    
    println!("  {}: API key detection - {}",
        if blocked { "✅" } else { "❌" },
        if blocked { "BLOCKED" } else { "ALLOWED" }
    );
    
    assert!(blocked, "Should block code with API keys");
}
