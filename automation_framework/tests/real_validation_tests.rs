//! REAL Validation Tests - Testing Actual Behavior

use automation_framework::{
    constraints::{ConstraintEnforcer, ConstraintContext, Change},
    resource::{ResourceManager, ResourceQuotas, ResourceUsage},
    governance::TriCameralGovernance,
    workflow::WorkflowBuilder,
    brittleness::BrittlenessAnalyzer,
};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

/// Test 1: Constraint validation with REAL malicious code
#[test]
fn test_constraint_detects_dangerous_patterns() {
    println!("\n🔍 Test 1: Constraint Detection");
    
    let enforcer = ConstraintEnforcer::new();
    
    // Test with actual dangerous code
    let test_cases = vec![
        ("eval('1+1')", true, "eval() should be detected"),
        ("exec('print(1)')", true, "exec() should be detected"),
        ("api_key = 'sk-secret'", true, "API key should be detected"),
        ("x = 1 + 1", false, "Safe code should pass"),
    ];
    
    for (code, should_violate, description) in test_cases {
        let change = Change {
            file_path: "test.py".to_string(),
            change_type: automation_framework::ChangeType::Modified,
            diff: format!("+{}", code),
            old_content: None,
            new_content: Some(code.to_string()),
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
        let has_critical = !result.critical_violations().is_empty();
        
        println!("  {}: {}", 
            if has_critical == should_violate { "✅" } else { "❌" },
            description
        );
        
        if has_critical != should_violate {
            println!("    Expected violation: {}, Got: {}", should_violate, has_critical);
            println!("    Code: {}", code);
        }
        
        assert_eq!(
            has_critical, should_violate,
            "{} - Code: {}", description, code
        );
    }
}

/// Test 2: Resource tracking ACCURACY
#[test]
fn test_resource_tracking_accuracy() {
    println!("\n🔍 Test 2: Resource Tracking Accuracy");
    
    let quotas = ResourceQuotas::default();
    let manager = ResourceManager::new(quotas);
    
    // Record exact amounts
    manager.record_api_call();
    manager.record_api_call();
    manager.record_api_call();
    manager.record_tokens(100);
    manager.record_tokens(200);
    
    let stats = manager.get_usage_stats();
    
    // Verify exact counts
    let api_calls_correct = stats.api_calls == 3;
    let tokens_correct = stats.tokens_consumed == 300;
    
    println!("  {}: API calls tracked correctly (expected 3, got {})",
        if api_calls_correct { "✅" } else { "❌" },
        stats.api_calls
    );
    println!("  {}: Tokens tracked correctly (expected 300, got {})",
        if tokens_correct { "✅" } else { "❌" },
        stats.tokens_consumed
    );
    
    assert_eq!(stats.api_calls, 3, "API calls should be exactly 3");
    assert_eq!(stats.tokens_consumed, 300, "Tokens should be exactly 300");
}

/// Test 3: Resource quota enforcement
#[test]
fn test_quota_enforcement() {
    println!("\n🔍 Test 3: Quota Enforcement");
    
    let quotas = ResourceQuotas {
        api_calls_per_minute: 5,
        tokens_per_hour: 1000,
        compute_seconds_per_day: 3600,
        storage_mb: 512,
    };
    
    let manager = ResourceManager::new(quotas);
    
    // Should pass initially
    let initial_check = manager.check_quotas().is_ok();
    println!("  {}: Initial check passes", if initial_check { "✅" } else { "❌" });
    assert!(initial_check, "Should pass initially");
    
    // Add calls up to limit
    for _ in 0..5 {
        manager.record_api_call();
    }
    
    // At limit should still pass
    let at_limit = manager.check_quotas().is_ok();
    println!("  {}: At limit (5/5) passes", if at_limit { "✅" } else { "❌" });
    assert!(at_limit, "Should pass at exactly the limit");
    
    // Exceed limit
    manager.record_api_call();
    
    // Should now fail
    let over_limit = manager.check_quotas().is_err();
    println!("  {}: Over limit (6/5) blocked", if over_limit { "✅" } else { "❌" });
    assert!(over_limit, "Should fail when over limit");
}

/// Test 4: Governance produces valid decisions
#[tokio::test]
async fn test_governance_decision_validity() {
    println!("\n🔍 Test 4: Governance Decision Validity");
    
    let governance = TriCameralGovernance::new();
    let workflow = WorkflowBuilder::new("test_workflow")
        .with_description("Test workflow")
        .build();
    
    let decision = governance.evaluate(&workflow).await
        .expect("Governance should evaluate");
    
    // Check confidence is valid
    let confidence_valid = decision.confidence >= 0.0 && decision.confidence <= 1.0;
    println!("  {}: Confidence in valid range ({})",
        if confidence_valid { "✅" } else { "❌" },
        decision.confidence
    );
    assert!(
        confidence_valid,
        "Confidence should be between 0 and 1, got {}",
        decision.confidence
    );
    
    // Check all branches voted
    let all_voted = !decision.cic_vote.reasoning.is_empty() 
        && !decision.aee_vote.reasoning.is_empty()
        && !decision.csf_vote.reasoning.is_empty();
    println!("  {}: All branches provided reasoning", if all_voted { "✅" } else { "❌" });
    assert!(all_voted, "All branches should provide reasoning");
    
    // Check decision is reasonable
    let decision_reasonable = decision.confidence > 0.2;
    println!("  {}: Decision confidence reasonable ({})",
        if decision_reasonable { "✅" } else { "❌" },
        decision.confidence
    );
    assert!(decision_reasonable, "Confidence should be reasonable");
}

/// Test 5: Brittleness score validity
#[test]
fn test_brittleness_score_validity() {
    println!("\n🔍 Test 5: Brittleness Score Validity");
    
    let analyzer = BrittlenessAnalyzer::new();
    let score = analyzer.get_brittleness_score();
    
    // Score should be in valid range
    let score_valid = score >= 0.0 && score <= 1.0;
    println!("  {}: Score in valid range [0,1] ({})",
        if score_valid { "✅" } else { "❌" },
        score
    );
    assert!(
        score_valid,
        "Score should be between 0 and 1, got {}",
        score
    );
    
    // Empty system should be stable
    let is_stable = score < 0.5;
    println!("  {}: Empty system marked as stable ({} < 0.5)",
        if is_stable { "✅" } else { "❌" },
        score
    );
    assert!(is_stable, "Empty system should have low brittleness");
    
    // Should provide suggestions
    let suggestions = analyzer.reduce_brittleness();
    let has_suggestions = !suggestions.is_empty();
    println!("  {}: Provides brittleness reduction suggestions",
        if has_suggestions { "✅" } else { "❌" }
    );
    assert!(!suggestions.is_empty(), "Should provide suggestions");
}

/// Test 6: Constraint summary accuracy
#[test]
fn test_constraint_summary_accuracy() {
    println!("\n🔍 Test 6: Constraint Summary Accuracy");
    
    let enforcer = ConstraintEnforcer::new();
    let context = ConstraintContext::default();
    
    let result = enforcer.validate(&context);
    let summary = result.summary();
    
    // Summary should contain meaningful info
    let has_passed_count = summary.contains("passed");
    let has_numbers = summary.chars().any(|c| c.is_digit(10));
    
    println!("  {}: Summary contains pass/fail info", if has_passed_count { "✅" } else { "❌" });
    println!("  {}: Summary contains numeric stats", if has_numbers { "✅" } else { "❌" });
    
    assert!(has_passed_count, "Summary should mention passed count");
    assert!(has_numbers, "Summary should contain numeric data");
    
    // Empty context should pass all
    let all_passed = result.is_valid();
    println!("  {}: Empty context passes all constraints", if all_passed { "✅" } else { "❌" });
    assert!(all_passed, "Empty context should pass");
}

/// Test 7: Hard constraints actually enforced
#[test]
fn test_hard_constraints_block_execution() {
    println!("\n🔍 Test 7: Hard Constraint Enforcement");
    
    let enforcer = ConstraintEnforcer::new();
    
    // Get hard constraints
    let hard_constraints = enforcer.hard_constraints();
    let has_hard_constraints = !hard_constraints.is_empty();
    
    println!("  {}: Has {} hard constraints defined",
        if has_hard_constraints { "✅" } else { "❌" },
        hard_constraints.len()
    );
    assert!(!hard_constraints.is_empty(), "Should have hard constraints");
    
    // Check that budget_limit is hard
    let has_budget_limit = hard_constraints.iter().any(|c| c.name == "budget_limit");
    println!("  {}: Budget limit is hard constraint", if has_budget_limit { "✅" } else { "❌" });
    assert!(has_budget_limit, "Budget limit should be hard constraint");
    
    // Check that no_eval_exec is hard
    let has_no_eval = hard_constraints.iter().any(|c| c.name == "no_eval_exec");
    println!("  {}: No eval/exec is hard constraint", if has_no_eval { "✅" } else { "❌" });
    assert!(has_no_eval, "No eval/exec should be hard constraint");
}

/// Test 8: Resource usage cloning preserves data
#[test]
fn test_resource_usage_clone_preserves_data() {
    println!("\n🔍 Test 8: Resource Usage Clone");
    
    let usage = ResourceUsage::default();
    usage.api_calls.store(42, Ordering::Relaxed);
    usage.tokens_consumed.store(1000, Ordering::Relaxed);
    
    let cloned = usage.clone();
    
    let api_calls_match = cloned.api_calls.load(Ordering::Relaxed) == 42;
    let tokens_match = cloned.tokens_consumed.load(Ordering::Relaxed) == 1000;
    
    println!("  {}: Cloned API calls match (expected 42, got {})",
        if api_calls_match { "✅" } else { "❌" },
        cloned.api_calls.load(Ordering::Relaxed)
    );
    println!("  {}: Cloned tokens match (expected 1000, got {})",
        if tokens_match { "✅" } else { "❌" },
        cloned.tokens_consumed.load(Ordering::Relaxed)
    );
    
    assert_eq!(
        cloned.api_calls.load(Ordering::Relaxed), 42,
        "Cloned API calls should match"
    );
    assert_eq!(
        cloned.tokens_consumed.load(Ordering::Relaxed), 1000,
        "Cloned tokens should match"
    );
}
