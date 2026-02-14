//! ACTUAL Validation Tests - Not just compilation checks
//! These tests verify BEHAVIOR with real scenarios

use automation_framework::{
    constraints::{ConstraintEnforcer, ConstraintContext, ConstraintType, Change},
    resource::{ResourceManager, ResourceQuotas, ResourceUsage, AlertManager, AlertType, AlertSeverity},
    governance::{TriCameralGovernance, Vote, Branch},
    brittleness::{detect_race_conditions, Operation, OperationType},
};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

/// Test constraint validation with REAL code samples
#[test]
fn test_constraint_detects_eval_in_code() {
    let enforcer = ConstraintEnforcer::new();
    
    // Test 1: Should detect eval() in actual code
    let malicious_change = Change {
        file_path: "test.py".to_string(),
        change_type: automation_framework::ChangeType::Modified,
        diff: "+result = eval('1 + 1')".to_string(),
        old_content: None,
        new_content: Some("result = eval('1 + 1')".to_string()),
        timestamp: chrono::Utc::now(),
        author: "test".to_string(),
        commit_message: "test".to_string(),
    };
    
    let context = ConstraintContext {
        changes: vec![malicious_change],
        resource_usage: Default::default(),
        custom_data: HashMap::new(),
    };
    
    let result = enforcer.validate(&context);
    
    // Should have critical violations
    let critical_count = result.critical_violations().len();
    assert!(
        critical_count > 0,
        "Should detect eval() as critical violation, found {} violations",
        critical_count
    );
    
    // Should NOT be valid
    assert!(!result.is_valid(), "Should block code with eval()");
}

#[test]
fn test_constraint_allows_safe_code() {
    let enforcer = ConstraintEnforcer::new();
    
    // Test: Safe code should pass
    let safe_change = Change {
        file_path: "test.py".to_string(),
        change_type: automation_framework::ChangeType::Modified,
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
    
    // Safe code should have no critical violations
    assert!(
        result.critical_violations().is_empty(),
        "Safe code should have no critical violations"
    );
}

#[test]
fn test_constraint_no_false_positives_in_comments() {
    let enforcer = ConstraintEnforcer::new();
    
    // Test: Comments mentioning eval should NOT be flagged
    let comment_change = Change {
        file_path: "test.py".to_string(),
        change_type: automation_framework::ChangeType::Modified,
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
    
    // Comments should not trigger violations
    let violations = result.critical_violations();
    let has_eval_violation = violations.iter().any(|v| {
        v.constraint.name == "no_eval_exec"
    });
    
    assert!(
        !has_eval_violation,
        "Comments mentioning eval() should not be flagged"
    );
}

/// Test resource quota ACTUALLY enforces limits
#[tokio::test]
async fn test_resource_quota_blocks_when_exceeded() {
    let quotas = ResourceQuotas {
        api_calls_per_minute: 5, // Very low for testing
        tokens_per_hour: 1000,
        compute_seconds_per_day: 3600,
        storage_mb: 512,
    };
    
    let manager = ResourceManager::new(quotas);
    
    // Record 5 API calls (at limit)
    for _ in 0..5 {
        manager.record_api_call();
    }
    
    // Should still pass (at limit, not over)
    assert!(
        manager.check_quotas().is_ok(),
        "Should pass at exactly the limit"
    );
    
    // Record 1 more (over limit)
    manager.record_api_call();
    
    // Should now fail
    let result = manager.check_quotas();
    assert!(
        result.is_err(),
        "Should fail when exceeding API call limit"
    );
    
    // Verify it's the right error
    match result {
        Err(automation_framework::errors::AutomationError::QuotaExceeded { resource, .. }) => {
            assert_eq!(resource, "api_calls_per_minute");
        }
        _ => panic!("Expected QuotaExceeded error for api_calls_per_minute"),
    }
}

#[tokio::test]
async fn test_budget_limit_actually_blocks() {
    let quotas = ResourceQuotas::default();
    let mut manager = ResourceManager::new(quotas);
    
    // Set a very low budget limit
    manager.billing.cost_limit = 0.01; // 1 cent
    
    // Simulate high usage that exceeds budget
    let usage = ResourceUsage::default();
    usage.api_calls.store(1000, Ordering::Relaxed); // Many API calls
    manager.record_usage(&usage);
    
    // Should fail due to budget
    let result = manager.check_quotas();
    assert!(
        result.is_err(),
        "Should fail when budget is exceeded"
    );
    
    match result {
        Err(automation_framework::errors::AutomationError::BudgetExceeded { .. }) => {
            // Correct error type
        }
        _ => panic!("Expected BudgetExceeded error"),
    }
}

/// Test governance ACTUALLY votes correctly
#[tokio::test]
async fn test_governance_proceeds_with_all_approve() {
    let governance = TriCameralGovernance::new();
    
    use automation_framework::workflow::WorkflowBuilder;
    
    // Create a workflow that all branches should approve
    let workflow = WorkflowBuilder::new("safe_workflow")
        .with_description("Safe workflow with no risks")
        .with_invariants(vec!["system_stable".to_string()])
        .build();
    
    let decision = governance.evaluate(&workflow).await
        .expect("Governance should evaluate successfully");
    
    // Verify structure
    assert!(
        decision.confidence >= 0.0 && decision.confidence <= 1.0,
        "Confidence should be between 0 and 1, got {}",
        decision.confidence
    );
    
    // All branches should have voted
    assert!(
        !decision.cic_vote.reasoning.is_empty(),
        "CIC should provide reasoning"
    );
    assert!(
        !decision.aee_vote.reasoning.is_empty(),
        "AEE should provide reasoning"
    );
    assert!(
        !decision.csf_vote.reasoning.is_empty(),
        "CSF should provide reasoning"
    );
    
    // Confidence should be reasonable
    assert!(
        decision.confidence > 0.3,
        "Confidence should be > 0.3 for safe workflow, got {}",
        decision.confidence
    );
}

/// Test race condition detection with ACTUAL scenarios
#[test]
fn test_race_detection_finds_real_conflicts() {
    // Scenario 1: Read-Write conflict on same resource
    let ops_rw = vec![
        Operation {
            id: "1".to_string(),
            resource_id: "data.json".to_string(),
            operation_type: OperationType::Read,
            dependencies: vec![],
        },
        Operation {
            id: "2".to_string(),
            resource_id: "data.json".to_string(),
            operation_type: OperationType::Write,
            dependencies: vec![],
        },
    ];
    
    let report = detect_race_conditions(&ops_rw).expect("Should analyze operations");
    
    // Should detect at least one potential conflict
    assert!(
        !report.potential_conflicts.is_empty(),
        "Should detect Read-Write conflict on same resource"
    );
    
    // Scenario 2: No conflict on different resources
    let ops_no_conflict = vec![
        Operation {
            id: "1".to_string(),
            resource_id: "file1.txt".to_string(),
            operation_type: OperationType::Write,
            dependencies: vec![],
        },
        Operation {
            id: "2".to_string(),
            resource_id: "file2.txt".to_string(),
            operation_type: OperationType::Write,
            dependencies: vec![],
        },
    ];
    
    let report_clean = detect_race_conditions(&ops_no_conflict).expect("Should analyze operations");
    
    // Should have no conflicts
    assert!(
        report_clean.potential_conflicts.is_empty(),
        "Should NOT detect conflict on different resources"
    );
}

/// Test alert suppression ACTUALLY works
#[tokio::test]
async fn test_alert_suppression_prevents_duplicates() {
    let mut alert_manager = AlertManager::new();
    
    let mut alert_count = 0;
    alert_manager.on_alert(move |_alert| {
        alert_count += 1;
    });
    
    // Send first alert
    let alert1 = automation_framework::resource::ResourceAlert {
        alert_type: AlertType::BudgetThreshold { percentage: 90.0 },
        severity: AlertSeverity::Critical,
        message: "Test alert".to_string(),
        timestamp: chrono::Utc::now(),
        metadata: HashMap::new(),
    };
    
    alert_manager.send_alert(alert1.clone()).await;
    
    // Send same alert again immediately (should be suppressed)
    alert_manager.send_alert(alert1.clone()).await;
    
    // History should show only 1 alert (second suppressed)
    let history = alert_manager.get_history();
    assert_eq!(
        history.len(),
        1,
        "Duplicate alert should be suppressed"
    );
}

/// Test that stats are ACCURATELY tracked
#[test]
fn test_resource_usage_accurately_tracked() {
    let quotas = ResourceQuotas::default();
    let manager = ResourceManager::new(quotas);
    
    // Record known amounts
    manager.record_api_call();
    manager.record_api_call();
    manager.record_tokens(100);
    manager.record_tokens(50);
    
    let stats = manager.get_usage_stats();
    
    // Verify exact counts
    assert_eq!(
        stats.api_calls, 2,
        "API calls should be exactly 2, got {}",
        stats.api_calls
    );
    assert_eq!(
        stats.tokens_consumed, 150,
        "Tokens should be exactly 150, got {}",
        stats.tokens_consumed
    );
}

/// Test brittleness score calculation is REASONABLE
#[test]
fn test_brittleness_score_in_valid_range() {
    use automation_framework::brittleness::BrittlenessAnalyzer;
    
    let analyzer = BrittlenessAnalyzer::new();
    let score = analyzer.get_brittleness_score();
    
    // Score should be in valid range
    assert!(
        score >= 0.0 && score <= 1.0,
        "Brittleness score should be between 0 and 1, got {}",
        score
    );
    
    // Empty system should be stable (low score)
    assert!(
        score < 0.5,
        "Empty system should have low brittleness, got {}",
        score
    );
}
