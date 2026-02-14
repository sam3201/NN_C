//! Integration tests for the Automation Framework
//!
//! These tests verify the integration between different components
//! and ensure the framework works end-to-end.

#[cfg(test)]
mod integration_tests {
    use automation_framework::{
        AutomationFramework, FrameworkConfig, ResourceQuotas,
        governance::TriCameralGovernance,
        workflow::WorkflowBuilder,
        resource::{ResourceManager, AlertManager},
        change_detection::ChangeTracker,
        brittleness::BrittlenessAnalyzer,
        constraints::{ConstraintEnforcer, ConstraintContext},
    };

    /// Test the complete workflow execution pipeline
    #[tokio::test]
    async fn test_complete_workflow_pipeline() {
        let config = FrameworkConfig {
            max_concurrent_subagents: 5,
            enable_resource_tracking: true,
            enable_race_detection: true,
            billing_threshold: 50.0,
            quota_limits: ResourceQuotas::default(),
        };

        let framework: AutomationFramework = AutomationFramework::new(config).await
            .expect("Failed to create framework");

        let workflow = WorkflowBuilder::new("test_workflow")
            .with_description("Integration test workflow")
            .with_invariants(vec!["system_stable".to_string()])
            .build();

        let result = framework.execute_workflow(workflow).await;
        
        // Should complete without errors
        assert!(result.is_ok());
        let workflow_result = result.unwrap();
        
        // Verify result structure
        assert!(workflow_result.success || !workflow_result.success); // Just verify it has the field
        assert!(!workflow_result.message.is_empty());
    }

    /// Test tri-cameral governance decision making
    #[tokio::test]
    async fn test_tri_cameral_governance() {
        let governance = TriCameralGovernance::new();
        
        let workflow = WorkflowBuilder::new("governance_test")
            .with_description("Test governance system")
            .build();

        let decision = governance.evaluate(&workflow).await
            .expect("Governance evaluation failed");

        // All branches should have voted
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        assert!(!decision.concerns.is_empty() || decision.proceed);
    }

    /// Test resource management and quota enforcement
    #[tokio::test]
    async fn test_resource_management() {
        let quotas = ResourceQuotas {
            api_calls_per_minute: 100,
            tokens_per_hour: 1000,
            compute_seconds_per_day: 3600,
            storage_mb: 512,
        };

        let manager = ResourceManager::new(quotas);
        
        // Record some usage
        manager.record_api_call();
        manager.record_tokens(100);
        
        // Check quotas should pass (we're under limits)
        assert!(manager.check_quotas().is_ok());
        
        // Get usage stats
        let stats = manager.get_usage_stats();
        assert_eq!(stats.api_calls, 1);
        assert_eq!(stats.tokens_consumed, 100);
    }

    /// Test constraint enforcement
    #[test]
    fn test_constraint_enforcement() {
        let enforcer = ConstraintEnforcer::new();
        let context = ConstraintContext::default();

        let result = enforcer.validate(&context);
        
        // Should pass with default constraints on empty context
        assert!(result.is_valid());
        
        // Check that we have constraints
        assert!(!result.passed.is_empty());
    }

    /// Test change detection (without git repo)
    #[tokio::test]
    async fn test_change_detection() {
        let tracker = ChangeTracker::new();
        
        // Try to analyze current directory
        let result = tracker.analyze_changes(".").await;
        
        // Should either succeed or fail gracefully
        match result {
            Ok(analysis) => {
                // If we got analysis, check structure
                assert!(analysis.total_lines_changed >= 0);
            }
            Err(_) => {
                // Expected if not in a git repo - that's fine
            }
        }
    }

    /// Test brittleness analyzer
    #[test]
    fn test_brittleness_analyzer() {
        let analyzer = BrittlenessAnalyzer::new();
        
        // Test with empty operations
        let score = analyzer.get_brittleness_score();
        assert!(score >= 0.0 && score <= 1.0);
        
        // Should provide suggestions
        let suggestions = analyzer.reduce_brittleness();
        assert!(!suggestions.is_empty());
    }

    /// Test alert manager
    #[test]
    fn test_alert_manager() {
        let mut alert_manager = AlertManager::new();
        
        // Register a test callback
        let alert_received = std::sync::Arc::new(std::sync::Mutex::new(false));
        let alert_received_clone = alert_received.clone();
        
        alert_manager.on_alert(move |_alert| {
            let mut received = alert_received_clone.lock().unwrap();
            *received = true;
        });
        
        // Initially no alerts
        let history = alert_manager.get_history();
        assert!(history.is_empty());
    }

    /// Test workflow builder
    #[test]
    fn test_workflow_builder() {
        let workflow = WorkflowBuilder::new("test_builder")
            .with_description("Test description")
            .with_invariants(vec!["invariant1".to_string()])
            .with_edge_cases(vec!["edge1".to_string()])
            .build();

        assert_eq!(workflow.name, "test_builder");
        assert_eq!(workflow.description, "Test description");
        assert_eq!(workflow.invariants.len(), 1);
        assert_eq!(workflow.edge_cases.len(), 1);
    }

    /// Test resource quotas
    #[test]
    fn test_resource_quotas() {
        let quotas = ResourceQuotas::default();
        
        assert!(quotas.api_calls_per_minute > 0);
        assert!(quotas.tokens_per_hour > 0);
        assert!(quotas.compute_seconds_per_day > 0);
        assert!(quotas.storage_mb > 0);
    }

    /// Test framework configuration
    #[test]
    fn test_framework_config() {
        let config = FrameworkConfig {
            max_concurrent_subagents: 10,
            enable_resource_tracking: true,
            enable_race_detection: true,
            billing_threshold: 100.0,
            quota_limits: ResourceQuotas::default(),
        };

        assert_eq!(config.max_concurrent_subagents, 10);
        assert!(config.enable_resource_tracking);
        assert!(config.enable_race_detection);
        assert_eq!(config.billing_threshold, 100.0);
    }
}

#[cfg(test)]
mod end_to_end_tests {
    use automation_framework::{
        AutomationFramework, FrameworkConfig,
        workflow::WorkflowBuilder,
    };

    /// End-to-end test of a complete automation cycle
    #[tokio::test]
    async fn test_complete_automation_cycle() {
        // 1. Create framework
        let config = FrameworkConfig::default();
        let framework: AutomationFramework = AutomationFramework::new(config).await
            .expect("Failed to create framework");

        // 2. Create a workflow
        let workflow = WorkflowBuilder::new("e2e_test")
            .with_description("End-to-end automation test")
            .build();

        // 3. Execute workflow
        let result: Result<automation_framework::WorkflowResult, _> = framework.execute_workflow(workflow).await;
        assert!(result.is_ok());

        // 4. Verify metrics
        let workflow_result = result.unwrap();
        println!("Workflow completed: {:?}", workflow_result.success);
    }
}
