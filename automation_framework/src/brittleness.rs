//! Brittleness detection and reduction with real race condition analysis

use crate::errors::{AutomationError, Result};
use crate::{ConflictType, Operation, OperationType, RaceCondition, RaceConditionReport, Severity};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Standalone function to detect race conditions
pub fn detect_race_conditions(operations: &[Operation]) -> Result<RaceConditionReport> {
    let analyzer = BrittlenessAnalyzer::new();
    analyzer.detect_race_conditions(operations)
}

/// Resource access tracker for detecting conflicts
#[derive(Debug, Clone)]
pub struct ResourceAccessTracker {
    /// Maps resource_id -> list of operations accessing it
    resource_accesses: HashMap<String, Vec<ResourceAccess>>,
    /// Current operation being tracked
    current_operations: Vec<Operation>,
}

#[derive(Debug, Clone)]
pub struct ResourceAccess {
    pub operation_id: String,
    pub access_type: AccessType,
    pub timestamp: std::time::Instant,
    pub thread_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AccessType {
    Read,
    Write,
    Delete,
}

/// Dependency graph for operation ordering
#[derive(Debug, Default)]
pub struct DependencyGraph {
    nodes: HashMap<String, OperationNode>,
    edges: Vec<(String, String, DependencyType)>,
}

#[derive(Debug, Clone)]
pub struct OperationNode {
    pub operation: Operation,
    pub in_degree: usize,
    pub out_degree: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DependencyType {
    Data,     // Operation B needs data from Operation A
    Control,  // Operation B can only run after A completes
    Resource, // Both operations need the same resource
}

/// Brittleness analyzer
pub struct BrittlenessAnalyzer {
    tracker: Arc<RwLock<ResourceAccessTracker>>,
    dependency_graph: Arc<RwLock<DependencyGraph>>,
}

/// Deadlock detection
#[derive(Debug, Default)]
pub struct DeadlockDetector {
    resource_wait_graph: HashMap<String, HashSet<String>>, // resource -> waiting operations
    operation_holdings: HashMap<String, HashSet<String>>,  // operation -> held resources
}

impl BrittlenessAnalyzer {
    pub fn new() -> Self {
        Self {
            tracker: Arc::new(RwLock::new(ResourceAccessTracker::new())),
            dependency_graph: Arc::new(RwLock::new(DependencyGraph::default())),
        }
    }

    /// Register an operation for tracking
    pub fn register_operation(&self, operation: Operation) {
        let mut tracker = self.tracker.write();

        // Track resource accesses
        let access_type = match operation.operation_type {
            OperationType::Read => AccessType::Read,
            OperationType::Write => AccessType::Write,
            OperationType::Delete => AccessType::Delete,
            _ => AccessType::Read,
        };

        let access = ResourceAccess {
            operation_id: operation.id.clone(),
            access_type: access_type.clone(),
            timestamp: std::time::Instant::now(),
            thread_id: None, // Could be set if tracking threads
        };

        tracker
            .resource_accesses
            .entry(operation.resource_id.clone())
            .or_insert_with(Vec::new)
            .push(access);

        tracker.current_operations.push(operation.clone());

        // Update dependency graph
        self.update_dependencies(&operation);
    }

    /// Update dependency graph with new operation
    fn update_dependencies(&self, operation: &Operation) {
        let mut graph = self.dependency_graph.write();

        // Add node
        graph.nodes.insert(
            operation.id.clone(),
            OperationNode {
                operation: operation.clone(),
                in_degree: operation.dependencies.len(),
                out_degree: 0,
            },
        );

        // Add edges from dependencies
        for dep_id in &operation.dependencies {
            graph
                .edges
                .push((dep_id.clone(), operation.id.clone(), DependencyType::Data));

            if let Some(node) = graph.nodes.get_mut(dep_id) {
                node.out_degree += 1;
            }
        }
    }

    /// Detect race conditions in current operations
    pub fn detect_race_conditions(&self, operations: &[Operation]) -> Result<RaceConditionReport> {
        let mut conflicts = Vec::new();
        let mut recommendations = Vec::new();

        // Group operations by resource
        let mut resource_groups: HashMap<String, Vec<&Operation>> = HashMap::new();
        for op in operations {
            resource_groups
                .entry(op.resource_id.clone())
                .or_insert_with(Vec::new)
                .push(op);
        }

        // Check each resource group for conflicts
        for (resource_id, ops) in resource_groups {
            if ops.len() < 2 {
                continue; // No conflict possible with single operation
            }

            // Check all pairs
            for i in 0..ops.len() {
                for j in (i + 1)..ops.len() {
                    let op1 = ops[i];
                    let op2 = ops[j];

                    if let Some(conflict) = self.check_conflict(op1, op2, &resource_id) {
                        conflicts.push(conflict);
                    }
                }
            }
        }

        // Generate recommendations
        if !conflicts.is_empty() {
            recommendations.push(format!(
                "Found {} potential conflicts. Consider:",
                conflicts.len()
            ));

            // Check if we need mutexes
            let write_write_count = conflicts
                .iter()
                .filter(|c| c.conflict_type == ConflictType::WriteWrite)
                .count();
            if write_write_count > 0 {
                recommendations.push(format!(
                    "- Add write locks for {} resources with write-write conflicts",
                    write_write_count
                ));
            }

            // Check for read-write conflicts
            let read_write_count = conflicts
                .iter()
                .filter(|c| c.conflict_type == ConflictType::ReadWrite)
                .count();
            if read_write_count > 0 {
                recommendations.push(format!(
                    "- Consider read-write locks for {} resources",
                    read_write_count
                ));
            }
        }

        Ok(RaceConditionReport {
            potential_conflicts: conflicts,
            recommendations,
        })
    }

    /// Check if two operations conflict
    fn check_conflict(
        &self,
        op1: &Operation,
        op2: &Operation,
        resource: &str,
    ) -> Option<RaceCondition> {
        // Check if they're already ordered via dependencies
        if self.has_dependency_path(&op1.id, &op2.id) || self.has_dependency_path(&op2.id, &op1.id)
        {
            return None; // Already ordered, no race
        }

        let conflict_type = match (op1.operation_type.clone(), op2.operation_type.clone()) {
            // Write-Write conflict
            (OperationType::Write, OperationType::Write) => ConflictType::WriteWrite,

            // Read-Write conflict (both directions)
            (OperationType::Read, OperationType::Write)
            | (OperationType::Write, OperationType::Read) => ConflictType::ReadWrite,

            // Delete conflicts
            (OperationType::Delete, _) | (_, OperationType::Delete) => ConflictType::DeleteAccess,

            // Execute conflicts
            (OperationType::Execute, _) | (_, OperationType::Execute) => ConflictType::Ordering,

            // Read-Read is fine
            (OperationType::Read, OperationType::Read) => return None,
        };

        let severity = match conflict_type {
            ConflictType::WriteWrite => Severity::Critical,
            ConflictType::DeleteAccess => Severity::Critical,
            ConflictType::ReadWrite => Severity::Warning,
            ConflictType::Ordering => Severity::Info,
        };

        Some(RaceCondition {
            operations: vec![op1.id.clone(), op2.id.clone()],
            resource: resource.to_string(),
            conflict_type,
            severity,
        })
    }

    /// Check if there's a dependency path from op1 to op2
    fn has_dependency_path(&self, from: &str, to: &str) -> bool {
        let graph = self.dependency_graph.read();

        // BFS to find path
        let mut visited = HashSet::new();
        let mut queue = vec![from];

        while let Some(current) = queue.pop() {
            if current == to {
                return true;
            }

            if visited.insert(current) {
                // Find edges from current
                for (src, dst, _) in &graph.edges {
                    if src == current {
                        queue.push(dst);
                    }
                }
            }
        }

        false
    }

    /// Get brittleness score (0.0 = stable, 1.0 = very brittle)
    pub fn get_brittleness_score(&self) -> f64 {
        let tracker = self.tracker.read();
        let graph = self.dependency_graph.read();

        let mut score: f64 = 0.0;

        // Factor 1: High concurrency on same resources
        for (resource, accesses) in &tracker.resource_accesses {
            if accesses.len() > 5 {
                score += 0.2;
                warn!(
                    "High contention on resource {}: {} accesses",
                    resource,
                    accesses.len()
                );
            }
        }

        // Factor 2: Missing dependencies
        let total_ops = graph.nodes.len();
        let ops_with_deps: usize = graph.nodes.values().filter(|n| n.in_degree > 0).count();

        if total_ops > 0 {
            let dependency_ratio = ops_with_deps as f64 / total_ops as f64;
            if dependency_ratio < 0.5 {
                score += 0.2;
                warn!(
                    "Low dependency declaration: {:.0}%",
                    dependency_ratio * 100.0
                );
            }
        }

        score.min(1.0)
    }

    /// Reduce brittleness by suggesting improvements
    pub fn reduce_brittleness(&self) -> Vec<String> {
        let mut suggestions = Vec::new();
        let score = self.get_brittleness_score();

        if score < 0.3 {
            suggestions.push("System is relatively stable".to_string());
            return suggestions;
        }

        suggestions.push(
            "Add explicit dependencies between operations accessing shared resources".to_string(),
        );

        suggestions.push("Consider batching operations to reduce contention".to_string());

        suggestions
    }
}

impl ResourceAccessTracker {
    pub fn new() -> Self {
        Self {
            resource_accesses: HashMap::new(),
            current_operations: Vec::new(),
        }
    }
}

impl Default for ResourceAccessTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BrittlenessAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
