//! Tri-cameral governance system (CIC/AEE/CSF)

use crate::errors::Result;
use crate::workflow::Workflow;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Tri-cameral governance with CIC, AEE, and CSF branches
pub struct TriCameralGovernance {
    cic: ConstructiveIntelligenceCore,
    aee: AdversarialExplorationEngine,
    csf: CoherenceStabilizationField,
    decision_history: HashMap<String, GovernanceDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceDecision {
    pub proceed: bool,
    pub confidence: f64,
    pub cic_vote: BranchVote,
    pub aee_vote: BranchVote,
    pub csf_vote: BranchVote,
    pub concerns: Vec<String>,
    pub recommendations: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchVote {
    pub branch: Branch,
    pub decision: Vote,
    pub reasoning: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Branch {
    CIC,
    AEE,
    CSF,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Vote {
    Approve,      // YES
    Conditional,  // YES with reservations
    Reject,       // NO
    Abstain,      // No opinion
}

impl TriCameralGovernance {
    pub fn new() -> Self {
        Self {
            cic: ConstructiveIntelligenceCore::new(),
            aee: AdversarialExplorationEngine::new(),
            csf: CoherenceStabilizationField::new(),
            decision_history: HashMap::new(),
        }
    }
    
    /// Evaluate a workflow using tri-cameral governance
    pub async fn evaluate(&self, workflow: &Workflow) -> Result<GovernanceDecision> {
        info!("Starting tri-cameral evaluation for workflow: {}", workflow.id);
        
        // Gather votes from all three branches concurrently
        let (cic_vote, aee_vote, csf_vote) = tokio::join!(
            self.cic.evaluate(workflow),
            self.aee.evaluate(workflow),
            self.csf.evaluate(workflow)
        );
        
        let cic_vote = cic_vote?;
        let aee_vote = aee_vote?;
        let csf_vote = csf_vote?;
        
        debug!("CIC vote: {:?} (confidence: {})", cic_vote.decision, cic_vote.confidence);
        debug!("AEE vote: {:?} (confidence: {})", aee_vote.decision, aee_vote.confidence);
        debug!("CSF vote: {:?} (confidence: {})", csf_vote.decision, csf_vote.confidence);
        
        // Apply decision matrix
        let (proceed, concerns, recommendations) = self.apply_decision_matrix(
            &cic_vote,
            &aee_vote,
            &csf_vote,
            workflow,
        );
        
        let confidence = (cic_vote.confidence + aee_vote.confidence + csf_vote.confidence) / 3.0;
        
        let decision = GovernanceDecision {
            proceed,
            confidence,
            cic_vote,
            aee_vote,
            csf_vote,
            concerns,
            recommendations,
            timestamp: chrono::Utc::now(),
        };
        
        if proceed {
            info!("Tri-cameral consensus: PROCEED (confidence: {:.2})", confidence);
        } else {
            warn!("Tri-cameral consensus: REJECT (confidence: {:.2})", confidence);
        }
        
        Ok(decision)
    }
    
    fn apply_decision_matrix(
        &self,
        cic: &BranchVote,
        aee: &BranchVote,
        csf: &BranchVote,
        workflow: &Workflow,
    ) -> (bool, Vec<String>, Vec<String>) {
        let mut concerns = Vec::new();
        let mut recommendations = Vec::new();
        let proceed;
        
        // Decision matrix logic
        match (cic.decision, aee.decision, csf.decision) {
            // All approve → Proceed
            (Vote::Approve, Vote::Approve, Vote::Approve) => {
                proceed = true;
                recommendations.push("Full consensus achieved. Proceed with implementation.".to_string());
            }
            
            // CIC+CSF approve, AEE rejects → Revise
            (Vote::Approve, Vote::Reject, Vote::Approve) => {
                proceed = false;
                concerns.push("AEE has identified risks that need addressing".to_string());
                recommendations.push("Address AEE concerns before proceeding".to_string());
                recommendations.push("Consider additional safety measures".to_string());
            }
            
            // CIC+AEE approve, CSF rejects → Reject
            (Vote::Approve, Vote::Approve, Vote::Reject) => {
                proceed = false;
                concerns.push("CSF: Invariant violations detected".to_string());
                recommendations.push("Workflow violates hard constraints. Must redesign.".to_string());
            }
            
            // CIC approves, AEE+CSF reject → Reject
            (Vote::Approve, Vote::Reject, Vote::Reject) => {
                proceed = false;
                concerns.push("Major issues identified by AEE and CSF".to_string());
                recommendations.push("Significant redesign required".to_string());
            }
            
            // AEE+CSF approve, CIC rejects → Refactor
            (Vote::Reject, Vote::Approve, Vote::Approve) => {
                proceed = false;
                concerns.push("CIC believes there's a better approach".to_string());
                recommendations.push("Consider AEE's alternative suggestions".to_string());
            }
            
            // Any conditional votes → Proceed with caution
            (_, Vote::Conditional, _) | (_, _, Vote::Conditional) => {
                proceed = true;
                concerns.push("Some branches have reservations".to_string());
                recommendations.push("Proceed with monitoring and contingency plans".to_string());
            }
            
            // Default: conservative reject
            _ => {
                proceed = false;
                concerns.push("No clear consensus reached".to_string());
                recommendations.push("Revisit planning phase".to_string());
            }
        }
        
        // Add workflow-specific recommendations
        if workflow.risk_level > 0.7 {
            recommendations.push("High-risk workflow: Implement additional safeguards".to_string());
        }
        
        if !workflow.invariants.is_empty() {
            recommendations.push(format!("Verify {} invariants during execution", workflow.invariants.len()));
        }
        
        (proceed, concerns, recommendations)
    }
    
    /// Get historical decisions
    pub fn get_decision_history(&self) -> &HashMap<String, GovernanceDecision> {
        &self.decision_history
    }
}

/// Constructive Intelligence Core - The Builder
struct ConstructiveIntelligenceCore;

impl ConstructiveIntelligenceCore {
    fn new() -> Self {
        Self
    }
    
    async fn evaluate(&self, workflow: &Workflow) -> Result<BranchVote> {
        // CIC is optimistic and growth-oriented
        let mut confidence: f64 = 0.8;
        let mut reasoning = "CIC: This workflow enables growth and expansion.".to_string();
        
        // Adjust based on workflow characteristics
        if workflow.innovation_score > 0.7 {
            confidence += 0.1;
            reasoning.push_str(" High innovation potential.");
        }
        
        if workflow.resource_requirements.available {
            confidence += 0.05;
            reasoning.push_str(" Resources available.");
        }
        
        // Cap confidence at 0.95
        confidence = confidence.min(0.95);
        
        let decision = if confidence > 0.7 {
            Vote::Approve
        } else {
            Vote::Conditional
        };
        
        Ok(BranchVote {
            branch: Branch::CIC,
            decision,
            reasoning,
            confidence,
        })
    }
}

/// Adversarial Exploration Engine - The Critic
struct AdversarialExplorationEngine;

impl AdversarialExplorationEngine {
    fn new() -> Self {
        Self
    }
    
    async fn evaluate(&self, workflow: &Workflow) -> Result<BranchVote> {
        // AEE is pessimistic and safety-oriented
        let mut concerns = Vec::new();
        let mut confidence: f64 = 0.5;
        
        // Check for edge cases
        if workflow.edge_cases.is_empty() {
            concerns.push("No edge cases identified - potential blind spot".to_string());
            confidence -= 0.1;
        }
        
        // Check risk level
        if workflow.risk_level > 0.6 {
            concerns.push(format!("High risk level: {:.2}", workflow.risk_level));
            confidence -= 0.2;
        }
        
        // Check for rollback strategy
        if !workflow.has_rollback_plan {
            concerns.push("No rollback plan specified".to_string());
            confidence -= 0.15;
        }
        
        let reasoning = if concerns.is_empty() {
            "AEE: No major concerns identified.".to_string()
        } else {
            format!("AEE: Identified {} concerns: {}", 
                concerns.len(),
                concerns.join("; ")
            )
        };
        
        let decision = if confidence > 0.7 {
            Vote::Approve
        } else if confidence > 0.4 {
            Vote::Conditional
        } else {
            Vote::Reject
        };
        
        Ok(BranchVote {
            branch: Branch::AEE,
            decision,
            reasoning,
            confidence: confidence.max(0.0),
        })
    }
}

/// Coherence Stabilization Field - The Guardian
struct CoherenceStabilizationField;

impl CoherenceStabilizationField {
    fn new() -> Self {
        Self
    }
    
    async fn evaluate(&self, workflow: &Workflow) -> Result<BranchVote> {
        // CSF is neutral and invariant-focused
        let mut violations = Vec::new();
        let mut confidence: f64 = 0.9;
        
        // Check each invariant
        for invariant in &workflow.invariants {
            if !self.check_invariant(invariant, workflow) {
                violations.push(invariant.clone());
                confidence -= 0.3;
            }
        }
        
        // Check consistency with existing architecture
        if !workflow.consistent_with_architecture {
            violations.push("Inconsistent with existing architecture".to_string());
            confidence -= 0.2;
        }
        
        let reasoning = if violations.is_empty() {
            "CSF: All invariants preserved. Architecture consistent.".to_string()
        } else {
            format!("CSF: {} invariant violations detected", violations.len())
        };
        
        let decision = if violations.is_empty() {
            Vote::Approve
        } else {
            Vote::Reject
        };
        
        Ok(BranchVote {
            branch: Branch::CSF,
            decision,
            reasoning,
            confidence: confidence.max(0.0),
        })
    }
    
    fn check_invariant(&self, invariant: &str, _workflow: &Workflow) -> bool {
        // Placeholder: In real implementation, check specific invariants
        // For now, assume most invariants pass
        !invariant.contains("violation")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowRequirements {
    pub available: bool,
    pub estimated_cost: f64,
    pub time_estimate_hours: f64,
}

impl Default for WorkflowRequirements {
    fn default() -> Self {
        Self {
            available: true,
            estimated_cost: 0.0,
            time_estimate_hours: 1.0,
        }
    }
}
