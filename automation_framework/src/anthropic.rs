//! Anthropic Claude integration for advanced reasoning and analysis
//!
//! Provides integration with Anthropic's Claude API for:
//! - Complex architectural decisions
//! - Safety and ethical analysis
//! - Long-context understanding
//! - Chain-of-thought reasoning

use crate::errors::{AutomationError, Result};
use crate::governance::{GovernanceDecision, TriCameralGovernance, BranchVote, Branch, Vote};
use crate::workflow::Workflow;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Anthropic API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f64,
    pub timeout_seconds: u64,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            model: "claude-3-5-sonnet-20241022".to_string(),
            max_tokens: 4096,
            temperature: 0.7,
            timeout_seconds: 60,
        }
    }
}

/// Claude reasoning capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningType {
    /// Constitutional AI reasoning with safety considerations
    Constitutional,
    /// Long-context document analysis
    LongContext,
    /// Step-by-step problem solving
    ChainOfThought,
    /// Nuanced interpretation
    NuancedInterpretation,
    /// Ethical analysis
    EthicalAnalysis,
}

/// Request for Claude consultation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeRequest {
    pub question: String,
    pub context: Option<String>,
    pub constraints: Vec<String>,
    pub reasoning_type: ReasoningType,
    pub reasoning_depth: ReasoningDepth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningDepth {
    Brief,
    Standard,
    Detailed,
    Exhaustive,
}

/// Claude's response with structured reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeResponse {
    pub answer: String,
    pub reasoning: Vec<String>,
    pub confidence: f64,
    pub risks: Vec<String>,
    pub recommendations: Vec<String>,
    pub usage: TokenUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost_usd: f64,
}

/// Anthropic integration client
pub struct AnthropicClient {
    config: AnthropicConfig,
    http_client: Client,
    usage_tracker: Arc<RwLock<UsageTracker>>,
}

#[derive(Debug, Default)]
struct UsageTracker {
    total_requests: u64,
    total_tokens: u64,
    total_cost_usd: f64,
    requests_by_model: std::collections::HashMap<String, u64>,
}

impl AnthropicClient {
    pub fn new(config: AnthropicConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(AutomationError::ConfigError {
                message: "ANTHROPIC_API_KEY not set".to_string()
            });
        }

        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| AutomationError::ConfigError { message: e.to_string() })?;

        Ok(Self {
            config,
            http_client,
            usage_tracker: Arc::new(RwLock::new(UsageTracker::default())),
        })
    }

    /// Consult Claude for complex decision support
    pub async fn consult(&self, request: ClaudeRequest) -> Result<ClaudeResponse> {
        info!("Consulting Claude for: {}", request.question);
        
        let prompt = self.build_prompt(&request);
        let response = self.call_api(&prompt).await?;
        
        // Parse and structure the response
        let parsed = self.parse_response(&response)?;
        
        // Track usage
        {
            let mut tracker = self.usage_tracker.write().await;
            tracker.total_requests += 1;
            tracker.total_tokens += (parsed.usage.input_tokens + parsed.usage.output_tokens) as u64;
            tracker.total_cost_usd += parsed.usage.cost_usd;
            *tracker.requests_by_model.entry(self.config.model.clone()).or_insert(0) += 1;
        }
        
        Ok(parsed)
    }

    /// Get advice for tri-cameral governance branches
    pub async fn advise_branch(
        &self,
        branch: &str,
        workflow: &Workflow,
        context: &str,
    ) -> Result<ClaudeResponse> {
        let request = ClaudeRequest {
            question: format!("Advise the {} branch on this workflow", branch),
            context: Some(context.to_string()),
            constraints: vec!["maintain_invariants".to_string()],
            reasoning_type: ReasoningType::Constitutional,
            reasoning_depth: ReasoningDepth::Detailed,
        };

        self.consult(request).await
    }

    /// Perform safety analysis on a proposal
    pub async fn safety_analysis(
        &self,
        feature: &str,
        concerns: Vec<String>,
    ) -> Result<SafetyAnalysis> {
        let request = ClaudeRequest {
            question: format!("Perform safety analysis for: {}", feature),
            context: Some(format!("Concerns: {:?}", concerns)),
            constraints: vec!["no_harm".to_string(), "explainability".to_string()],
            reasoning_type: ReasoningType::EthicalAnalysis,
            reasoning_depth: ReasoningDepth::Detailed,
        };

        let response = self.consult(request).await?;
        
        Ok(SafetyAnalysis {
            feature: feature.to_string(),
            risk_level: self.assess_risk_level(&response),
            concerns: response.risks,
            mitigations: response.recommendations,
            approval_recommended: response.confidence > 0.7,
        })
    }

    /// Architecture review with Claude
    pub async fn architecture_review(
        &self,
        component: &str,
        design_doc: &str,
        focus_areas: Vec<String>,
    ) -> Result<ArchitectureReview> {
        let request = ClaudeRequest {
            question: format!("Review architecture for: {}", component),
            context: Some(format!("Design: {}\n\nFocus areas: {:?}", design_doc, focus_areas)),
            constraints: vec!["scalability".to_string(), "safety".to_string()],
            reasoning_type: ReasoningType::ChainOfThought,
            reasoning_depth: ReasoningDepth::Exhaustive,
        };

        let response = self.consult(request).await?;
        
        Ok(ArchitectureReview {
            component: component.to_string(),
            score: response.confidence,
            strengths: response.reasoning.clone(),
            weaknesses: response.risks,
            recommendations: response.recommendations,
        })
    }

    /// Build the appropriate prompt based on reasoning type
    fn build_prompt(&self, request: &ClaudeRequest) -> String {
        let base = match request.reasoning_type {
            ReasoningType::Constitutional => {
                "You are Claude, an AI assistant with Constitutional AI principles. \
                 You think through problems step-by-step with built-in safety considerations. \
                 Identify potential harms, consider multiple perspectives, and evaluate long-term consequences."
            }
            ReasoningType::ChainOfThought => {
                "Think through this problem step-by-step. Show your reasoning explicitly."
            }
            ReasoningType::EthicalAnalysis => {
                "Perform an ethical analysis considering: beneficence, non-maleficence, autonomy, and justice. \
                 Evaluate against safety, fairness, transparency, and long-term societal impact."
            }
            ReasoningType::LongContext => {
                "Analyze this information carefully, considering all details in the context provided."
            }
            ReasoningType::NuancedInterpretation => {
                "Provide a nuanced interpretation, considering ambiguity, conflicting constraints, \
                 and subtle implications."
            }
        };

        let depth_instructions = match request.reasoning_depth {
            ReasoningDepth::Brief => "Keep your response brief and focused.",
            ReasoningDepth::Standard => "Provide a standard level of detail.",
            ReasoningDepth::Detailed => "Provide detailed analysis with specific examples.",
            ReasoningDepth::Exhaustive => "Provide exhaustive analysis covering all edge cases and implications.",
        };

        format!(
            "{}\n\n{}\n\nQuestion: {}\n\nConstraints: {:?}\n\n{}",
            base,
            depth_instructions,
            request.question,
            request.constraints,
            request.context.as_deref().unwrap_or("No additional context provided.")
        )
    }

    /// Call Anthropic API
    async fn call_api(&self, prompt: &str) -> Result<serde_json::Value> {
        let url = "https://api.anthropic.com/v1/messages";
        
        let body = json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        });

        let response = self.http_client
            .post(url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AutomationError::ExternalService(e.to_string()))?;

        if !response.status().is_success() {
            let error_text: String = response.text().await.unwrap_or_default();
            return Err(AutomationError::ExternalService(format!(
                "Anthropic API error: {}", error_text
            )));
        }

        let json_response: serde_json::Value = response.json().await
            .map_err(|e: reqwest::Error| AutomationError::ExternalService(e.to_string()))?;
        
        Ok(json_response)
    }

    /// Parse API response into structured format
    fn parse_response(&self, raw: &serde_json::Value) -> Result<ClaudeResponse> {
        let content = raw["content"][0]["text"]
            .as_str()
            .unwrap_or("No response content");

        let usage = raw["usage"].clone();
        let input_tokens = usage["input_tokens"].as_u64().unwrap_or(0) as u32;
        let output_tokens = usage["output_tokens"].as_u64().unwrap_or(0) as u32;
        
        // Estimate cost (Claude 3.5 Sonnet: $3/M input, $15/M output)
        let cost_usd = (input_tokens as f64 * 3.0 / 1_000_000.0) + 
                       (output_tokens as f64 * 15.0 / 1_000_000.0);

        // Simple parsing - extract sections
        let reasoning = self.extract_section(content, "Reasoning", "Risks");
        let risks = self.extract_section(content, "Risks", "Recommendations");
        let recommendations = self.extract_section(content, "Recommendations", "");

        Ok(ClaudeResponse {
            answer: content.to_string(),
            reasoning,
            confidence: 0.8, // Placeholder - could use actual analysis
            risks,
            recommendations,
            usage: TokenUsage {
                input_tokens,
                output_tokens,
                cost_usd,
            },
        })
    }

    fn extract_section(&self, content: &str, start: &str, end: &str) -> Vec<String> {
        let start_lower = start.to_lowercase();
        let end_lower = end.to_lowercase();
        
        let lines: Vec<&str> = content.lines().collect();
        let mut in_section = false;
        let mut section_lines = Vec::new();

        for line in &lines {
            let line_lower = line.to_lowercase();
            
            if line_lower.contains(&start_lower) {
                in_section = true;
                continue;
            }
            
            if !end.is_empty() && line_lower.contains(&end_lower) {
                break;
            }
            
            if in_section && !line.trim().is_empty() {
                section_lines.push(line.trim().to_string());
            }
        }

        section_lines
    }

    fn assess_risk_level(&self, response: &ClaudeResponse) -> RiskLevel {
        let risk_score = response.risks.len() as f64 * 0.1;
        if risk_score > 0.7 {
            RiskLevel::High
        } else if risk_score > 0.3 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    /// Get usage statistics
    pub async fn get_usage_stats(&self) -> UsageStats {
        let tracker = self.usage_tracker.read().await;
        UsageStats {
            total_requests: tracker.total_requests,
            total_tokens: tracker.total_tokens,
            total_cost_usd: tracker.total_cost_usd,
            requests_by_model: tracker.requests_by_model.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UsageStats {
    pub total_requests: u64,
    pub total_tokens: u64,
    pub total_cost_usd: f64,
    pub requests_by_model: std::collections::HashMap<String, u64>,
}

#[derive(Debug, Clone)]
pub struct SafetyAnalysis {
    pub feature: String,
    pub risk_level: RiskLevel,
    pub concerns: Vec<String>,
    pub mitigations: Vec<String>,
    pub approval_recommended: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct ArchitectureReview {
    pub component: String,
    pub score: f64,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Extension trait for tri-cameral governance
#[async_trait::async_trait]
pub trait ClaudeGovernanceExt {
    async fn consult_claude(
        &self,
        client: &AnthropicClient,
        question: &str,
        context: Option<&str>,
    ) -> Result<GovernanceDecision>;
}

#[async_trait::async_trait]
impl ClaudeGovernanceExt for TriCameralGovernance {
    async fn consult_claude(
        &self,
        client: &AnthropicClient,
        question: &str,
        context: Option<&str>,
    ) -> Result<GovernanceDecision> {
        let request = ClaudeRequest {
            question: question.to_string(),
            context: context.map(|s| s.to_string()),
            constraints: vec!["maintain_invariants".to_string()],
            reasoning_type: ReasoningType::Constitutional,
            reasoning_depth: ReasoningDepth::Detailed,
        };

        let response = client.consult(request).await?;
        
        // Convert Claude's response to governance decision
        let should_proceed = response.confidence > 0.6 && 
                            !response.risks.iter().any(|r| r.contains("critical") || r.contains("severe"));

        let vote = if should_proceed { Vote::Approve } else { Vote::Reject };

        Ok(GovernanceDecision {
            proceed: should_proceed,
            confidence: response.confidence,
            cic_vote: BranchVote {
                branch: Branch::CIC,
                decision: vote,
                reasoning: response.answer.clone(),
                confidence: response.confidence,
            },
            aee_vote: BranchVote {
                branch: Branch::AEE,
                decision: vote,
                reasoning: "Advised by Claude".to_string(),
                confidence: response.confidence,
            },
            csf_vote: BranchVote {
                branch: Branch::CSF,
                decision: vote,
                reasoning: "Constitutional AI check passed".to_string(),
                confidence: response.confidence,
            },
            concerns: response.risks,
            recommendations: response.recommendations,
            timestamp: chrono::Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_anthropic_config_default() {
        let config = AnthropicConfig::default();
        assert_eq!(config.model, "claude-3-5-sonnet-20241022");
        assert_eq!(config.max_tokens, 4096);
    }

    #[test]
    fn test_prompt_building() {
        let config = AnthropicConfig::default();
        let client = AnthropicClient::new(config);
        
        if let Ok(client) = client {
            let request = ClaudeRequest {
                question: "Test question".to_string(),
                context: Some("Test context".to_string()),
                constraints: vec!["test".to_string()],
                reasoning_type: ReasoningType::ChainOfThought,
                reasoning_depth: ReasoningDepth::Detailed,
            };

            let prompt = client.build_prompt(&request);
            assert!(prompt.contains("Test question"));
            assert!(prompt.contains("Test context"));
        }
    }
}
