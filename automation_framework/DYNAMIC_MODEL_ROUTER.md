# Automation Framework - Dynamic Model Router

## Overview

The Automation Framework includes an intelligent **Dynamic Model Router** that automatically selects the best AI model for each task based on:
- Task characteristics (type, complexity, context size)
- Model capabilities (reasoning, coding, speed, cost)
- Budget constraints and usage quotas
- Historical performance

## Features

### 🎯 Automatic Model Selection

The router analyzes each task and selects the optimal model from available options:

```python
from automation_bridge import select_best_model

# Automatically selects best model
model = select_best_model("Implement a secure authentication system")
# Returns: 'claude-3-5-sonnet' (high reliability for security tasks)
```

### 🔄 Dynamic Model Switching

Automatically switch models between tasks:

```python
from automation_bridge import auto_switch_model

# Task 1: Quick check - Uses fast, cheap model
model1 = auto_switch_model("Quick status check")
# Returns: 'claude-3-haiku'

# Task 2: Deep analysis - Uses powerful model
model2 = auto_switch_model("Complex architectural reasoning")
# Returns: 'claude-3-5-sonnet'

# Task 3: Safety-critical - Uses most reliable model
model3 = auto_switch_model("Security audit of authentication code")
# Returns: 'claude-3-5-sonnet' (highest reliability)
```

### 💰 Budget-Aware Routing

The router automatically adjusts model selection based on budget usage:

- **Low usage (< 30%)**: Prioritize quality, cost is secondary
- **Medium usage (30-70%)**: Balance cost and quality
- **High usage (> 70%)**: Prioritize cost-effective models

```python
from automation_bridge import ModelRouter

router = ModelRouter()

# When budget is tight, automatically uses cheaper models
router.daily_budget = 50.0
router.current_spend = 45.0  # 90% used

model = router.select_model("General analysis task")
# Returns: 'local-llm' or 'claude-3-haiku' (cheaper options)
```

## Task Analysis

The router automatically analyzes tasks to determine:

### Task Types
- **Coding**: Implementation, debugging, code review
- **Analysis**: Review, check, analyze
- **Reasoning**: Logic, deduction, complex thinking
- **Creative**: Design, create, innovate
- **Safety-Critical**: Security, safety, invariants

### Complexity Scoring
Tasks are scored 0.0-1.0 based on:
- Keywords in task description
- Context size
- Safety requirements
- Time sensitivity

## Model Profiles

### Registered Models

#### Claude 3.5 Sonnet
```json
{
  "reasoning": 0.95,
  "coding": 0.92,
  "creativity": 0.88,
  "analysis": 0.94,
  "speed": 0.75,
  "cost_per_1k": 0.003,
  "context_window": 200000,
  "reliability": 0.97,
  "specialty": ["reasoning", "analysis", "coding", "safety"]
}
```

#### Claude 3 Haiku
```json
{
  "reasoning": 0.80,
  "coding": 0.78,
  "creativity": 0.75,
  "analysis": 0.82,
  "speed": 0.95,
  "cost_per_1k": 0.00025,
  "context_window": 200000,
  "reliability": 0.90,
  "specialty": ["quick_response", "analysis"]
}
```

#### GPT-4
```json
{
  "reasoning": 0.93,
  "coding": 0.90,
  "creativity": 0.90,
  "analysis": 0.92,
  "speed": 0.70,
  "cost_per_1k": 0.03,
  "context_window": 8000,
  "reliability": 0.95,
  "specialty": ["reasoning", "coding", "creative"]
}
```

#### Local LLM (Ollama)
```json
{
  "reasoning": 0.70,
  "coding": 0.75,
  "creativity": 0.65,
  "analysis": 0.72,
  "speed": 0.85,
  "cost_per_1k": 0.0,
  "context_window": 4000,
  "reliability": 0.80,
  "specialty": ["quick_response", "coding"]
}
```

## Scoring Algorithm

Models are scored using weighted factors:

```
Total Score = 
  Capability Match × 40% +
  Specialty Match × 20% +
  Context Window Fit × 15% +
  Cost Optimization × 15% +
  Historical Performance × 10%
```

### Adjustments

- **Safety-Critical Tasks**: +10% if model reliability > 0.95
- **Time-Sensitive**: +5% if model speed > 0.8
- **Budget Low**: Cost factor weighted higher
- **Context Large**: Context window fit weighted higher

## Integration with Tri-Cameral Governance

The model router integrates seamlessly with tri-cameral governance:

```python
from automation_bridge import AutomationFramework, WorkflowConfig

framework = AutomationFramework()

config = WorkflowConfig(
    name="Implement authentication",
    high_level_plan="Add secure auth system",
    risk_level=0.9  # High risk
)

# Automatically selects most reliable model
result = asyncio.run(framework.execute_workflow(config))
print(f"Model used: {result['model_used']}")
# Output: 'claude-3-5-sonnet' (highest reliability for high-risk tasks)
```

## Usage Examples

### Basic Model Selection

```python
from automation_bridge import select_best_model

# Coding task
model = select_best_model("Implement a Python function to calculate Fibonacci")
print(model)  # claude-3-5-sonnet (excellent coding)

# Quick response
model = select_best_model("Quick yes/no question")
print(model)  # claude-3-haiku (fast, cheap)

# Creative task
model = select_best_model("Design a new UI layout")
print(model)  # gpt-4 (high creativity)
```

### Workflow Integration

```python
from automation_bridge import AutomationFramework, WorkflowConfig

async def run_workflow():
    framework = AutomationFramework()
    
    # Workflow 1: Safety critical
    result1 = await framework.execute_workflow(WorkflowConfig(
        name="Security audit",
        risk_level=0.95
    ))
    print(f"Security audit used: {result1['model_used']}")
    
    # Workflow 2: Quick task
    result2 = await framework.execute_workflow(WorkflowConfig(
        name="Quick check",
        risk_level=0.3
    ))
    print(f"Quick check used: {result2['model_used']}")

asyncio.run(run_workflow())
```

### Custom Model Registration

```python
from automation_framework.src.model_router import ModelRouter, ModelProfile, ModelCapabilities

router = ModelRouter()

# Register custom model
router.register_model(
    "custom-llm".to_string(),
    ModelProfile {
        name: "My Custom LLM".to_string(),
        provider: ModelProvider::Custom("my-api".to_string()),
        capabilities: ModelCapabilities {
            reasoning: 0.85,
            coding: 0.90,
            creativity: 0.70,
            analysis: 0.80,
            long_context: 0.75,
            speed: 0.80,
        },
        cost_per_1k_tokens: 0.001,
        latency_ms: 600,
        context_window: 16000,
        reliability_score: 0.88,
        specialty: vec![TaskType::Coding, TaskType::Analysis],
        quota_remaining: None,
    }
)
```

## Performance Tracking

The router tracks model performance over time:

```rust
// Record model usage
router.record_usage(
    "claude-3-5-sonnet",
    1500,    // tokens used
    800,     // latency in ms
    true     // success
);

// Get performance stats
let stats = router.get_stats();
println!("Total calls: {}", stats.total_calls);
println!("Cost tier: {}", stats.cost_tier);
```

## Budget Management

Set and manage budgets:

```python
from automation_bridge import ModelRouter

router = ModelRouter()
router.daily_budget = 50.0  # $50/day

# Check remaining budget
stats = router.get_stats()
remaining = stats['budget_remaining']
print(f"Budget remaining: ${remaining:.2f}")

# Router automatically switches to cheaper models when budget is low
```

## Best Practices

### 1. Let the Router Decide
Don't manually specify models unless absolutely necessary. The router optimizes for:
- Task requirements
- Cost efficiency
- Performance history

### 2. Provide Context
More context leads to better model selection:

```python
# Good - provides context
model = select_best_model(
    "Review code",
    context="Security-critical authentication module"
)

# Less optimal - no context
model = select_best_model("Review code")
```

### 3. Monitor Usage
Track which models are being used:

```python
stats = framework.get_model_stats()
print(f"Current model: {stats['current_model']}")
print(f"Total spend: ${stats['current_spend']:.2f}")
```

### 4. Set Appropriate Risk Levels
Help the router make better decisions:

```python
# High risk → High reliability model
config = WorkflowConfig(
    name="Security audit",
    risk_level=0.95
)

# Low risk → Fast, cheap model
config = WorkflowConfig(
    name="Generate ideas",
    risk_level=0.3
)
```

## API Reference

### `select_best_model(task: str, context: Optional[str] = None) -> str`
Selects the best model for a given task.

### `auto_switch_model(task: str, context: Optional[str] = None) -> str`
Automatically switches to the best model and tracks the switch.

### `ModelRouter.select_model(task: str, context: Optional[str]) -> Result<String>`
Analyzes task and returns best model name.

### `ModelRouter.auto_switch(task: str, context: Optional[str]) -> Result<String>`
Switches to best model and updates current_model.

### `ModelRouter.record_usage(model: &str, latency_ms: u64, cost: f64, success: bool)`
Records model usage for performance tracking.

### `ModelRouter.get_stats() -> RouterStats`
Returns router statistics including current model, usage, and costs.

## Integration with Complete System

The model router works with all other automation features:

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTOMATION FRAMEWORK                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌────────────┐  │
│  │ Task Input   │────▶│ Model Router │────▶│ Best Model │  │
│  └──────────────┘     └──────────────┘     └────────────┘  │
│         │                                              │    │
│         ▼                                              ▼    │
│  ┌──────────────┐     ┌──────────────┐     ┌────────────┐  │
│  │ Tri-Cameral  │◀───▶│  Workflow    │◀───▶│ Subagents  │  │
│  │ Governance   │     │  Execution   │     │ (Parallel) │  │
│  └──────────────┘     └──────────────┘     └────────────┘  │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Resource Management                      │  │
│  │  • Budget tracking  • Quota enforcement  • Billing   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Future Enhancements

- [ ] Multi-model ensemble for critical tasks
- [ ] Predictive cost estimation
- [ ] A/B testing framework for model comparison
- [ ] Automatic model fine-tuning triggers
- [ ] Integration with model marketplaces
- [ ] Real-time performance monitoring dashboard
