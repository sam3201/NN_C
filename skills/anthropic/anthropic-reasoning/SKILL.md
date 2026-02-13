---
name: anthropic-reasoning
description: Anthropic Claude superpowers for complex reasoning and analysis
license: MIT
compatibility: opencode
metadata:
  audience: researchers
  workflow: reasoning
---

## What I Do

Leverage Anthropic Claude's advanced reasoning capabilities for:
- Complex architectural decisions
- Multi-step problem solving
- Ethical and safety analysis
- Long-context understanding
- Nuanced interpretation

## Claude Superpowers

### 1. Constitutional AI Reasoning
Claude thinks through problems step-by-step with built-in safety considerations:
- Identifies potential harms
- Considers multiple stakeholder perspectives
- Evaluates long-term consequences
- Maintains helpful, harmless, honest principles

### 2. Long Context Understanding
Process and reason across very long documents:
- Analyze 100K+ token contexts
- Synthesize information from multiple sources
- Track complex relationships across time
- Maintain coherence in extended reasoning

### 3. Chain-of-Thought Reasoning
Explicit step-by-step thinking:
```
Problem: Should we add self-modification to Phase 2?

Step 1: Identify requirements
  - Must integrate with existing Power/Control
  - Must not break invariants
  - Must be testable

Step 2: Analyze risks
  - Risk of runaway self-modification
  - Risk of breaking existing systems
  - Risk of untestable behavior

Step 3: Evaluate benefits
  - Could optimize performance
  - Could discover better architectures
  - Could adapt to new requirements

Step 4: Balance tradeoffs
  - Benefits outweigh risks if properly constrained
  - Recommend: Add with strict invariant checking
```

### 4. Nuanced Interpretation
Understand and work with:
- Ambiguous requirements
- Conflicting constraints
- Context-dependent meanings
- Subtle implications

### 5. Ethical Analysis
Evaluate decisions against:
- Safety considerations
- Fairness and bias
- Transparency and explainability
- Long-term societal impact

## Integration Patterns

### Pattern 1: Complex Decision Support
```
consult_claude({
  question: "Should we proceed with Phase 2 implementation?",
  context: "Current system has Id/Ego/Superego implemented",
  constraints: ["maintain_invariants", "no_breaking_changes"],
  reasoning_depth: "detailed"
})
```

### Pattern 2: Architecture Review
```
claude_architecture_review({
  component: "Power/Control system",
  design_doc: "path/to/design.md",
  review_focus: ["scalability", "safety", "maintainability"]
})
```

### Pattern 3: Risk Assessment
```
claude_risk_assessment({
  proposal: "Add self-modifying code capability",
  scenarios: ["best_case", "worst_case", "edge_cases"],
  mitigation_required: true
})
```

### Pattern 4: Requirements Clarification
```
claude_clarify({
  ambiguous_requirement: "Make the system more intelligent",
  context: "Phase 2 planning",
  clarification_questions: 5
})
```

## Usage with Tri-Cameral System

Claude can act as an advisor to all three branches:

**For CIC (Builder)**:
- "What optimizations would improve this architecture?"
- "How can we make this more extensible?"
- "What features should we prioritize?"

**For AEE (Critic)**:
- "What could go wrong with this approach?"
- "What edge cases are we missing?"
- "How might this be exploited?"

**For CSF (Guardian)**:
- "Does this violate any invariants?"
- "What safety measures should we add?"
- "How do we ensure stability?"

## Best Practices

1. **Provide Context**: Always give Claude sufficient background
2. **Be Specific**: Ask precise questions for better answers
3. **Challenge Assumptions**: Ask "What if we're wrong about...?"
4. **Consider Tradeoffs**: Ask for pros/cons, not just recommendations
5. **Iterate**: Use follow-up questions to drill deeper

## Example Workflows

**Safety Analysis**:
```
claude_safety_analysis({
  feature: "Self-modifying neural architecture",
  concerns: ["unbounded_growth", "loss_of_control", "emergent_behavior"],
  safeguards: ["invariant_preservation", "human_oversight"]
})
```

**Design Tradeoffs**:
```
claude_tradeoff_analysis({
  options: [
    "Option A: Add Power to existing DriveSystem",
    "Option B: Create separate PowerSystem",
    "Option C: Integrate with MetaController"
  ],
  criteria: ["complexity", "performance", "maintainability", "safety"]
})
```

**Ethical Review**:
```
claude_ethical_review({
  feature: "Autonomous goal modification",
  stakeholders: ["users", "developers", "society"],
  principles: ["beneficence", "non-maleficence", "autonomy", "justice"]
})
```

## Integration

- **MCP**: Use context7 for documentation context
- **Skills**: Combine with tri-cameral-orchestrator
- **Tools**: Use subagent_orchestrator for parallel reasoning
- **OpenClaw**: Execute Claude's recommendations locally

## When to Use Claude

✅ Complex architectural decisions  
✅ Safety-critical evaluations  
✅ Long-context analysis  
✅ Ethical considerations  
✅ Multi-stakeholder scenarios  
✅ Ambiguous requirement clarification  

⚠️ Simple coding tasks (overkill)  
⚠️ Well-defined problems with clear solutions  
⚠️ Time-sensitive quick decisions
