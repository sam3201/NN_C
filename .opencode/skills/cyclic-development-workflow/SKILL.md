---
name: cyclic-development-workflow
description: Cyclic Plan-Analyze-Build-Analyze-Test workflow with decision gates
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: iterative
---

## What I Do

Implement a cyclic development workflow with mandatory analysis gates between each phase:

```
┌──────────────────────────────────────────────────────────────┐
│                  CYCLIC DEVELOPMENT FLOW                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   START                                                      │
│     │                                                        │
│     ▼                                                        │
│  ┌──────────────┐                                            │
│  │     PLAN     │◄─────────────────────┐                     │
│  └──────────────┘                      │                     │
│     │                                  │                     │
│     ▼                                  │                     │
│  ┌──────────────┐  NO ┌──────────┐    │                     │
│  │   ANALYZE 1  │────→│  REVISE  │────┘                     │
│  └──────────────┘     └──────────┘                           │
│     │ YES                                                    │
│     ▼                                                        │
│  ┌──────────────┐                                            │
│  │    BUILD     │                                            │
│  └──────────────┘                                            │
│     │                                                        │
│     ▼                                                        │
│  ┌──────────────┐  NO ┌──────────┐                           │
│  │   ANALYZE 2  │────→│ REFACTOR │────┐                      │
│  └──────────────┘     └──────────┘    │                      │
│     │ YES                             │                      │
│     ▼                                 │                      │
│  ┌──────────────┐                     │                      │
│  │  TEST/EXEC   │─────────────────────┘                      │
│  └──────────────┘                                            │
│     │                                                        │
│     ▼                                                        │
│  ┌──────────────┐  NO ┌──────────┐                           │
│  │   ANALYZE 3  │────→│  DEBUG   │────┐                     │
│  └──────────────┘     └──────────┘    │                     │
│     │ YES                             │                     │
│     ▼                                 │                     │
│  ┌──────────────┐                     │                     │
│  │   COMPLETE   │◄────────────────────┘                     │
│  └──────────────┘                                            │
│                                                              │
│  OR                                                          │
│                                                              │
│     │ NO (fundamental flaw)                                  │
│     ▼                                                        │
│  ┌──────────────┐                                            │
│  │  ABANDON →   │──→ Return to PLAN with lessons learned    │
│  └──────────────┘                                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Phase Descriptions

### Phase 1: PLAN
**Activities**:
- Define objectives and requirements
- Draft architecture and design
- Break down into subtasks
- Estimate effort and resources

**Deliverables**:
- Implementation plan document
- Architecture diagrams
- Task breakdown

### Gate 1: ANALYZE PLAN
**Checks**:
- Feasibility assessment
- Resource availability
- Risk identification
- Alignment with invariants

**Decision**: Proceed → BUILD | Revise → Back to PLAN

### Phase 2: BUILD
**Activities**:
- Write code
- Create documentation
- Implement tests
- Build integrations

**Deliverables**:
- Working code
- Unit tests
- Documentation drafts

### Gate 2: ANALYZE BUILD
**Checks**:
- Code quality
- Test coverage
- Documentation completeness
- Performance benchmarks

**Decision**: Proceed → TEST | Refactor → Back to BUILD

### Phase 3: TEST/EXECUTE
**Activities**:
- Run test suite
- Integration testing
- Performance testing
- User acceptance testing

**Deliverables**:
- Test results
- Performance metrics
- Bug reports

### Gate 3: ANALYZE TEST RESULTS
**Checks**:
- All tests passing?
- Performance acceptable?
- No critical bugs?
- Meets requirements?

**Decision**: Complete → DONE | Debug → Back to BUILD | Abandon → Back to PLAN

## High-Level vs Low-Level Planning

### High-Level (Strategic)
- Architecture decisions
- Technology choices
- Resource allocation
- Timeline planning
- Risk assessment

### Low-Level (Tactical)
- Function signatures
- Algorithm selection
- Variable naming
- Implementation details
- Edge case handling

## Hard vs Soft Constraints

### Hard Constraints (Invariants)
- Must never be violated
- System integrity depends on them
- CSF enforces strictly

**Examples**:
- Identity continuity
- No breaking API changes
- Security requirements
- Legal compliance

### Soft Constraints (Guidelines)
- Should be followed
- Can be relaxed with justification
- AEE challenges when violated

**Examples**:
- Code style preferences
- Performance targets
- Documentation completeness
- Test coverage goals

## Integration with Tri-Cameral System

At each ANALYZE gate, all three branches participate:

**CIC (Optimistic)**:
- "This approach will work because..."
- Identifies opportunities
- Proposes optimizations

**AEE (Pessimistic)**:
- "This will fail because..."
- Finds edge cases
- Identifies risks

**CSF (Neutral)**:
- "Checking invariants..."
- Measures against standards
- Validates constraints

## Usage Examples

**Start full cycle:**
```
cyclic_workflow({
  task: "Implement Power/Control systems",
  start_phase: "PLAN",
  high_level: "Add P_t and C_t classes",
  low_level: "Integrate with existing DriveSystem",
  constraints: ["maintain_API_compat", "no_perf_regression"]
})
```

**Run specific phase:**
```
workflow_phase({
  phase: "BUILD",
  plan_document: "phase2_implementation_plan.md",
  analysis_gate: true
})
```

**Force gate decision:**
```
analysis_gate({
  gate_number: 2,
  phase: "BUILD",
  decision: "REFACTOR",
  reason: "Performance below threshold"
})
```

## Automation Level

**Level 1 - Manual**: Human reviews at each gate
**Level 2 - Assisted**: AI suggests, human decides
**Level 3 - Semi-Auto**: AI decides, human can override
**Level 4 - Full-Auto**: Autonomous progression

## Integration Points

- **MCP Servers**: context7 for docs, github-search for patterns
- **Skills**: tri-cameral-orchestrator for governance
- **Custom Tools**: subagent_orchestrator for parallel analysis
- **Anthropic**: Claude for complex reasoning
- **OpenClaw**: Local execution and testing
