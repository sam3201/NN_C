---
name: tri-cameral-orchestrator
description: Tri-cameral governance workflow orchestrator with CIC/AEE/CSF branches
license: MIT
compatibility: opencode
metadata:
  audience: architects
  workflow: governance
---

## What I Do

I implement the S³-D³ Tri-Cameral Governance Model for comprehensive automation:

**Three Branches:**
- **CIC (Constructive Intelligence Core)**: Plans growth, proposes changes, builds features
- **AEE (Adversarial Exploration Engine)**: Challenges plans, finds edge cases, tests limits
- **CSF (Coherence Stabilization Field)**: Validates invariants, ensures stability, vetoes unsafe changes

## Workflow Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                    TRI-CAMERAL CYCLE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │   PLAN   │───→│ ANALYZE  │───→│ DECISION │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       ↑                              │                      │
│       │                              ↓                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  CYCLE   │←───│  CHECK   │←───│  BUILD   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       ↑                              │                      │
│       │                              ↓                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ COMPLETE │←───│  VERIFY  │←───│  TEST    │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                             │
│  At each ANALYZE/CHECK/VERIFY: All 3 branches vote          │
│  • CIC votes: "Proceed with growth"                         │
│  • AEE votes: "This will break here..."                     │
│  • CSF votes: "Invariant check: PASS/FAIL"                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Branch Responsibilities

### CIC - Constructive Intelligence Core (The Builder)
**Role**: Proposer, Optimistic, Growth-oriented
**Actions**:
- Draft implementation plans
- Write code and documentation
- Design new features
- Optimize performance
- Propose architectural improvements

**Keywords**: create, build, implement, add, optimize, improve, expand

### AEE - Adversarial Exploration Engine (The Critic)
**Role**: Challenger, Pessimistic, Safety-oriented
**Actions**:
- Find edge cases and failure modes
- Identify security vulnerabilities
- Challenge assumptions
- Test boundary conditions
- Predict potential regressions

**Keywords**: test, challenge, verify, check, critique, question, break

### CSF - Coherence Stabilization Field (The Guardian)
**Role**: Validator, Neutral, Stability-oriented
**Actions**:
- Enforce invariants
- Check consistency
- Validate against requirements
- Measure system health
- Veto unsafe changes

**Keywords**: validate, enforce, measure, verify, ensure, maintain, protect

## Decision Matrix

| CIC | AEE | CSF | Decision | Action |
|-----|-----|-----|----------|--------|
| YES | YES | YES | ✅ PROCEED | Execute plan |
| YES | NO  | YES | ⚠️ REVISE  | Address AEE concerns |
| YES | YES | NO  | 🛑 REJECT  | Violates invariants |
| NO  | YES | YES | 🔄 REFACTOR| AEE found better way |
| YES | NO  | NO  | 🛑 REJECT  | Unstable growth |
| NO  | YES | NO  | 🛑 HALT    | Critical issues |
| NO  | NO  | YES | ⏸️ PAUSE   | Reconsider approach |

## Usage

**Start a tri-cameral cycle:**
```
tri_cameral_cycle({
  task: "Implement Phase 2 Power/Control systems",
  high_level_plan: "Add P_t and C_t to sam_cores.py with resources tracking",
  branches: ["CIC", "AEE", "CSF"]
})
```

**Single branch consultation:**
```
consult_branch({
  branch: "AEE",
  proposal: "Add new C module for power calculation",
  context: "Current system has Id/Ego/Superego implemented"
})
```

**Force consensus:**
```
force_consensus({
  topic: "Should we proceed with Phase 2?",
  min_agreement: 2,  // At least 2 of 3 branches must agree
  timeout: 300
})
```

## Integration Points

- **MCP Servers**: context7 for docs, github-search for examples
- **Custom Tools**: subagent_orchestrator for parallel analysis
- **Anthropic Skills**: Claude's reasoning for complex decisions
- **OpenClaw**: Local execution environment

## Example Session

**User**: "I want to add Power and Control systems to sam_cores.py"

**CIC**: "Great! I'll draft the implementation plan with P_t and C_t classes, integrate with existing Drive/Emotion/Wisdom systems..."

**AEE**: "Wait - how will this affect the existing Phase 1 systems? What if P_t conflicts with Superego constraints? We need to test for..."

**CSF**: "Checking invariants: 1) Identity continuity preserved? 2) No breaking changes to existing API? 3) Performance within bounds?"

**Consensus**: Proceed with CIC plan, incorporating AEE edge cases, with CSF monitoring invariants.
