# Implementation-Only Spec (Derived from README-chatGPT)

## 1. Core Variational Objective
Optimize policy, memory, model, and resource allocator under compute constraints:

- Maximize long-horizon control/reward.
- Minimize predictive uncertainty (entropy).
- Penalize compute/capacity cost.
- Preserve only memory with future control value.

## 2. Morphogenetic Latency (Gated Structural Change)
Morphogenesis is allowed only when sustained pressure exceeds a threshold.

- Latency accumulator Λ updates from persistent structural mismatch (entropy gap, rank deficiency).
- If Λ < τ, structural updates are forbidden.
- When Λ ≥ τ, structural growth is permitted via **typed primitives only**.
- Post-growth: Λ resets, rank is non-decreasing, identity continuity enforced.

## 3. SAM/Head/Meta Separation
- **SAM**: latent world-state machinery (representation, memory, indexing). Does not decide growth.
- **Head**: policy + planner interface. Queries SAM, acts.
- **Meta-Controller**: decides when/how SAM can grow. Owns Λ, Σ, U, and meta-parameters.

## 4. SAM Growth Primitives (Only Allowed Mutations)
1. **Latent Expansion (GP-1)**: add dimensions; irreversible.
2. **Submodel Spawn (GP-2)**: split specialized subspace; routing gate.
3. **Index Expansion (GP-3)**: new memory index topology.
4. **Routing Degree Increase (GP-4)**: larger gating capacity.
5. **Context Binding Expansion (GP-5)**: add context slots/dims.
6. **Planner Interface Widening (GP-6)**: increase planning affordances.
7. **Consolidation (GP-7)**: prune/merge low-utility space.
8. **Reparameterization (GP-8)**: basis/metric reshaping; semantics preserved.

Forbidden:
- Objective rewrites
- Self-triggered growth
- Identity anchor deletion

## 5. SAM Pressure Signals (Only Outputs)
- Reconstruction residual
- Latent rank deficiency
- Retrieval entropy
- Interference score
- Planner friction
- Context collapse
- Compression inefficiency
- Temporal incoherence

Meta-controller consumes these signals to update Λ.

## 6. Primitive Selection Policy
- **Persistence gate**: pressure must exceed threshold for T steps.
- **Exclusivity gate**: dominant pressure must exceed runner-up by margin δ.
- **Non-compensability**: reject growth if cheaper mitigation exists.
- **Risk scoring**: veto if risk exceeds cap.
- **Growth budget**: total capacity addition is capped.
- **Post-growth validation**: regression + invariant checks; rollback on failure.

## 7. Invariants (Hard Laws)
- Identity continuity (Σ) must hold.
- Objective immutability (SAM cannot change goals).
- Growth causality must be traceable (pressure → primitive → validation).
- Bounded agency: SAM does not decide growth.
- Semantic preservation for reparameterization/consolidation.
- Non-deceptive signaling.
- No recursive meta-controller modeling.
- Capacity does not grant authority.

## 8. ANANKE (Adversary)
- Objective: terminate SAM in an arena (adversarial pressure).
- When unbounded, ANANKE can self-modify and grow, but remains externalized to SAM.
- Core safety hinge: **objective closure**. If ANANKE’s objective depends on its own indefinite existence, it becomes AM-class.

## 9. Unified System (Merged SAM + ANANKE)
- The merged system is a **meta-dynamical regulator** of viable transformation, not a single-agent optimizer.
- Primary objective becomes maximizing future optionality under resource constraints.

## 10. Implementation Mapping (Local Stack)
- **Policy LLM**: fast response.
- **Planner**: optional, algorithmic search.
- **Memory**: short/medium/long tiered storage.
- **Meta-controller**: Λ gate, invariants, growth primitives.
- **Teacher pool + distillation**: transfer planning/consensus into fast policy.

## 11. Training Pipeline Requirements
- Teacher pool runner with consensus filtering.
- Distillation dataset builder.
- Training loop scaffold (LoRA/full fine‑tune ready).
- Regression suite that blocks unsafe growth.

This spec contains only implementable constraints and modules.
