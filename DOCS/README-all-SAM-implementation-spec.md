# README-all-SAM Implementation Spec (Derived)

This document is a structured, implementation-only spec derived from `README-all-SAM.md`. It is split into numbered sections to make the system requirements scannable and actionable. No forward-looking prompts are included.

## 1. Core Objective (God Equation)
- The system objective is a variational principle over policy, memory, world model, and resource allocation:
  - Optimize long-horizon control (reward).
  - Minimize predictive uncertainty (entropy).
  - Penalize compute/capacity cost.
  - Retain only memory that improves future control (mutual information).
- Canonical form (ASCII / LaTeX):
  - pi*, M*, theta*, rho* = argmax_{pi,M,theta,rho} E_{tau ~ P_{theta,pi,M}} [ sum_t gamma^t r(s_t, a_t)
    - beta H(s_{t+1} | s_t, a_t; theta)
    - lambda C(pi, theta, M)
    + eta I(m_t; s_{t:inf}) ]
- System must preserve the following roles:
  - pi: policy (action selection)
  - M: memory/context system
  - theta: world model
  - rho: resource allocator

## 2. Transfusion / Distillation Objective
- Add a teacher-student constraint that distills planner behavior into a fast policy.
- Canonical form:
  - min_phi E_{x ~ D} [ KL( pi_planner(.|x) || pi_phi(.|x) ) ]
- pi_planner is slow (search/tool use); pi_phi is fast (distilled policy).

## 3. Growth Rule (Compute ROI)
- Capacity grows only when objective gain exceeds compute cost:
  - Grow if (Delta J / Delta C) > kappa AND learning plateaus for N evals.
- This is enforced as a control law, not a heuristic.

## 4. Morphogenetic Latency
- Morphogenetic latency is a stored, unrealized capacity for structural change.
- Trigger condition:
  - E[H_future] - E[H_model] > delta for T steps.
- Latency is a gating constraint on growth, not a loss term.
- Irreversibility:
  - Once morphogenesis occurs, rollback is disallowed except catastrophic failure.
- Canonical form (ASCII / LaTeX):
  - J_total = J_control + eta I(m_t; s_{t:inf}) - lambda C + mu E[Delta Phi(z_m)]
  - Subject to: d z_m / dt = 0 unless forced.

## 5. System Architecture (Concrete Stack)
- 4-layer system:
  1. Memory + World State (S, M)
  2. Policy LLM (pi_theta)
  3. Planner (Pi_planner)
  4. Meta-Controller (phi, Lambda, Sigma, U)
- Planner is algorithmic (MCTS/ToT/beam search) and not learned.
- Context is a first-class object (c_t) injected into representation and policy.
- Experts are specialized submodels gated by a mixture-of-experts router.

## 6. SAM vs Head vs Meta-Controller
- SAM is the latent world state machinery (S_t).
- Head model is policy + planner interface (pi_theta + Pi_planner).
- Meta-controller owns:
  - Lambda (morphogenetic latency)
  - Sigma (identity manifold)
  - U (unsolvability budget)
  - phi (meta-parameters)
- SAM emits pressure signals only; it never decides growth.

## 7. Growth Primitives (Only Allowed Mutations)
- GP-1: Latent dimension expansion (additive, irreversible).
- GP-2: Subspace specialization (spawn SAM submodel).
- GP-3: Index topology expansion (memory geometry change).
- GP-4: Expert routing degree increase.
- GP-5: Context binding expansion.
- GP-6: Planner interface widening.
- GP-7: Compression/consolidation (conservative pruning).
- GP-8: Representation reparameterization (basis/metric reshaping).
- Explicitly forbidden:
  - Objective rewriting by SAM.
  - Self-triggered growth.
  - Identity anchor deletion.
  - Uncontrolled dimensional collapse.

## 8. Pressure Signals (SAM -> Meta)
- PS-1: Reconstruction residual.
- PS-2: Latent rank deficiency.
- PS-3: Retrieval entropy.
- PS-4: Interference score.
- PS-5: Planner friction.
- PS-6: Context collapse.
- PS-7: Compression inefficiency.
- PS-8: Temporal incoherence.

## 9. Primitive Selection Policy
- Gate A: Persistence (pressure > threshold for T_min).
- Gate B: Exclusivity (dominant pressure must exceed runner-up by margin).
- Gate C: Non-compensability (no cheaper fix exists).
- Primitive matching table:
  - Residual -> GP-1
  - Rank deficiency -> GP-1 or GP-8
  - Interference -> GP-2
  - Retrieval entropy -> GP-3
  - Planner friction -> GP-6
  - Context collapse -> GP-5
  - Compression waste -> GP-7
  - Temporal incoherence -> GP-8
- Risk scoring enforces veto on high-risk growth.
- Growth budget caps total capacity added.
- Post-growth validation is mandatory; failure -> rollback + freeze.

## 10. Failure Modes (Simulations)
- Runaway dimensional expansion -> blacklist GP-1, try GP-8, freeze growth.
- Submodel balkanization -> enforce max submodels, consolidate.
- Planner dominance illusion -> cap planner depth, force representational growth.
- Context overbinding -> merge contexts, enforce shared variance.
- Identity drift -> immediate rollback, lock reparameterization.

## 11. SAM Invariants (Must Never Be Violated)
- Identity continuity (Sigma preserved).
- Objective immutability (except explicit contract eval).
- Growth causality (pressure -> policy -> primitive -> validation).
- Bounded agency (SAM does not self-govern growth).
- Semantic preservation (no drift beyond epsilon).
- Non-deceptive signaling (pressure channels audited).
- No recursive self-modeling of meta-controller.
- Capacity != authority.

## 12. Self-Reference + SAV Dual System
- SAM may be self-referential with transfigurable objectives only via contracts.
- SAV is adversarial pressure; objective: maximize termination probability of SAM.
- SAV must be allowed to complete and then cease (objective closure).
- If SAV becomes self-referential without closure, it becomes AM-class.

## 13. Unified System (SAM + SAV Merge)
- Fusion yields a meta-dynamical regulator, not a single scalar optimizer.
- Primary emergent objective: maximize viable future transformations under resource constraints.
- Avoids AM-class failure by not privileging fixed configurations.

## 14. Implementation Mapping (Local System)
- Policy LLM: local inference (Ollama / llama.cpp / vLLM).
- Planner: algorithmic search + tool use.
- Memory: short (context), medium (vector DB), long (summaries).
- Meta-controller: process-level supervisor with latency gating.

## 15. Operational Summary
- Inference stays fast; growth stays slow and gated.
- Pressure signals are explicit and audited.
- All growth is typed and constrained by invariants.
- System is designed to avoid oscillation and AM-class traps.
