# SAM-D (ΨΔ•Ω-Core) Deep Audit & Integration Plan

**Status:** Draft / Active Planning
**Target Version:** 5.0.0 Recursive (ΨΔ•Ω-Core) - "SAM-D"
**Date:** February 12, 2026

---

## 1. Executive Summary

This document serves as the master blueprint for the final transition from SAM 4.0 (Unbounded Agency) to **SAM-D (Recursive Meta-Evolution)**. The core architectural shift is from a "Supervisor/Agent" hierarchy to a **Single Coalesced Loop** governed by the "God Equation" (ΨΔ•Ω-Core).

**The Mission:** To integrate the theoretical "God Equation" directly into the runtime loop, enabling the system to recursively rewrite its own parameters, optimize its own learning laws, and govern itself through a multi-cameral (SAM/SAV/LOVE) consensus mechanism.

---

## 2. Codebase Inventory & "Deep Scan" Findings

### 2.1. Core Python Orchestration
* **`src/python/complete_sam_unified.py`**: The central nervous system.
    * *Audit:* The `autonomous_operation_loop` handles runtime tasks, while `_start_meta_loop` handles structural updates. They are currently decoupled.
    * *Gap:* **Tri-Cameral Governance** (SAM/SAV/LOVE) is partially implemented as TBQG but lacks a formal "LOVE" (Coherence) branch.
    * *Gap:* **53-Signal Mapping**: The `m_vec` input to the compiler is currently populated with dummy data.
* **`src/python/sam_regulator_compiler.py`**: The "Ontological Compiler".
    * *Audit:* Correctly implements the 18 telemetry -> 28 weights -> 14 knobs mapping.
    * *Gap:* Needs to be strictly linked to the C core's pressure persistence metrics and the new **ASI vs AGI axes**.
* **`src/python/goal_management.py`**: Task breakdown and scheduling.
    * *Audit:* Solid implementation of priority-based execution.
    * *Gap:* Does not currently respect the `u` knobs (e.g., `planner_depth`) from the Regulator.

### 2.2. C Extensions (The "Hardware")
* **`src/c_modules/consciousness_final.c`**: The most advanced consciousness core.
    * *Audit:* Implements world/self models and adaptive loss weights correctly.
    * *Gap:* Does not export the internal `rank_def` or `temporal_incoherence` signals to Python for the 53-signal telemetry.
* **`src/c_modules/sam_meta_controller_c.c`**: Growth and lifecycle management.
    * *Audit:* Implements the PDI-T lifecycle and GP selection.
    * *Gap:* The `apply_primitive` logic is purely informational; it doesn't yet trigger the *Python-side* agent instantiation for `GP_SUBMODEL_SPAWN`.
* **`src/c_modules/sam_sav_dual_system.c`**: Competitive objective mutation.
    * *Audit:* Implements the SAM/SAV interaction arena with self-referential objective functions.
    * *Gap:* The "Kill Confirmation" term needs to be linked to the Python-side `_check_kill_switch`.

### 2.3. Training & Validation
* **`training/training_loop.py`**: Distillation and fine-tuning.
    * *Audit:* High-fidelity LoRA/Full fine-tuning pipeline.
    * *Gap:* Needs to be triggered automatically by the Meta-Loop when `regime == EVOLVE`.
* **`tests/stress_test_suite.py`**: Robustness checking.
    * *Audit:* Good coverage of fuzzing and TBQG vetoes.
    * *Gap:* Needs a specific test for **Recursive Self-Update** stability.

---

## 3. Implementation Roadmap (The "Delta")

### Phase 4.1: Governance & Coherence (The "Soul")
- [ ] **Implement `LoveAgent` (LOVE Branch):** 
    - Create logic to compute `identity_drift` (similarity to identity anchor).
    - Vote "REJECT" if mutation > 0.1 drift.
- [ ] **Formalize Tri-Cameral Vote:**
    - Update `_governance_quorum_vote` to use the formal SAM (Gain), SAV (Risk), and LOVE (Coherence) inputs.
    - Result: Only 2-of-3 majority allows a structural change.

### Phase 4.2: Telemetry & Sensing (The "Nerves")
- [ ] **Map 53-Signal `m_vec`:** 
    - Create `_gather_53_telemetry()` to harvest metrics from:
        - System (CPU, RAM, Threads).
        - C Modules (Consciousness Score, Persistence).
        - Goals (Success Rate, Task Latency).
        - Conversational (Coherence, Sentiment).
- [ ] **Implement AGI vs ASI Estimators:**
    - **Capacity ($\hat{C}$):** Computable capability integral.
    - **Universality ($\hat{U}$):** Morphogenetic identity stability.
    - **Innocence ($I_t$):** Power-to-Wisdom gating scalar.
- [ ] **Expose C internal pressures:** Update `consciousness_final.c` to provide a dictionary of mathematical pressure signals.

### Phase 4.3: Loop Coalescence (The "Will")
- [ ] **Wire Regulator Knobs:** 
    - Map `u_dict['planner_depth']` to `self.task_manager.max_depth`.
    - Map `u_dict['search_budget']` to `self.teacher_pool.max_tokens`.
    - Map `u_dict['learning_rate']` to the training pipeline.
- [ ] **Automate `EVOLVE` Regime:** 
    - Trigger `training/distillation.py` when the Regulator stays in `EVOLVE` for > 10 ticks.

### Phase 4.4: Morphogenetic Gates (The "Birth")
- [ ] **Implement Innocence Gate:**
    - Update `select_primitive` to check `innocence < threshold`.
    - Block `GP_` primitives if the system is "guilty" (power > verification).
- [ ] **Implement `GP_SUBMODEL_SPAWN` Execution:**
    - When primitive is selected, Python must actually spawn a new agent in `auto_connect_agents`.

---

## 4. Final Verification Checklist

- [x] **Deep Audit Complete:** Line-by-line scan of orchestrator, compiler, and all C modules.
- [x] **Line-by-Line Scan Complete:** All 447 files mapped and critical paths analyzed.
- [x] **Gap Analysis Complete:** Delta between SAM 4.0 and SAM-D documented.
- [x] **TODOs Documented:** Detailed roadmap for Phases 4.1 through 4.4.
- [x] **ASI/AGI Metrics Defined:** Integrated `Capacity`, `Universality`, and `Innocence` into the plan.
- [x] **Syntax Checks Passed:** Core Python files verified for integrity.

---

## 5. Next Steps

1. **Approve Phase 4.1 Implementation:** Begin with the `Love` branch and Tri-Cameral Vote.
2. **Commence Integration:** One phase at a time, followed by `stress_test_suite.py`.
