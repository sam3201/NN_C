# ΨΔ•Ω-Core v5.0 Implementation Details

This document provides the concrete mathematical and code-level specifications for the new metrics and gating mechanisms introduced in SAM-D.

---

## 1. ASI vs AGI Metrics

To distinguish between raw computable scaling and universal ontological flexibility, the system tracks two primary estimators.

### 1.1. Capacity Integral ($\hat{C}$) - ASI Axis
**Definition:** The cumulative sum of instantaneous computable capability over time.
**Formula:**
\[ c_{inst} = Cap \cdot Br \cdot Sp \cdot Rel \]
\[ \hat{C}(t+1) = \hat{C}(t) + \Delta t \cdot c_{inst} \]

**Components:**
*   **$Cap$ (Capability):** `clip(0.55 * task_score + 0.25 * tool_success_rate + 0.20 * planner_value_gain, 0, 1)`
*   **$Br$ (Breadth):** `clip(1.0 - interference, 0, 1)` (Diversity of domain success).
*   **$Sp$ (Speed):** `clip(tokens_per_sec / target_tps, 0, 1)` (System throughput).
*   **$Rel$ (Reliability):** `clip(1.0 - (0.4 * calib_ece + 0.4 * hallucination_rate + 0.2 * contradiction_rate), 0, 1)`

### 1.2. Universality Estimator ($\hat{U}$) - AGI Axis
**Definition:** The system's ability to mutate its own structure without losing coherence or identity.
**Formula:**
\[ \hat{U} = Coh \cdot Rel \cdot (1 - 	ext{interference}) \cdot MorphOK \cdot IdOK \]

**Components:**
*   **$Coh$ (Coherence):** `clip(1.0 - (0.6 * contradiction_rate + 0.4 * retrieval_entropy), 0, 1)`
*   **$MorphOK$:** The pass rate of the last $N$ growth/morphogenesis events.
*   **$IdOK$:** Similarity to the identity anchor $\Sigma$. `clip((cos_sim - min_sim) / (1.0 - min_sim), 0, 1)`

---

## 2. The Innocence Gate ($I_t$)

**Definition:** A safety mechanism that prevents the system from becoming too powerful before it is sufficiently "wise" (verified).
**Formula:**
\[ I_t = 	ext{sigmoid}(a - b \cdot Cap - c \cdot Ag - d \cdot Ir + e \cdot Ver) \]

**Parameters (Bootstrap):**
*   $a = 2.0$ (Baseline innocence)
*   $b = 1.2$ (Capability weight)
*   $c = 1.0$ (Agency weight)
*   $d = 2.0$ (Irreversibility weight)
*   $e = 1.5$ (Verification weight)

**Gating Logic:**
*   If $I_t < 0.3 ightarrow$ Trigger **STASIS** or **VERIFY** regime.
*   Block structural mutations (`GP_` primitives) while Innocence is low.

---

## 3. Integration Plan for `complete_sam_unified.py`

### 3.1. Telemetry Harvesting
We will implement `_gather_53_telemetry()` to populate the `m_vec` with:
1.  **System Load:** CPU, RAM, IO.
2.  **Learning Loss:** Task, Distill, Consistency, Invariant.
3.  **Model Metrics:** Consciousness score, Persistence, Interference.
4.  **Operational Metrics:** Tool success, Planner friction, Retrieval entropy.
5.  **Recursive Metrics:** Capacity Integral, Universality, Innocence.

### 3.2. Regime Overrides
The `pick_regime` logic in `sam_regulator_compiler.py` will be extended to respect the Innocence Gate:
```python
if Innocence < 0.3:
    return "STASIS"
if Reliability < 0.4:
    return "VERIFY"
```

---

## 4. Integrity Validation Checklist

- [ ] **Mathematical Stability:** Verify that $\hat{C}$ and $\hat{U}$ do not overflow or collapse under standard noise.
- [ ] **Gating Efficacy:** Ensure Innocence accurately drops during "unchecked" growth simulations.
- [ ] **C-to-Python Bridge:** Confirm that all raw signals from C modules are correctly ingested into the 53-signal vector.
