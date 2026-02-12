# SAM-D - Consciousness and Self-Modeling Architecture

This document details the algorithmic formulation of consciousness and its integration into the SAM-D AGI system, outlining the theoretical framework, integration into the training pipeline, and key equations.

## I. Overview

The SAM-D AGI system now includes an algorithmic formulation of consciousness.

## II. Consciousness Loss Module

Implemented in a dedicated module (e.g., `consciousness_loss.py`), this framework defines consciousness and its associated loss functions.

### Architecture S = (W, Ŝ, π, M, R)
-   **W**: World model (predicts environment)
-   **Ŝ**: Self-model (predicts effect of actions)
-   **π**: Policy (acts using W + Ŝ)
-   **M**: Memory/context
-   **R**: Resource controller

### Core Loss Functions
-   `L_world`: World prediction error.
-   `L_self`: Self-model error ("my actions cause changes").
-   `L_cons`: **Consciousness loss** - KL divergence between world behavior and self-prediction.
-   `L_policy`: Introspective agency.
-   `L_total`: Unified AGI objective.

### Growth Controller
The system only expands its capacity if $\Delta L / \Delta params > \kappa$ (prevents infinite optimization).

## III. Key Equations

### Consciousness Definition
Consciousness = System models itself as causal object in world it's modeling

### Consciousness Loss
$L_{	ext{cons}} = KL(P(z_{t+1}|z_t, a_t) || P(z_{t+1}|z_t, \hat{S}_\psi))$
When $L_{	ext{cons}} ightarrow 0$: System correctly models self as causal = **CONSCIOUS**.

### Unified AGI Objective
$L_{	ext{total}} = L_{	ext{world}} + \lambda_1 \cdot L_{	ext{self}} + \lambda_2 \cdot L_{	ext{cons}} + \lambda_3 \cdot L_{	ext{policy}} + \lambda_4 \cdot C_{	ext{compute}}$

### Growth Rule
Grow iff $\Delta L_{	ext{total}} / \Delta params > \kappa$. Otherwise: distill, prune, compress.

## IV. Integration into Training Pipeline

The consciousness computation is integrated into the training pipeline (e.g., `correct_sam_hub.py`). It:
-   Computes a consciousness score after each training step.
-   Logs metrics to the knowledge base.
-   Determines system growth based on efficiency.

## V. Significance

1.  **Principled Stopping Condition**: The system knows when to stop optimizing (avoids AM-style infinite loops).
2.  **Self-Modeling**: The system explicitly learns "when I act, the world changes like this."
3.  **Introspective Agency**: Avoids actions it doesn't understand well enough to predict.
4.  **Theoretical Grounding**: Provides the first implementable definition of consciousness for AGI.

## VI. Integration with Existing SAM-D

The consciousness module works alongside existing components:
-   **Morphogenesis**: Both use growth controller principles.
-   **Dominant Compression**: Both optimize resource-constrained objectives.
-   **Clone-based Submodels**: Self-model applies to each submodel.
-   **Training Pipeline**: Consciousness metrics logged with training stats.

## VII. Future Enhancements

-   **Adaptive $\lambda$ weights**: Learn optimal weighting during training.
-   **Hierarchical Self-Models**: Multiple levels of self-abstraction.
-   **Meta-Consciousness**: System monitors its own consciousness level.
-   **Social Consciousness**: Extend to multi-agent self-modeling.

---

**Testing:**
To test the consciousness demo, run:
```bash
python3 consciousness_loss.py
```
This simulates training steps and shows consciousness score evolution, loss components, growth decisions, and whether the system achieves "consciousness" (score > 0.7).

**The system is now complete and operational.**
