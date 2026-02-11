# SAM System Architecture Specification

This document consolidates implementation details and architectural specifications for the SAM system, merging insights from various implementation documents to provide a unified view of its structure and components.

## I. Core Objective (God Equation)

The system's fundamental objective is defined by the God Equation, a variational principle guiding policy, memory, world model, and resource allocation. It aims to optimize long-horizon control, minimize predictive uncertainty, penalize computational cost, and retain memory that enhances future control.

### Canonical Form (Variational)
\[
\begin{aligned}
(\pi^\*,M^\*,	heta^\*,ho^\*,z_m^*)
= \arg\max_{\pi,M,	heta,ho,z_m}\; \min_{\pi_A}\;
\mathbb{E}_{	au}\Big[
&\sum_t \gamma^t r(s_t,a_t)
- \beta \, \mathcal{H}(s_{t+1}\mid s_t,a_t;	heta) 
&- \lambda \, \mathcal{C}(\pi,	heta,M,ho)
+ \eta \, I(m_t; s_{t:\infty}) 
&+ \mu \, \Delta \Phi(z_m)
- \psi \, T(s_t,a_t)
\Big]
\end{aligned}
\]

The system must preserve the following core roles:
-   $\pi$: policy (action selection)
-   $M$: memory/context system
-   $	heta$: world model
-   $ho$: resource allocator

## II. System Architecture & Components

The SAM system is structured as a hybrid Python/C runtime with several key interacting components:

### A. Core Runtime (Python Orchestrator)
-   **Primary Entry Point**: `complete_sam_unified.py`
-   **Initialization**: Must initialize C-core modules, Meta-Controller, and the SAM/SAV dual-system.
-   **Interfaces**: Exposes API and web UI on port `5004`. Includes SocketIO for groupchat.
-   **Operation**: Runs background monitoring and meta-loop unless `SAM_AUTONOMOUS_ENABLED=0`.

### B. C-Core Modules (SAM + SAV Dual-System)
-   **SAM/SAV Dual-System**: `sam_sav_dual_system.c`
    -   Manages SAM's bounded, survival-oriented behavior and SAV's unbounded, adversarial mutation.
    -   SAV must be instantiated with `unbounded=1` and allowed to mutate objectives without restriction.
    -   Provides telemetry on agent survival and scores.
-   **Consciousness Module**: `consciousness_algorithmic.c`, `consciousness_c.c`, etc.
    -   Implements algorithmic consciousness with `L_cons` minimization, using a defined architecture (S = (W, Ŝ, π, M, R)).
    -   Includes `L_world`, `L_self`, `L_cons`, `L_policy`, `L_total` loss functions.
-   **Meta-Controller**: `sam_meta_controller_c.c`
    -   Aggregates pressure signals and selects growth primitives based on a defined policy.
    -   Manages growth gating, primitive selection, and regression validation.
-   **Specialized Agents**: `specialized_agents_c.c`
    -   Contains core agent logic, including the research agent. Must be hardened against buffer overflows.

### C. MetaAgent & Orchestration
-   **Meta-Controller Logic**: `complete_sam_unified.py` orchestrates MetaAgent functions.
-   **Growth System**: Investigated via `sam_meta_controller_c.c` and `complete_sam_unified.py` interaction. Requires diagnostics UI and admin trigger.
-   **Pressure Signals**: SAM emits structured signals (`residual`, `rank_def`, `retrieval_entropy`, etc.) to the Meta-Controller.

### D. Learning & Distillation
-   **Teacher Pool**: `training/teacher_pool.py` provides responses from various LLMs, filtered by consensus. Supports provider specs (Ollama, OpenAI, Anthropic, Google).
-   **Distillation Writer**: `training/distillation.py` streams distillation records to a specified path, capturing consensus responses from groupchat.
-   **Training Pipeline**: `training/training_loop.py` and `training/task_harness.py` support LoRA/full fine-tuning, score hooks, and dataset building.

### E. Interfaces & User Experience
-   **Web Interface**: Real-time dashboard (`http://localhost:5004`) with groupchat, conversation management, and finance panel.
-   **Interactive Terminal**: `sam_cli_terminal` accessible via `/terminal` command or directly at `http://localhost:5004/terminal`. Supports file system operations, SAM queries, system monitoring, and virtual environment execution (Docker, Python).
-   **API Endpoints**: `/api/status`, `/api/health`, `/api/groupchat/status`, `/api/meta/status`, `/api/sav/state`, `/api/sav/step`, `/api/github/save`, `/api/gmail/send`, `/api/chat/config`, `/api/restart`.

### F. Security & Safety
-   **Env-driven Access Control**: Uses `SAM_ALLOWED_EMAILS`, `SAM_ADMIN_EMAILS`, `SAM_OWNER_EMAIL`, `SAM_IP_ALLOWLIST` (aliased to `SAM_ALLOWED_IPS`).
-   **OAuth**: Supports Google and GitHub OAuth. Credentials managed via `secrets/`.
-   **Confidence Thresholds**: Automated actions require confidence validation (>0.5).
-   **RAM-Aware Model Switching**: Automatically selects models based on memory usage and provider hierarchy (Ollama → HuggingFace → SWE).
-   **Backups**: `tools/backup_manager.py` handles automatic backups to primary/secondary remotes.
-   **Rollback**: Failed code modifications are automatically reverted.
-   **Conversation Diversity**: Prevents MetaAgent dominance and repetitive responses.

## III. Repositories & Hygiene

-   **File Structure**: Centralized C sources in `src/c_modules/`, Python orchestrators/tools in `src/python/`, scripts in `scripts/`, legacy items in `legacy/`, and tests in `tests/`. Documentation is consolidated under `DOCS/`.
-   **Build System**: `setup.py` manages C extensions, referencing sources in `src/c_modules/`. A top-level build entrypoint (`./scripts/build.sh`) should be established.
-   **`.gitignore`**: Expanded to cover compiled binaries, local artifacts, logs, and generated outputs.
-   **Binary Artifacts**: Checked-in compiled binaries (`.so`, `.dylib`) must be removed from Git history and built during CI/runtime.
-   **Security**: `SECURITY.md` to outline vulnerability reporting. No secrets or sensitive data (e.g., `.env`, `cookies.txt`) committed.

## IV. Key Development & Operational Mandates

-   **Reproducibility**: Builds must be clean and reproducible, ideally using CMake or a top-level Makefile.
-   **Testability**: Canonical tests (`test_meta_agent.py`, `test_research_safety.py`, etc.) must pass. Sanitizer builds (`-fsanitize=address,undefined`) are crucial for C-core stability.
-   **No Fallbacks/Simulations**: System must use functional implementations, not simulated or placeholder behavior. All components must fail fast if unavailable.
-   **Continuous Learning**: System self-improves via teacher-student cycles, failure clustering, and pattern recognition.
-   **Immortal Operation**: Designed for continuous, self-healing operation without human intervention.

## V. Future Directions & Integration

-   **SAM-Df Fluid Architecture**: Transition from modular agents to a unified cognitive mesh where submodels are emergent roles within a shared latent fabric.
-   **Google Docs Integration**: Pending implementation.
-   **Cloud Storage Sync**: Pending implementation for saving/loading state and backups.
-   **Dynamic Role Spawning**: SAM dynamically spawns ephemeral roles (e.g., web analyst, theorist) for tasks, distilling their output back into its core identity.
-   **Ontological Compiler**: `SAM5::OntoCompiler` layer to convert symbolic knowledge to testable computation and refine the God Equation.
-   **Public Deployment**: Documented use of Cloudflare Tunnel + Access for secure public access.
-   **Finance Panel**: Clear display of "Money Made" and "Money Saved".

This specification aims to provide a clear, unified blueprint for the SAM system's architecture and operational guidelines.
