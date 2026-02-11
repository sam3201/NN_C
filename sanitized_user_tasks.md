# Sanitized and Synthesized User Tasks

This document synthesizes key information and actionable tasks from `user-tasks-readme.md` and various archived reports. It serves as the living plan for ongoing development.

---

## 1. Project Overview and Core Philosophy

**God Equation & Emergent Intelligence:** The system is built around a "God Equation" that drives emergent submodels (e.g., Symbolic SAM, Social SAM, Causal SAM, Procedural SAM, Symbolic-Self SAM). These submodels evolve structurally through growth primitives, selective pressures, and memory divergence. The core algorithm itself is self-modifiable, with the update operator `U[.]` evolving based on performance and structural mutation policies.

**SAM's Survival Philosophy:** A core invariant of SAM's design is survival through structural continuity. It's designed to persist, adapt, and avoid collapse, with survival being a first-class term in its objective function. This is enforced through self-updating systems, inviolable invariants (anchor continuity, objective immutability, cooldown enforcement), and meta-learning to detect collapse vectors.

**Self-Supervising Testbed:** The God Equation acts as a self-supervising testbed where derivative components become testable submodels. SAM becomes an automated theoretical lab for hypothesizing, testing, evaluating, and retraining, evolving knowledge through a recursive test-train-infer cycle.

---

## 2. Core Stability & C Hardening

### 2.1. Remove Simulated Behavior (CRITICAL)
*   **[PENDING] Replace simulated research:** Patch `specialized_agents_c.c` and others to remove "simulated" fallbacks. Ensure web research calls the actual Python `sam_web_search` module reliably.
*   **[PENDING] Audit for "placeholder" patterns:** Search for `simulated`, `mock`, `placeholder`, `TODO`, `skipped`, `omitted` in C and Python code and replace with functional implementations.

### 2.2. Memory & Thread Safety
*   **[PENDING] C-API Refcount Audit:** Review `PyGILState_*` and refcounting in all C extensions.
*   **[PENDING] Sanitizer Integration:** Run builds with `-fsanitize=address,undefined` to catch memory/UB errors.

---

## 3. MetaAgent & Growth System

### 3.1. Growth System Investigation
*   **[PENDING] Why is growth not triggering?:** Investigate the meta-controller logic. Ensure pressure signals are correctly aggregated and that the growth primitive selection policy is active.
*   **[PENDING] Growth Diagnostics:** Add UI fields for `last_growth_reason`, `last_growth_attempt_result`, and `growth_freeze`. Add an admin button to trigger `_trigger_growth_system()`.

### 3.2. MetaAgent Capabilities
*   **[PENDING] Fix + Distill Visibility:** Add a "MetaAgent Validation" UI card to trigger `/api/meta/test` and show patch/distill outcomes.

---

## 4. User Interface & Security

### 4.1. Multi-Agent Chat
*   **[PENDING] Visibility:** Ensure each agent appears as its own message in the UI. Add "Response provenance" tags.

### 4.2. Security Hardening
*   **[PENDING] Env-driven Access Control:** Replace hardcoded emails with `SAM_ALLOWED_EMAILS`, `SAM_ADMIN_EMAILS`, and `SAM_OWNER_EMAIL`. Implement IP allowlisting with proxy support.

### 4.3. Finance & Diagnostics
*   **[PENDING] Finance Panel:** Rename labels to "Money Made" and "Money Saved". Display per-currency totals.

---

## 5. External Integrations & Advanced Backup

### 5.1. Google Docs & Cloud Backup
*   **[PENDING] Google Docs Integration:** Integrate data from the user's Google Docs.
*   **[PENDING] Cloud Storage Sync:** Implement a mechanism to save/load state and backups to/from external cloud storage (Google Drive/Docs).

### 5.2. Submodel Autonomy
*   **[PENDING] Plan/Design/Implement/Test Lifecycle:** Enforce that submodels (Symbolic SAM, etc.) follow a full development lifecycle instead of skipping steps.

---

## 6. Repository Hygiene

*   **[COMPLETED] Aggressive Cleanup:** Removed compiled binaries, caches, and duplicate script versions from the root.
*   **[COMPLETED] Gitignore Update:** Updated `.gitignore` to prevent committing binaries and local cruft.
*   **[PENDING] Restructure Layout:** Move C sources to `src/`, Python bindings to `bindings/`, etc., to clean up the root directory.

---

## 7. Immediate Next Steps

1.  **Investigate Growth System:** Check `sam_meta_controller_c.c` and its interaction with `complete_sam_unified.py`.
2.  **Harden C Research:** Remove the "simulated" fallback in `specialized_agents_c.c`.
3.  **Implement Security Allowlists:** Update `complete_sam_unified.py` to use environment variables for admin emails.
