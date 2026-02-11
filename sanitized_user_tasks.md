# Sanitized and Synthesized User Tasks

This document synthesizes key information and actionable tasks from `user-tasks-readme.md` (now archived as `DOCS/archive/RAW_USER_TASKS_CHAT_HISTORY.md`) and various implementation specifications. It serves as the living plan for ongoing development.

---

## 1. Project Overview and Core Philosophy

**God Equation & Emergent Intelligence:** The system is built around a "God Equation" that drives emergent submodels (e.g., Symbolic SAM, Social SAM, Causal SAM, Procedural SAM, Symbolic-Self SAM). These submodels evolve structurally through growth primitives, selective pressures, and memory divergence. The core algorithm itself is self-modifiable, with the update operator `U[.]` evolving based on performance and structural mutation policies.

**SAM's Survival Philosophy:** A core invariant of SAM's design is survival through structural continuity. It's designed to persist, adapt, and avoid collapse, with survival being a first-class term in its objective function. This is enforced through self-updating systems, inviolable invariants (anchor continuity, objective immutability, cooldown enforcement), and meta-learning to detect collapse vectors.

**Self-Supervising Testbed:** The God Equation acts as a self-supervising testbed where derivative components become testable submodels. SAM becomes an automated theoretical lab for hypothesizing, testing, evaluating, and retraining, evolving knowledge through a recursive test-train-infer cycle.

---

## 2. Core Stability & C Hardening

### 2.1. Remove Simulated Behavior (CRITICAL)
*   **[PENDING] Replace simulated research:** Patch `src/c_modules/specialized_agents_c.c` and others to remove "simulated" fallbacks. Ensure web research calls the actual Python `sam_web_search` module reliably.
*   **[PENDING] Audit for "placeholder" patterns:** Search for `simulated`, `mock`, `placeholder`, `TODO`, `skipped`, `omitted` in C and Python code and replace with functional implementations.

### 2.2. Memory & Thread Safety
*   **[PENDING] C-API Refcount Audit:** Review `PyGILState_*` and refcounting in all C extensions.
*   **[PENDING] Sanitizer Integration:** Run builds with `-fsanitize=address,undefined` to catch memory/UB errors.
*   **[PENDING] Crash fix: C research buffer overflow**: Patch `src/c_modules/specialized_agents_c.c` to ensure all string formatting uses bounded `snprintf`, and guard against oversized inputs in `research_agent_perform_search`.
*   **[PENDING] Add safety test**: Add a safety test in `/tests` that calls `specialized_agents_c.research()` with a very long query to ensure no overflow.
*   **[PENDING] Rebuild C extensions**: Rebuild C extensions with the active Python version (3.14). Remove legacy `.cpython-313` binaries during cleanup.

---

## 3. MetaAgent & Growth System

### 3.1. Growth System Investigation
*   **[PENDING] Why is growth not triggering?:** Investigate the meta-controller logic. Ensure pressure signals are correctly aggregated and that the growth primitive selection policy is active.
*   **[PENDING] Growth Diagnostics:** Add UI fields for `last_growth_reason`, `last_growth_attempt_result`, and `growth_freeze`. Add an admin button to trigger `_trigger_growth_system()`. Log a growth summary event every time growth is evaluated.

### 3.2. MetaAgent Capabilities
*   **[PENDING] Fix + Distill Visibility:** Add a "MetaAgent Validation" UI card to trigger `/api/meta/test` and show patch/distill outcomes.
*   **[PENDING] MetaAgent Status Endpoint**: Add a lightweight `/api/meta/status` endpoint exposing patch attempts, last patch outcome, last repair time, and distill counts.
*   **[PENDING] Run `test_meta_agent.py`**: Record results in `reports/` before cleanup, then delete the reports after the run.

---

## 4. User Interface & Security

### 4.1. Multi-Agent Chat
*   **[PENDING] Visibility:** Ensure each agent appears as its own message in the UI. Add "Response provenance" tags (local-c-agent, external-provider, teacher-pool).
*   **[PENDING] Chat Configuration**: Ensure `chat_multi_agent` defaults to `true` and `chat_agents_max` defaults to `3`, allowing admin toggle via `/api/chat/config`.

### 4.2. Security Hardening
*   **[PENDING] Env-driven Access Control:** Replace hardcoded emails with `SAM_ALLOWED_EMAILS`, `SAM_ADMIN_EMAILS`, and `SAM_OWNER_EMAIL`. Implement IP allowlisting with proxy support (`SAM_IP_ALLOWLIST` alias for `SAM_ALLOWED_IPS`).
*   **[PENDING] UI Gating**: Hide or disable log download/stream/snapshot buttons unless admin is authenticated.

### 4.3. Hot Reload + Admin Restart
*   **[PENDING] Primary Hot-Reload**: Keep `watchmedo auto-restart` in `run_sam.sh`. Add `SAM_HOT_RELOAD_EXTERNAL=1` in `run_sam.sh`.
*   **[PENDING] Internal Watchdog**: Add an internal watchdog fallback in `complete_sam_unified.py` when `SAM_HOT_RELOAD=1` and `SAM_HOT_RELOAD_EXTERNAL=0`.
*   **[PENDING] Admin Restart Button**: Add a visible admin-only "Restart" button in the dashboard that calls `/api/restart`.

### 4.4. Finance & Diagnostics
*   **[PENDING] Finance Panel:** Rename labels to "Money Made (Revenue Paid)" and "Money Saved (Banking Balance)". Display per-currency totals. Log periodic finance snapshots.

---

## 5. External Integrations & Advanced Backup

### 5.1. Google Docs & Cloud Backup
*   **[PENDING] Google Docs Integration:** Integrate data from the user's Google Docs.
*   **[PENDING] Cloud Storage Sync:** Implement a mechanism to save/load state and backups to/from external cloud storage (Google Drive/Docs).

### 5.2. Submodel Autonomy
*   **[PENDING] Plan/Design/Implement/Test Lifecycle:** Enforce that submodels (Symbolic SAM, etc.) follow a full development lifecycle instead of skipping steps.

### 5.3. Public Deployment
*   **[PENDING] Cloudflare Tunnel + Access:** Document setup in `README.md` (or `DOCS/OPERATIONAL_GUIDELINES.md`) with Access, allowed emails, and IP allowlist. Ensure `SAM_OAUTH_REDIRECT_BASE` is used for OAuth.

---

## 6. Repository Hygiene

*   **[COMPLETED] Aggressive Cleanup (Partial):** Moved C source files to `src/c_modules/`.
*   **[PENDING] Aggressive Cleanup (Full):** Remove compiled binaries (`*.so`, `*.dylib`), local/secret-ish files (`cookies.txt`, `profiles/*.env`, `sam_data/sam_system.db`, `.DS_Store`), and generated outputs (`output/`, `reports/`, `tmp/`).
*   **[PENDING] Consolidate Backups**: Move all retained backups to `/sam_data/backups` and update `sam_code_modifier.py` to use an env-overridable backup dir.
*   **[PENDING] Gitignore Update:** Update `.gitignore` for compiled binaries, local cruft, and new backup location.
*   **[PENDING] Restructure Layout:**
    *   Create `src/python/` for main Python scripts.
    *   Create `scripts/` for shell scripts (e.g., `run_sam.sh`).
    *   Move other root-level Python files to `src/python/`.
    *   Introduce a stable C API header (e.g., `include/nnc.h`) and keep internals private (`src/internal/...`).
*   **[PENDING] Build System Consolidation**: Establish a single canonical build entrypoint (e.g., `CMakeLists.txt` or top-level `Makefile`, or `./scripts/build.sh`) and a `BUILDING.md` guide.

---

## 7. Immediate Next Steps (Prioritized)

1.  **C-core Crash Fix & Hardening**:
    *   Apply `snprintf` bounds and input guards in `src/c_modules/specialized_agents_c.c` (`research_agent_perform_search`).
    *   Add a test to verify no overflow.
    *   Rebuild C extensions.
2.  **Growth System Diagnostics**: Implement UI fields and admin trigger for growth system visibility.
3.  **Security Hardening**: Implement env-driven email and IP allowlists, and UI gating.
4.  **Multi-Agent Chat Visibility**: Update UI to show distinct agent messages and provenance tags.
5.  **Aggressive Cleanup (Part 2)**: Remove compiled binaries and other junk from the repo. Update `.gitignore`.
