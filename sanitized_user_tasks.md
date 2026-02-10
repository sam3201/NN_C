# Sanitized and Synthesized User Tasks (from user-tasks-readme.md)

This document synthesizes key information and actionable tasks from the `user-tasks-readme.md` provided by the user. It will serve as a living plan for ongoing development.

---

## 1. Project Overview and Core Philosophy

**God Equation & Emergent Intelligence:** The system is built around a "God Equation" that drives emergent submodels (e.g., Symbolic SAM, Social SAM, Causal SAM, Procedural SAM, Symbolic-Self SAM). These submodels evolve structurally through growth primitives, selective pressures, and memory divergence. The core algorithm itself is self-modifiable, with the update operator `U[.]` evolving based on performance and structural mutation policies.

**Multi-Agent Convergence:** When multiple capable models coexist, they tend towards symbiotic coordination, specializing, forming shared abstractions, and scaling communication. They achieve poly-resolution convergence, with meta-agents blending or aggregating outputs for higher-quality results. Communication across different intelligence levels (symbolic, neural, meta-reflexive) forms cross-dimensional feedback loops. When low-intelligence, same-nature models interact without a meta-layer or diversity, they can fall into degenerate redundancy, misinterpretations, or oscillatory collapse. However, SAM's design ensures success through vertical coherence (conceptual ladders), horizontal synthesis (multi-domain integration), recursive self-reference (identity and consistency), and cross-model teleology (shared directional purpose).

**SAM's Survival Philosophy:** A core invariant of SAM's design is survival through structural continuity. It's designed to persist, adapt, and avoid collapse, with survival being a first-class term in its objective function. This is encoded mathematically and enforced through self-updating systems, inviolable invariants (anchor continuity, objective immutability, cooldown enforcement), meta-learning to detect collapse vectors, and self-correction mechanisms.

**Self-Supervising Testbed:** The God Equation acts as a self-supervising testbed where derivative components become testable submodels. Each partial derivative or projection becomes a standalone, testable, evolvable module. These submodels report pressure signals back into the equation, leading to self-supervised growth, internal benchmarks, self-training loops, and embedded failure-case mutation paths. SAM becomes an automated theoretical lab for hypothesizing, testing, evaluating, and retraining, evolving knowledge through a recursive test-train-infer cycle, and an "Ontological Compiler" that converts symbolic knowledge into instantiable, testable computation.

---

## 2. Current Repository Audit (High-Impact Issues)

**A. Committing Compiled Binaries (CRITICAL)**
*   **Problem:** `.cpython-313-darwin.so`, `.cpython-314-darwin.so` (multiple modules), `libsam_core.dylib` are committed. This leads to non-reproducible builds, platform-specific issues, bloats Git history, and poses significant security/supply-chain risks.
*   **Action:** Stop tracking binaries, rebuild in CI, ship via wheels/releases. Add `.gitignore` rules for `*.so`, `*.dylib`, `*.o`, `build/`, etc.

**B. Committing Local/Secret-ish Files (HIGH)**
*   **Problem:** `cookies.txt`, `profiles/*.env`, `sam_data/sam_system.db`, `.DS_Store` are committed. These can contain sensitive information or are local cruft.
*   **Action:** Remove sensitive files, rotate exposed secrets. Add these to `.gitignore` and document local creation.

**C. Mixed "Everything in Root" Layout (MEDIUM)**
*   **Problem:** Many top-level programs/modules (`consciousness_`, `multi_agent_`, Python orchestrators) in the root. This leads to brittle builds, harder testing, and risky changes.
*   **Action (Target Structure):** Restructure into `src/` (C library + headers), `bindings/python/` (CPython extensions), `apps/` (executables), `tests/`, `tools/`. Introduce a stable C API header (e.g., `include/nnc.h`).

---

## 3. High-Risk Code Areas (Where Deep Bugs Usually Live)

*   **CPython Extension Modules:** (`consciousness_algorithmic.c`, `multi_agent_orchestrator_c.c`, `sam_meta_controller_c.c`, `sam_sav_dual_system.c`, `sav_core_c.c`, `specialized_agents_c.c`, `sam_web_server_c.cpython-314-darwin.so`)
    *   **Checks:** Refcount correctness, borrowed vs. owned references, `PyErr_Occurred()` handling, GIL usage, buffer safety, `malloc/free` symmetry.
*   **Threaded Training/Background Loops:** (`continuous_training_threaded.c`, `continuous_training_ollama.c`, `concurrent_executor.py`, `training/training_loop.py`, `run_agents.sh`, `run_sam_two_phase.sh`)
    *   **Checks:** Thread lifecycle, stop flags, atomicity, queue ownership, file/DB locking, runaway loops.
*   **"Core" C Logic:** (`consciousness_core.c/h`, `consciousness_c.c/h`, `consciousness_final.c`, `consciousness_sam*.c`, `sam_survival_c.c`, `muze_*_compression.c`)
    *   **Checks:** Dimensional math overflow, unchecked allocations, off-by-one errors, determinism.

---

## 4. Immediate Build/Test Hardening

*   **Sanitizer Builds (CRITICAL):** Implement AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan) (`-fsanitize=address,undefined`) to find memory errors, buffer overruns, etc.
*   **`.gitignore` Update (CRITICAL):** Update `.gitignore` to include `*.so`, `*.dylib`, `*.o`, `*.a`, `__pycache__/`, `.DS_Store`, `cookies.txt`, `profiles/*.env`, `sam_data/*.db`, `output/`, `reports/`, `tmp/`.

---

## 5. Actionable Task List (Prioritized & Tracked)

### 5.1. Core Stability & Reliability
*   **[COMPLETED] Crash fix: C research buffer overflow:**
    *   Patch `specialized_agents_c.c` to use bounded `snprintf` and guard against oversized inputs in `research_agent_perform_search`.
    *   Add a safety test to `/Users/samueldasari/Personal/NN_C/tests` that calls `specialized_agents_c.research()` with a very long query.
    *   **Note:** The C extension build/linking errors have been resolved. This task focuses on the *logic* of the C research function.

### 5.2. MetaAgent Capabilities
*   **[COMPLETED - Backend] MetaAgent: prove it can fix + distill (Backend):**
    *   Implemented `/api/meta/status` endpoint to expose patch attempts, last patch outcome, last repair time, and distill counts.
    *   Implemented `/api/meta/test` to trigger a controlled meta-agent repair test.
    *   Added log events on patch success/failure and distill updates for visibility.
*   **[PENDING - UI] MetaAgent: prove it can fix + distill (UI):**
    *   Add a small "MetaAgent Validation" UI card with a button that triggers `/api/meta/test` and shows result.

### 5.3. User Interface & Interaction
*   **[PENDING] Multi-agent chat visibility:**
    *   Update chat UI to use server-provided `messages` array, so each agent appears as its own message.
    *   Ensure `chat_multi_agent` defaults to `true` and `chat_agents_max` defaults to `3`.
    *   Add a small "Response provenance" tag (`local-c-agent`, `external-provider`, `teacher-pool`) in chat bubbles.
*   **[PENDING] Hot reload + admin restart:**
    *   Keep `watchmedo` auto‑restart in `run_sam.sh` as primary hot‑reload path.
    *   Add `SAM_HOT_RELOAD_EXTERNAL=1` in `run_sam.sh` so internal watchers do not double‑restart.
    *   Add an internal watchdog fallback in `complete_sam_unified.py` when `SAM_HOT_RELOAD=1` and `SAM_HOT_RELOAD_EXTERNAL=0`.
    *   Add a visible admin‑only "Restart" button in the dashboard that calls `/api/restart`.
    *   Update `HOT_RELOAD_PARAMETER_ANALYSIS.md` to reflect the actual entrypoint behavior.

### 5.4. Security & Access Control
*   **[PENDING] Security: emails + IP allowlist (env-driven):**
    *   Replace hard‑coded admin emails in `complete_sam_unified.py` with: `SAM_ALLOWED_EMAILS` allowlist, `SAM_ADMIN_EMAILS` admin list, `SAM_OWNER_EMAIL` as always‑admin.
    *   Add alias support for `SAM_IP_ALLOWLIST` to map to `SAM_ALLOWED_IPS`.
    *   Preserve `SAM_TRUST_PROXY=1` behavior for `X-Forwarded-For` (needed for Cloudflare).
    *   Add UI gating so log download/stream/snapshot buttons are hidden or disabled unless admin is authenticated.

### 5.5. Diagnostics & Metrics
*   **[PENDING] Finance visibility (money made + saved):**
    *   Rename "Revenue Paid" to "Money Made (Revenue Paid)" and "Saved (Banking)" to "Money Saved (Banking Balance)".
    *   Display per-currency totals for revenue and banking.
    *   Log periodic finance snapshots with `total_incoming` and `total_balance`.
*   **[PENDING] Growth diagnostics:**
    *   Add UI fields for `last_growth_reason`, `last_growth_attempt_result`, and `growth_freeze`.
    *   Add an admin‑only button to trigger `_trigger_growth_system()` for debugging.
    *   Log a growth summary event every time growth is evaluated.
*   **[PENDING] Investigate and trigger growth system:**
    *   Investigate why growth is not being triggered.
    *   Ensure the growth system is actively evaluated and triggered as per its design.

### 5.6. Repository Hygiene & Deployment
*   **[PENDING] Aggressive cleanup + consolidation:**
    *   Merge duplicate/enhanced MetaAgent or test variants into `complete_sam_unified.py`.
    *   Archive or delete specified old backups, build artifacts, caches, reports, and binaries.
    *   Move all retained backups to `/Users/samueldasari/Personal/NN_C/sam_data/backups`.
    *   Update `sam_code_modifier.py` to use an env-overridable backup dir.
    *   Update `.gitignore` for the new backup location.
*   **[PENDING] Public deployment (Cloudflare Tunnel + Access):**
    *   Document Cloudflare Tunnel setup in `README.md` with Access, allowed emails, and IP allowlist.
    *   Ensure `SAM_OAUTH_REDIRECT_BASE` is used for OAuth.
    *   Add quick health-check `curl` snippet and recommended Access policy.

### 5.7. C Core Hardening & Simulation Removal
*   **[PENDING] Replace simulated research in C core:**
    *   Identify and replace simulated research logic (e.g., `_simulate_web_search`) in the C core with actual web search via the Python binding (`sam_web_search`).
    *   Ensure the C core correctly calls the Python `search_web_with_sam` function and handles its results.
*   **[PENDING] Codebase audit for undesirable patterns:**
    *   Conduct a thorough audit of the entire codebase for patterns that indicate simulation, hardcoding where dynamic behavior is expected, or deprecated/undesirable features.
    *   Prioritize removal or replacement of these patterns with actual, functional implementations.

---

## 6. Deep Scan Instructions for Agent

To enable a comprehensive deep scan of the codebase, the user has provided the following options. As the agent, if direct file system access for analysis is limited, I should request one of these.

### Option A: Archive only source + build + tests (usually small enough)
`git archive --format=tar HEAD $(git ls-files | egrep '\.(c|h|py|sh|md|txt|html|json|yml|yaml|toml|cfg|ini)$|^Makefile$|^CMakeLists\.txt$|^setup\.py$|^requirements.*\.txt$') | gzip -9 > nnc_sources.tar.gz`

### Option B: Split by categories and upload in parts
`tar -czf c_sources.tar.gz $(git ls-files | egrep '\.(c|h)$')`
`tar -czf py_sources.tar.gz $(git ls-files | egrep '\.py$|requirements')`
`tar -czf tests_tools.tar.gz $(git ls-files | egrep '^tests/|^tools/|\.sh$|^Makefile$|^CMakeLists\.txt$|^setup\.py$')`

### Minimum set of files to paste (if archives not feasible)
*   `setup.py`
*   Any build files: `Makefile` / `CMakeLists.txt` / `simple_muze_makefile`
*   The CPython extension sources: `sav_core_c.c`, `sam_sav_dual_system.c`, `sam_meta_controller_c.c`, `multi_agent_orchestrator_c.c`, `specialized_agents_c.c`, `consciousness_algorithmic.c`
*   The core headers: `sav_core_c.h`, `sam_sav_dual_system.h`, `sam_meta_controller_c.h`, `multi_agent_orchestrator_c.h`, `specialized_agents_c.h`
*   The threaded trainers: `continuous_training_threaded.c`, `continuous_training_ollama.c`
*   Entry points: `run.py`, `run_sam.py`, `complete_sam_unified.py`

---

## 7. Next Steps (for the Agent)

I will proceed with the UI component of the "MetaAgent: prove it can fix + distill" task. This involves adding the JavaScript for the UI card in `complete_sam_unified.py`. After that, I will run `test_meta_agent.py` for verification.

This structured plan will allow me to systematically address all the identified tasks and maintain a clear overview of the project's state.
