# SAM-D (ΨΔ•Ω-Core v5.0.0 Recursive) Clean Codebase Plan

## 1. Structural Targets
- **Consolidation:** 
    - Merge all `automation_master*.py` variants into `src/python/automation/core.py`.
    - Archive all `run_*.sh` scripts into a single `tools/launcher.sh` with subcommands.
    - Move all `.py` files from the root directory to `src/python/` unless they are entry-point scripts.
- **Decomposition:**
    - Decompose the 18k+ line `complete_sam_unified.py` into modular components:
        - `src/python/core/orchestrator.py`
        - `src/python/core/governance.py` (CIC/AEE/CSF logic)
        - `src/python/api/routes.py`
        - `src/python/api/socket_handlers.py`

## 2. Robustness & Completeness
- **Zero-Fallback Policy:** 
    - Audit all `pass` statements and implement real logic or proper error handling.
    - Implement real logic for `_extract_assumptions_from_response` and `_extract_unknowns_from_response` (COMPLETED).
    - Flesh out the `IntelligentIssueResolver` to handle complex multi-file patches.
- **Real Automation:** 
    - Replace mock data in `.openclaw/openclaw_bridge.py` with calls to the actual `AutomationCore`.
    - Implement real `PLAN` and `BUILD` phase execution in the cyclic workflow.

## 3. Security & RBAC (Finalization)
- **Financial Lockdown:**
    - (COMPLETED) Restrict all Revenue Ops and Banking Sandbox write-actions to `Owner` only.
    - (COMPLETED) Restrict read-access to sensitive financial metrics to `Owner` only.
- **Auditing:**
    - Implement automated log rotation for `logs/audit.jsonl` to prevent disk exhaustion.
    - Add real-time dashboard alerts for blocked elevation attempts.
- **Command Allowlisting:**
    - Implement a strict regex-based allowlist for all shell commands executed by subagents.

## 4. Coding Standards
- **Naming:** 
    - (COMPLETED) Rename `LoveAgent` to `Love` (Stability Branch).
    - Eliminate all personality-coded naming (e.g., `MoneyMaker` -> `RevenueEngine`).
- **Style:**
    - Adopt `ruff` for linting and `black` for formatting.
    - Remove all emojis from logs and code comments.
- **Documentation:**
    - Maintain `docs/CHANGELOG_AUDIT.md` for every significant system change.

## 5. Execution Roadmap
1.  **Milestone 1: Repository Consolidation** (Move legacy to `legacy/`, establish `src/` layout).
2.  **Milestone 2: Modular Decomposition** (Break down `complete_sam_unified.py`).
3.  **Milestone 3: OpenClaw Wiring** (Connect bridge to real automation core).
4.  **Milestone 4: Security Hardening** (Command allowlisting + Secret rotation).

**Status:** Awaiting final review of Deep Scan findings before proceeding with implementation.
