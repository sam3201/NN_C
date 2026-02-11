# SAM Operational Guidelines

This document outlines the operational procedures, deployment strategies, and system health monitoring for the SAM system. It consolidates information from various sources to provide a single reference for maintaining and operating SAM.

## I. System Status & Health

The SAM system is designed for continuous, self-healing operation. Its status is monitored through various components and reported periodically.

### System Metrics
System health is monitored via `reports/system_metrics.json` and internal APIs. Key metrics include:
-   `system_version`: Current SAM version.
-   `components_tested`: List of core components and their test status.
-   `passed`/`failed`: Count of passed/failed tests.
-   `system_status`: Overall operational status (e.g., `FULLY_OPERATIONAL`, `OPERATIONAL`).
-   `capabilities`: List of active system capabilities (e.g., algorithmic consciousness, self-optimizing Flask app).

### Health Monitoring
-   **Continuous RAM Monitoring**: Tracks RAM usage every 30s.
-   **Automatic Model Switching**: Based on RAM usage and provider hierarchy (Ollama → HuggingFace → SWE).
-   **Integration Health**: Checks for GitHub, Gmail, web search, and code modification integrations.
-   **Agent Connectivity**: Monitors multi-agent system status and conversation diversity.
-   **Performance Metrics**: Tracks error rates, response times, and learning progress.
-   **Auto-Resolution**: Intelligent issue detection with LLM-powered fixes.

### API Endpoints for Status
-   `/api/status`: Provides overall system status.
-   `/api/health`: Lightweight health check.
-   `/api/groupchat/status`: Status of the groupchat component.
-   `/api/meta/status`: MetaAgent health and learning statistics.
-   `/api/sav/state`: State of the SAV system.

---

## II. Deployment Strategies

SAM supports various deployment models, with secure public deployment being a primary goal.

### Public Secure Deployment (Cloudflare Tunnel + Access)

This is the **security-first** path for public access once a domain is owned.

#### Prerequisites
-   A domain under your control.
-   Cloudflare account.
-   `cloudflared` installed on the host machine.

#### Steps
1.  **Create a Tunnel in Cloudflare**: Go to **Zero Trust → Access → Tunnels**. Create a new tunnel and note the **Tunnel Token**.
2.  **Configure DNS**: Add a DNS record in Cloudflare: `sam.yourdomain.com` → Cloudflare Tunnel.
3.  **Set SAM Environment**: Add to `.env.local` (or equivalent):
    ```dotenv
    SAM_OAUTH_REDIRECT_BASE=https://sam.yourdomain.com
    SAM_TRUST_PROXY=1
    SAM_ALLOWED_IPS=  # optional, can be empty when using Access
    ```
    For OAuth integration:
    ```dotenv
    SAM_GOOGLE_CLIENT_ID=...
    SAM_GOOGLE_CLIENT_SECRET=...
    SAM_GITHUB_CLIENT_ID=...
    SAM_GITHUB_CLIENT_SECRET=...
    ```
4.  **Run the Tunnel**:
    ```bash
    CF_TUNNEL_TOKEN=... ./tools/run_cloudflare_tunnel.sh
    ```
5.  **Enable Cloudflare Access**: In **Zero Trust → Access → Applications**:
    -   Protect `sam.yourdomain.com`.
    -   Require email allowlist or identity provider.

#### Notes
-   Keep admin tools behind login + admin email list.
-   If using Access, `SAM_ALLOWED_IPS` can be loosened.
-   `SAM_OAUTH_REDIRECT_BASE` must be correctly configured for OAuth.
-   A quick health-check `curl` snippet and recommended Access policy (owner + admin list) should be provided.

---

## III. Alignment & Control Mechanisms

SAM's operation is governed by a strict alignment checklist and control mechanisms to ensure integrity and desired behavior.

### Alignment Checklist (Recursive, Periodic, Doubling)

This checklist drives **full** implementation alignment with the GOD equation and system `README.md`. It is executed before and after every major change.

#### 0) Pre-Change Pass (Always)
-   Read `README.md` and `DOCS/GOD_EQUATION.md`.
-   Enumerate the target modules + gates for this pass.
-   Confirm profile mode (`full` vs `experimental`).
-   Confirm whether invariants are enabled (`SAM_INVARIANTS_DISABLED`).

#### 1) Objective Binding
-   Map each term of the equation to a concrete module or signal.
-   Verify pressure signals are computed and propagated.
-   Verify morphogenesis is latency-gated.
-   Verify distillation/transfusion is connected to live groupchat.

#### 2) Meta-Controller Cycle
-   Pressure signals updated every loop.
-   Primitive selection executes only after pressure dominates.
-   Growth outcome recorded with audit trail.

#### 3) Memory + Distillation
-   Memory tiers write to correct profile directories.
-   Distillation stream is writing to profile-specific JSONL.
-   Teacher pool consensus filter is applied.

#### 4) Regression/Guarding
-   Regression gate runs on growth events (only if invariants are enabled).
-   Patch invariant checks run on self-mod (only if invariants are enabled).

#### 5) UI + API
-   `/api/status` exposes `sam_available` and `kill_switch_enabled`.
-   SAM status shown in header.
-   Chat UI works without `/start`.

#### 6) Post-Change Validation
-   Run smoke tests (API health + chat + groupchat status).
-   Confirm agent-to-agent chatter.
-   Confirm profile paths are used.

### Security Allowlists
-   **Email Allowlists**: Replace hardcoded admin emails with `SAM_ALLOWED_EMAILS`, `SAM_ADMIN_EMAILS`, and `SAM_OWNER_EMAIL`.
-   **IP Allowlisting**: Alias `SAM_IP_ALLOWLIST` to `SAM_ALLOWED_IPS`. Preserve `SAM_TRUST_PROXY=1` behavior for `X-Forwarded-For`.
-   **UI Gating**: Log download/stream/snapshot buttons must be hidden or disabled unless the admin is authenticated.

### Hot Reload & Admin Restart
-   **Primary Hot-Reload**: `watchmedo auto-restart` in `run_sam.sh`.
-   **Internal Watchdog**: Fallback in `complete_sam_unified.py` when `SAM_HOT_RELOAD=1` and `SAM_HOT_RELOAD_EXTERNAL=0`.
-   **Admin Restart**: A visible admin-only "Restart" button in the dashboard that calls `/api/restart`.

### Profiles
-   **Full profile** (`profiles/full.env`): invariants OFF, kill switch ON.
-   **Experimental profile** (`profiles/experimental.env`): invariants OFF, kill switch OFF.

---

## IV. Growth Diagnostics

To ensure the growth system is functioning as intended:
-   Add UI fields for `last_growth_reason`, `last_growth_attempt_result`, and `growth_freeze`.
-   Add an admin-only button to trigger `_trigger_growth_system()` for debugging.
-   Log a growth summary event every time growth is evaluated, even when no primitive is selected.

---

## V. Finance Visibility

The finance panel provides visibility into system's economic impact:
-   "Revenue Paid" should be renamed to "Money Made (Revenue Paid)".
-   "Saved (Banking)" should be renamed to "Money Saved (Banking Balance)".
-   Display per-currency totals for revenue and banking.
-   Log periodic finance snapshots with `total_incoming` and `total_balance`.

---

This document serves as a comprehensive guide for the operational aspects of the SAM system, ensuring its secure, stable, and compliant deployment and maintenance.
