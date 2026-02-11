# SAM 2.1 AGI (ΨΔ-Core)

SAM 2.1 is a hybrid Python/C multi-agent system with a web dashboard, slash-command interface, and C-accelerated cores for meta-control and dual-system simulation. It features the advanced ΨΔ-Core morphogenesis and PDI-T submodel lifecycle.

## What's Included
- Python orchestration, API server, and CLI: `complete_sam_unified.py`
- Web dashboard and terminal UI served by the API
- C extensions for speed: `sam_sav_dual_system`, `sam_meta_controller_c`, `consciousness_*`, `multi_agent_orchestrator_c`, `specialized_agents_c`
- Support scripts and runners: `run_sam.sh`, `setup.py`

## Requirements
- Python 3.10+
- A C compiler toolchain compatible with Python extensions
- Optional local model backend: Ollama (if using local models)
- Optional hosted model backends: set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GITHUB_TOKEN`
- Gmail OAuth dependencies (see Gmail section)

## Quick Start
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Build C extensions
```bash
python setup.py build_ext --inplace
```
3. (Optional) Install Ollama and pull at least one model
```bash
ollama --version
ollama pull codellama:latest
```
4. Run
```bash
./run_sam.sh
# or
python3 complete_sam_unified.py
```

## Profiles (Full vs Experimental)
This repo ships two execution profiles:
- **Full** (stable + kill switch enabled)
- **Experimental** (no kill switch by design)

Profile configs live in `profiles/`:
- `profiles/full.env`
- `profiles/experimental.env`

Launch scripts:
```bash
./run_sam_full.sh         # full profile (kill switch enabled)
./run_sam_experimental.sh # experimental profile (no kill switch)
```

By default, `run_sam.sh` loads the **full** profile. To override:
```bash
SAM_PROFILE=experimental ./run_sam.sh
```

## ChatGPT Research Archive
The raw ChatGPT research transcript (sanitized) and a cleaned summary are available:
- `README-chatGPT-raw.md` (verbatim, private info redacted)
- `README-chatGPT-clean.md` (structured summary)

## Canonical Equation + Alignment
- `DOCS/GOD_EQUATION.md` — full, canonical objective formulation
- `DOCS/ALIGNMENT.md` — recursive alignment checklist (full vs experimental)

### SAM-D / OmniSynapse Equation (Top-Level)
G = Σ_{i=1}^{∞} [ α_i · F_i(x⃗, t) + β_i · dF_i/dt + γ_i · ∇_{F_i} L + δ_i · μ_i + ζ_i · Φ(G) ]

Recursive layer
Φ(G) = lim_{n→∞} ( G_n + λ · dG_n/dn + ρ · d^2 G_n/dn^2 + … )

Full canonical expansion and appendix live in `DOCS/GOD_EQUATION.md`.

## Score Handling (N/A)
When agent outputs do not include a `score:` field, the system marks the result as **pending** and records a reason + action in the JSONL log (`logs/sam_runtime.jsonl`) under event `score_unusable` or `score_missing`. Actions include `retry_later`, `retry_score_inference`, or `investigate` depending on the reason (initializing, timeout, rate limit, error, or missing score field).

## Live Log Panel
The dashboard includes a **Live Event Log** panel. It streams `logs/sam_runtime.jsonl` in real time and provides a snapshot that summarizes the moving window, counts by level, and top event types.

Finance summary is available in the dashboard and via:
- `/api/finance/summary` (combined revenue + banking)
- `/api/revenue/metrics`, `/api/banking/metrics`
- Snapshot interval is configurable in the UI or via `POST /api/finance/config` (admin).

## Secure Remote Access (Free)
Recommended: **Tailscale** for private, secure access with no public exposure.
1. Install Tailscale on the host machine.
2. Join the same tailnet on your device.
3. Access the app via the host’s Tailscale IP and port 5004.
4. Use the app login (email + password) for access control.

### Cloudflare Tunnel + Access
For a permanent public URL and enhanced security, Cloudflare Tunnel is recommended.

1.  **Set up a Cloudflare Tunnel:**
    *   Install `cloudflared` on your SAM host: `brew install cloudflare/cloudflare/cloudflared` (macOS).
    *   Authenticate `cloudflared` with your Cloudflare account.
    *   Create a new Tunnel: `cloudflared tunnel create sam-tunnel`.
    *   Configure `~/.cloudflared/config.yml` to point to your SAM instance (e.g., `url: http://localhost:5004`).
    *   Route a DNS record (e.g., `sam.yourdomain.com`) to your Tunnel.
    *   Run the Tunnel: `cloudflared tunnel run sam-tunnel`.

2.  **Configure Cloudflare Access:**
    *   In your Cloudflare Dashboard, navigate to `Access -> Applications`.
    *   Add a new application for your SAM domain (e.g., `sam.yourdomain.com`).
    *   **Session Duration:** `24 hours` (recommended).
    *   **Identity Providers:** Configure Google or GitHub (matching your SAM OAuth setup).
    *   **Rules:**
        *   **Owner Access:** `Include: Emails -> your@owner.email`. Action: `Allow`.
        *   **Admin Access:** `Include: Emails -> Emails in SAM_ADMIN_EMAILS`. Action: `Allow`.
        *   **Allowed Users:** `Include: Emails -> Emails in SAM_ALLOWED_EMAILS`. Action: `Allow`.
        *   **IP Allowlist:** `Include: IPs -> IPs in SAM_ALLOWED_IPS / SAM_IP_ALLOWLIST`. Action: `Allow`.
        *   **Fallback:** `Block all others`.
    *   **CORS (Optional, if using API from different origins):** Enable CORS for your API routes.

3.  **SAM Environment Variables for Cloudflare Integration:**
    *   `SAM_TRUST_PROXY=1`: Essential for Flask to correctly interpret `X-Forwarded-For` headers from Cloudflare.
    *   `SAM_OAUTH_REDIRECT_BASE=https://sam.yourdomain.com`: **Crucial** for OAuth redirects to work correctly with your public domain. Ensure this matches your configured OAuth redirect URIs in Google/GitHub.

4.  **Health Check Snippet:**
    You can verify your SAM instance through the tunnel:
    ```bash
    curl -s https://sam.yourdomain.com/api/health
    # Expected output: {"c_core": "active", "kill_switch_enabled": true, "python_orchestration": "active", "sam_available": true, "status": "ok", "timestamp": ..., "web_interface": "active"}
    ```

When you purchase a domain, switch to Cloudflare Tunnel + Access for a permanent public URL.

## Login + OAuth + IP Allowlist
Set these in `.env.local`:
- `SAM_LOGIN_PASSWORD` — required for password login
- `SAM_LOGIN_PASSWORD_FILE` — optional file path (read and trimmed at runtime)
- `SAM_LOGIN_PASSWORD_KEYCHAIN_SERVICE` + `SAM_LOGIN_PASSWORD_KEYCHAIN_ACCOUNT` — macOS Keychain lookup (preferred)
- `SAM_ALLOWED_EMAILS` — comma‑separated allowlist
- `SAM_OWNER_EMAIL` — always treated as admin
- `SAM_ADMIN_EMAILS` — comma‑separated admin list
- `SAM_SESSION_SECRET` — session secret
- `SAM_ALLOWED_IPS` — optional allowlist (comma‑separated IPs/CIDRs)
- `SAM_TRUST_PROXY=1` — use `X-Forwarded-For` when behind proxy

Keychain (macOS) example:
```
security add-generic-password -s "SAM_LOGIN" -a "you@example.com" -w
SAM_LOGIN_PASSWORD_KEYCHAIN_SERVICE=SAM_LOGIN
SAM_LOGIN_PASSWORD_KEYCHAIN_ACCOUNT=you@example.com
```

OAuth (optional):
- `SAM_GOOGLE_CLIENT_ID`, `SAM_GOOGLE_CLIENT_SECRET`
- `SAM_GITHUB_CLIENT_ID`, `SAM_GITHUB_CLIENT_SECRET`
- `SAM_OAUTH_REDIRECT_BASE` — e.g. `http://localhost:5004`

OAuth helper:
- `GET /api/oauth/help` returns the exact redirect URIs to paste into Google/GitHub.

## Interfaces
- Dashboard: http://localhost:5004
- Terminal: http://localhost:5004/terminal
- Health/API: `/api/health`, `/api/agents`, `/api/command`, `/api/terminal/execute`
- Groupchat: SocketIO (`/api/groupchat/status`)

## Slash Commands (subset)
- `/help`, `/status`, `/agents`
- `/connect <agent_id>`, `/disconnect <agent_id>`, `/clone <agent_id> [name]`, `/spawn <type> <name> [personality]`
- `/research <topic>`, `/code <task>`, `/finance <query>`, `/websearch <query>`
- `/revenue` (queue / approve / reject / submit / leads / invoices / sequences)
- `/start`, `/stop`, `/clear`

## Dual System Implementation (SAM + SAV)
The C extension `sam_sav_dual_system` implements a self-referential dual-system arena optimized for speed:
- Fast RNG (xorshift64*) and fixed-size arenas
- Internal state and long-term memory vectors per system
- Self-alignment and memory-energy metrics integrated into objective scoring
- Objective mutation with structural term changes and self-reference gain
- SAV kill confirmation term for adversarial termination pressure
- SAV unbounded mode (aggressive mutation + action scaling)
- SAM unbounded mode (self-referential + unrestricted mutation)
- Arena pressure feedback loop and adversarial interaction
- Python bindings for creation, stepping, mutation, and telemetry

## Meta-Controller (C)
The `sam_meta_controller_c` extension provides:
- Pressure aggregation across residuals, interference, retrieval entropy, and more
- Growth primitive selection (latent expansion, submodel spawn, routing, consolidation)
- Identity anchoring and optional invariant checks (disabled by default in current profiles)
- Objective contract evaluation (minimax-style)
- Policy gates: persistence thresholds, dominance margin, cooldowns, and risk caps

### Pressure Signals (SAM → Meta)
SAM emits only structured pressure channels:
- `residual`, `rank_def`, `retrieval_entropy`, `interference`
- `planner_friction`, `context_collapse`, `compression_waste`, `temporal_incoherence`

### Growth Primitives (Only Allowed Mutations)
- `GP_LATENT_EXPAND` (add latent dimensions)
- `GP_SUBMODEL_SPAWN` (split into specialized sub-models)
- `GP_INDEX_EXPAND` (expand memory index topology)
- `GP_ROUTING_INCREASE` (increase routing degree)
- `GP_CONTEXT_EXPAND` (expand context binding)
- `GP_PLANNER_WIDEN` (planner depth/width)
- `GP_CONSOLIDATE` (compression/pruning)
- `GP_REPARAM` (representation reparameterization)

### Invariants (Optional, Disabled in Current Profiles)
These constraints are available when `SAM_INVARIANTS_DISABLED=0`.
- Growth causality: every mutation must follow a valid pressure → selection → apply path
- Identity continuity: anchor similarity must remain above threshold
- Cooldown enforcement: structural changes are rate-limited
- Objective immutability (outside explicit contract evaluation)

## Repository Highlights
- `complete_sam_unified.py` — main orchestrator, API, and UI server
- `sam_sav_dual_system.c` — dual-system arena
- `sam_meta_controller_c.c` — meta-controller core
- `multi_agent_orchestrator_c.c` — agent coordination
- `specialized_agents_c.c` — specialized agent primitives
- `consciousness_*.c` — consciousness-related modules

## Smoke Test
```bash
python3 -c "import sam_sav_dual_system, sam_meta_controller_c; print('C extensions import OK')"
python3 -c "from complete_sam_unified import UnifiedSAMSystem; print('System import OK')"
```

## Comprehensive Tests
```bash
SAM_TEST_MODE=1 ./venv/bin/python -c "from SAM_AGI import CompleteSAMSystem; s=CompleteSAMSystem(); s.run_comprehensive_tests()"
```

## Recursive Checks (README‑Aligned)
Run the recursive alignment checks + regression gate:
```bash
./tools/run_recursive_checks.sh
```

## State Persistence
On shutdown, SAM saves a state snapshot and reloads it on next start.
State file is profile‑scoped:
- `sam_data/full/state.json`
- `sam_data/experimental/state.json`

## Training Pipeline
### 1) Install training requirements
```bash
pip install -r requirements_training.txt
```

### 2) Build a distillation dataset (teacher consensus)
```bash
python -m training.distillation \
  --tasks training/tasks/default_tasks.jsonl \
  --output training/distilled.jsonl \
  --teacher ollama:mistral:latest \
  --teacher ollama:llama3:latest \
  --n-per-teacher 1 \
  --min-similarity 0.72 \
  --min-votes 1
```

### 3) Train (LoRA or full fine‑tune)
```bash
# LoRA
python -m training.training_loop \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset training/distilled.jsonl \
  --output training/output_lora \
  --lora

# Full fine‑tune
python -m training.training_loop \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset training/distilled.jsonl \
  --output training/output_full \
  --full
```

### 4) Regression gate (blocks unsafe growth)
```bash
python -m training.regression_suite \
  --tasks training/tasks/default_tasks.jsonl \
  --provider ollama:mistral:latest \
  --min-pass 0.7
```

Environment overrides:
- `SAM_POLICY_PROVIDER` (default: `ollama:qwen2.5-coder:7b`)
- `SAM_POLICY_PROVIDER_PRIMARY` (default: `SAM_POLICY_PROVIDER`)
- `SAM_POLICY_PROVIDER_FALLBACK` (default: `ollama:qwen2.5-coder:7b`)
- `SAM_PROVIDER_AUTO_SWITCH` (default: `1`)
- `SAM_PROVIDER_RAM_THRESHOLD` (default: `0.85`)
- `SAM_PROVIDER_RAM_RECOVER` (default: `0.75`)
- `SAM_CHAT_PROVIDER` (default: empty) — override chat UI provider (e.g. `ollama:qwen2.5-coder:7b`)
- `SAM_CHAT_TIMEOUT_S` (default: `60`)
- `SAM_CHAT_MAX_TOKENS` (default: `512`)
- `SAM_REGRESSION_TASKS` (default: `training/tasks/default_tasks.jsonl`)
- `SAM_REGRESSION_MIN_PASS` (default: `0.7`)
- `SAM_REGRESSION_ON_GROWTH` (default: `1`)
- `SAM_REGRESSION_TIMEOUT_S` (default: `120`)
- `SAM_REQUIRE_SELF_MOD` (default: `1`)
- `SAM_TWO_PHASE_BOOT` (default: `0`) — start meta-only then auto-promote to full
- `SAM_TWO_PHASE_DELAY_S` (default: `5`)
- `SAM_TWO_PHASE_TIMEOUT_S` (default: `180`)
- `SAM_AUTONOMOUS_LOOP_INTERVAL_S` (default: `2`) — throttle autonomous loop to avoid CPU spin
- `SAM_AUTOCONNECT_OLLAMA_MAX` (default: `8`) — cap Ollama auto-connections
- `SAM_AUTOCONNECT_HF_MAX` (default: `6`) — cap HF auto-connections
- HF provider (local LoRA) syntax: `hf:<base_model>@<adapter_path>`
  - Example: `hf:Qwen/Qwen2.5-1.5B@training/output_lora_qwen2.5_1.5b_fp16_v2`
  - Optional env: `SAM_HF_DEVICE_MAP` (default: `auto`), `SAM_HF_DTYPE` (default: `float16`), `SAM_HF_FORCE_GREEDY` (default: `1`)

### Live Groupchat Distillation
The real-time groupchat loop can stream teacher-pool consensus responses directly into a distillation dataset.

Environment overrides:
- `SAM_TEACHER_POOL_ENABLED` (default: `1`)
- `SAM_TEACHER_POOL` (default: `ollama:mistral:latest`)
  - `SAM_TEACHER_POOL_PRIMARY` (default: `SAM_TEACHER_POOL`)
  - `SAM_TEACHER_POOL_FALLBACK` (default: `ollama:mistral:latest`)
  - HF local LoRA example: `hf:Qwen/Qwen2.5-1.5B@training/output_lora_qwen2.5_1.5b_fp16_v2`
- `SAM_TEACHER_N_PER` (default: `1`)
- `SAM_TEACHER_MIN_SIM` (default: `0.72`)
- `SAM_TEACHER_MIN_VOTES` (default: `1`)
- `SAM_TEACHER_TEMP` (default: `0.2`)
- `SAM_TEACHER_MAX_TOKENS` (default: `512`)
- `SAM_TEACHER_TIMEOUT_S` (default: `60`)
- `SAM_DISTILL_PATH` (default: `training/distilled/groupchat.jsonl`)
- `SAM_DISTILL_INCLUDE_CANDIDATES` (default: `0`)

## Revenue Ops Pipeline (Approval + Audit)
Revenue actions (CRM updates, email sequences, invoicing) are queued for explicit approval and audited.

Environment overrides:
- `SAM_REVENUE_OPS_ENABLED` (default: `1`)
- `SAM_REVENUE_DATA_DIR` (default: `sam_data/revenue_ops`)
- `SAM_REVENUE_QUEUE_PATH` (default: `sam_data/revenue_ops/queue.json`)
- `SAM_REVENUE_AUDIT_LOG` (default: `logs/revenue_ops_audit.jsonl`)
- `SAM_REVENUE_AUTOPLANNER_ENABLED` (default: `1`)
- `SAM_REVENUE_AUTOPLANNER_INTERVAL_S` (default: `600`)
- `SAM_REVENUE_AUTOPLANNER_MAX_PENDING` (default: `10`)
- `SAM_REVENUE_AUTOPLANNER_SEQUENCE_ID` (default: unset; uses first available sequence)
- `SAM_REVENUE_SEQUENCE_EXECUTOR_ENABLED` (default: `1`)
- `SAM_REVENUE_SEQUENCE_EXECUTOR_INTERVAL_S` (default: `120`)
- `SAM_REVENUE_DEFAULT_INVOICE_AMOUNT` (default: `0` -> disabled unless set)

## Implementation Spec (Derived)
- `DOCS/README-chatGPT-implementation-spec.md` — strict, implementation-only spec distilled from `DOCS/README-chatGPT-source.md` (no forward-looking prompts).

## Auto Backup (GitHub)


